"""Hardware-specific native runtime cache. Standard library only; never downloads."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile


def abi():
    machine=platform.machine().lower()
    if machine=='amd64': machine='x86_64'
    if machine=='aarch64': machine='arm64'
    system=platform.system()
    return dict(system=system,machine=machine,libc=list(platform.libc_ver()) if system=='Linux' else [],
                os_version=platform.mac_ver()[0] if system=='Darwin' else platform.win32_ver()[1] if system=='Windows' else '')


def signature(hardware):
    devices={json.dumps({k:d[k] for k in ('backend','arch','driver') if k in d},sort_keys=True) for d in hardware['devices']}
    return dict(abi=abi(),cpu=hardware['cpu'],devices=[json.loads(d) for d in sorted(devices)])


def cuda_choice(architectures, arch):
    """Nearest validated cubin at or below arch in the same major; no PTX JIT."""
    return max((a for a in architectures if a//10==arch//10 and a<=arch),default=None)


def parse_probe(text):
    """Parse hardware-probe output into (cpu, devices); cpu is None if unreported."""
    devices=[]; cpu=None
    for line in text.splitlines():
        f=line.split('\t')
        if f[:2]==['cpu','avx2-fma'] and len(f)==3: cpu='avx2-fma' if f[2]=='1' else 'baseline'
        elif len(f)==4 and f[0]=='cuda': devices.append(dict(backend='cuda',index=int(f[1]),arch=int(f[2]),driver=int(f[3])))
        elif len(f)==3 and f[0] in ('hip','opencl'): devices.append(dict(backend=f[0],index=int(f[1]),arch=f[2]))
    # ponytail: every Apple Silicon Mac has Metal; no separate Metal probe.
    if platform.system()=='Darwin' and platform.machine()=='arm64': devices.append(dict(backend='metal',index=0,arch='apple'))
    return cpu, devices


def probe(package):
    suffix='.exe' if os.name=='nt' else ''
    binary=package/'bin'/('hardware-probe'+suffix)
    output=[]
    backends=('cpu-only','cuda','hip','opencl') if sys.platform.startswith('linux') else ('cpu-only','cuda','opencl') if os.name=='nt' else ('cpu-only',)
    for backend in backends:
        try:
            result=subprocess.run([str(binary),'--'+backend],capture_output=True,text=True,timeout=30)
        except (OSError,subprocess.TimeoutExpired):
            continue
        if not result.returncode: output.append(result.stdout)
    cpu,devices=parse_probe('\n'.join(output))
    if abi()['machine']=='x86_64' and cpu is None:
        raise RuntimeError('the installed hardware probe cannot run on this host')
    return signature(dict(cpu=cpu or 'baseline',devices=devices))


def _parse_version(value):
    if not isinstance(value, str) or not value:
        return None
    parts = value.split('.')
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        return None


def abi_compatible(built_abi, current_abi, portable=False):
    if built_abi == current_abi:
        return True
    if not portable:
        return False
    if built_abi.get('system') != current_abi.get('system') or built_abi.get('machine') != current_abi.get('machine'):
        return False
    system = built_abi['system']
    if system == 'Windows':
        # UCRT64 builds link only Windows 10/11 system DLLs (kernel32/ucrtbase),
        # so a wheel built on Windows Server 2022 (10.0.20348) runs cleanly on
        # Windows 10 and Windows 11 (10.0.*) without requiring MSYS2.
        if built_abi.get('libc', []) != current_abi.get('libc', []):
            return False
        bv = _parse_version(built_abi.get('os_version', ''))
        cv = _parse_version(current_abi.get('os_version', ''))
        return bv is not None and cv is not None and len(bv) >= 2 and len(cv) >= 2 and bv[:2] == cv[:2]
    if system == 'Darwin':
        # Executables target MACOSX_DEPLOYMENT_TARGET=13.2 and link only system libs.
        if built_abi.get('libc', []) != current_abi.get('libc', []):
            return False
        bv = _parse_version(built_abi.get('os_version', ''))
        cv = _parse_version(current_abi.get('os_version', ''))
        return bv is not None and cv is not None and cv >= (13, 2)
    if system == 'Linux':
        if built_abi.get('os_version', '') != current_abi.get('os_version', ''):
            return False
        bl = built_abi.get('libc', [])
        cl = current_abi.get('libc', [])
        if len(bl) == 2 and len(cl) == 2 and bl[0] == cl[0] == 'glibc':
            bv, cv = _parse_version(bl[1]), _parse_version(cl[1])
            return bv is not None and cv is not None and cv >= bv
    return False


def compatible(manifest, current):
    built=manifest['runtime_environment']
    fixed=manifest['build_policy']['target']!='auto'
    if not abi_compatible(built['abi'],current['abi'],portable=fixed): return False
    if manifest['cpu_target']=='avx2-fma' and current['cpu']!='avx2-fma': return False
    # Explicit build targets remain fixed. Baseline CPU code can travel to
    # CPUs lacking AVX2, and driver disappearance is handled by native code.
    if fixed: return True
    if built==current: return True
    policy=manifest['artifact_policy']
    cuda=set()
    hip=set()
    for name,record in manifest['artifacts'].items():
        if name.startswith('cuda_modules/'): cuda.update(record['architectures'])
        if name.startswith('hip_artifacts/'): hip.add(record['architecture'])
    for device in current['devices']:
        backend=device['backend']
        if backend=='cuda' and device['driver']>=policy['cuda_minimum_driver']:
            choice=cuda_choice(policy['cuda_architectures'],device['arch'])
            if choice is not None and choice not in cuda: return False
        elif backend=='hip' and device['arch'] in policy['hip_architectures'] and device['arch'] not in hip: return False
        elif backend=='metal' and 'METAL=1' not in manifest['options']: return False
        elif backend=='opencl' and 'OPENCL=1' not in manifest['options']: return False
    return True


def unpack_source(archive, destination):
    """Extract only regular members beneath the single distribution root."""
    roots=set()
    with tarfile.open(archive,'r:gz') as source:
        members=source.getmembers()
        for member in members:
            parts=PurePosixPath(member.name).parts
            if not member.isfile() or len(parts)<2 or any(p in ('..','') for p in parts) or member.name.startswith('/') or '\\' in member.name or ':' in member.name:
                raise ValueError('invalid member in bundled source archive')
            roots.add(parts[0])
        if len(roots)!=1: raise ValueError('invalid bundled source root')
        for member in members:
            path=destination.joinpath(*PurePosixPath(member.name).parts)
            path.parent.mkdir(parents=True,exist_ok=True)
            with source.extractfile(member) as reader, path.open('wb') as writer: shutil.copyfileobj(reader,writer)
    return destination/next(iter(roots))


def cache_root():
    if os.name=='nt': base=Path(os.environ.get('LOCALAPPDATA',Path.home()/'AppData/Local'))
    elif sys.platform=='darwin': base=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'Library/Caches'))
    else: base=Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))
    return base/'brainchop/runtimes'


@contextmanager
def lock(path):
    # Advisory OS locks release on process death. Never delete the lock file:
    # doing so could split waiters between distinct inodes on shared storage.
    with path.open('a+b') as handle:
        if handle.seek(0,os.SEEK_END)==0:
            handle.write(b'0'); handle.flush()
        deadline=time.monotonic()+600
        while True:
            try:
                if os.name=='nt':
                    import msvcrt
                    handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic()>deadline: raise RuntimeError('timed out waiting for a hardware runtime build')
                time.sleep(.1)
        try: yield
        finally:
            if os.name=='nt':
                handle.seek(0); msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def cached(path, capsule_hash, current, policy):
    try:
        manifest=json.loads((path/'build.json').read_text())
        if manifest['build_policy']!=policy or manifest['source_capsule']['sha256']!=capsule_hash or not compatible(manifest,current): return None
        for name,record in manifest['executables'].items():
            binary=path/'bin'/name
            if not binary.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest()!=record['sha256']: return None
        return manifest
    except (OSError,ValueError,KeyError): return None


def select(package, manifest):
    if 'source_capsule' not in manifest: return package,manifest  # older local development installs
    # Prebuilt fixed-target wheels do not depend on host GPU discovery; skip
    # spawning GPU vendor probes when the host ABI (and CPU ISA, if required)
    # already satisfies the wheel.
    if (manifest.get('build_policy',{}).get('target','auto')!='auto'
            and abi_compatible(manifest['runtime_environment']['abi'],abi(),portable=True)):
        if manifest.get('cpu_target')=='baseline':
            return package,manifest
        if manifest.get('cpu_target')=='avx2-fma':
            suffix='.exe' if os.name=='nt' else ''
            binary=package/'bin'/('hardware-probe'+suffix)
            try:
                result=subprocess.run([str(binary),'--cpu-only'],capture_output=True,text=True,timeout=10)
                if result.returncode==0 and parse_probe(result.stdout)[0]=='avx2-fma':
                    return package,manifest
            except (OSError,subprocess.TimeoutExpired):
                pass
    capsule=package/manifest['source_capsule']['file']
    digest=manifest['source_capsule']['sha256']
    try:
        current=probe(package)
    except RuntimeError:
        # A cached executable may require a newer libc. Compile a fresh probe
        # from the self-contained source before choosing a native runtime.
        current=None
    if current is not None and compatible(manifest,current): return package,manifest
    if hashlib.sha256(capsule.read_bytes()).hexdigest()!=digest: raise ValueError('bundled source integrity check failed')
    root=cache_root()
    if current is not None:
        key=hashlib.sha256(json.dumps(dict(source=digest,hardware=current,policy=manifest['build_policy']),sort_keys=True).encode()).hexdigest()
        record=cached(root/key,digest,current,manifest['build_policy'])
        if record is not None: return root/key,record
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.build-',dir=root) as temp:
        work=Path(temp)
        source=None
        if current is None:
            source=unpack_source(capsule,work)
            script="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from brainchop_hardware import detect,tool_environment;from brainchop_cli.adaptive import signature;print(json.dumps(signature(detect(Path(sys.argv[2]),Path(sys.argv[3]),tool_environment()))))"
            result=subprocess.run([sys.executable,'-I','-c',script,str(source/'packaging'),str(source),str(work)],capture_output=True,text=True,timeout=120)
            if result.returncode: raise RuntimeError('cannot compile a hardware probe; a native C compiler and make are required: '+result.stderr)
            current=json.loads(result.stdout.splitlines()[-1])
        key=hashlib.sha256(json.dumps(dict(source=digest,hardware=current,policy=manifest['build_policy']),sort_keys=True).encode()).hexdigest()
        destination=root/key
        record=cached(destination,digest,current,manifest['build_policy'])
        if record is not None: return destination,record
        with lock(root/(key+'.lock')):
            record=cached(destination,digest,current,manifest['build_policy'])
            if record is not None: return destination,record
            print('brainchop: building a native runtime for this hardware; subsequent runs reuse it.',file=sys.stderr,flush=True)
            if source is None: source=unpack_source(capsule,work)
            output=work/'wheels'; output.mkdir()
            log=root/(key+'.log')
            settings=manifest['build_policy']['settings']
            script="import sys,json;sys.path.insert(0,sys.argv[1]);import brainchop_build;brainchop_build.build_wheel(sys.argv[2],json.loads(sys.argv[3]))"
            try:
                with log.open('w') as stream:
                    result=subprocess.run([sys.executable,'-I','-c',script,str(source/'packaging'),str(output),json.dumps(settings)],stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            except subprocess.TimeoutExpired as error: raise RuntimeError('native rebuild timed out; see '+str(log)) from error
            if result.returncode:
                raise RuntimeError('native rebuild failed; an ordinary C compiler, make and zlib development files are required (no GPU compiler). See '+str(log))
            wheels=list(output.glob('*.whl'))
            if len(wheels)!=1: raise RuntimeError('native rebuild produced an invalid wheel')
            staged=work/'runtime'; staged.mkdir()
            with zipfile.ZipFile(wheels[0]) as wheel:
                for entry in wheel.infolist():
                    parts=PurePosixPath(entry.filename).parts
                    if not parts or parts[0]!='brainchop_cli': continue
                    if len(parts)<2 or any(p=='..' for p in parts): raise ValueError('invalid rebuilt wheel path')
                    if parts[1]!='bin' and parts[1]!='build.json': continue
                    path=staged.joinpath(*parts[1:]); path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(wheel.read(entry)); path.chmod(0o755 if parts[1]=='bin' else 0o644)
            record=cached(staged,digest,current,manifest['build_policy'])
            if record is None: raise RuntimeError('rebuilt runtime does not match this hardware/source; see '+str(log))
            if destination.exists(): shutil.rmtree(destination)
            os.replace(staged,destination)
            return destination,record
