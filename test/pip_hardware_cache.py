"""Demonstrate that ordinary pip installation skips hardware detection on cache hit."""
import functools, http.server, json, os, pathlib, subprocess, sys, sysconfig, tarfile, tempfile, threading
BACKEND = r'''
import os, pathlib, sysconfig, zipfile
NAME = 'hardware_cache_probe-1.0'
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    tag = 'py3-none-' + sysconfig.get_platform().replace('-', '_').replace('.', '_')
    wheel = NAME + '-' + tag + '.whl'
    hardware = os.environ['PROBE_HARDWARE']
    with open(os.environ['PROBE_BUILD_LOG'], 'a') as f: f.write(hardware + '\n')
    info = NAME + '.dist-info/'
    with zipfile.ZipFile(pathlib.Path(wheel_directory) / wheel, 'w') as z:
        z.writestr('hardware_cache_probe.py', 'HARDWARE = ' + repr(hardware))
        z.writestr(info + 'METADATA', 'Metadata-Version: 2.1\nName: hardware-cache-probe\nVersion: 1.0\n')
        z.writestr(info + 'WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: ' + tag + '\n')
        z.writestr(info + 'RECORD', '')
    return wheel
'''
def main():
    output = pathlib.Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'hardware_cache_probe-1.0'
    source.mkdir()
    (source / 'backend.py').write_text(BACKEND)
    (source / 'pyproject.toml').write_text('[build-system]\nrequires = []\nbuild-backend = "backend"\nbackend-path = ["."]\n')
    index = output / 'simple' / 'hardware-cache-probe'
    index.mkdir(parents=True)
    archive = index / 'hardware_cache_probe-1.0.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar: tar.add(source, arcname=source.name)
    (index / 'index.html').write_text('<a href="' + archive.name + '">' + archive.name + '</a>')
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(output)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    rows = []
    try:
        for label, hardware, cache in [('cold-cpu','cpu','cache'), ('warm-gpu','gpu','cache'), ('cold-gpu','gpu','fresh-cache')]:
            venv = output / label
            subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
            python = venv / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
            env = dict(os.environ, PIP_CACHE_DIR=str(output/cache), PROBE_HARDWARE=hardware, PROBE_BUILD_LOG=str(output/'builds.log'), PIP_DISABLE_PIP_VERSION_CHECK='1')
            command = [str(python), '-m', 'pip', 'install', '--index-url', f'http://127.0.0.1:{server.server_port}/simple', 'hardware-cache-probe']
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            (output/(label+'.log')).write_text(result.stdout + result.stderr)
            detected = subprocess.check_output([str(python), '-c', 'import hardware_cache_probe; print(hardware_cache_probe.HARDWARE)'], text=True).strip()
            rows.append(dict(case=label, visible=hardware, installed=detected, cached='Using cached' in result.stdout))
        assert [r['installed'] for r in rows] == ['cpu', 'cpu', 'gpu'], rows
        assert (output/'builds.log').read_text().splitlines() == ['cpu','gpu']
        (output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(rows, indent=2))
    finally: server.shutdown()
if __name__ == '__main__': main()
