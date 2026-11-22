"""Launch the environment's own native tools, never an unrelated PATH binary."""
from pathlib import Path
import subprocess
import os
import sys
from .native import find_executable

def main():
    name=Path(sys.argv[0]).name.removesuffix('.exe').removesuffix('-script.py')
    model=name.removeprefix('brainchop-')
    try:
        executable=find_executable(model,adapt=not any(a in ('-h','--help','-v','--version') for a in sys.argv[1:]))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ValueError,OSError,RuntimeError) as error:
        print('brainchop: '+str(error),file=sys.stderr)
        raise SystemExit(2)
    if os.name!='nt':
        os.execv(executable,[name,*sys.argv[1:]])
    try:
        result=subprocess.run([name,*sys.argv[1:]],executable=executable)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(result.returncode if result.returncode>=0 else 128-result.returncode)
