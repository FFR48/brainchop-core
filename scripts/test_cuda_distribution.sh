#!/bin/bash
# Focused manual CI gate. GPU execution is a separate hardware test.
set -eu
cd "$(dirname "$0")/.."
mkdir -p dist
# A poison compiler detects accidental probes as well as accidental compilation.
bc_poison="$PWD/dist/nvcc-forbidden"
cat > "$bc_poison" <<'EOF'
#!/bin/sh
echo 'ERROR: user build invoked nvcc' >&2
touch "$(dirname "$0")/nvcc-was-called"
exit 97
EOF
chmod +x "$bc_poison"
make -C src -j2 CC=gcc METAL=0 OPENCL=0 NVCC="$bc_poison" models
make -C src CC=gcc METAL=0 OPENCL=0 NVCC="$bc_poison" test
make -C src CC=gcc METAL=0 OPENCL=0 NVCC="$bc_poison" VERSION=ci native-package
test ! -e dist/nvcc-was-called
# Verify the archive's contents and execute its extracted copies, not src/.
python3 - <<'PY'
import shutil
from pathlib import Path
archives = list(Path('dist').glob('*.zip')) + list(Path('dist').glob('*.tar.gz'))
assert len(archives) == 1, archives
shutil.unpack_archive(str(archives[0]), 'dist/extracted')
PY
python3 test/cuda_no_driver.py dist/extracted/brainchop-ci-*
rm -f "$bc_poison"
