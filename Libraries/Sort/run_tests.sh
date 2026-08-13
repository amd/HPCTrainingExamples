#!/bin/bash
# Single-APU build + smoke tests. Run under srun on an MI300A compute node.
set -e
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
           /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash; do
    [ -f "$f" ] && source "$f" && break
  done
fi
module load rocm
export HSA_XNACK=1
export OFFLOAD_ARCH=gfx942

echo "=== node: $(hostname) ==="
rocminfo | grep -m1 'gfx' || true
echo

echo "=== build (zip_sort, name_sort) + names.txt ==="
make zip_sort name_sort names.txt

echo
echo "=== zip_sort correctness (small) ==="
./zip_sort 1024
echo "=== zip_sort 1M ==="
./zip_sort 1000000
echo "=== zip_sort 10M ==="
./zip_sort 10000000

echo
echo "=== name_sort on names.txt ==="
./name_sort names.txt
