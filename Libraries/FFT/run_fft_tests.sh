#!/bin/bash
# Single-APU FFT build + smoke tests. Run under a single-task srun allocation.
set -e
if ! command -v module >/dev/null 2>&1; then
  for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
           /usr/share/lmod/lmod/init/bash /usr/share/modules/init/bash; do
    [ -f "$f" ] && source "$f" && break
  done
fi
module load rocm/7.2.4
export HSA_XNACK=1
export OFFLOAD_ARCH=gfx942

echo "=== node: $(hostname) ==="
echo "=== build ==="
hipcc -O3 --offload-arch=gfx942 rocfft_c2c.hip -lrocfft -o rocfft_c2c
hipcc -O3 --offload-arch=gfx942 rocfft_3d.hip  -lrocfft -o rocfft_3d
hipcc -O3 --offload-arch=gfx942 hipfft_c2c.hip -lhipfft -o hipfft_c2c

echo "=== rocFFT 1D batched ==="
./rocfft_c2c 1048576 1
./rocfft_c2c 65536 64
echo "=== rocFFT 3D ==="
./rocfft_3d 128
./rocfft_3d 256
echo "=== hipFFT 1D batched (portable API) ==="
./hipfft_c2c 1048576 1
