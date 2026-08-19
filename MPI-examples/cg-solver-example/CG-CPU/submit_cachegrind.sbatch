#!/bin/bash
#SBATCH --job-name=cg_cachegrind
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=cachegrind_%j.out
#SBATCH --error=cachegrind_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

SEED=${CG_SEED:-12345}
OUT=cachegrind_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
BIG=../CG-GPU/src/poisson80.pm            # 512,000-row 3D Poisson (larger case)

echo "=== toolchain ==="
command -v mpicxx valgrind cg_annotate callgrind_annotate
valgrind --version

echo "=== build (cachegrind.md: -O2 -g for clean annotation) ==="
make clean >/dev/null 2>&1
make CXXFLAGS="-O2 -g -std=c++17" || { echo BUILD FAILED; exit 1; }

run_cachegrind () {   # $1=label  $2=matrix
  local label=$1 mat=$2
  echo; echo "############ cachegrind: $label  ($mat) ############"
  /usr/bin/time -v valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
      --cachegrind-out-file="$OUT/cg.$label.out" \
      ./cg_cpu "$mat" "$SEED" > "$OUT/run.$label.txt" 2>&1
  echo "--- solver + cachegrind summary ($label) ---"
  grep -E "seed|iterations|2-norm|CG solve|refs|misses|miss rate|Branches|Mispred" \
      "$OUT/run.$label.txt"
  echo "--- cg_annotate: per-function D1/LLd misses ($label) ---"
  cg_annotate "$OUT/cg.$label.out" > "$OUT/annotate.$label.func.txt" 2>&1
  sed -n '1,60p' "$OUT/annotate.$label.func.txt"
  echo "--- cg_annotate: cg.cpp line-by-line (top of file) ($label) ---"
  cg_annotate "$OUT/cg.$label.out" src/cg.cpp > "$OUT/annotate.$label.cgcpp.txt" 2>&1
}

# current example size
run_cachegrind dubcova2 src/Dubcova2.pm
# larger case (only if present)
if [ -f "$BIG" ]; then
  run_cachegrind poisson80 "$BIG"
else
  echo "[skip] larger matrix $BIG not found"
fi

echo; echo "############ callgrind (call graph) : dubcova2 ############"
valgrind --tool=callgrind --cache-sim=yes \
    --callgrind-out-file="$OUT/callgrind.dubcova2.out" \
    ./cg_cpu src/Dubcova2.pm "$SEED" > "$OUT/run.callgrind.txt" 2>&1
grep -E "seed|iterations|2-norm|CG solve" "$OUT/run.callgrind.txt" || true
callgrind_annotate --tree=both "$OUT/callgrind.dubcova2.out" \
    > "$OUT/callgrind.annotate.txt" 2>&1
echo "--- callgrind_annotate (top functions) ---"
sed -n '1,50p' "$OUT/callgrind.annotate.txt"

echo; echo "############ render call-graph PNG (headless) ############"
python3 -m pip install --user --break-system-packages --quiet matplotlib 2>&1 | tail -2
python3 render_callgraph.py "$OUT/callgrind.dubcova2.out" "$OUT/callgraph.dubcova2.png" 14 \
    2>&1 | tail -5
ls -la "$OUT"/*.png 2>/dev/null || echo "no PNG produced"

date; echo "DONE -> $OUT/"
