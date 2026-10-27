#!/bin/bash
# Sweep MPI ranks per GPU against GPU_MAX_HW_QUEUES for the HIP Ghost Exchange
# versions, recording run time and the hardware queues each rank actually holds.
#
# All ranks are placed on a single MI300A (ROCR_VISIBLE_DEVICES=0) so that adding
# ranks oversubscribes one device instead of spreading across the node.
#
# Usage: ./sweep.sh [-v "Ver1 Ver6 Ver8"] [-r "1 2 4 8 11 12 16"] [-q "1 2 4"]
#                   [-i 10000] [-I 100] [-o results.csv] [-b]
#   -r   NRANKS values to sweep
#   -q   GPU_MAX_HW_QUEUES values to sweep
#   -b   build the versions first

set -u

VERSIONS="Ver1 Ver6 Ver8"
RANKS="1 2 4 8 11 12 16"
QUEUES="1 2 4"
SIZE=10000
ITERS=100
OUT=results.csv
BUILD=0

while getopts "v:r:q:i:I:o:bh" c; do
   case $c in
      v) VERSIONS=$OPTARG ;;
      r) RANKS=$OPTARG ;;
      q) QUEUES=$OPTARG ;;
      i) SIZE=$OPTARG ;;
      I) ITERS=$OPTARG ;;
      o) OUT=$OPTARG ;;
      b) BUILD=1 ;;
      h) sed -n '2,12p' "$0"; exit 0 ;;
      *) exit 1 ;;
   esac
done

HIPDIR=$(cd "$(dirname "$0")/../GhostExchange_ArrayAssign_HIP" && pwd)

# Ghost Exchange needs an explicit process grid; keep it as square as possible.
grid_for() {
   case $1 in
      1) echo "1 1" ;;  2) echo "2 1" ;;  3) echo "3 1" ;;  4) echo "2 2" ;;
      6) echo "3 2" ;;  8) echo "4 2" ;;  9) echo "3 3" ;; 11) echo "11 1" ;;
     12) echo "4 3" ;; 16) echo "4 4" ;; 20) echo "5 4" ;; 24) echo "6 4" ;;
      *) echo "$1 1" ;;
   esac
}

# Versions relying on OS page migration need XNACK; Ver6 allocates with hipMalloc.
xnack_for() {
   case $1 in
      Ver6) echo 0 ;;
      *)    echo 1 ;;
   esac
}

if [ "$BUILD" = 1 ]; then
   for v in $VERSIONS; do
      echo "building $v"
      ( cd "$HIPDIR/$v" && rm -rf build && mkdir -p build && cd build \
        && cmake .. > cmake.log 2>&1 && make -j 16 > make.log 2>&1 ) \
        || { echo "build failed for $v (see $HIPDIR/$v/build/*.log)"; exit 1; }
   done
fi

# Counts the KFD user-mode queues held by every rank of the running job. The
# driver exposes one directory per queue under /sys/class/kfd/kfd/proc/<pid>.
sample_queues() {
   local total=0 n procs
   procs=$(pgrep -u "$USER" -x GhostExchange 2>/dev/null)
   for p in $procs; do
      n=$(ls "/sys/class/kfd/kfd/proc/$p/queues" 2>/dev/null | wc -l)
      total=$((total + n))
   done
   echo "$total"
}

echo "version,ranks,gpu_max_hw_queues,queues_total,queues_per_rank,total_s,stencil_s,ghost_s" > "$OUT"

for v in $VERSIONS; do
   bin="$HIPDIR/$v/build/GhostExchange"
   [ -x "$bin" ] || { echo "skipping $v: no binary (run with -b)"; continue; }
   for r in $RANKS; do
      read -r px py <<< "$(grid_for "$r")"
      for q in $QUEUES; do
         log=$(mktemp)
         (
            export ROCR_VISIBLE_DEVICES=0
            export GPU_MAX_HW_QUEUES=$q
            export HSA_XNACK=$(xnack_for "$v")
            mpirun -n "$r" --map-by core:OVERSUBSCRIBE --bind-to core \
                   "$bin" -x "$px" -y "$py" -i "$SIZE" -j "$SIZE" \
                   -h 1 -t -c -I "$ITERS" > "$log" 2>&1
         ) &
         runner=$!

         peak=0
         while kill -0 $runner 2>/dev/null; do
            s=$(sample_queues)
            [ "$s" -gt "$peak" ] && peak=$s
            sleep 0.2
         done
         wait $runner

         tot=$(awk '/Total:/{print $2}' "$log" | tail -1)
         sten=$(awk '/Solution Advancement:/{print $3}' "$log" | tail -1)
         ghost=$(awk '/Ghost Cell Update:/{print $4}' "$log" | tail -1)
         [ -z "${tot:-}" ] && tot=FAILED
         per=$(awk -v t="$peak" -v r="$r" 'BEGIN{if(r>0) printf "%.2f", t/r; else print 0}')

         echo "$v,$r,$q,$peak,$per,${tot},${sten:-},${ghost:-}" | tee -a "$OUT"
         rm -f "$log"
      done
   done
done

echo "results in $OUT"
