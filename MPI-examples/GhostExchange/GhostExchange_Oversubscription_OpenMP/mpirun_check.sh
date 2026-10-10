#!/bin/bash
# Launch wrapper that checks MPI ranks per GPU against the amdgpu runlist budgets
# before handing the command to the real launcher.
#
#   mpirun_check [--recommended-settings] [--set-affinity[=gpu|cpu|both]] \
#                [--affinity-order=round-robin|block] [--strict] [--quiet] \
#                <launcher args...>
#
# Symlink or copy it under either name; the wrapped launcher follows the name:
#   ln -s mpirun_check.sh mpirun_check     -> wraps mpirun
#   ln -s mpirun_check.sh srun_check       -> wraps srun
# Override with MPIRUN_CHECK_LAUNCHER=<path>.
#
#   --recommended-settings  apply the fixes instead of only reporting them: set
#                           GPU_MAX_HW_QUEUES to the largest value that keeps the job
#                           inside the budget, and MPICH_GPU_SUPPORT_ENABLED=1 when a
#                           Cray MPICH binary would otherwise refuse device pointers
#   --strict                refuse to launch when the budget is exceeded
#   --quiet                 print nothing when the configuration is already fine
#   --set-affinity[=WHAT]   WHAT is gpu (the default), cpu, or both.
#                           gpu: give each rank one GPU, round-robin by local rank, so
#                           the ranks really are spread over the visible GPUs. Without
#                           it a code calling hipSetDevice(0), or an OpenMP code with no
#                           omp_set_default_device, puts every rank on GPU 0 however
#                           many are visible, and the counts below are wrong.
#                           cpu: also pin each rank to the NUMA node its GPU sits on.
#                           Measured worth nothing on a GPU-bound code; it is here for
#                           host-heavy ones, and it will fight an existing CPU binding.
#   --affinity-order=ORDER  how local ranks map onto the GPUs: round-robin (default,
#                           rank k -> GPU k mod N) or block (consecutive ranks share a
#                           GPU). Worth up to a factor of 1.5 on a halo exchange, and
#                           which one wins depends on the code and the domain
#                           decomposition, so measure both rather than assuming.
#
# Budgets are per GPU family and were measured on AAC6; see README.md.
#   a rank holds min(GPU_MAX_HW_QUEUES, streams it uses) + 1 compute queues
#   gfx942 (MI300A): 24 queues, and <= 6 ranks per partitioned (CPX) GPU
#   gfx90a (MI250):  21 queues; its process limit is unreachable without first
#                    exceeding the queue budget, so only the queue rule applies
# "Streams it uses" counts the MPI library's as well as the application's, so the
# worst case is GPU_MAX_HW_QUEUES+1 whatever the MPI. Open MPI reaches it on its own,
# measured at every GPU_MAX_HW_QUEUES from 1 to 8 and independent of how many streams
# the application uses, so that is what this check assumes. Cray MPICH contributes only
# one stream of its own, for device-buffer collectives, so an application using few
# streams costs less there; the check says so rather than guessing the application's
# stream count. Those Cray numbers assume GPU-aware MPI is on, which needs both
# libmpi_gtl_hsa linked into the executable and MPICH_GPU_SUPPORT_ENABLED=1; the
# check reports either one missing.

set -u

say() { echo "mpirun_check: $*" >&2; }

RECOMMEND=0 STRICT=0 QUIET=0 AFF_GPU=0 AFF_CPU=0 AFF_ORDER=round-robin
while [ $# -gt 0 ]; do
   case $1 in
      --recommended-settings) RECOMMEND=1; shift ;;
      --strict)               STRICT=1;    shift ;;
      --quiet)                QUIET=1;     shift ;;
      --set-affinity)         AFF_GPU=1;   shift ;;
      --affinity-order=*)
         case ${1#*=} in
            round-robin|rr) AFF_ORDER=round-robin ;;
            block|blk)      AFF_ORDER=block ;;
            *) say "unknown --affinity-order value: ${1#*=} (use round-robin or block)"; exit 2 ;;
         esac
         shift ;;
      --set-affinity=*)
         case ${1#*=} in
            gpu)      AFF_GPU=1 ;;
            cpu)      AFF_CPU=1 ;;
            both|gpu,cpu|cpu,gpu) AFF_GPU=1; AFF_CPU=1 ;;
            *) say "unknown --set-affinity value: ${1#*=} (use gpu, cpu or both)"; exit 2 ;;
         esac
         shift ;;
      --help|-h) sed -n '2,36p' "$0"; exit 0 ;;
      *) break ;;
   esac
done

case $(basename "$0") in
   srun_check*) DEFAULT_LAUNCHER=srun ;;
   *)           DEFAULT_LAUNCHER=mpirun ;;
esac
LAUNCHER=${MPIRUN_CHECK_LAUNCHER:-$DEFAULT_LAUNCHER}


# --- how many ranks? ----------------------------------------------------------
# Read the launcher's own rank flags; fall back to Slurm's view of the job. Only
# the first numeric match counts, so a flag of the same name belonging to the
# application ("... bash -c 'cmd'") cannot be mistaken for a rank count. mpirun's
# -c alias for -n is deliberately not recognized, for the same reason.
NRANKS= NNODES= PER_NODE=
set_num() { case $2 in ''|*[!0-9]*) return ;; esac; [ -n "${!1}" ] || printf -v "$1" %s "$2"; }
prev=
for a in "$@"; do
   [ "$a" = "--" ] && break
   case $prev in
      -n|-np|--n|--np|--ntasks) set_num NRANKS   "$a" ;;
      -N|--nodes)               set_num NNODES   "$a" ;;
      --ntasks-per-node)        set_num PER_NODE "$a" ;;
   esac
   case $a in
      --ntasks=*)          set_num NRANKS   "${a#*=}" ;;
      --nodes=*)           set_num NNODES   "${a#*=}" ;;
      --ntasks-per-node=*) set_num PER_NODE "${a#*=}" ;;
   esac
   prev=$a
done
[ -z "$NRANKS" ]   && NRANKS=${SLURM_NTASKS:-}
[ -z "$NNODES" ]   && NNODES=${SLURM_JOB_NUM_NODES:-1}
[ -z "$PER_NODE" ] && PER_NODE=${SLURM_NTASKS_PER_NODE:-}
PER_NODE=${PER_NODE%%(*}                  # Slurm may report "4(x2)"
case $PER_NODE in *[!0-9]*) PER_NODE= ;; esac

case $NNODES in ''|*[!0-9]*|0) NNODES=1 ;; esac
case $NRANKS in *[!0-9]*) NRANKS= ;; esac
RANKS_PER_NODE=
if [ -n "$PER_NODE" ]; then
   RANKS_PER_NODE=$PER_NODE
elif [ -n "$NRANKS" ]; then
   RANKS_PER_NODE=$(( (NRANKS + NNODES - 1) / NNODES ))
fi

# --- how many logical GPUs, and is the device partitioned? --------------------
# A KFD topology node with simd_count > 0 is a logical GPU. Its compute units are
# simd_count/simd_per_cu: a full MI300A reports 228, a CPX slice 38.
KFD=/sys/class/kfd/kfd/topology/nodes
NGPU= CU_PER_GPU= GFXVER=
if [ -d $KFD ]; then
   read -r NGPU CU_PER_GPU GFXVER <<EOF
$(awk '/^simd_count/{s=$2} /^simd_per_cu/{p=$2;
        if (s>0 && p>0) {n++; cu=s/p}; s=0}
      /^gfx_target_version/{if ($2>0) v=$2}
      END{print n+0, cu+0, v+0}' $KFD/*/properties)
EOF
fi

# Each family has its own runlist budget and its own notion of partitioning.
case ${GFXVER:-0} in
   90402) FAMILY="gfx942"; QUEUE_BUDGET=24; FULL_CU=228; SDMA_RANKS=8 ;;
   90010) FAMILY="gfx90a"; QUEUE_BUDGET=21; FULL_CU=104; SDMA_RANKS= ;;
   *)     FAMILY=;         QUEUE_BUDGET=24; FULL_CU=;     SDMA_RANKS= ;;
esac
# An explicit mask or a Slurm allocation narrows what the ranks can actually see.
VIS=${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-}}
if [ -n "$VIS" ]; then
   NGPU=$(awk -F, '{print NF}' <<< "$VIS")
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
   NGPU=$SLURM_GPUS_ON_NODE
fi
case ${NGPU:-0} in ''|*[!0-9]*|0) NGPU= ;; esac

# Only gfx942 has compute partitions; on other families rocm-smi reports nothing and
# a small compute-unit count means a small GPU, not a slice.
PARTITIONED=0
MODE=$(rocm-smi --showcomputepartition 2>/dev/null | awk '/^GPU\[/ && /Compute Partition/{print $NF; exit}')
case ${MODE:-} in
   CPX|TPX|DPX) PARTITIONED=1 ;;
   SPX)         PARTITIONED=0 ;;
   *) MODE=
      [ -n "$FULL_CU" ] && [ "${CU_PER_GPU:-0}" -gt 0 ] && [ "$FULL_CU" = 228 ] \
         && [ "$CU_PER_GPU" -lt "$FULL_CU" ] && PARTITIONED=1 ;;
esac

if [ -z "$RANKS_PER_NODE" ] || [ -z "$NGPU" ]; then
   say "cannot determine ranks (${RANKS_PER_NODE:-?}) or GPUs (${NGPU:-?}); not checking"
   exec $LAUNCHER "$@"
fi

RANKS_PER_GPU=$(( (RANKS_PER_NODE + NGPU - 1) / NGPU ))

# --- which MPI? -----------------------------------------------------------------
# The application's NEEDED entries are decisive; a loaded module says nothing about
# what the binary was linked against, so it is only the fallback. Cray MPICH names its
# library after the compiler that built it (libmpi_cray, libmpi_amd, libmpi_gnu), so
# match those rather than libmpi_* in general: Open MPI's own libmpi_cxx and
# libmpi_mpifh would otherwise look the same.
MPI_STACK= IS_CRAY=0 HAS_GTL=0 APPBIN= OFFLOAD= EXTRA_Q=0
for a in "$@"; do
   case $a in -*) continue ;; esac
   [ -f "$a" ] && [ -x "$a" ] && { APPBIN=$a; break; }
done
if [ -n "$APPBIN" ] && command -v objdump >/dev/null 2>&1; then
   NEEDED=" $(objdump -p "$APPBIN" 2>/dev/null | awk '/NEEDED/{print $2}' | tr '\n' ' ')"
   case $NEEDED in
      *libmpi_cray*|*libmpi_amd*|*libmpi_gnu*|*libmpi_gtl_*|*libmpich.so*)
         MPI_STACK="Cray MPICH"; IS_CRAY=1 ;;
      *libmpi.so.4*|*libmpi_cxx*|*libmpi_mpifh*|*libopen-rte*|*libopen-pal*)
         MPI_STACK="Open MPI" ;;
   esac
   case $NEEDED in *libmpi_gtl_hsa*) HAS_GTL=1 ;; esac
   # The OpenMP offload runtime holds one stream of its own on top of whatever the
   # application and MPI use, so a rank costs one queue more than the HIP equivalent.
   case $NEEDED in *libomptarget*) OFFLOAD="OpenMP offload"; EXTRA_Q=1 ;; esac
fi
if [ -z "$MPI_STACK" ]; then
   if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
      MPI_STACK="Cray MPICH, assumed from the loaded module"; IS_CRAY=1
   else MPI_STACK="Open MPI, assumed"; fi
fi

# --- the budgets --------------------------------------------------------------
PROC_BUDGET=6                       # partitioned gfx942 only
Q=${GPU_MAX_HW_QUEUES:-4}
case $Q in ''|*[!0-9]*) Q=4 ;; esac

qpr() { echo $(( $1 + 1 + EXTRA_Q )); }
QPR=$(qpr "$Q")
USED=$(( RANKS_PER_GPU * QPR ))
# Cray MPICH supplies one stream of its own and only when a device-buffer collective
# runs, so its demand spans a range the launcher cannot resolve: 2 per rank for a
# single-stream code up to GPU_MAX_HW_QUEUES+1. Warn only when even the low end is
# over budget, and flag the uncertain middle separately.
CRAY_LOW=$(( 2 + EXTRA_Q ))
LOW_USED=$(( RANKS_PER_GPU * CRAY_LOW ))

# Largest GPU_MAX_HW_QUEUES that still fits, and the rank ceiling it implies. Under
# Cray MPICH the demand does not depend on it, so there is nothing to recommend.
BEST=0
for c in 4 3 2 1; do
   [ $(( RANKS_PER_GPU * $(qpr $c) )) -le $QUEUE_BUDGET ] && { BEST=$c; break; }
done
# Most ranks this GPU can take under any setting, and the setting that gets there.
MAX_RANKS=$(( QUEUE_BUDGET / $(qpr 1) ))
[ "$PARTITIONED" = 1 ] && [ "$MAX_RANKS" -gt "$PROC_BUDGET" ] && MAX_RANKS=$PROC_BUDGET
MAX_RANKS_Q=1
for c in 4 3 2 1; do
   [ $(( MAX_RANKS * $(qpr $c) )) -le $QUEUE_BUDGET ] && { MAX_RANKS_Q=$c; break; }
done

OVER_QUEUES=0 MAYBE_QUEUES=0
if [ "$IS_CRAY" = 1 ]; then
   [ "$LOW_USED" -gt "$QUEUE_BUDGET" ] && OVER_QUEUES=1
   [ "$OVER_QUEUES" = 0 ] && [ "$USED" -gt "$QUEUE_BUDGET" ] && MAYBE_QUEUES=1
else
   [ "$USED" -gt "$QUEUE_BUDGET" ] && OVER_QUEUES=1
fi
OVER_PROCS=0
[ "$PARTITIONED" = 1 ] && [ "$RANKS_PER_GPU" -gt "$PROC_BUDGET" ] && OVER_PROCS=1

# --- report -------------------------------------------------------------------
RANKS_DESC="$RANKS_PER_NODE rank(s) over $NGPU logical GPU(s) = $RANKS_PER_GPU per GPU"
DESC="$RANKS_DESC"
DESC="$DESC, GPU_MAX_HW_QUEUES=$Q -> up to $QPR queues each, $USED of $QUEUE_BUDGET"
[ -n "$OFFLOAD" ] && DESC="$DESC ($OFFLOAD)"
TAG="${FAMILY:-unknown GPU}"
[ -n "${MODE:-}" ] && TAG="$TAG $MODE"
DESC="$DESC [$TAG]"
[ -n "$FAMILY" ] || say "note: unrecognized GPU family, assuming a $QUEUE_BUDGET-queue budget"

if [ "$OVER_QUEUES" = 0 ] && [ "$OVER_PROCS" = 0 ]; then
   if [ "$MAYBE_QUEUES" = 1 ]; then
      say "note: $RANKS_DESC, $LOW_USED to $USED queues of $QUEUE_BUDGET [$TAG]"
      say "  under Cray MPICH the cost depends on how many streams the code uses: $CRAY_LOW per rank"
      say "  for a single-stream code, which fits $(( QUEUE_BUDGET / CRAY_LOW )), rising to $QPR per rank for several"
   else
      [ "$QUIET" = 1 ] || say "ok: $DESC"
   fi
else
   say "WARNING: the amdgpu runlist will be oversubscribed and every rank slows down."
   say "  $DESC"
   [ "$OVER_QUEUES" = 1 ] && \
      say "  over the $QUEUE_BUDGET-queue budget: expect \"Runlist is getting oversubscribed due to too many queues\" in dmesg"
   [ "$OVER_PROCS" = 1 ] && \
      say "  over $PROC_BUDGET processes on a partitioned GPU: expect \"... due to too many processes\" in dmesg"
   if [ "$OVER_PROCS" = 1 ] || [ "$BEST" = 0 ]; then
      if [ "$OVER_PROCS" = 1 ]; then
         say "  fix: the process limit cannot be raised with GPU_MAX_HW_QUEUES"
      else
         say "  fix: no GPU_MAX_HW_QUEUES value fits $RANKS_PER_GPU ranks on one GPU"
      fi
      say "       at most $MAX_RANKS rank(s) per GPU with GPU_MAX_HW_QUEUES=$MAX_RANKS_Q, so $(( MAX_RANKS * NGPU )) rank(s) over the $NGPU visible GPU(s)"
   else
      say "  fix: GPU_MAX_HW_QUEUES=$BEST keeps this rank count inside the budget"
      [ "$RECOMMEND" = 1 ] || say "  or re-run with --recommended-settings to apply it"
   fi
   [ "$IS_CRAY" = 1 ] && \
      say "  note: over even at Cray MPICH's best case of $CRAY_LOW queues per rank ($LOW_USED of $QUEUE_BUDGET)"
   if [ "$STRICT" = 1 ]; then
      say "refusing to launch (--strict)"
      exit 1
   fi
fi

# --- where do the ranks actually land? -------------------------------------------
# Everything above divides ranks by the number of visible GPUs, which is only true if
# something assigns them. Neither language does by default: a bare hipSetDevice(0), or
# an OpenMP code with no omp_set_default_device, puts every rank on GPU 0.
if [ "$NGPU" -gt 1 ] && [ "$AFF_GPU" = 0 ]; then
   if [ -n "$OFFLOAD" ]; then
      CULPRIT="A code with no omp_set_default_device"
   else
      CULPRIT="A code that calls hipSetDevice(0)"
   fi
   say "note: this assumes the $RANKS_PER_NODE rank(s) are spread over all $NGPU GPUs. $CULPRIT"
   say "  puts them all on one, which would be $(( RANKS_PER_NODE * QPR )) queues of $QUEUE_BUDGET."
   say "  pass --set-affinity to make the assumption true."
fi

# --- the separate SDMA pool ------------------------------------------------------
# Measured on gfx942: 16 SDMA queues per GPU and 2 per rank, so from the ninth rank
# the driver logs "No more SDMA queue to allocate" and those ranks fall back to a
# slower copy path. It is a soft limit, unlike the compute-queue budget.
if [ -n "${SDMA_RANKS:-}" ] && [ "$RANKS_PER_GPU" -gt "$SDMA_RANKS" ]; then
   say "note: past $SDMA_RANKS rank(s) per GPU the 16 SDMA queues run out, so $(( RANKS_PER_GPU - SDMA_RANKS )) rank(s) will"
   say "  log \"No more SDMA queue to allocate\" and fall back to a slower copy path"
fi

# --- is GPU-aware MPI actually turned on? ---------------------------------------
# Cray MPICH needs both halves: the GPU transport layer linked into the executable
# (a craype-accel-amd-* module makes the wrappers do it) and MPICH_GPU_SUPPORT_ENABLED=1
# at run time. It is not set by default. With the variable set but no transport layer
# linked, MPI_Init aborts outright rather than falling back.
GPU_AWARE_FIX=0
if [ "$IS_CRAY" = 1 ]; then
   GPU_ON=0
   case ${MPICH_GPU_SUPPORT_ENABLED:-} in 1) GPU_ON=1 ;; esac
   if [ -n "$APPBIN" ] && [ "$HAS_GTL" = 0 ]; then
      say "WARNING: $APPBIN does not link libmpi_gtl_hsa, so GPU-aware MPI is unavailable."
      say "  rebuild with a craype-accel-amd-gfx* module loaded so the wrappers add it"
      [ "$GPU_ON" = 1 ] &&          say "  MPICH_GPU_SUPPORT_ENABLED=1 without it aborts: \"GPU_SUPPORT_ENABLED is requested, but GTL library is not linked\""
   elif [ "$GPU_ON" = 0 ]; then
      say "WARNING: MPICH_GPU_SUPPORT_ENABLED is not set, so Cray MPICH will not accept device pointers."
      say "  fix: export MPICH_GPU_SUPPORT_ENABLED=1"
      [ "$RECOMMEND" = 1 ] || say "  or re-run with --recommended-settings to apply it"
      GPU_AWARE_FIX=1
   fi
fi

if [ "$RECOMMEND" = 1 ]; then
   if [ "$GPU_AWARE_FIX" = 1 ]; then
      say "setting MPICH_GPU_SUPPORT_ENABLED=1"
      export MPICH_GPU_SUPPORT_ENABLED=1
   fi
   if [ "$BEST" != 0 ] && [ "$OVER_PROCS" = 0 ]; then
      if [ "$BEST" != "$Q" ]; then
         say "setting GPU_MAX_HW_QUEUES=$BEST"
         export GPU_MAX_HW_QUEUES=$BEST
      fi
   else
      say "no GPU_MAX_HW_QUEUES value fits this rank count; leaving it at $Q"
   fi
fi

if [ "$AFF_GPU" = 1 ] || [ "$AFF_CPU" = 1 ]; then
   # Build the shim as text and insert it ahead of the executable rather than writing
   # a file. The device is chosen per rank from whichever local-rank variable the
   # launcher sets, so this works under both mpirun and srun.
   # round-robin spreads neighbouring ranks over devices, block keeps them together.
   # Which is faster is code- and decomposition-dependent; see the README.
   if [ "$AFF_ORDER" = block ]; then
      PICK='n=${OMPI_COMM_WORLD_LOCAL_SIZE:-'"${RANKS_PER_NODE:-0}"'}
per=$(( (n + '"$NGPU"' - 1) / '"$NGPU"' ))
g=$(( per > 0 ? r / per : 0 ))
if [ $g -ge '"$NGPU"' ]; then g=$(( '"$NGPU"' - 1 )); fi
'
   else
      PICK='g=$(( r % '"$NGPU"' ))
'
   fi
   SHIM='r=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${MPI_LOCALRANKID:-0}}}
'"$PICK"
   [ "$AFF_GPU" = 1 ] && SHIM="$SHIM"'export ROCR_VISIBLE_DEVICES=$g
'
   if [ "$AFF_CPU" = 1 ]; then
      # GPU k sits on one NUMA node; read the map rather than assuming k -> node k.
      NUMA_MAP=$(for d in /sys/class/drm/renderD*/device/numa_node; do
                    [ -e "$d" ] && cat "$d"; done | tr '\n' ' ')
      case " $NUMA_MAP" in
         *" -1"*|" ") say "--set-affinity=cpu: no NUMA node reported for the GPUs; skipping CPU pinning"
                      AFF_CPU=0 ;;
         *) SHIM="$SHIM"'m=('"$NUMA_MAP"'); n=${m[$g]}
cpus=$(cat /sys/devices/system/node/node$n/cpulist)
export OMP_PROC_BIND=close OMP_PLACES=cores
exec taskset -c "$cpus" "$@"
' ;;
      esac
   fi
   [ "$AFF_CPU" = 1 ] || SHIM="$SHIM"'exec "$@"'
   ARGS=(); INSERTED=0
   for a in "$@"; do
      if [ "$INSERTED" = 0 ] && [ -n "$APPBIN" ] && [ "$a" = "$APPBIN" ]; then
         ARGS+=(bash -c "$SHIM" _); INSERTED=1
      fi
      ARGS+=("$a")
   done
   if [ "$INSERTED" = 1 ]; then
      WHAT="one GPU each, $AFF_ORDER over the $NGPU visible"
      [ "$AFF_GPU" = 0 ] && WHAT="CPU pinning only"
      [ "$AFF_GPU" = 1 ] && [ "$AFF_CPU" = 1 ] && WHAT="$WHAT, each pinned to its GPU's NUMA node"
      [ "$AFF_GPU" = 0 ] && [ "$AFF_CPU" = 1 ] && WHAT="each rank pinned to its GPU's NUMA node"
      say "setting affinity: $WHAT"
      exec $LAUNCHER "${ARGS[@]}"
   fi
   say "--set-affinity: no executable found in the command line; launching unchanged"
fi

exec $LAUNCHER "$@"
