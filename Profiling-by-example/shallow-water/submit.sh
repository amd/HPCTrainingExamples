#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
source "${ROOT}/env.sh"
exec sbatch -p "${SLURM_PARTITION}" "$@"
