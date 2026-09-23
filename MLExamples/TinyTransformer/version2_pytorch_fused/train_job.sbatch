#!/bin/bash
#SBATCH --job-name=tinyllama-train
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#
# Batch submission for tiny_llama_v2.py (quality-tracking mode) on AAC6. Submit with:
#   sbatch train_job.sbatch
#
# Override any config below without editing the file, e.g. a longer/bigger run:
#   sbatch --export=ALL,NUM_STEPS=100000,WIKI_NUM_DOCS=50000 train_job.sbatch

set -eo pipefail  # no -u: module/Lmod functions reference unset vars internally
cd "$SLURM_SUBMIT_DIR"  # sbatch copies the script elsewhere; run from the submit dir
mkdir -p slurm_logs

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
amd-smi list 2>&1 || true

# sbatch runs as a non-login shell, so the module must be loaded explicitly.
module load pytorch/2.12.0

# --- Model / data config (production-size text pool; vocab size is fixed by the pretrained tokenizer) ---
HIDDEN_DIM=${HIDDEN_DIM:-768}
NUM_LAYERS=${NUM_LAYERS:-12}
NUM_HEADS=${NUM_HEADS:-12}
SEQ_LEN=${SEQ_LEN:-128}
WIKI_NUM_DOCS=${WIKI_NUM_DOCS:-100000}

# --- Training config ---
NUM_STEPS=${NUM_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_BATCHES=${EVAL_BATCHES:-15}
GENERATE_EVERY=${GENERATE_EVERY:-2000}
OUTPUT_DIR=${OUTPUT_DIR:-./quality_runs/batch_run}
DROPOUT=${DROPOUT:-0.1}

FUSION_ARGS=(--enable-torch-compile --torch-compile-mode default)

echo "Config: hidden=$HIDDEN_DIM layers=$NUM_LAYERS heads=$NUM_HEADS seq=$SEQ_LEN docs=$WIKI_NUM_DOCS"
echo "        steps=$NUM_STEPS batch=$BATCH_SIZE lr=$LEARNING_RATE dropout=$DROPOUT output=$OUTPUT_DIR"

python3 -u tiny_llama_v2.py \
    --dataset wikipedia \
    --hidden-dim "$HIDDEN_DIM" --num-layers "$NUM_LAYERS" \
    --num-heads "$NUM_HEADS" --seq-len "$SEQ_LEN" --dropout "$DROPOUT" \
    --wiki-num-docs "$WIKI_NUM_DOCS" \
    --num-steps "$NUM_STEPS" --batch-size "$BATCH_SIZE" --learning-rate "$LEARNING_RATE" \
    --eval-interval "$EVAL_INTERVAL" --eval-batches "$EVAL_BATCHES" --generate-every "$GENERATE_EVERY" \
    "${FUSION_ARGS[@]}" \
    --save-checkpoints --output-dir "$OUTPUT_DIR"

echo "Job finished $(date)"
