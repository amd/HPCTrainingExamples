#!/usr/bin/env bash
# Submit an end-to-end job: grab an 8-GPU node, serve Muse-Spark-Open, hit the API.
# ./3-run-serve-test.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
mkdir -p "${LOGDIR}"

SB="${BASE}/serve-test-muse.sbatch"
cat > "${SB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=muse-serve-test
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --exclusive --gres=${GPU_GRES} --nodes=1 --mem=0 --time=00:50:00
#SBATCH --output=${LOGDIR}/muse-serve-test-%j.log
set -uo pipefail
LOG=${LOGDIR}/muse-server-\${SLURM_JOB_ID}.log
VLLM_MODULE="${VLLM_MODULE}" VLLM_MODULEPATH="${VLLM_MODULEPATH}" RBK_HF_HOME="${HF_HOME}" \
    "${SERVE}" > "\${LOG}" 2>&1 &
SPID=\$!
echo "waiting for /v1/models (up to ~60 min for load+graph capture)..."
for i in \$(seq 1 180); do
    curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    kill -0 "\${SPID}" 2>/dev/null || { echo "SERVER DIED:"; tail -40 "\${LOG}"; exit 1; }
    [[ \$i -eq 180 ]] && { echo "TIMEOUT:"; tail -60 "\${LOG}"; kill "\${SPID}"; exit 1; }
    sleep 20
done
echo "=== /v1/models ===";          curl -sS http://127.0.0.1:8000/v1/models | python3 -m json.tool | head -30
echo "=== /v1/chat/completions ==="; curl -sS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"${SERVED_NAME}","messages":[{"role":"user","content":"Reply with exactly: rocBudAI online"}],"max_tokens":512,"temperature":0,"chat_template_kwargs":{"reasoning_effort":"minimal"}}' | python3 -m json.tool
echo "smoke test done; leaving server up 10 min for a manual module-load test."
sleep 600; kill "\${SPID}" 2>/dev/null || true
EOF

JOB=$(sbatch --parsable "${SB}")
echo "job ${JOB} queued. Watch:  squeue -u \$USER  /  tail -f ${LOGDIR}/muse-serve-test-${JOB}.log"
