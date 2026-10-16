#!/usr/bin/env bash
# Submit one project command to Slurm (never run training on the login node). Follows the site's HPC skill
# (docs/hpcskill.md): sbatch, one log per job, stale jobs of the same name cancelled first.
#
#   scripts/hpc/submit.sh <job-name> <partition> <walltime> -- <command ...>
#   scripts/hpc/submit.sh har_smoke gpu 00:30:00 -- uv run --no-sync python scripts/train.py --models s4d --datasets utd_mhad --splits 0 --max-epochs 1
#   scripts/hpc/submit.sh har_analyses normal 00:20:00 -- uv run --no-sync python scripts/run_analyses.py
#
# Partitions (2026-10-15): gpu = hpc21-24, 1 GPU each (often full; jobs queue); normal/small/tiny = hpc01-20, CPU
# only; dgx-* (A100) was down. Override resources with CPUS, MEM, GPUS (default 1 on gpu partitions), NODE;
# DEPEND passes --dependency (e.g. DEPEND=afterok:86828), QOS --qos (the gpu partition accepts every QOS:
# QOS=tiny GPUS=0 runs CPU work on the GPU nodes' idle cores, at most 2 h).
set -euo pipefail
[[ $# -ge 5 && $4 == "--" ]] || { sed -n '2,10p' "$0"; exit 2; }
NAME=$1; PART=$2; WALL=$3; shift 4
PROJECT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
LOG_DIR="$PROJECT_DIR/experiments/logs/hpc"
mkdir -p "$LOG_DIR"
CPUS=${CPUS:-8}; MEM=${MEM:-32G}

OLD=$(squeue -u "$USER" -h -n "$NAME" -o "%A" || true)
if [[ -n "$OLD" ]]; then echo "cancelling stale $NAME: $OLD"; echo "$OLD" | xargs -r scancel; fi

ARGS=(-p "$PART" --job-name="$NAME" --cpus-per-task="$CPUS" --mem="$MEM" --time="$WALL"
      --output="$LOG_DIR/${NAME}_%j.out" --chdir="$PROJECT_DIR")
if [[ $PART == gpu || $PART == dgx-* ]]; then ARGS+=(--gres="gpu:${GPUS:-1}"); fi
if [[ -n ${NODE:-} ]]; then ARGS+=(--nodelist="$NODE"); fi
if [[ -n ${DEPEND:-} ]]; then ARGS+=(--dependency="$DEPEND"); fi
if [[ -n ${QOS:-} ]]; then ARGS+=(--qos="$QOS"); fi

CMD=$(printf '%q ' "$@")
# no GPU requested: hide the node's GPUs, which Slurm does not fence off for a job without a GPU allocation
HIDE=""
if [[ ${GPUS:-1} == 0 || ( $PART != gpu && $PART != dgx-* ) ]]; then HIDE="export CUDA_VISIBLE_DEVICES='' && "; fi
sbatch "${ARGS[@]}" --wrap="${HIDE}source scripts/hpc/env.sh && echo \"node \$(hostname), job \$SLURM_JOB_ID, \$(date)\" && (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true) && $CMD"
echo "log: $LOG_DIR/${NAME}_<jobid>.out   monitor: squeue -u $USER; tail -f $LOG_DIR/${NAME}_<jobid>.out"
