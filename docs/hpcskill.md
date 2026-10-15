---
mode: agent
description: Reusable HPC submission and monitoring skill for training jobs in any project
---

# Skill: Submit & Monitor Training on HPC

This skill is for safely running long training jobs on SLURM HPC and reusing the same process across different files/projects.

Use this skill when user asks to:
- train on HPC
- submit/run experiment on cluster
- monitor training progress
- rerun with updated hyperparameters

## Goals

- Never train on login/gateway CPU.
- Submit through SLURM (`sbatch` preferred for robustness).
- Avoid duplicate jobs.
- Keep one clear log file per job.
- Make workflow reusable for any project and entry command.
- Always check the sucessful of the submission job, if fail or not run, find another HPC
- Always keep monitoring the traing process

## Input Variables (Set Before Running)

Set these placeholders first so the same flow works for any project.

- `PROJECT_DIR`: absolute project folder on HPC.
- `ENTRY_CMD`: command to run training.
- `ENV_SETUP`: environment activation command. Source should be user input first, otherwise current active env.
- `PARTITION`: optional user override. If not provided, auto-select GPU first, then CPU.
- `NODE_HINT`: optional fixed node (for example `hpc03`).
- `CPUS`, `MEM`, `WALLTIME`: resource request.
- `JOB_NAME`: short unique name.
- `LOG_DIR`: log folder under project directory. Default is `$PROJECT_DIR/log`.

Example values:

```bash
PROJECT_DIR="/path/to/project"
ENTRY_CMD="python train.py"
ENV_SETUP='source /opt/hpc/anaconda3/etc/profile.d/conda.sh && conda activate batlife'
PARTITION=""
NODE_HINT="hpc03"
CPUS="8"
MEM="32G"
WALLTIME="8:00:00"
JOB_NAME="train_cpu"
LOG_DIR="$PROJECT_DIR/log"
```

Environment selection priority:
- 1) User-provided env command/path.
- 2) Current active environment (`$CONDA_DEFAULT_ENV` or current Python venv).
- 3) If neither is clear, stop and ask user for env before submission.

Log folder policy:
- Always write logs under project folder: `$PROJECT_DIR/log`.
- Do not default to home-level log directories.

## Part 1: Inspect Cluster and Choose Resources

CPU nodes:

```bash
sinfo -p normal -N -o "%N %T %C %m"
```

GPU nodes:

```bash
sinfo -p gpu -N -o "%N %T %G %C"
```

Queue snapshot:

```bash
squeue -u "$USER"
```

Selection rules:
- If user explicitly requested CPU/GPU, honor that.
- If user did not specify, prefer GPU when available.
- If GPU is not available, automatically fall back to CPU.
- Do not run `python train.py` directly on login node.

Auto-select example when user does not set `PARTITION`:

```bash
if [[ -z "$PARTITION" ]]; then
	GPU_AVAIL=$(sinfo -p gpu -N -h -o "%N %T %G %C" | awk '$2 ~ /idle|mix/ {print $0}' | wc -l)
	if [[ "$GPU_AVAIL" -gt 0 ]]; then
		PARTITION="gpu"
	else
		PARTITION="normal"
	fi
fi
```

## Part 2: Prevent Duplicate Jobs

Before new submission, cancel stale job(s) with same name.

```bash
OLD_IDS=$(squeue -u "$USER" -h -n "$JOB_NAME" -o "%A")
if [[ -n "$OLD_IDS" ]]; then
	echo "$OLD_IDS" | xargs -r scancel
fi
```

Then verify only intended jobs remain:

```bash
squeue -u "$USER"
```

## Part 3: Submit Batch Job (Preferred)

Use `sbatch` so job continues if terminal disconnects.

```bash
mkdir -p "$LOG_DIR"

SBATCH_CMD=(
	sbatch
	-p "$PARTITION"
	--cpus-per-task="$CPUS"
	--mem="$MEM"
	--time="$WALLTIME"
	--job-name="$JOB_NAME"
	--output="$LOG_DIR/${JOB_NAME}_%j.out"
	--chdir="$PROJECT_DIR"
)

if [[ -n "$NODE_HINT" ]]; then
	SBATCH_CMD+=(--nodelist="$NODE_HINT")
fi

"${SBATCH_CMD[@]}" --wrap="$ENV_SETUP && $ENTRY_CMD"
```

If `ENV_SETUP` is not provided, derive it from active env first:

```bash
if [[ -z "$ENV_SETUP" ]]; then
	if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
		ENV_SETUP="source /opt/hpc/anaconda3/etc/profile.d/conda.sh && conda activate $CONDA_DEFAULT_ENV"
	elif [[ -n "$VIRTUAL_ENV" ]]; then
		ENV_SETUP="source $VIRTUAL_ENV/bin/activate"
	else
		echo "No active env detected. Please provide ENV_SETUP." >&2
		exit 1
	fi
fi
```

Expected output:

```text
Submitted batch job <JOB_ID>
```

## Part 4: Confirm Run Started

```bash
squeue -j <JOB_ID>
tail -n 50 "$LOG_DIR/${JOB_NAME}_<JOB_ID>.out"
```

Look for signals like:
- model/dataset print
- data loading started
- first epoch line

## Part 5: Monitor Progress

Quick monitor:

```bash
tail -f "$LOG_DIR/${JOB_NAME}_<JOB_ID>.out"
```

Structured monitor (epoch summary):

```bash
grep -E "^Epoch|Best model saved|EarlyStopping|Early stopping triggered|Final checkpoint" \
	"$LOG_DIR/${JOB_NAME}_<JOB_ID>.out" | tail -n 40
```

## Part 6: Stop or Rerun

Cancel one job:

```bash
scancel <JOB_ID>
```

Cancel all jobs by name:

```bash
squeue -u "$USER" -h -n "$JOB_NAME" -o "%A" | xargs -r scancel
```

For rerun with changed config:
- update config/file first
- cancel stale job(s)
- submit again with same template

## Common Failure Checks

- `ModuleNotFoundError`:
	- wrong environment or env not activated in `ENV_SETUP`.
- `Broken pipe` after completion:
	- common when parent shell exits; if final metrics are printed and files are saved, run likely completed.
- job stuck `PD`:
	- inspect reason with `squeue -j <JOB_ID> -o "%i %T %R"`.
- no log updates:
	- confirm job running and output path is correct.

## Output to Report Back to User

After submission, always return:
- job id
- partition/node
- log file path
- current state (`R`/`PD`/`CG`)
- exact monitor command user can run

## Minimal One-Command Template

Use this when user gives only "submit training" and details are known:

```bash
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/log}" && \
mkdir -p "$LOG_DIR" && \
OLD_IDS=$(squeue -u "$USER" -h -n "$JOB_NAME" -o "%A") && \
[[ -z "$OLD_IDS" ]] || echo "$OLD_IDS" | xargs -r scancel && \
if [[ -z "$PARTITION" ]]; then GPU_AVAIL=$(sinfo -p gpu -N -h -o "%N %T %G %C" | awk '$2 ~ /idle|mix/ {print $0}' | wc -l); [[ "$GPU_AVAIL" -gt 0 ]] && PARTITION="gpu" || PARTITION="normal"; fi && \
sbatch -p "$PARTITION" --cpus-per-task="$CPUS" --mem="$MEM" --time="$WALLTIME" \
	${NODE_HINT:+--nodelist="$NODE_HINT"} \
	--job-name "$JOB_NAME" --output "$LOG_DIR/${JOB_NAME}_%j.out" --chdir "$PROJECT_DIR" \
	--wrap "$ENV_SETUP && $ENTRY_CMD"
```

This skill is intentionally reusable for any future file/project by changing only the input variables.