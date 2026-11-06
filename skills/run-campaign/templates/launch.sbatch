#!/bin/bash
#SBATCH --job-name=@JOB_NAME@
#SBATCH --partition=@PARTITION@
#SBATCH --nodes=1
@GRES_LINE@
#SBATCH --cpus-per-task=@CPUS@
#SBATCH --mem=@MEM@
@TIME_LINE@
#SBATCH --output=@LOG_DIR@/%x-%j.out
# Written by skills/run-campaign/scripts/write_launch.py. One agsci run on one node.
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} start=$(date -Is)"
nvidia-smi -L || true
@ENV_EXPORTS@
cd @AGSCI_REPO@
exec .venv/bin/agsci run @TASK_DIR@ @OVERRIDES@
