#!/bin/bash
# Written by skills/run-campaign/scripts/write_launch.py. Start it detached, with output piped to a log
# (a piped stdout makes the pipeline write heartbeats that campaign_status.py relies on):
#   @START_COMMAND@
set -euo pipefail
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi -L || true
@ENV_EXPORTS@
cd @AGSCI_REPO@
exec .venv/bin/agsci run @TASK_DIR@ @OVERRIDES@
