#!/usr/bin/env bash
# Sequentially run a list of config JSONs through run_one_dylan.py.
# Usage: bash sweep_runner.sh <config1.json> [config2.json ...]
#
# Logs to experiments/sweep.log; per-run logs at experiments/runs/<id>/train.log.
# Single-process by design — PPO already uses 4 vectorized envs and pegs the CPU.

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

PY=/Users/dylanmassaro/axiom_tilt/.venv/bin/python
LOG=experiments/sweep_dylan.log
mkdir -p experiments/runs

ts() { date +'%Y-%m-%dT%H:%M:%S'; }

for cfg in "$@"; do
    name=$(basename "$cfg" .json)
    out=experiments/runs/d_${name#d_}
    mkdir -p "$out"
    echo "[$(ts)] START $cfg -> $out/train.log" | tee -a "$LOG"
    PYTHONPATH=. "$PY" -u experiments/run_one_dylan.py "$cfg" \
        > "$out/train.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        last=$(grep -E '^  (excess_sharpe|ppo_sharpe|ppo_turnover):' "$out/train.log" | tail -3 | tr '\n' ' ')
        echo "[$(ts)] DONE  $cfg  rc=$rc  $last" | tee -a "$LOG"
    else
        echo "[$(ts)] FAIL  $cfg  rc=$rc  (see $out/train.log)" | tee -a "$LOG"
    fi
done

echo "[$(ts)] sweep complete." | tee -a "$LOG"
