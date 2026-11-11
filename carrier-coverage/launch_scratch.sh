#!/usr/bin/env bash
set -euo pipefail

root=${1:?provide the run directory}
baseline=runs/author64-exposure-equal-20261110
coverage=runs/author64-carrier-coverage-20261110
exposure=runs/author64-exposure-matched-20261110
py=/mnt/vos-79jtuvax/conda_envs/rgwm/bin/python
export PYTHONPATH="$PWD/$root:$PWD/$baseline:$PWD/$coverage:$PWD/$exposure:$PWD/runs/author64-decoder-probe-20261109:$PWD/scripts:$PWD/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

trap 'rc=$?; echo "$rc" > "$root/pipeline.exit"' EXIT
date -Iseconds > "$root/launched-at.txt"
common=(--config configs/M.yaml
    --assets reports/author64-watermark-runtime-v1-2threads/assets.yaml
    --design runs/author64-message-sampling/dynamic-full/design.json
    --baseline "$baseline"
    --identity-pairs reports/author64-watermark-acceptance-v3-2threads/pairs_val.jsonl
    --seed 0)
"$py" "$root/train_scratch.py" "${common[@]}" --arm control --device cuda:0 \
    --output "$root/control" > "$root/control.log" 2>&1 &
control_pid=$!
echo "$control_pid" > "$root/control.pid"
"$py" "$root/train_scratch.py" "${common[@]}" --arm ofea --device cuda:1 \
    --output "$root/ofea" > "$root/ofea.log" 2>&1 &
ofea_pid=$!
echo "$ofea_pid" > "$root/ofea.pid"
status=0
wait "$control_pid" || status=1
wait "$ofea_pid" || status=1
exit "$status"
