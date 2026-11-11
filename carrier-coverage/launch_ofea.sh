#!/usr/bin/env bash
set -euo pipefail

# Run from the project root after copying the updated scripts into code_dir.
code_dir=${1:?provide the directory containing the OFEA scripts}
steps=${2:?provide the number of paired continuation updates}
control_device=${3:?provide the control device, for example cuda:0}
ofea_device=${4:?provide the OFEA device, for example cuda:1}
if ! [[ $steps =~ ^[1-9][0-9]{0,4}$ ]] || (( steps > 39000 )); then
    echo 'Continuation length must be an integer from 1 to 39000' >&2
    exit 2
fi
baseline=runs/author64-exposure-equal-20261110
coverage=runs/author64-carrier-coverage-20261110
exposure=runs/author64-exposure-matched-20261110
trial="$code_dir/steps-$steps"
py=/mnt/vos-79jtuvax/conda_envs/rgwm/bin/python
end=$((117000 + steps))
export PYTHONPATH="$PWD/$code_dir:$PWD/$baseline:$PWD/$exposure:$PWD/$coverage:$PWD/runs/author64-decoder-probe-20261109:$PWD/scripts:$PWD/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

mkdir -p "$trial"
trap 'rc=$?; echo "$rc" > "$trial/exit-code"' EXIT
common=(--config configs/author64_B1_1000.yaml
    --assets reports/author64-watermark-runtime-v1-2threads/assets.yaml
    --design runs/author64-message-sampling/dynamic-full/design.json
    --parent "$baseline" --plan-root "$baseline/exposure"
    --resume-from "$baseline/exposure/n416-seed0/step-117000.pt"
    --carriers 416 --through-step "$end")
pairs=reports/author64-watermark-acceptance-v3-2threads/pairs_val.jsonl

if [[ ! -s "$trial/probe.json" ]]; then
    "$py" "$code_dir/train_exposure.py" "${common[@]}" --output "$trial/probe" \
        --device "$ofea_device" --probe-ofea > "$trial/probe.tmp"
    mv "$trial/probe.tmp" "$trial/probe.json"
fi
weight=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["suggested_ofea_weight"])' "$trial/probe.json")
echo "OFEA weight $weight"

run_arm() {
    local name=$1 weight=$2 device=$3
    "$py" "$code_dir/train_exposure.py" "${common[@]}" --output "$trial/$name" \
        --device "$device" --ofea-weight "$weight" --identity-pairs "$pairs"
    "$py" "$code_dir/audit_exposure.py" --root "$baseline/exposure" --parent "$baseline" \
        --carriers 416 --through-step "$end" --run-dir "$trial/$name"
}

if [[ $control_device == "$ofea_device" ]]; then
    run_arm control 0 "$control_device" > "$trial/control.log" 2>&1
    run_arm ofea "$weight" "$ofea_device" > "$trial/ofea.log" 2>&1
    exit 0
fi
run_arm control 0 "$control_device" > "$trial/control.log" 2>&1 &
control_pid=$!
run_arm ofea "$weight" "$ofea_device" > "$trial/ofea.log" 2>&1 &
ofea_pid=$!
status=0
wait "$control_pid" || status=1
wait "$ofea_pid" || status=1
exit "$status"
