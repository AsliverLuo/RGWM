#!/usr/bin/env bash
set -euo pipefail

# Run from the project root. Each recipe has its own checkpoints and logs.
root=${1:?provide the new run directory}
coverage=runs/author64-carrier-coverage-20261110
exposure=runs/author64-exposure-matched-20261110
py=/mnt/vos-79jtuvax/conda_envs/rgwm/bin/python
cfg=configs/author64_B1_1000.yaml
assets=reports/author64-watermark-runtime-v1-2threads/assets.yaml
design=runs/author64-message-sampling/dynamic-full/design.json
checkpoint=runs/author64-swap-recovery-v2/latest.pt
historical=runs/author64-experiment-a-20261109/author64-swap-recovery-v2/summary.json
export PYTHONPATH="$PWD/$root:$PWD/$exposure:$PWD/$coverage:$PWD/runs/author64-decoder-probe-20261109:$PWD/scripts:$PWD/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

trap 'rc=$?; echo "$rc" > "$root/pipeline.exit"' EXIT
date -Iseconds > "$root/launched-at.txt"
cp "$coverage/shared-plan.json" "$root/"
cp "$exposure/"{carrier-identities.json,n64-schedule.json,n64-schedule.npz,n416-schedule.json,n416-schedule.npz} "$root/"

coverage_arm() {
    local size=$1 gpu=$2
    "$py" "$root/run_joint.py" --config "$cfg" --assets "$assets" --design "$design" \
        --historical "$historical" --checkpoint "$checkpoint" --coverage-plan "$root/shared-plan.json" \
        --carrier-count "$size" --output "$root/n$size-seed0" --device "cuda:$gpu" \
        --arm j1 --message-recipe swap-equal --steps 6000
}

exposure_arm() {
    local size=$1 through=$2 gpu=$3
    "$py" "$root/train_exposure.py" --config "$cfg" --assets "$assets" --design "$design" \
        --parent "$root" --plan-root "$root" --output "$root/exposure/n$size-seed0" \
        --carriers "$size" --through-step "$through" --device "cuda:$gpu"
    "$py" "$root/audit_exposure.py" --root "$root/exposure" --parent "$root" \
        --carriers "$size" --through-step "$through"
}

wait_all() {
    local pid rc=0
    for pid in "$@"; do
        wait "$pid" || rc=1
    done
    return "$rc"
}

pids=()
for arm in 16:0 64:1 416:3; do
    size=${arm%:*}
    gpu=${arm#*:}
    coverage_arm "$size" "$gpu" > "$root/n$size-6000.log" 2>&1 &
    pids+=("$!")
    echo "N$size 50:50 coverage GPU $gpu PID $!"
done
wait_all "${pids[@]}"
"$py" "$root/analyze_joint.py" --root "$root" --comparison coverage
for size in 16 64 416; do
    cmp "$root/n$size-seed0/plan.json" "$coverage/n$size-seed0/plan.json"
    cmp "$root/n$size-seed0/validation-0000.jsonl" "$coverage/n$size-seed0/validation-0000.jsonl"
done
echo COVERAGE_COMPLETE

mkdir "$root/exposure"
for name in carrier-identities.json n64-schedule.json n64-schedule.npz n416-schedule.json n416-schedule.npz; do
    ln "$root/$name" "$root/exposure/$name"
done

(exposure_arm 64 24000 1; echo N64_COMPLETE) > "$root/n64-24000.log" 2>&1 &
pid64=$!
(
    exposure_arm 416 39000 3
    decision=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["continue_to_156000"])' \
        "$root/exposure/n416-seed0/audit-39000.json")
    if [ "$decision" = True ]; then
        echo N416_GATE_PASSED
        exposure_arm 416 156000 3 > "$root/n416-156000.log" 2>&1
    elif [ "$decision" = False ]; then
        echo N416_GATE_NOT_PASSED
    else
        echo INVALID_GATE_RESULT
        exit 1
    fi
    echo N416_COMPLETE
) > "$root/n416-39000.log" 2>&1 &
pid416=$!
echo "N64 and N416 exposure jobs: $pid64 $pid416"
wait_all "$pid64" "$pid416"
echo PIPELINE_COMPLETE
