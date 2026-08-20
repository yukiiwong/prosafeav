#!/bin/bash
# ---------------------------------------------------------------------------
# Zero-shot transfer evaluation: take ONE trained checkpoint and evaluate it on
# several tasks it was never trained on.
#
#   tools/eval_matrix.sh <checkpoint> <gpu> <task1> [task2 ...]
#   tools/eval_matrix.sh logdir/prosafeav_s0/checkpoint.ckpt 0 \
#       carla_overtake_d05 carla_overtake_d15 carla_overtake_d30 carla_overtake_d45
#
# Environment:
#   EPISODES=200   evaluation steps per task           (default 2e5 steps)
#   OUT=logdir/eval_matrix   where per-task logdirs go
#   BASE_PORT=3000 first CARLA port
#
# This is what separates "trained at each density" from "trained once, deployed
# at every density" in Table IV.  The first shows the method can fit each
# regime; only the second shows it generalises.
#
# The EVT model is frozen during evaluation.  A policy must be scored against
# the risk model it was trained under, otherwise the tail model keeps adapting
# to the evaluation episodes and the reported risk is not comparable across
# tasks.  Pass --env.evt.load_from <fitted.json> to pin it explicitly.
# ---------------------------------------------------------------------------
set -uo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <checkpoint> <gpu> <task1> [task2 ...]"
    exit 1
fi

REPO="/home/yukai/CarDreamer_prosafeav"
CKPT=$1
GPU=$2
shift 2
TASKS=("$@")

STEPS="${STEPS:-2e5}"
OUT="${OUT:-${REPO}/logdir/eval_matrix}"
BASE_PORT="${BASE_PORT:-3000}"
EVT_MODEL="${EVT_MODEL:-}"

: "${CARLA_ROOT:?CARLA_ROOT is not set; point it at the CARLA 0.9.15 install}"

if [ ! -e "$CKPT" ]; then
    echo "checkpoint not found: $CKPT"
    exit 1
fi

mkdir -p "$OUT"
echo "checkpoint : $CKPT"
echo "tasks      : ${TASKS[*]}"
echo "output     : $OUT"
echo

idx=0
for task in "${TASKS[@]}"; do
    port=$((BASE_PORT + idx * 4))
    logdir="${OUT}/${task}"
    mkdir -p "$logdir"
    echo ">>> evaluating on ${task} (port ${port})"

    extra=()
    [ -n "$EVT_MODEL" ] && extra+=(--env.evt.load_from "$EVT_MODEL")

    # Start CARLA for this task, wait for it, evaluate, then tear it down; the
    # runs are sequential so a single checkpoint is never evaluated twice at once.
    fuser -k "${port}/tcp" >/dev/null 2>&1
    CUDA_VISIBLE_DEVICES="$GPU" "$CARLA_ROOT/CarlaUE4.sh" \
        -RenderOffScreen -carla-port="$port" -benchmark -fps=10 \
        >> "${logdir}/carla.log" 2>&1 &
    carla_pid=$!

    waited=0
    while ! nc -z localhost "$port" 2>/dev/null; do
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -ge 180 ]; then
            echo "    CARLA failed to start; skipping ${task}"
            kill -TERM "$carla_pid" 2>/dev/null
            continue 2
        fi
    done

    CUDA_VISIBLE_DEVICES="$GPU" python -u "${REPO}/dreamerv3/eval.py" \
        --task "$task" \
        --env.world.carla_port "$port" \
        --env.eval True \
        --dreamerv3.logdir "$logdir" \
        --dreamerv3.run.from_checkpoint "$CKPT" \
        --dreamerv3.run.steps "$STEPS" \
        "${extra[@]}" >> "${logdir}/eval.log" 2>&1

    status=$?
    kill -TERM "$carla_pid" 2>/dev/null
    fuser -k "${port}/tcp" >/dev/null 2>&1
    echo "    exit ${status}; metrics at ${logdir}/metrics.jsonl"
    idx=$((idx + 1))
done

echo
echo "aggregate with:"
echo "  python tools/collect_results.py --logdir ${OUT} --pattern '*' \\"
echo "      --latex ${OUT}/transfer_table.tex"
