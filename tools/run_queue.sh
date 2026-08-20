#!/bin/bash
# ---------------------------------------------------------------------------
# Bounded-concurrency queue for the ProSafeAV revision experiment matrix.
#
#   tools/run_queue.sh list                     # show the matrix
#   tools/run_queue.sh start main               # launch a group
#   tools/run_queue.sh start main,ablation      # several groups
#   tools/run_queue.sh status                   # what is running / queued / done
#   tools/run_queue.sh stop                     # stop everything this queue owns
#
# Environment:
#   GPUS=0            comma-separated GPU ids to use          (default 0)
#   JOBS=2            concurrent runs                          (default 2)
#   SEEDS="0 1 2"     seeds per configuration                  (default 0 1 2)
#   STEPS=3e5         training steps                           (default 3e5)
#   BASE_PORT=2000    first CARLA port; each run takes port+4k  (default 2000)
#
# One CARLA server plus one trainer is roughly 8-12 GB of VRAM, so JOBS=2 fits
# comfortably on a single A100 80GB alongside another user's work; raise it only
# after checking `nvidia-smi`.
#
# The queue writes its state under logdir/.queue so `status` and `stop` work from
# a fresh shell.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="/home/yukai/CarDreamer_prosafeav"
LOGROOT="${REPO}/logdir"
QDIR="${LOGROOT}/.queue"

GPUS="${GPUS:-0}"
JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-3e5}"
BASE_PORT="${BASE_PORT:-2000}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"

# ---------------------------------------------------------------------------
# The matrix: group | name | entry | task | extra args
# ---------------------------------------------------------------------------
matrix() {
cat <<'EOF'
main|prosafeav|dreamerv3/train.py|carla_overtake_prosafeav|
density|prosafeav_d05|dreamerv3/train.py|carla_overtake_d05|
density|prosafeav_d15|dreamerv3/train.py|carla_overtake_d15|
density|prosafeav_d30|dreamerv3/train.py|carla_overtake_d30|
density|prosafeav_d45|dreamerv3/train.py|carla_overtake_d45|
scenario|prosafeav_fourlane|dreamerv3/train.py|carla_four_lane_prosafeav|
scenario|prosafeav_fourlane_dense|dreamerv3/train.py|carla_four_lane_prosafeav_dense|
scenario|prosafeav_merge|dreamerv3/train.py|carla_lane_merge_prosafeav|
scenario|prosafeav_rightturn|dreamerv3/train.py|carla_right_turn_prosafeav|
scenario|prosafeav_roundabout|dreamerv3/train.py|carla_roundabout_prosafeav|
perception|prosafeav_fov|dreamerv3/train.py|carla_overtake_fov|
perception|prosafeav_noisy|dreamerv3/train.py|carla_overtake_noisy|
ablation|prosafeav_noevt|dreamerv3/train.py|carla_overtake_noevt|
ablation|prosafeav_evtenv|dreamerv3/train.py|carla_overtake_evtenv|
ablation|prosafeav_evtimag|dreamerv3/train.py|carla_overtake_evtimag|
ablation|prosafeav_lonly|dreamerv3/train.py|carla_overtake_lonly|
sensitivity|prosafeav_frank|dreamerv3/train.py|carla_overtake_frank|
sensitivity|prosafeav_thrq|dreamerv3/train.py|carla_overtake_thrq|
sensitivity|prosafeav_w1|dreamerv3/train.py|carla_overtake_w1|
sensitivity|prosafeav_w10|dreamerv3/train.py|carla_overtake_w10|
wm|prosafeav_dv3_s|dreamerv3/train.py|carla_overtake_prosafeav|--dreamerv3.rssm.deter 256 --dreamerv3.rssm.stoch 16
wm|prosafeav_gauss|dreamerv3/train.py|carla_overtake_prosafeav|--dreamerv3.rssm.classes 0
wm|prosafeav_rssm|dreamerv3/train_prosafeav_rssm.py|carla_overtake_prosafeav|
wm|prosafeav_det|dreamerv3/train_prosafeav_deterministic.py|carla_overtake_prosafeav|
wm|prosafeav_transformer|dreamerv3/train_transformer_wm.py|carla_overtake_prosafeav|
wm|prosafeav_tdmpc|dreamerv3/train_tdmpc.py|carla_overtake_prosafeav|
wm|baseline_planet|dreamerv3/train_planet.py|carla_overtake_prosafeav|
wm|baseline_worldmodels|dreamerv3/train_worldmodels.py|carla_overtake_prosafeav|
wm|baseline_simple|dreamerv3/train_simple.py|carla_overtake_prosafeav|
modelfree|baseline_dqn|dreamerv3/train_dqn.py|carla_overtake_prosafeav|
modelfree|baseline_sac|dreamerv3/train_sac.py|carla_overtake_prosafeav|
modelfree|baseline_td3|dreamerv3/train_td3.py|carla_overtake_prosafeav|
modelfree|baseline_ppo|dreamerv3/train_ppo.py|carla_overtake_prosafeav|
critical|prosafeav_critical|dreamerv3/train.py|carla_overtake_critical|
critical|prosafeav_calm|dreamerv3/train.py|carla_overtake_calm|
legacy|prosafeav_legacy|dreamerv3/train.py|carla_overtake_legacy|
EOF
}

selected() {
    local groups="$1"
    if [ "$groups" = "all" ]; then matrix; return; fi
    local IFS=','
    for g in $groups; do matrix | awk -F'|' -v g="$g" '$1==g'; done
}

cmd_list() {
    local groups="${1:-all}"
    printf '%-12s %-24s %-40s %s\n' GROUP NAME TASK ENTRY
    selected "$groups" | while IFS='|' read -r group name entry task extra; do
        printf '%-12s %-24s %-40s %s\n' "$group" "$name" "$task" "$entry"
    done
    local n
    n=$(selected "$groups" | wc -l)
    local s
    s=$(echo "$SEEDS" | wc -w)
    echo
    echo "$n configurations x $s seeds = $((n * s)) runs; JOBS=$JOBS on GPU(s) $GPUS"
}

cmd_start() {
    local groups="${1:-all}"
    if [ -z "${CARLA_ROOT:-}" ] || [ ! -x "${CARLA_ROOT}/CarlaUE4.sh" ]; then
        echo "CARLA_ROOT is not set or does not contain an executable CarlaUE4.sh."
        echo "  CARLA_ROOT=${CARLA_ROOT:-<unset>}"
        echo "Training cannot start without the simulator."
        exit 1
    fi
    mkdir -p "$QDIR"
    local todo="${QDIR}/todo"
    : > "$todo"
    local idx=0
    while IFS='|' read -r group name entry task extra; do
        for seed in $SEEDS; do
            echo "${idx}|${group}|${name}_s${seed}|${entry}|${task}|${seed}|${extra}" >> "$todo"
            idx=$((idx + 1))
        done
    done < <(selected "$groups")

    echo "queued $(wc -l < "$todo") runs; running at most ${JOBS} at a time"
    nohup bash "$0" __worker "$todo" > "${QDIR}/worker.log" 2>&1 &
    echo "worker pid $! (logs: ${QDIR}/worker.log)"
}

cmd_worker() {
    local todo="$1"
    mkdir -p "$QDIR"
    echo $$ > "${QDIR}/worker.pid"
    : > "${QDIR}/running"
    while IFS='|' read -r idx group name entry task seed extra; do
        # Block until a slot frees up.
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 20; done

        local port=$((BASE_PORT + idx * 4))
        local gpu=${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}
        local logdir="${LOGROOT}/${name}"
        mkdir -p "$logdir"

        echo "[$(date '+%F %T')] launching ${name} (group ${group}) port ${port} gpu ${gpu}"
        # shellcheck disable=SC2086
        bash "${REPO}/tools/run_one.sh" "$port" "$gpu" "$logdir" "${REPO}/${entry}" \
            --task "$task" \
            --dreamerv3.seed "$seed" \
            --dreamerv3.run.steps "$STEPS" \
            $extra &
        echo "${name}|$!|${port}|${gpu}|$(date '+%F %T')" >> "${QDIR}/running"
        sleep 5
    done < "$todo"
    wait
    echo "[$(date '+%F %T')] queue drained"
}

cmd_status() {
    if [ ! -d "$QDIR" ]; then echo "no queue state under ${QDIR}"; return; fi
    if [ -f "${QDIR}/worker.pid" ] && kill -0 "$(cat "${QDIR}/worker.pid")" 2>/dev/null; then
        echo "worker: running (pid $(cat "${QDIR}/worker.pid"))"
    else
        echo "worker: not running"
    fi
    echo
    printf '%-26s %-8s %-6s %-5s %s\n' RUN PID PORT GPU STATE
    if [ -f "${QDIR}/running" ]; then
        while IFS='|' read -r name pid port gpu started; do
            local state="finished"
            kill -0 "$pid" 2>/dev/null && state="running"
            printf '%-26s %-8s %-6s %-5s %s (since %s)\n' "$name" "$pid" "$port" "$gpu" "$state" "$started"
        done < "${QDIR}/running"
    fi
    echo
    if [ -f "${QDIR}/todo" ]; then
        echo "queued total: $(wc -l < "${QDIR}/todo")"
    fi
}

cmd_stop() {
    if [ -f "${QDIR}/worker.pid" ]; then
        kill -TERM "$(cat "${QDIR}/worker.pid")" 2>/dev/null && echo "worker stopped"
    fi
    if [ -f "${QDIR}/running" ]; then
        while IFS='|' read -r name pid port gpu started; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null
                echo "stopped ${name} (pid ${pid})"
            fi
        done < "${QDIR}/running"
    fi
}

case "${1:-list}" in
    list)    cmd_list "${2:-all}" ;;
    start)   cmd_start "${2:-all}" ;;
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    __worker) shift; cmd_worker "$@" ;;
    *) echo "usage: $0 {list|start|status|stop} [groups]"; exit 1 ;;
esac
