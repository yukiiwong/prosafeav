#!/bin/bash
# ---------------------------------------------------------------------------
# Supervise ONE training run: its own CARLA server on its own port, restarted if
# either side dies, and everything scoped to this run's logdir.
#
#   tools/run_one.sh <carla_port> <gpu> <logdir> <entry> [training args...]
#
# ``entry`` is either "dreamerv3/train.py" for the JAX backbone or one of the
# PyTorch entry points such as "dreamerv3/train_tdmpc.py".
#
# Why this exists rather than train_prosafeav.sh: that script decides whether the
# trainer is alive with `pgrep -f dreamerv3/train.py`, which matches *any* run on
# the machine.  With two concurrent runs each supervisor sees the other's process
# and concludes its own is healthy, so a crashed run is never restarted.  Here the
# supervisor tracks its own child PID, so runs can be launched in parallel.
# ---------------------------------------------------------------------------
set -uo pipefail

if [ $# -lt 4 ]; then
    echo "Usage: $0 <carla_port> <gpu> <logdir> <entry> [additional args...]"
    exit 1
fi

CARLA_PORT=$1
GPU=$2
LOGDIR=$3
ENTRY=$4
shift 4
EXTRA=("$@")

: "${CARLA_ROOT:?CARLA_ROOT is not set; point it at the CARLA 0.9.15 install}"

mkdir -p "$LOGDIR"
LOG="${LOGDIR}/run.log"
TM_PORT=$((CARLA_PORT + 6000))

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

launch_carla() {
    if ! nc -z localhost "$CARLA_PORT" 2>/dev/null; then
        log "starting CARLA on port ${CARLA_PORT} (gpu ${GPU})"
        fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1
        CUDA_VISIBLE_DEVICES="$GPU" "$CARLA_ROOT/CarlaUE4.sh" \
            -RenderOffScreen -carla-port="$CARLA_PORT" -benchmark -fps=10 \
            >> "${LOGDIR}/carla.log" 2>&1 &
        CARLA_PID=$!
        local waited=0
        while ! nc -z localhost "$CARLA_PORT" 2>/dev/null; do
            sleep 2
            waited=$((waited + 2))
            if [ "$waited" -ge 180 ]; then
                log "CARLA failed to come up within 180s"
                return 1
            fi
        done
        log "CARLA up after ${waited}s"
    fi
    return 0
}

start_training() {
    launch_carla || return 1
    log "starting ${ENTRY}"
    CUDA_VISIBLE_DEVICES="$GPU" python -u "$ENTRY" \
        --env.world.carla_port "$CARLA_PORT" \
        --env.world.traffic.tm_seed "$CARLA_PORT" \
        --dreamerv3.logdir "$LOGDIR" \
        "${EXTRA[@]}" >> "$LOG" 2>&1 &
    TRAIN_PID=$!
    log "trainer pid ${TRAIN_PID}"
    return 0
}

cleanup() {
    log "shutting down"
    [ -n "${TRAIN_PID:-}" ] && kill -TERM "$TRAIN_PID" 2>/dev/null
    [ -n "${CARLA_PID:-}" ] && kill -TERM "$CARLA_PID" 2>/dev/null
    fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1
    exit 0
}
trap cleanup SIGINT SIGTERM

: > "$LOG"
log "run: port=${CARLA_PORT} gpu=${GPU} entry=${ENTRY} args=${EXTRA[*]}"
start_training || { log "initial start failed"; exit 1; }

RESTARTS=0
MAX_RESTARTS="${MAX_RESTARTS:-20}"
while true; do
    sleep 60
    # Track this run's own child rather than pattern-matching every trainer on
    # the machine, so concurrent runs cannot mistake each other for themselves.
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        wait "$TRAIN_PID" 2>/dev/null
        code=$?
        if [ "$code" -eq 0 ]; then
            log "training finished cleanly"
            cleanup
        fi
        RESTARTS=$((RESTARTS + 1))
        if [ "$RESTARTS" -gt "$MAX_RESTARTS" ]; then
            log "exceeded ${MAX_RESTARTS} restarts, giving up"
            cleanup
        fi
        log "trainer exited with ${code}; restart ${RESTARTS}/${MAX_RESTARTS}"
        start_training || { log "restart failed"; cleanup; }
    fi
done
