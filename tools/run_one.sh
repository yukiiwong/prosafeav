#!/bin/bash
# ---------------------------------------------------------------------------
# Supervise ONE training run: its own CARLA server on its own port, restarted if
# either side dies, and everything scoped to this run's logdir.
#
#   tools/run_one.sh <carla_port> <gpu> <logdir> <entry> [training args...]
#
# STALL_SECONDS (default 900) is how long metrics.jsonl may go unwritten before
# the run is considered hung and restarted.
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

# A non-interactive shell does not run the conda hook, so bare `python` is not on
# PATH and every run would die with 127 and be restarted forever.  Resolve the
# interpreter explicitly, and fail immediately if it is missing rather than
# discovering it one restart at a time.
PYTHON="${PYTHON:-/home/yukai/.conda/envs/cardreamer/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "python interpreter not found or not executable: $PYTHON" >&2
    echo "set PYTHON=/path/to/env/bin/python" >&2
    exit 1
fi
export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$LOGDIR"
LOG="${LOGDIR}/run.log"
TM_PORT=$((CARLA_PORT + 6000))

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# An open TCP port is not proof of a healthy simulator: a CARLA that is shutting
# down still holds the socket, and a trainer started against it dies on the first
# load_world.  Probe with a real client handshake instead.
carla_ready() {
    "$PYTHON" - "$CARLA_PORT" <<'PYEOF' >/dev/null 2>&1
import sys
import carla
client = carla.Client("127.0.0.1", int(sys.argv[1]))
client.set_timeout(8.0)
client.get_server_version()
PYEOF
}

launch_carla() {
    if carla_ready; then
        return 0
    fi
    log "starting CARLA on port ${CARLA_PORT} (gpu ${GPU})"
    # Clear any half-dead server still holding the port before rebinding it.
    [ -n "${CARLA_PID:-}" ] && kill -9 "$CARLA_PID" 2>/dev/null
    fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1
    sleep 3
    CUDA_VISIBLE_DEVICES="$GPU" "$CARLA_ROOT/CarlaUE4.sh" \
        -RenderOffScreen -carla-port="$CARLA_PORT" -benchmark -fps=10 \
        >> "${LOGDIR}/carla.log" 2>&1 &
    CARLA_PID=$!
    local waited=0
    until carla_ready; do
        sleep 5
        waited=$((waited + 5))
        if ! kill -0 "$CARLA_PID" 2>/dev/null; then
            log "CARLA process died during startup; see carla.log"
            return 1
        fi
        if [ "$waited" -ge 240 ]; then
            log "CARLA did not answer a client handshake within 240s"
            return 1
        fi
    done
    log "CARLA ready after ${waited}s (pid ${CARLA_PID})"
    return 0
}

start_training() {
    launch_carla || return 1
    log "starting ${ENTRY}"
    # CUDA_VISIBLE_DEVICES pins the trainer to one card: JAX otherwise
    # preallocates on every visible GPU and would intrude on other users' work.
    # XLA_PYTHON_CLIENT_MEM_FRACTION caps this job's share so several runs fit on
    # the same card; jaxagent applies its 0.8 default only when this is unset,
    # and 0.8 leaves no room for a second concurrent run.
    CUDA_VISIBLE_DEVICES="$GPU" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION:-0.35}" \
    "$PYTHON" -u "$ENTRY" \
        --env.world.carla_port "$CARLA_PORT" \
        --env.world.traffic.tm_seed "$CARLA_PORT" \
        --dreamerv3.logdir "$LOGDIR" \
        "${EXTRA[@]}" >> "$LOG" 2>&1 &
    TRAIN_PID=$!
    TRAIN_STARTED_AT=$(date +%s)
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
# A hung CARLA leaves the trainer alive but spinning: the process check passes
# forever while no step is taken.  Treat a metrics file that has stopped growing
# as a failure too, otherwise a hang silently burns the rest of the run.
STALL_SECONDS="${STALL_SECONDS:-900}"
while true; do
    sleep 60

    stalled=0
    metrics="${LOGDIR}/metrics.jsonl"
    if [ -f "$metrics" ]; then
        # Measure from whichever is later: the last metric written, or the moment
        # this trainer started.  Using the file mtime alone kills a resumed run
        # on sight, because the metrics file it inherits from the previous
        # attempt is already older than the limit and it is never granted a
        # grace period in which to write its first line.
        last=$(stat -c %Y "$metrics")
        if [ "${TRAIN_STARTED_AT:-0}" -gt "$last" ]; then
            last="$TRAIN_STARTED_AT"
        fi
        age=$(( $(date +%s) - last ))
        if [ "$age" -ge "$STALL_SECONDS" ]; then
            log "no metrics written for ${age}s (limit ${STALL_SECONDS}s); treating as hung"
            kill -9 "$TRAIN_PID" 2>/dev/null
            stalled=1
        fi
    fi

    # Track this run's own child rather than pattern-matching every trainer on
    # the machine, so concurrent runs cannot mistake each other for themselves.
    if [ "$stalled" -eq 1 ] || ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        wait "$TRAIN_PID" 2>/dev/null
        code=$?
        if [ "$code" -eq 0 ] && [ "$stalled" -eq 0 ]; then
            log "training finished cleanly"
            # The replay buffer is about half the footprint of a finished run and
            # is only needed to resume one.  Over a hundred-run sweep it is the
            # difference between fitting on the disk and filling a volume other
            # people are also using.  Opt-in, because pruning forfeits resuming.
            if [ "${PRUNE_REPLAY:-0}" = "1" ] && [ -d "${LOGDIR}/replay" ]; then
                local freed
                freed=$(du -sh "${LOGDIR}/replay" 2>/dev/null | cut -f1)
                rm -rf "${LOGDIR}/replay"
                log "pruned the replay buffer (${freed} reclaimed)"
            fi
            cleanup
        fi
        RESTARTS=$((RESTARTS + 1))
        if [ "$RESTARTS" -gt "$MAX_RESTARTS" ]; then
            log "exceeded ${MAX_RESTARTS} restarts, giving up"
            cleanup
        fi
        log "trainer exited with ${code}; restart ${RESTARTS}/${MAX_RESTARTS}"
        # A trainer crash is usually a simulator problem, so recycle CARLA rather
        # than reconnecting a fresh trainer to a sick server.
        [ -n "${CARLA_PID:-}" ] && kill -9 "$CARLA_PID" 2>/dev/null
        fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1
        CARLA_PID=""
        sleep 5
        start_training || { log "restart failed"; cleanup; }
    fi
done
