#!/bin/bash
# ---------------------------------------------------------------------------
# ProSafeAV -- TITS R1 revision experiment matrix.
#
# Every run is one line.  Each needs its own CARLA port and GPU slot, so the
# script prints the commands by default and only executes them with --run.
#
#   ./tools/run_revision_experiments.sh                # show the matrix
#   ./tools/run_revision_experiments.sh --run main     # launch one group
#   ./tools/run_revision_experiments.sh --run all
#
# Groups map onto the reviewer comments:
#
#   main      the headline ProSafeAV result on the new stochastic scenario
#   density   R1.6  generalisation across traffic density
#   perception R1.3 BEV obtainable from onboard perception (FOV limited, noisy)
#   wm        R1.2  cross-architecture world model comparison
#   ablation  R2/R3 where the EVT term acts, and the world-model capacity sweep
#   sensitivity  R2.2/R2.3  copula family, threshold rule, EVT weight
#   seeds     three seeds per headline configuration
#
# Each configuration should be run with THREE seeds before anything is reported;
# the tables in the manuscript quote mean +/- standard error over seeds.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="/home/yukai/CarDreamer_prosafeav"
LOGROOT="${REPO}/logdir"
STEPS="${STEPS:-3e5}"
SEEDS="${SEEDS:-0 1 2}"
BASE_PORT="${BASE_PORT:-2000}"
GPUS=(0 1)

MODE="print"
GROUP="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) MODE="run"; GROUP="${2:-all}"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

port_counter=0
launch() {
  local group="$1" name="$2" task="$3"; shift 3
  local extra="$*"
  if [[ "$GROUP" != "all" && "$GROUP" != "$group" ]]; then return; fi
  for seed in $SEEDS; do
    local port=$((BASE_PORT + port_counter * 4))
    local gpu=${GPUS[$((port_counter % ${#GPUS[@]}))]}
    port_counter=$((port_counter + 1))
    local logdir="${LOGROOT}/${name}_s${seed}"
    local cmd="./train_prosafeav.sh ${port} ${gpu} \
--task ${task} \
--dreamerv3.logdir ${logdir} \
--dreamerv3.seed ${seed} \
--dreamerv3.run.steps ${STEPS} ${extra}"
    if [[ "$MODE" == "print" ]]; then
      echo "[${group}] ${cmd}"
    else
      echo ">>> launching ${name} seed ${seed} on port ${port} gpu ${gpu}"
      mkdir -p "${logdir}"
      (cd "${REPO}" && eval "${cmd}")
    fi
  done
}

# --- headline result ------------------------------------------------------- #
launch main prosafeav carla_overtake_prosafeav

# --- R1.6 traffic density -------------------------------------------------- #
for d in 05 15 30 45; do
  launch density "prosafeav_d${d}" "carla_overtake_d${d}"
done
# Zero-shot transfer: train at the medium density, evaluate at the others.
# See tools/eval_matrix.sh, which reuses one checkpoint across the density tasks.

# --- R1.3 perception realism ----------------------------------------------- #
launch perception prosafeav_fov   carla_overtake_fov
launch perception prosafeav_noisy carla_overtake_noisy

# --- R1.2 cross-architecture world models ---------------------------------- #
# DreamerV3 backbone at three capacities.  "rssm.classes 0" replaces the
# categorical latent with a diagonal Gaussian, which is the DreamerV2-style
# state space; it is a genuinely different latent family, not merely a smaller net.
launch wm prosafeav_dv3    carla_overtake_prosafeav
launch wm prosafeav_rssm_s carla_overtake_prosafeav \
  --dreamerv3.rssm.deter 256 --dreamerv3.rssm.stoch 16
launch wm prosafeav_gauss  carla_overtake_prosafeav \
  --dreamerv3.rssm.classes 0
# DreamerV2 backbone (separate entry point, kept for reference):
#   python dreamerv2/train.py --task carla_overtake_prosafeav ...

# --- ablations ------------------------------------------------------------- #
launch ablation prosafeav_noevt    carla_overtake_noevt
launch ablation prosafeav_evtenv   carla_overtake_evtenv
launch ablation prosafeav_evtimag  carla_overtake_evtimag
launch ablation prosafeav_lonly    carla_overtake_lonly

# --- sensitivity ----------------------------------------------------------- #
launch sensitivity prosafeav_frank carla_overtake_frank
launch sensitivity prosafeav_thrq  carla_overtake_thrq
launch sensitivity prosafeav_w1    carla_overtake_w1
launch sensitivity prosafeav_w10   carla_overtake_w10

# --- legacy reproduction --------------------------------------------------- #
launch legacy prosafeav_legacy carla_overtake_legacy

if [[ "$MODE" == "print" ]]; then
  cat <<'EOF'

---------------------------------------------------------------------------
Offline analyses that need no GPU:

  # Reviewer 1.4 -- fit the same EVT model to real trajectories
  python tools/evt_realdata_validation.py --max-frames 0 \
      --out logdir/evt_real_tgsim.json

  # ... and compare it against the model fitted during a training run
  python tools/evt_realdata_validation.py --max-frames 0 \
      --compare logdir/prosafeav_s0/evt_model.json \
      --out logdir/evt_real_vs_sim.json

  # Aggregate every evaluation run into one table
  python tools/collect_results.py --logdir logdir --pattern 'prosafeav_*' \
      --latex logdir/results_table.tex

  # Correctness checks
  python tools/test_prosafeav.py
  python tools/test_evt_jax.py
---------------------------------------------------------------------------
EOF
fi
