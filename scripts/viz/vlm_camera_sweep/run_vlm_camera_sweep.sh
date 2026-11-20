#!/usr/bin/env bash
# What does the Psi0 VLM latent say about where the robot stands and looks? Two tasks, one command each
# (no policy server and no controller needed):
#
# TASK=faucet (default): experiment 3, is there an in-distribution signal in the VLA latent?
#   Could a navigation policy read from the latent when the robot's view matches what the VLA started from in
#   training? For the released open-faucet checkpoint: at most BUDGET (300) base poses around the working table, in
#   the scene of training episode EPISODE, are rendered and scored against the first frame of every training episode.
#   0. prepare  Psi0 venv:   the checkpoint, its train pack and the raw teleop pack (downloaded if missing; stops
#               unless the pack is the checkpoint's training set), every training start's pelvis pose and joints;
#               SIMPLE venv: a top-down map of the room (scene_sweep/room_map.py, no Isaac process); then the plan:
#               the training start, a fine lattice over the training range of base poses, a medium one around it, a
#               coarse one over the floor around the table, the start turned on the spot, and poses facing the faucet
#               from around the table ($OUT/prep/plan.png, plan.json)                                      (~1 min)
#   1. render   SIMPLE venv: the scene reset as recorded, the robot in the posture of the training start, the pelvis
#               moved to every plan pose, the head camera rendered                                         (~4 min)
#   2. extract  Psi0 venv:   the checkpoint's frozen VLM, all 29 layers, and its action expert with a recording
#               attention processor, on the training start frames and on the renders: the image tokens averaged,
#               averaged with the action tokens' attention, and kept in place; the attention itself         (~4 min)
#   3. plot     Psi0 venv:   per latent and layer a score against the training starts (kNN, PCA subspace, token
#               match), thresholded on the training starts' own scores; 2 clusters without labels; whether following
#               the score leads back to the training range; the result page                                (~1 min)
#   Result: $OUT/index.html, one self-contained page: a conclusion per section over one chart, a pose's camera view
#   on hover. Detail figures of every signal and layer, summary.json, candidates.csv: $OUT/exp3_indist_signal/figures.
#
# TASK=carrybox: experiments 1 and 2, how does the latent react when the head camera moves? (as before)
#   0. check      the anchor frame is a training sample of the checkpoint: pixel-identical image and identical
#                 state in the pack the checkpoint was fine-tuned on (stops the run otherwise)    (<1 min)
#   experiment 1  camera sweep in the training scene: the head camera moved one DoF at a time around the
#                 anchor, plus the floor grid and the re-rendered episodes (figures 1-8)         (~12 min)
#   experiment 2  the same camera sweeps in a scene that is not in the training set (lighting +
#                 materials re-drawn, a table no training episode uses), figure 1b               (~6 min)
#   overview      <OUT>/index.html: the anchor check and both experiments on one page
# Each of these two experiments runs three stages:
#   render   SIMPLE venv: reset the episode's recorded scene, put the robot at the anchor frame, re-render
#            the head camera under pose offsets
#   extract  Psi0 venv:   the served checkpoint's frozen VLM (exactly like /act) -> latents, and the action
#            expert's predicted chunk for each render
#   plot     Psi0 venv:   figures + summary.json + index.html
#
# Data (carrybox): the checkpoint was fine-tuned on G1WholebodyXMoveBendCarryBoxSonic-v0 (50 episodes; TRAIN_PACK), which is
# G1WholebodyXMoveBendPickVariant1Teleop-v0 post-processed: same images and states, but only the source keeps
# each episode's scene (table material, lighting) in meta/episodes.jsonl. So DATA_DIR is the source and stage 0
# proves the anchor is the same sample in TRAIN_PACK. Both are downloaded from USC-PSI-Lab/psi-data on first use.
# .../G1WholebodyXMoveBendCarryBoxSonic-v0/dr-level-0 is the 5-episode *eval* pack (held out; fails stage 0).
#
# Usage:
#   scripts/viz/vlm_camera_sweep/run_vlm_camera_sweep.sh                 # faucet, experiment 3, defaults below
#   EPISODE=17 .../run_vlm_camera_sweep.sh                               # the grid in another training scene
#   BUDGET=200 .../run_vlm_camera_sweep.sh                               # fewer base poses (the coarse lattice thins)
#   PLAN_ARGS="--facing-step 0 --yaws ''" .../run_vlm_camera_sweep.sh    # lattices at the training heading only
#   EXTRACT_ARGS="--spatial-layers all" .../run_vlm_camera_sweep.sh      # keep every layer's image tokens (4 GB)
#   PLOT_ARGS="--band 0.05 --k 5" SKIP_RENDER=1 SKIP_EXTRACT=1 .../run_vlm_camera_sweep.sh   # re-plot only
#   PLOT_ARGS="--fragment <file>" SKIP_RENDER=1 SKIP_EXTRACT=1 .../run_vlm_camera_sweep.sh   # + the page without its head
#   RUN_DIR=<run dir> CKPT_STEP=40000 .../run_vlm_camera_sweep.sh        # another checkpoint of the same train pack
#   TASK=carrybox .../run_vlm_camera_sweep.sh                            # experiments 1 and 2, defaults below
#   TASK=carrybox EPISODE=2 ANCHOR_FRAME=250 .../run_vlm_camera_sweep.sh # another training frame
#   TASK=carrybox MODE=camera .../run_vlm_camera_sweep.sh                # move only the camera
#   EXPERIMENTS=1 .../run_vlm_camera_sweep.sh      # only the camera sweep in the training scene (implies carrybox)
#   TASK=carrybox SCENE2_SEED=7 .../run_vlm_camera_sweep.sh              # another second scene for experiment 2
#   TASK=carrybox RENDER_ARGS="--steps 25 --yaw-range -60 60" .../run_vlm_camera_sweep.sh
#   SKIP_RENDER=1 SKIP_EXTRACT=1 OUT=<existing dir> .../run_vlm_camera_sweep.sh   # re-plot only
#   TASK=carrybox CHECK_ANCHOR=0 DATA_DIR=<other dataset> .../run_vlm_camera_sweep.sh   # anchor outside the training set
#   .../run_vlm_camera_sweep.sh --tmux                                   # detached tmux session
#   OPEN=1 .../run_vlm_camera_sweep.sh                                   # open index.html at the end
# Only one Isaac process fits in this machine's memory: don't run another sweep, replay or teleop at the same time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI0_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIMPLE_DIR="${SIMPLE_DIR:-$(cd "$PSI0_DIR/../SIMPLE" && pwd)}"

# experiments 1 and 2 are the carry-box camera sweep, experiment 3 the faucet in-distribution signal
TASK="${TASK:-$([[ "${EXPERIMENTS:-3}" =~ ^[12\ ]+$ ]] && echo carrybox || echo faucet)}"
case "$TASK" in
  faucet)
    SIM_TASK="${SIM_TASK:-G1WholebodyOpenFaucetTeleop-v0}"   # the SIMPLE env; its released checkpoint and packs
    ENV_ID="${ENV_ID:-simple/$SIM_TASK}"
    RUN_DIR_GIVEN="${RUN_DIR:-}"       # empty = the task's released checkpoint (downloaded if missing)
    EXPERIMENTS="${EXPERIMENTS-3}"
    BUDGET="${BUDGET:-300}"            # most base poses rendered
    PLAN_ARGS="${PLAN_ARGS:-}"         # extra flags for prepare_base_grid.py plan (see its --help)
    EXTRACT_ARGS="${EXTRACT_ARGS:-}"   # extra flags for extract_grid_latents.py
    PLOT_ARGS="${PLOT_ARGS:-}"         # extra flags for plot_indist_signal.py
    OUT="${OUT:-$PSI0_DIR/.runs/vlm_camera_sweep/faucet_ep${EPISODE:-0}_indist}"
    CHECK_ANCHOR=0 ;;                  # (experiment 3 checks the train pack against the checkpoint itself)
  carrybox) ;;
  *) echo "unknown TASK '$TASK' (faucet or carrybox)" >&2; exit 1 ;;
esac
ENV_ID="${ENV_ID:-simple/G1WholebodyXMoveBendCarryBoxSonic-v0}"
TRAIN_DATA="$SIMPLE_DIR/data/simple/G1WholebodyXMoveBendPickVariant1Teleop-v0/level-0"
TRAIN_DATA_URL="https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/main/G1WholebodyXMoveBendPickVariant1Teleop-v0.zip"
TRAIN_PACK="${TRAIN_PACK:-$SIMPLE_DIR/data/psi-data/simple/G1WholebodyXMoveBendCarryBoxSonic-v0}"
TRAIN_PACK_URL="https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/main/simple/G1WholebodyXMoveBendCarryBoxSonic-v0.zip"
DATA_DIR="${DATA_DIR:-$TRAIN_DATA}"
DATA_LABEL="${DATA_LABEL:-$([[ "$DATA_DIR" == "$TRAIN_DATA" ]] && echo "training set" || basename "$(dirname "$DATA_DIR")")}"
DATA_TAG="${DATA_TAG:-$([[ "$DATA_DIR" == "$TRAIN_DATA" ]] && echo train || echo data)}"
RUN_DIR="${RUN_DIR:-$PSI0_DIR/.runs/finetune/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223}"
CKPT_STEP="${CKPT_STEP:-40000}"
EPISODE="${EPISODE:-0}"
ANCHOR_FRAME="${ANCHOR_FRAME:-100}"  # training renders settle after ~10 frames; 100 is still the initial stand
MODE="${MODE:-rigid}"            # rigid: robot moves with the camera | camera: only the camera moves
SEED="${SEED:-0}"                # table-material draw, only used with DR_LEVEL=0
DR_LEVEL="${DR_LEVEL:--1}"       # -1: each episode's recorded scene (as trained on) | 0: re-draw the table (eval)
GI="${GI:-on}"                   # RTX indirect diffuse: on like the training renderer
TRAJ_EPISODES="${TRAJ_EPISODES:-0,1,2,3,4}"  # re-rendered episodes (probes); the anchor episode is always in
EXPERIMENTS="${EXPERIMENTS-1 2}"         # 1: camera sweep, training scene | 2: + scene not in the training set
CHECK_ANCHOR="${CHECK_ANCHOR:-1}"        # stage 0: stop unless the anchor is a training sample
SCENE2_SEED="${SCENE2_SEED:-100}"        # experiment 2's scene; bumped past every training table material
SCENE2_DR_LEVEL="${SCENE2_DR_LEVEL:-1}"  # 1: re-draw lighting + table/robot materials, layout kept
RENDER_ARGS="${RENDER_ARGS:-}"   # extra flags for render_camera_sweep.py (see its --help)
OUT="${OUT:-$PSI0_DIR/.runs/vlm_camera_sweep/${DATA_TAG}_ep${EPISODE}_f${ANCHOR_FRAME}_${MODE}}"
if [[ "$TASK" == "faucet" ]]; then RUN_DIR="$RUN_DIR_GIVEN"; fi  # (the default above is the carry-box checkpoint)

if [[ "${1:-}" == "--tmux" ]]; then
  session="vlm_sweep_$(date +%H%M%S)"
  env_vars=""
  for v in TASK SIM_TASK BUDGET PLAN_ARGS EXTRACT_ARGS PLOT_ARGS \
           ENV_ID DATA_DIR DATA_LABEL DATA_TAG TRAIN_PACK RUN_DIR CKPT_STEP EPISODE ANCHOR_FRAME MODE SEED DR_LEVEL \
           GI TRAJ_EPISODES EXPERIMENTS CHECK_ANCHOR SCENE2_SEED SCENE2_DR_LEVEL RENDER_ARGS OUT \
           SKIP_RENDER SKIP_EXTRACT OPEN CUDA_VISIBLE_DEVICES; do
    [[ -n "${!v:-}" ]] && env_vars+="$v=$(printf '%q' "${!v}") "
  done
  tmux new-session -d -s "$session" \
    "env $env_vars bash $(printf '%q' "${BASH_SOURCE[0]}"); echo; echo '[done - press Enter to close]'; read"
  echo "started tmux session '$session'   attach: tmux attach -t $session"
  echo "results will be in: $OUT/index.html"
  exit 0
fi

[[ -x "$SIMPLE_DIR/.venv/bin/python" ]] || { echo "SIMPLE venv missing: $SIMPLE_DIR/.venv" >&2; exit 1; }
[[ -x "$PSI0_DIR/.venv-psi/bin/python" ]] || { echo "Psi0 venv missing: $PSI0_DIR/.venv-psi" >&2; exit 1; }
[[ "$TASK" == "faucet" || -d "$RUN_DIR" ]] || { echo "checkpoint run dir missing: $RUN_DIR" >&2; exit 1; }
fetch() {  # <url> <unzip into> <what>
  echo "downloading $3 -> $2"
  local zip; zip="$(mktemp --suffix=.zip)"
  curl -fL --retry 3 -o "$zip" "$1" && mkdir -p "$2" && unzip -q -o "$zip" -d "$2"
  rm -f "$zip"
}
if [[ "$TASK" == "carrybox" ]]; then
  [[ -d "$DATA_DIR" || "$DATA_DIR" != "$TRAIN_DATA" ]] || fetch "$TRAIN_DATA_URL" "$SIMPLE_DIR/data/simple" "the training-set source (180 MB)"
  [[ -d "$DATA_DIR" ]] || { echo "dataset missing: $DATA_DIR" >&2; exit 1; }
  [[ "$CHECK_ANCHOR" != "1" || -d "$TRAIN_PACK" ]] || fetch "$TRAIN_PACK_URL" "$(dirname "$TRAIN_PACK")" "the training pack (178 MB)"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "output: $OUT"

if [[ "$CHECK_ANCHOR" == "1" ]]; then
  echo "=== [0] is the anchor a training sample of the checkpoint?"
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/check_anchor.py" --data-dir "$DATA_DIR" \
      --train-pack "$TRAIN_PACK" --run-dir "$RUN_DIR" --episode "$EPISODE" --frame "$ANCHOR_FRAME" --out "$OUT") \
    || { echo "stopping: episode $EPISODE frame $ANCHOR_FRAME is not a training sample ($OUT/anchor_check.json)" >&2; exit 1; }
elif [[ "$TASK" == "carrybox" ]]; then
  rm -f "$OUT/anchor_check.json"
fi

gpu_check() {  # the model needs ~8 GB next to nothing else
  local free_mb
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null || echo 99999)
  if (( free_mb < 9000 )); then
    echo "WARNING: only ${free_mb} MiB GPU memory free; the model needs ~8 GB (is a policy server running?)" >&2
  fi
}

run_indist() {  # <dir>: experiment 3, see the header
  local dir="$1" prep="$OUT/prep" pid usd room yaw ox oy
  mkdir -p "$dir" "$prep"
  echo
  echo "##### experiment 3: is there an in-distribution signal in the VLA latent? -> $dir"
  if [[ "${SKIP_RENDER:-0}" != "1" || ! -f "$prep/plan.json" ]]; then
    echo "=== [0/3] prepare (log: $prep/prepare.log)"
    (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/prepare_base_grid.py" data --task "$SIM_TASK" \
        --psi0-dir "$PSI0_DIR" --simple-dir "$SIMPLE_DIR" --out "$prep" --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP" \
        --episode "$EPISODE") 2>&1 | tee "$prep/prepare.log" | grep --line-buffered -E "^\[prepare\]|Traceback|Error|rror:" || true
    [[ "${PIPESTATUS[0]}" == "0" ]] || { echo "prepare failed, see $prep/prepare.log" >&2; tail -n 5 "$prep/prepare.log" >&2; exit 1; }
    if ! cmp -s "$prep/room.json" "$prep/room_map.room.json" || [[ ! -f "$prep/room_map.npz" ]]; then
      read -r room yaw ox oy < <("$SIMPLE_DIR/.venv/bin/python" -c \
        'import json, sys; r = json.load(open(sys.argv[1])); print(r["room"], r["yaw"], *r["offset"])' "$prep/room.json")
      usd="$(ls -d "$SIMPLE_DIR"/.venv/lib/python3*/site-packages/isaacsim/extscache/omni.usd.libs-* | head -n 1)"
      (cd "$SIMPLE_DIR" && PYTHONPATH="$usd" LD_LIBRARY_PATH="$usd/bin:${LD_LIBRARY_PATH:-}" .venv/bin/python -u \
          "$SCRIPT_DIR/../scene_sweep/room_map.py" --simple-dir "$SIMPLE_DIR" --room "$room" --yaw "$yaw" \
          --offset "$ox" "$oy" --out "$prep/room_map.npz" --png "$prep/room_map.png") >> "$prep/prepare.log" 2>&1 \
        || { echo "room_map.py failed, see $prep/prepare.log" >&2; exit 1; }
      cp "$prep/room.json" "$prep/room_map.room.json"
      grep "^\[room_map\]" "$prep/prepare.log" | tail -n 1 || true
    fi
    # shellcheck disable=SC2086
    (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/prepare_base_grid.py" plan --task "$SIM_TASK" \
        --psi0-dir "$PSI0_DIR" --simple-dir "$SIMPLE_DIR" --out "$prep" --episode "$EPISODE" --budget "$BUDGET" \
        $PLAN_ARGS) 2>&1 | tee -a "$prep/prepare.log" | grep --line-buffered -E "^\[plan\]|Traceback|Error|rror:" || true
    [[ "${PIPESTATUS[0]}" == "0" ]] || { echo "the plan failed, see $prep/prepare.log" >&2; tail -n 5 "$prep/prepare.log" >&2; exit 1; }
    echo "the plan, top-down: $prep/plan.png"
  fi
  if [[ "${SKIP_RENDER:-0}" != "1" ]]; then
    echo "=== [1/3] render (Isaac Sim, log: $dir/render.log)"
    rm -rf "$dir/img" "$dir/index.jsonl" "$dir/meta.json"
    # shellcheck disable=SC2086
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python -u "$SCRIPT_DIR/render_base_grid.py" \
        --env-id "$ENV_ID" --prep "$prep" --out "$dir" --gi "$GI" $RENDER_ARGS) > "$dir/render.log" 2>&1 &
    pid=$!
    tail -n +1 -f "$dir/render.log" --pid "$pid" 2>/dev/null \
      | grep --line-buffered -E "^(anchor|grid|wrote|WARNING:)|Traceback|Error:" || true
    wait "$pid" || { echo "render failed, see $dir/render.log" >&2; tail -n 30 "$dir/render.log" >&2; exit 1; }
    grep -q "^wrote " "$dir/render.log" || { echo "render did not finish, see $dir/render.log" >&2; exit 1; }
  fi
  if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
    echo "=== [2/3] extract the latents (log: $dir/extract.log)"
    gpu_check
    rm -rf "$dir/latents" "$dir/latents.json"
    # shellcheck disable=SC2086
    (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/extract_grid_latents.py" --renders "$dir" \
        --prep "$prep" $EXTRACT_ARGS) > "$dir/extract.log" 2>&1 &
    pid=$!
    tail -n +1 -f "$dir/extract.log" --pid "$pid" 2>/dev/null | grep --line-buffered -E "^\[[0-9t]|^saved|Traceback|Error" || true
    wait "$pid" || { echo "extraction failed, see $dir/extract.log" >&2; tail -n 30 "$dir/extract.log" >&2; exit 1; }
  fi
  echo "=== [3/3] plot"
  # shellcheck disable=SC2086
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_indist_signal.py" --renders "$dir" --prep "$prep" \
      --report "$OUT/index.html" $PLOT_ARGS)
}

run_experiment() {  # <dir> <title> <scene2 seed> <extra render args>
  local dir="$1" title="$2" scene2="$3" extra="$4" pid
  mkdir -p "$dir"
  echo
  echo "##### $title -> $dir"
  if [[ "${SKIP_RENDER:-0}" != "1" ]]; then
    echo "=== render (Isaac Sim, log: $dir/render.log)"
    rm -rf "$dir/img" "$dir/index.jsonl" "$dir/meta.json"
    # shellcheck disable=SC2086
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python -u "$SCRIPT_DIR/render_camera_sweep.py" \
        --env-id "$ENV_ID" --data-dir "$DATA_DIR" --out "$dir" --episode "$EPISODE" \
        --anchor-frame "$ANCHOR_FRAME" --mode "$MODE" --seed "$SEED" --dr-level "$DR_LEVEL" --gi "$GI" \
        --data-label "$DATA_LABEL" --traj-episodes "$TRAJ_EPISODES" \
        --scene2-seed "$scene2" --scene2-dr-level "$SCENE2_DR_LEVEL" $RENDER_ARGS $extra) \
      > "$dir/render.log" 2>&1 &
    pid=$!
    # Isaac prints a lot; show only this script's progress lines
    tail -n +1 -f "$dir/render.log" --pid "$pid" 2>/dev/null \
      | grep --line-buffered -E "^(anchor|sweep|grid|traj|scene2|wrote|WARNING:)|Traceback|Error:" || true
    wait "$pid" || { echo "render failed, see $dir/render.log" >&2; tail -n 30 "$dir/render.log" >&2; exit 1; }
    grep -q "^wrote " "$dir/render.log" || { echo "render did not finish, see $dir/render.log" >&2; exit 1; }
  fi
  if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
    echo "=== extract VLM latents (log: $dir/extract.log)"
    local free_mb
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null || echo 99999)
    if (( free_mb < 9000 )); then
      echo "WARNING: only ${free_mb} MiB GPU memory free; the model needs ~8 GB (is a policy server running?)" >&2
    fi
    (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/extract_sweep_latents.py" \
        --renders "$dir" --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP") > "$dir/extract.log" 2>&1 &
    pid=$!
    tail -n +1 -f "$dir/extract.log" --pid "$pid" 2>/dev/null | grep --line-buffered -E "^\[|^saved|Traceback|Error" || true
    wait "$pid" || { echo "extraction failed, see $dir/extract.log" >&2; tail -n 30 "$dir/extract.log" >&2; exit 1; }
  fi
  echo "=== plot"
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_sweep.py" --renders "$dir" --data-dir "$DATA_DIR")
}

for e in $EXPERIMENTS; do
  case "$e" in
    1|2) [[ "$TASK" == "carrybox" ]] || { echo "experiments 1 and 2 are the carry-box task's (TASK=carrybox)" >&2; exit 1; } ;;&
    1) run_experiment "$OUT/exp1_camera_sweep" "experiment 1: camera sweep in the training scene" -1 "" ;;
    # the training-scene sweeps again (same anchor, the blue curves) + the same sweeps in scene 2; no grid/episodes
    2) run_experiment "$OUT/exp2_scene_change" "experiment 2: the same sweeps in a scene not in the training set" \
         "$SCENE2_SEED" "--grid-steps 0 --traj-episodes none" ;;
    3) [[ "$TASK" == "faucet" ]] || { echo "experiment 3 is the faucet task's (TASK=faucet)" >&2; exit 1; }
       run_indist "$OUT/exp3_indist_signal" ;;
    *) echo "unknown experiment '$e' (EXPERIMENTS takes 1 and/or 2 with TASK=carrybox, 3 with TASK=faucet)" >&2; exit 1 ;;
  esac
done

if [[ "$TASK" == "carrybox" ]]; then  # the faucet task's plot stage writes $OUT/index.html itself
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/overview.py" --out "$OUT")
fi
echo
echo "results: $OUT/index.html"
if [[ "${OPEN:-0}" == "1" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$OUT/index.html" >/dev/null 2>&1 || true
fi
