#!/usr/bin/env bash
# Experiment 4: does the paper's failure detector tell a training-like standing pose from a far or turned one?
#
# The detector is LLMD of VLA-FAIL (arXiv 2606.21386; docs/papers/vla_fail.pdf in SIMPLE): the Mahalanobis distance
# of the action expert's last-layer features (one pass at the noise end of the flow, fixed noise) to the training
# data. Task: open the faucet, the released Psi0 checkpoint. It is judged on the 281 base poses of experiment 3
# (scripts/viz/vlm_camera_sweep/run_vlm_camera_sweep.sh, the scene of training episode 0: close, far, sideways,
# turned), against experiment 3's best signal on the same poses. The paper's second detector (action chunk
# consistency) is not tested: it compares consecutive chunks of a moving robot.
#
#   0. prepare  Psi0 venv:   the plan of 175 new renders (every training start in its own scene, the starts of the
#               teleop recordings that were not trained on, 19 more poses in the grid scene), and the first
#               observation + outcome of the 810 closed-loop trials of the trajectory and target-pose sweeps  (~1 min)
#   1. render   SIMPLE venv: Isaac Sim, one scene reset per start, renderer only, in passes of 40 scenes       (~50 min)
#   2. extract  Psi0 venv:   the hook check (timestep 1000 = pure noise, the hooked tensor is the final layer's
#               input, our pass = the served policy's first step), then the features of all 35,808 training
#               frames and of every test image, raw and through the training videos' H.264 round trip          (~40 min)
#   3. fit      Psi0 venv:   per-token statistics in float64 with ridge 1e-6, every training episode left out in
#               turn; the paper's detector, the start-referenced ones (tied covariance, Ledoit-Wolf), thresholds
#               from the re-rendered training starts, the pass/fail rule fixed in advance                      (~25 min)
#   4. report   Psi0 venv:   $OUT/index.html, one conclusion per chart
# Needs experiment 3's output (its renders and its index of the training set); run that first if it is missing.
#
# Usage:
#   scripts/viz/llmd_indist/run_llmd_indist.sh                    # everything, defaults below
#   scripts/viz/llmd_indist/run_llmd_indist.sh --tmux             # detached tmux session
#   SKIP_RENDER=1 SKIP_EXTRACT=1 .../run_llmd_indist.sh           # fit and report again from the saved features
#   SKIP_RENDER=1 SKIP_EXTRACT=1 SKIP_FIT=1 .../run_llmd_indist.sh   # the page only
#   SETS=grid,grid_h264 .../run_llmd_indist.sh                    # extract only some observation sets
#   REPORT_ARGS="--fragment <file>" ...                           # also the page without its document head
#   OPEN=1 .../run_llmd_indist.sh                                 # open index.html at the end
# Disk: about 45 GB under $OUT/feats (the cached VLM output of the training frames is 13 GB of it; KEEP_VLM=0
# deletes that cache after the features are extracted).
# Only one Isaac process fits in this machine's memory: don't run another sweep, replay or teleop at the same time.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI0_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIMPLE_DIR="${SIMPLE_DIR:-$(cd "$PSI0_DIR/../SIMPLE" && pwd)}"

EPISODE="${EPISODE:-0}"                  # training episode whose scene experiment 3's grid is in
EXP3="${EXP3:-$PSI0_DIR/.runs/vlm_camera_sweep/faucet_ep${EPISODE}_indist}"
OUT="${OUT:-$PSI0_DIR/.runs/llmd_indist/faucet_ep${EPISODE}}"
ENV_ID="${ENV_ID:-simple/G1WholebodyOpenFaucetTeleop-v0}"
SETS="${SETS:-all}"                      # observation sets to extract (see extract_llmd_features.py)
SEEDS="${SEEDS:-0,1,2}"                  # seeds of the fixed noise
BATCH="${BATCH:-16}"
MAX_SCENES="${MAX_SCENES:-40}"           # scene resets per Isaac process
WORKERS="${WORKERS:-5}"                  # tokens fitted in parallel
THREADS="${THREADS:-4}"                  # BLAS threads of each
KEEP_VLM="${KEEP_VLM:-1}"
RENDER_ARGS="${RENDER_ARGS:-}"
EXTRACT_ARGS="${EXTRACT_ARGS:-}"
FIT_ARGS="${FIT_ARGS:-}"
REPORT_ARGS="${REPORT_ARGS:-}"

if [[ "${1:-}" == "--tmux" ]]; then
  session="llmd_indist_$(date +%H%M%S)"
  env_vars=""
  for v in SIMPLE_DIR EPISODE EXP3 OUT ENV_ID SETS SEEDS BATCH MAX_SCENES WORKERS THREADS KEEP_VLM RENDER_ARGS \
           EXTRACT_ARGS FIT_ARGS REPORT_ARGS SKIP_RENDER SKIP_EXTRACT SKIP_FIT OPEN CUDA_VISIBLE_DEVICES; do
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
for f in prep/train_index.json prep/plan.json prep/room_map.npz exp3_indist_signal/index.jsonl exp3_indist_signal/meta.json \
         exp3_indist_signal/figures/summary.json; do
  [[ -f "$EXP3/$f" ]] || { echo "experiment 3's output is missing ($EXP3/$f); run first:" >&2
                           echo "  EPISODE=$EPISODE $PSI0_DIR/scripts/viz/vlm_camera_sweep/run_vlm_camera_sweep.sh" >&2; exit 1; }
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
PY="$PSI0_DIR/.venv-psi/bin/python"
mkdir -p "$OUT/prep" "$OUT/logs"
echo "output: $OUT"

follow() {  # <log> <pid> <grep pattern> [first line]: show a stage's progress lines while it runs
  tail -n "+${4:-1}" -f "$1" --pid "$2" 2>/dev/null | grep --line-buffered -E "$3" || true
}

if [[ "${SKIP_RENDER:-0}" != "1" || ! -f "$OUT/prep/render_plan.json" ]]; then
  echo "=== [0/4] prepare (log: $OUT/logs/prepare.log)"
  (cd "$PSI0_DIR" && "$PY" -u "$SCRIPT_DIR/prepare_llmd.py" --exp3 "$EXP3" --simple-dir "$SIMPLE_DIR" --out "$OUT/prep") \
    2>&1 | tee "$OUT/logs/prepare.log" | grep --line-buffered -E "^\[prepare\]|Traceback|Error|rror:" || true
  [[ "${PIPESTATUS[0]}" == "0" ]] || { echo "prepare failed, see $OUT/logs/prepare.log" >&2; exit 1; }
fi
if [[ ! -f "$OUT/trials/trials.jsonl" ]]; then
  echo "=== [0/4] index the earlier closed-loop trials (log: $OUT/logs/trials.log)"
  (cd "$PSI0_DIR" && "$PY" -u "$SCRIPT_DIR/index_trials.py" --psi0-dir "$PSI0_DIR" --simple-dir "$SIMPLE_DIR" \
      --exp3 "$EXP3" --out "$OUT/trials") > "$OUT/logs/trials.log" 2>&1 \
    || { echo "WARNING: the earlier trials could not be indexed (see $OUT/logs/trials.log); continuing without them" >&2
         rm -f "$OUT/trials/trials.jsonl"; }
  tail -n 3 "$OUT/logs/trials.log" || true
fi

if [[ "${SKIP_RENDER:-0}" != "1" ]]; then
  echo "=== [1/4] render (Isaac Sim, log: $OUT/logs/render.log)"
  : > "$OUT/logs/render.log"
  for pass in $(seq 1 12); do
    from=$(( $(wc -l < "$OUT/logs/render.log") + 1 ))  # show only this pass's lines
    # shellcheck disable=SC2086
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" .venv/bin/python -u "$SCRIPT_DIR/render_starts.py" \
        --env-id "$ENV_ID" --plan "$OUT/prep/render_plan.json" --out "$OUT/renders" --max-scenes "$MAX_SCENES" $RENDER_ARGS) \
      >> "$OUT/logs/render.log" 2>&1 &
    pid=$!
    follow "$OUT/logs/render.log" "$pid" "^(scene|done|wrote|WARNING:)|Traceback|Error:" "$from"
    wait "$pid" || { echo "render failed, see $OUT/logs/render.log" >&2; tail -n 30 "$OUT/logs/render.log" >&2; exit 1; }
    tail -n 5 "$OUT/logs/render.log" | grep -q "^done: .*wrote all" && break
  done
  tail -n 5 "$OUT/logs/render.log" | grep -q "^done: .*wrote all" || { echo "render did not finish, see $OUT/logs/render.log" >&2; exit 1; }
fi

if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
  echo "=== [2/4] extract the features (log: $OUT/logs/extract.log)"
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null || echo 99999)
  (( free_mb >= 9000 )) || echo "WARNING: only ${free_mb} MiB GPU memory free; the model needs ~8 GB (is a policy server running?)" >&2
  # shellcheck disable=SC2086
  (cd "$PSI0_DIR" && "$PY" -u "$SCRIPT_DIR/extract_llmd_features.py" --out "$OUT" --exp3 "$EXP3" --simple-dir "$SIMPLE_DIR" \
      --sets "$SETS" --seeds "$SEEDS" --batch "$BATCH" $EXTRACT_ARGS) > "$OUT/logs/extract.log" 2>&1 &
  pid=$!
  follow "$OUT/logs/extract.log" "$pid" "^\[|^saved|Traceback|Error"
  wait "$pid" || { echo "extraction failed, see $OUT/logs/extract.log" >&2; tail -n 30 "$OUT/logs/extract.log" >&2; exit 1; }
  if [[ "$KEEP_VLM" != "1" ]]; then rm -f "$OUT/feats/train/vlm.npy"; fi
fi

if [[ "${SKIP_FIT:-0}" != "1" ]]; then
  echo "=== [3/4] fit and score (log: $OUT/logs/fit.log)"
  # shellcheck disable=SC2086
  (cd "$PSI0_DIR" && OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
      "$PY" -u "$SCRIPT_DIR/fit_llmd.py" --out "$OUT" --exp3 "$EXP3" --workers "$WORKERS" $FIT_ARGS) 2>&1 \
    | tee "$OUT/logs/fit.log" | grep --line-buffered -E "^\[fit\]|^wrote|Traceback|Error" || true
  [[ "${PIPESTATUS[0]}" == "0" ]] || { echo "the fit failed, see $OUT/logs/fit.log" >&2; exit 1; }
fi

echo "=== [4/4] report"
# shellcheck disable=SC2086
(cd "$PSI0_DIR" && "$PY" -u "$SCRIPT_DIR/llmd_report.py" --out "$OUT" --exp3 "$EXP3" $REPORT_ARGS)
echo
echo "results: $OUT/index.html"
if [[ "${OPEN:-0}" == "1" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$OUT/index.html" >/dev/null 2>&1 || true
fi
