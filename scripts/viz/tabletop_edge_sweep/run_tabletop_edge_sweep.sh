#!/usr/bin/env bash
# Does the tabletop-grasp policy reach for the object, or for a spot set by the table's near edge?
#
# The OpenFaucet sweeps (../scene_sweep) found the legs stop a fixed distance from the visible table edge. This asks
# the same of an arms-only task: closed-loop trials of the G1WholebodyTabletopGraspMP-v0 Psi0 checkpoint (default: the
# task's released one) in two sets of TRIALS scenes each. Trial i starts from the same scene in both sets: one of the
# task's eval-pack scenes (the robot spawn) randomized the way SIMPLE's highest replay level does it (distractors,
# lighting, and the poses of the chips can and the distractors within the task's ranges), seeded per trial. The table
# is drawn as in the training images (the room's own counter; SIMPLE's evaluation draws a thin slab with a random
# material instead, which the training images never show; LOOK=simple keeps that). Sets:
#   c0_baseline   nothing changed
#   e1_edge_x     the counter's near edge drawn FARTHER farther to CLOSER closer (spread over the trials); the can,
#                 the distractors, the robot and the physics stay (the robot and the objects collide only with the
#                 recorded table); the texture keeps its scale
# Trials run index by index (c0 0, e1 0, c0 1, ...), so a partial run compares like with like. Success is SIMPLE's
# test: the can lifted 1.8 cm. Output: $OUT/report/index.html (success per set and pair by pair; how far the right
# hand, 1.2 s in, follows the edge (same scene, edge moved minus baseline) and the can (across scenes, against the
# demos); a top-down map of every trial's drawn edge and can; every trial's video) and $OUT/report/results.json
# (with every trial's 10 Hz trace in report/traces). Once every trial has finished, the report folder is all OUT keeps.
#
#   0. prepare  Psi0 venv:   checkpoint, eval pack, training pack, training geometry (prepare_edge_sweep.py);
#               SIMPLE venv: every trial's scene file (make_edge_trials.py, CPU, about a minute) and the demos' own
#               reach against the can (demo_reach.py: forward kinematics of their recorded joints)
#   1. serve    Psi0 venv:   serve_psi0_simple with the checkpoint (port $PORT, log server.log)
#   2. eval     SIMPLE venv: SIMPLE's MP eval loop (eval.py, agent psi0), unchanged, driven by eval_edge_sweep.py:
#               one episode per trial, CHUNK trials per Isaac process; a crash is resumed, re-running continues
#   3. report   Psi0 venv:   report/ (plot_edge_sweep.py); then, if every trial has finished, the rest of OUT goes
#
# Usage:  run_tabletop_edge_sweep.sh [TASK] [TRIALS] [--tmux]
#   scripts/viz/tabletop_edge_sweep/run_tabletop_edge_sweep.sh --tmux      # 50 + 50 trials, in tmux
#   .../run_tabletop_edge_sweep.sh 2                                       # 2 + 2, for a quick look
#   CLOSER=0.10 FARTHER=0.15 .../run_tabletop_edge_sweep.sh                # the edge range, m
#   SETS=c0_baseline .../run_tabletop_edge_sweep.sh                        # one set only
#   LOOK=simple .../run_tabletop_edge_sweep.sh                             # SIMPLE's own table drawing
#   RUN_DIR=<run dir> CKPT_STEP=40000 .../run_tabletop_edge_sweep.sh       # another checkpoint
#   KEEP=1 ...   keep everything in OUT;  SKIP_EVAL=1 OUT=<dir> ...   rebuild the report (KEEP=1 runs)
# A trial takes about 1-2 min when the can is lifted early, 4-5 min when it runs the 800-step budget. Only one Isaac
# process fits in this machine's memory: don't run another sweep at the same time. Delete OUT to start over.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI0_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIMPLE_DIR="${SIMPLE_DIR:-$(cd "$PSI0_DIR/../SIMPLE" && pwd)}"

use_tmux=0
for a in "$@"; do
  case "$a" in
    --tmux) use_tmux=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $a" >&2; exit 1 ;;
    *[!0-9]*) TASK="$a" ;;
    *) TRIALS="$a" ;;
  esac
done

TASK="${TASK:-G1WholebodyTabletopGraspMP-v0}"
TASK="${TASK#simple/}"
TRIALS="${TRIALS:-50}"
SETS="${SETS:-c0_baseline,e1_edge_x}"
CLOSER="${CLOSER:-0.10}"
FARTHER="${FARTHER:-0.15}"
LOOK="${LOOK:-training}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-}"
CKPT_STEP="${CKPT_STEP:-40000}"
BUDGET_STEPS="${BUDGET_STEPS:-0}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
INSTRUCTION="${INSTRUCTION:-}"
PORT="${PORT:-8016}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
RTC="${RTC:-1}"
ATTEMPTS="${ATTEMPTS:-3}"
CHUNK="${CHUNK:-40}"
REPORT_EVERY="${REPORT_EVERY:-1200}"
name="${TASK}_edge_${TRIALS}x_seed${SEED}"
[[ "$SETS" != "c0_baseline,e1_edge_x" ]] && name+="_$(echo "$SETS" | tr ',' '+')"
[[ "$LOOK" != "training" ]] && name+="_look-$LOOK"
[[ -n "$RUN_DIR" ]] && name+="_$(basename "$RUN_DIR")"
OUT="$(realpath -m "${OUT:-$PSI0_DIR/.runs/tabletop_edge_sweep/$name}")"
if [[ -f "$OUT/report/results.json" && ! -f "$OUT/plan.json" && "${SKIP_EVAL:-0}" != "1" ]]; then
  echo "$OUT holds only the report of a finished sweep; set OUT to another folder, delete it to run it again, or" \
       "SKIP_EVAL=1 to rebuild its report" >&2
  exit 1
fi

if [[ "$use_tmux" == "1" ]]; then
  session="edge_sweep_$(date +%H%M%S)"
  mkdir -p "$(dirname "$OUT")"  # for the console log next to OUT
  unset_args=() set_args=()
  for v in TASK TRIALS SETS CLOSER FARTHER LOOK SEED RUN_DIR CKPT_STEP BUDGET_STEPS SAVE_VIDEO INSTRUCTION PORT \
           ACTION_EXEC_HORIZON RTC ATTEMPTS CHUNK REPORT_EVERY OUT SKIP_EVAL KEEP REUSE_SERVER HEADLESS SIMPLE_DIR \
           CUDA_VISIBLE_DEVICES HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE; do
    if [[ -n "${!v:-}" ]]; then set_args+=("$v=${!v}"); else unset_args+=(-u "$v"); fi
  done
  tmux new-session -d -s "$session" "env $(printf '%q ' "${unset_args[@]}" "${set_args[@]}")bash \
$(printf '%q' "${BASH_SOURCE[0]}") 2>&1 | tee -a $(printf '%q' "$OUT.log"); echo; echo '[done - press Enter to close]'; read"
  echo "started tmux session '$session'   attach: tmux attach -t $session   log: $OUT.log"
  echo "report will be: $OUT/report/index.html (rebuilt every $((REPORT_EVERY / 60)) min while it runs)"
  exit 0
fi

[[ -x "$SIMPLE_DIR/.venv/bin/python" ]] || { echo "SIMPLE venv missing: $SIMPLE_DIR/.venv" >&2; exit 1; }
[[ -x "$PSI0_DIR/.venv-psi/bin/python" ]] || { echo "Psi0 venv missing: $PSI0_DIR/.venv-psi" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "task: $TASK   output: $OUT"
settings="TASK=$TASK TRIALS=$TRIALS SETS=$SETS CLOSER=$CLOSER FARTHER=$FARTHER LOOK=$LOOK SEED=$SEED RUN_DIR=$RUN_DIR"
settings+=" CKPT_STEP=$CKPT_STEP ACTION_EXEC_HORIZON=$ACTION_EXEC_HORIZON RTC=$RTC BUDGET_STEPS=$BUDGET_STEPS"
settings+=" INSTRUCTION=$INSTRUCTION"
if [[ -f "$OUT/settings.txt" && "$(cat "$OUT/settings.txt")" != "$settings" ]]; then
  echo "$OUT holds a sweep with other settings ($(cat "$OUT/settings.txt")); delete it or set OUT" >&2; exit 1
fi
echo "$settings" > "$OUT/settings.txt"

health() { curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }
SERVER_PID=
REPORT_PID=
cleanup() {
  status=$?
  trap - EXIT INT TERM
  [[ -n "$REPORT_PID" ]] && kill "$REPORT_PID" 2>/dev/null || true
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "stopping the policy server"
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 100); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  if [[ ! -f "$OUT/plan.json" ]]; then
    echo "=== [0/3] prepare (log: $OUT/prepare.log)"
    (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/prepare_edge_sweep.py" --task "$TASK" \
        --psi0-dir "$PSI0_DIR" --simple-dir "$SIMPLE_DIR" --out "$OUT" --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP") \
      2>&1 | tee -a "$OUT/prepare.log"
    plan_settings="$(python3 -c 'import json, sys; k = sys.argv[1::2]; print(json.dumps(dict(zip(k, sys.argv[2::2]))))' \
      run_dir "$(cat "$OUT/run_dir.txt")" ckpt_step "$CKPT_STEP" look "$LOOK" action_exec_horizon "$ACTION_EXEC_HORIZON" \
      rtc "$RTC" budget_steps "$BUDGET_STEPS" instruction "$INSTRUCTION")"
    (cd "$SIMPLE_DIR" && .venv/bin/python -u "$SCRIPT_DIR/make_edge_trials.py" --out "$OUT" --task "$TASK" \
        --simple-dir "$SIMPLE_DIR" --base-scenes "$(cat "$OUT/base_scenes.txt")" --trials "$TRIALS" --sets "$SETS" \
        --seed "$SEED" --closer "$CLOSER" --farther "$FARTHER" --settings "$plan_settings") >> "$OUT/prepare.log" 2>&1 \
      || { rm -f "$OUT/plan.json"; echo "make_edge_trials.py failed, see $OUT/prepare.log" >&2; exit 1; }
    grep -E "^\[trials\]" "$OUT/prepare.log" | tail -n 3 || true
    # the demos' own reach against the can (forward kinematics of their joints), the report's reference
    (cd "$SIMPLE_DIR" && .venv/bin/python -u "$SCRIPT_DIR/demo_reach.py" --pack "$(cat "$OUT/train_pack.txt")" \
        --out "$OUT/demo_reach.json") >> "$OUT/prepare.log" 2>&1 \
      || echo "demo_reach.py failed (see $OUT/prepare.log); the report goes without the demos' reference" >&2
  fi
  run_dir="$(cat "$OUT/run_dir.txt")"

  start_server() {
    rtc_flag=""; [[ "$RTC" == "1" ]] && rtc_flag="--rtc"
    # shellcheck disable=SC2086
    (cd "$PSI0_DIR" && exec setsid .venv-psi/bin/serve_psi0_simple --policy psi0 --port "$PORT" \
        --run-dir "$run_dir" --ckpt-step "$CKPT_STEP" --action-exec-horizon "$ACTION_EXEC_HORIZON" $rtc_flag) \
      >> "$OUT/server.log" 2>&1 &
    SERVER_PID=$!
    waited=0
    until health; do
      kill -0 "$SERVER_PID" 2>/dev/null || { echo "policy server exited, see $OUT/server.log" >&2; tail -n 30 "$OUT/server.log" >&2; return 1; }
      (( waited >= 600 )) && { echo "policy server not up after 600 s, see $OUT/server.log" >&2; return 1; }
      sleep 5; waited=$((waited + 5))
    done
    echo "server up after ${waited} s"
  }

  echo "=== [1/3] policy server (log: $OUT/server.log)"
  if health; then
    [[ "${REUSE_SERVER:-0}" == "1" ]] || {
      echo "something already answers on port $PORT; stop it, pick another PORT, or set REUSE_SERVER=1" >&2; exit 1; }
    echo "reusing the server on port $PORT"
  else
    start_server || exit 1
  fi
  curl -sf "http://127.0.0.1:$PORT/info" > "$OUT/server_info.json" || true

  n_trials="$(cat "$OUT/n_trials.txt")"
  n_done() { local n; n="$(grep -c . "$OUT/results.jsonl" 2>/dev/null)" || true; echo "${n:-0}"; }
  echo "=== [2/3] closed-loop eval: $(n_done)/$n_trials done (log: $OUT/eval.log; each Isaac start takes a few minutes)"
  if (( REPORT_EVERY > 0 )); then
    (trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM
     while true; do
       sleep "$REPORT_EVERY" & wait $!
       [[ "$OUT/results.jsonl" -nt "$OUT/report/index.html" ]] || continue
       (cd "$PSI0_DIR" && exec nice -n 10 .venv-psi/bin/python -u "$SCRIPT_DIR/plot_edge_sweep.py" --results "$OUT") \
         >> "$OUT/report.log" 2>&1 &
       wait $! && echo "[report] rebuilt at $(date +%H:%M) ($(n_done)/$n_trials done): $OUT/report/index.html" || true
     done) &
    REPORT_PID=$!
  fi
  eval_args=(--out "$OUT" --trials "$n_trials" --max-new "$CHUNK" --budget-steps "$BUDGET_STEPS" --host 127.0.0.1
             --port "$PORT" --look "$LOOK")
  [[ "$SAVE_VIDEO" == "1" ]] && eval_args+=(--save-video) || eval_args+=(--no-save-video)
  [[ "${HEADLESS:-1}" == "1" ]] && eval_args+=(--headless) || eval_args+=(--no-headless)
  [[ -n "$INSTRUCTION" ]] && eval_args+=(--instruction "$INSTRUCTION")
  fails=0
  while (( $(n_done) < n_trials )); do
    before=$(n_done)
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" exec .venv/bin/python -u "$SCRIPT_DIR/eval_edge_sweep.py" \
        "${eval_args[@]}") >> "$OUT/eval.log" 2>&1 &
    pid=$!
    tail -n 0 -f "$OUT/eval.log" --pid "$pid" 2>/dev/null | grep --line-buffered -E "^\[sweep\]|Traceback|Error:" || true
    rc=0; wait "$pid" || rc=$?
    if (( $(n_done) > before )); then fails=0; else fails=$((fails + 1)); fi
    if (( rc != 0 )); then
      echo "eval exited with $rc at $(date '+%F %H:%M'), see $OUT/eval.log" >&2
      tail -n 20 "$OUT/eval.log" >&2
    fi
    (( fails >= ATTEMPTS )) && { echo "no trial finished in $ATTEMPTS attempts in a row; giving up" >&2; break; }
    if ! health; then
      echo "the policy server is down, see $OUT/server.log" >&2
      [[ -n "$SERVER_PID" ]] || break
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
      echo "restarting it"
      start_server || break
    fi
  done
  [[ -n "$REPORT_PID" ]] && { kill "$REPORT_PID" 2>/dev/null || true; wait "$REPORT_PID" 2>/dev/null || true; }
  REPORT_PID=
  (( $(n_done) >= n_trials )) || echo "WARNING: the sweep is incomplete ($(n_done)/$n_trials); reporting what finished (re-run to resume)" >&2
fi

echo "=== [3/3] report"
(cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_edge_sweep.py" --results "$OUT")

n_done="$(grep -c . "$OUT/results.jsonl" 2>/dev/null || true)"
n_trials="$(cat "$OUT/n_trials.txt" 2>/dev/null || echo 1000000)"
if [[ "${KEEP:-0}" != "1" && -f "$OUT/report/index.html" && -f "$OUT/report/results.json" ]] \
    && (( ${n_done:-0} >= n_trials )); then
  find "$OUT" -mindepth 1 -maxdepth 1 ! -name report -exec rm -rf {} +
  echo "kept only the report ($(du -sh "$OUT/report" | cut -f1)); KEEP=1 keeps everything"
fi
echo
echo "report: $OUT/report/index.html"
