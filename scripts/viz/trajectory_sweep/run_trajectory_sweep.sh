#!/usr/bin/env bash
# Does the Psi0 policy succeed from the states it was trained on, and from a little beside them?
#
# Closed-loop trials of a SIMPLE Teleop task's released Psi0 checkpoint, every one in a training episode's own
# recorded scene and with the training prompt ("turn the faucet"):
#   start  the robot at the training start pose, moved 5 or 10 cm in 8 directions, or turned 20 or 40 degrees left or
#          right (its spawn moved before the reset; it stands up there as in any eval). The start pose is the same in
#          every episode, so each start condition runs REPEATS (5) times in all, once in each chosen episode's scene
#   path   5 training episodes, chosen far apart from each other: each one's state at 1/3 and 2/3 of the way to the
#          faucet and at the reach point (0.5 s before the handle starts to turn), REPEATS times each. The robot gets
#          there by replaying the episode's recorded actions through the policy's own action queue (never moved by
#          hand), so its joints and camera match the training frame; then the policy takes over
# 21 start conditions x 5 + 5 episodes x 3 points x 5 = 180 trials, about 12 h. Output: $OUT/report/index.html, a
# self-contained page (top-down success map, success by condition, every trial's video), refreshed every
# REPORT_EVERY s while the sweep runs, and $OUT/report/results.json. Once every trial has finished, the report folder
# is all OUT keeps.
#
# One command, five stages:
#   0. prepare    Psi0 venv:   download the checkpoint, the train pack and the raw teleop pack if missing; choose the
#                 episodes; write the trial plan (prepare_trajectory_sweep.py)
#   1. serve      Psi0 venv:   serve_psi0_simple with the checkpoint (port $PORT, log server.log)
#   2. calibrate  SIMPLE venv: replay each chosen episode whole (no policy) to find its path points; a replay counts
#                 only if it opens the faucet and its pelvis stays within 4 cm / 6 deg of the teleop recording at the
#                 path points; an episode gets 2 replays, then its slot moves on to the nearest backup
#                 (eval_trajectory_sweep.py --calibrate)
#   3. eval       SIMPLE venv: SIMPLE's Teleop eval loop (eval_decoupled_wbc, agent psi0_decoupled_wbc), unchanged,
#                 driven by eval_trajectory_sweep.py: one episode per trial, CHUNK trials per Isaac process. Each trial
#                 is written to results.jsonl as it finishes; a crash is resumed (given up after ATTEMPTS crashes in a
#                 row without a finished trial; a dead policy server is restarted), and re-running the command
#                 continues
#   4. report     Psi0 venv:   report/ (index.html, results.json, web-encoded videos, images) (plot_trajectory_sweep.py);
#                 then, if every trial has finished, the rest of OUT is deleted
#
# Usage:  run_trajectory_sweep.sh [TASK] [TRIALS] [--tmux]
#   scripts/viz/trajectory_sweep/run_trajectory_sweep.sh --tmux       # OpenFaucet, every trial, in a tmux session
#   scripts/viz/trajectory_sweep/run_trajectory_sweep.sh 5            # only the first 5 trials (one of each kind)
#   EPISODES=28,90,76,52,2 .../run_trajectory_sweep.sh                # these training episodes instead of choosing
#   RADII=0.05,0.10 YAWS=20,40 REPEATS=5 .../run_trajectory_sweep.sh  # the defaults
#   INSTRUCTION="move forward to the faucet and open it" ...          # another prompt (default: the training one)
#   KEEP=1 ...                  # keep the rest of OUT too (needed to rebuild the report later, or to add trials)
#   SKIP_EVAL=1 OUT=<dir> ...   # rebuild the report only (KEEP=1 runs)
#   REUSE_SERVER=1 ...          # a server for this checkpoint already answers on $PORT
#   OPEN=1 ...                  # open the report at the end
# The plan is fixed by the settings and SEED; delete OUT to start over (re-running the same command continues).
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

TASK="${TASK:-G1WholebodyOpenFaucetTeleop-v0}"
TASK="${TASK#simple/}"
TRIALS="${TRIALS:-0}"            # 0 = every trial of the plan
N_EPISODES="${N_EPISODES:-5}"    # training episodes to use (chosen far apart)
EPISODES="${EPISODES:-}"         # or these, comma separated
RADII="${RADII:-0.05,0.10}"      # m, start moves (8 directions each)
YAWS="${YAWS:-20,40}"            # deg, start turns (left and right each)
REPEATS="${REPEATS:-5}"          # trials per condition
REACH_LEAD="${REACH_LEAD:-25}"   # frames between the reach point and the faucet handle starting to turn
MIN_WALK="${MIN_WALK:-0.12}"     # m, shortest start-to-reach walk a chosen episode may have
INSTRUCTION="${INSTRUCTION:-turn the faucet}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-}"           # empty = the task's released checkpoint (downloaded if missing)
CKPT_STEP="${CKPT_STEP:-40000}"
BUDGET_STEPS="${BUDGET_STEPS:-0}"  # 0 = the task's max_episode_steps (replayed frames count)
SAVE_VIDEO="${SAVE_VIDEO:-1}"
PORT="${PORT:-8015}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
RTC="${RTC:-1}"
ATTEMPTS="${ATTEMPTS:-3}"        # crashes in a row without a finished trial before giving up
CHUNK="${CHUNK:-40}"             # trials per Isaac process (0 = all in one)
REPORT_EVERY="${REPORT_EVERY:-1200}"  # s between report rebuilds while the sweep runs (0 = only at the end)
custom=""
[[ "$N_EPISODES|$EPISODES|$RADII|$YAWS|$REACH_LEAD|$MIN_WALK" != "5||0.05,0.10|20,40|25|0.12" ]] && custom="_custom"
[[ "$INSTRUCTION" != "turn the faucet" ]] && custom+="_prompt"
OUT="${OUT:-$PSI0_DIR/.runs/trajectory_sweep/${TASK}_x${REPEATS}${custom}_seed${SEED}}"

if [[ "$use_tmux" == "1" ]]; then
  session="trajectory_$(date +%H%M%S)"
  # every setting passed explicitly, the unset ones unset: a tmux server keeps the environment it was started from
  # (e.g. an earlier run's EPISODES), and its sessions would inherit it
  unset_args=() set_args=()
  for v in TASK TRIALS N_EPISODES EPISODES RADII YAWS REPEATS REACH_LEAD MIN_WALK INSTRUCTION SEED RUN_DIR CKPT_STEP \
           BUDGET_STEPS SAVE_VIDEO PORT ACTION_EXEC_HORIZON RTC ATTEMPTS CHUNK REPORT_EVERY OUT SKIP_EVAL \
           KEEP REUSE_SERVER OPEN HEADLESS SIMPLE_DIR CUDA_VISIBLE_DEVICES; do
    if [[ -n "${!v:-}" ]]; then set_args+=("$v=${!v}"); else unset_args+=(-u "$v"); fi
  done
  tmux new-session -d -s "$session" "env $(printf '%q ' "${unset_args[@]}" "${set_args[@]}")bash \
$(printf '%q' "${BASH_SOURCE[0]}"); echo; echo '[done - press Enter to close]'; read"
  echo "started tmux session '$session'   attach: tmux attach -t $session"
  echo "report will be: $OUT/report/index.html (rebuilt every $((REPORT_EVERY / 60)) min while it runs)"
  exit 0
fi

[[ -x "$SIMPLE_DIR/.venv/bin/python" ]] || { echo "SIMPLE venv missing: $SIMPLE_DIR/.venv" >&2; exit 1; }
[[ -x "$PSI0_DIR/.venv-psi/bin/python" ]] || { echo "Psi0 venv missing: $PSI0_DIR/.venv-psi" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "task: $TASK   output: $OUT"

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
  echo "=== [0/4] prepare"
  (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/prepare_trajectory_sweep.py" --task "$TASK" \
      --psi0-dir "$PSI0_DIR" --simple-dir "$SIMPLE_DIR" --out "$OUT" --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP" \
      --n-episodes "$N_EPISODES" --episodes "$EPISODES" --min-walk "$MIN_WALK" --radii "$RADII" --yaws "$YAWS" \
      --repeats "$REPEATS" --reach-lead "$REACH_LEAD" --instruction "$INSTRUCTION" --trials "$TRIALS" --seed "$SEED")
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

  echo "=== [1/4] policy server (log: $OUT/server.log)"
  if health; then
    [[ "${REUSE_SERVER:-0}" == "1" ]] || {
      echo "something already answers on port $PORT; stop it, pick another PORT, or set REUSE_SERVER=1" >&2; exit 1; }
    echo "reusing the server on port $PORT"
  else
    start_server || exit 1
  fi
  curl -sf "http://127.0.0.1:$PORT/info" > "$OUT/server_info.json" || true
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" \
    | sed 's/^/GPU memory used, free: /' || true

  run_eval() {  # one Isaac process of eval_trajectory_sweep.py with the given extra args; sets rc
    local common=(--out "$OUT" --budget-steps "$BUDGET_STEPS" --host 127.0.0.1 --port "$PORT")
    [[ "$SAVE_VIDEO" == "1" ]] && common+=(--save-video) || common+=(--no-save-video)
    [[ "${HEADLESS:-1}" == "1" ]] && common+=(--headless) || common+=(--no-headless)
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" exec .venv/bin/python -u "$SCRIPT_DIR/eval_trajectory_sweep.py" \
        "${common[@]}" "$@") >> "$OUT/eval.log" 2>&1 &
    local pid=$!
    # Isaac prints a lot; show only the sweep's progress lines
    tail -n 0 -f "$OUT/eval.log" --pid "$pid" 2>/dev/null \
      | grep --line-buffered -E "^\[sweep\]|Traceback|Error:" || true
    rc=0; wait "$pid" || rc=$?
  }

  echo "=== [2/4] calibrate: replay the chosen episodes (log: $OUT/eval.log; each Isaac start takes a few minutes)"
  # the status comes from a run without the simulator: Isaac ends its process with status 0 whatever happened
  n_attempts() { local n; n="$(grep -o '"wall_seconds"' "$OUT/calib.json" 2>/dev/null | grep -c .)" || true; echo "${n:-0}"; }
  fails=0
  while true; do
    run_eval --calib-status
    case "$rc" in
      0) break ;;
      3) ;;
      4) echo "a slot ran out of episodes whose replay opens the faucet and stays on the recording; see $OUT/calib.json" >&2
         exit 1 ;;
      *) echo "calibration status check failed ($rc), see $OUT/eval.log" >&2; tail -n 20 "$OUT/eval.log" >&2; exit 1 ;;
    esac
    before=$(n_attempts)
    run_eval --calibrate
    if (( $(n_attempts) > before )); then fails=0; else fails=$((fails + 1))
      echo "calibration made no progress (exit $rc), see $OUT/eval.log" >&2; tail -n 20 "$OUT/eval.log" >&2
      (( fails >= ATTEMPTS )) && { echo "calibration failed $ATTEMPTS times in a row; giving up" >&2; exit 1; }
    fi
  done

  n_trials="$(cat "$OUT/n_trials.txt")"
  n_done() { local n; n="$(grep -c . "$OUT/results.jsonl" 2>/dev/null)" || true; echo "${n:-0}"; }
  echo "=== [3/4] closed-loop eval: $(n_done)/$n_trials done (log: $OUT/eval.log)"
  if (( REPORT_EVERY > 0 )); then  # a fresh report every REPORT_EVERY s, so a long sweep can be looked at as it goes
    (trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM
     while true; do
       sleep "$REPORT_EVERY" & wait $!
       [[ "$OUT/results.jsonl" -nt "$OUT/report/index.html" ]] || continue
       (cd "$PSI0_DIR" && exec nice -n 10 .venv-psi/bin/python -u "$SCRIPT_DIR/plot_trajectory_sweep.py" --results "$OUT") \
         >> "$OUT/report.log" 2>&1 &
       wait $! && echo "[report] rebuilt at $(date +%H:%M) ($(n_done)/$n_trials done): $OUT/report/index.html" || true
     done) &
    REPORT_PID=$!
  fi
  fails=0
  while (( $(n_done) < n_trials )); do
    before=$(n_done)
    run_eval --trials "$n_trials" --max-new "$CHUNK" --instruction "$INSTRUCTION"
    if (( $(n_done) > before )); then fails=0; else fails=$((fails + 1)); fi
    if (( rc != 0 )); then
      echo "eval exited with $rc at $(date '+%F %H:%M'), see $OUT/eval.log" >&2
      tail -n 20 "$OUT/eval.log" >&2
    fi
    (( fails >= ATTEMPTS )) && { echo "no trial finished in $ATTEMPTS attempts in a row; giving up" >&2; break; }
    if ! health; then
      echo "the policy server is down, see $OUT/server.log" >&2
      [[ -n "$SERVER_PID" ]] || break  # not ours (REUSE_SERVER) to restart
      kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
      echo "restarting it"
      start_server || break
    fi
  done
  [[ -n "$REPORT_PID" ]] && { kill "$REPORT_PID" 2>/dev/null || true; wait "$REPORT_PID" 2>/dev/null || true; }
  REPORT_PID=
  (( $(n_done) >= n_trials )) || echo "WARNING: the sweep is incomplete ($(n_done)/$n_trials); plotting what finished (re-run to resume)" >&2
fi

echo "=== [4/4] report"
(cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_trajectory_sweep.py" --results "$OUT")

# The report (with results.json) is self-contained; the rest is only needed to resume, so it goes once every trial is in
n_done="$(grep -c . "$OUT/results.jsonl" 2>/dev/null || true)"
n_trials="$(cat "$OUT/n_trials.txt" 2>/dev/null || echo 1000000)"
n_plan="$(cd "$PSI0_DIR" && .venv-psi/bin/python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['trials']))" \
          "$OUT/plan.json" 2>/dev/null || echo 1000000)"
if [[ "${KEEP:-0}" != "1" && -f "$OUT/report/index.html" && -f "$OUT/report/results.json" ]] \
    && (( ${n_done:-0} >= n_trials && n_trials >= n_plan )); then
  find "$OUT" -mindepth 1 -maxdepth 1 ! -name report -exec rm -rf {} +
  echo "kept only the report ($(du -sh "$OUT/report" | cut -f1)); KEEP=1 keeps everything"
fi

echo
echo "report: $OUT/report/index.html"
if [[ "${OPEN:-0}" == "1" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$OUT/report/index.html" >/dev/null 2>&1 || true
fi
