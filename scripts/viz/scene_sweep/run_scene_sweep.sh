#!/usr/bin/env bash
# Does the Psi0 policy keep its success rate when the table or the room around the target changes?
#
# Closed-loop trials of a SIMPLE Teleop task's Psi0 checkpoint (default: the task's released one) in sets of TRIALS
# scenes each (room 50, table 20, stop 50). Trial i starts from the same scene in every set: one of the task's
# eval-pack scenes (robot spawn, room, the recorded table) randomized the way SIMPLE's highest replay level does it
# (distractors, materials, lighting, and the poses of the faucet and the clutter within the task's ranges), seeded
# per trial; see-through table materials are re-drawn. Each set changes one thing. Three presets:
#   room (default)
#     c0_baseline          nothing
#     c1_1_table_y         the table, with its clutter, slides sideways; same distance to the robot, same height
#     c1_2_table_width     the table gets narrower or wider (sideways); clutter it no longer holds is removed
#     c1_3_room_shift      robot, table and faucet move together to another spot of the room
#     c1_4_room_rotation   robot, table and faucet turn together in the room, about the table centre
#   table
#     c0_baseline          nothing: taken from the room sweep's report (BASELINE) instead of run again, as trial i
#                          is the same scene there; run here when that report is missing (or BASELINE=none)
#     c2_1_table_x         the table, with its clutter, comes 0-5 cm closer to the robot
#     c2_2_table_x_visual  the table, with its clutter, is drawn 0-25 cm closer; the robot collides only with the
#                          recorded table, so the part in front of the recorded edge is there for the camera only
#     c3_1_width_left      the table grows 5-150 cm on the robot's left; its texture keeps the recorded scale
#     c3_2_width_right     the same on the robot's right
#     c3_3_width_both      the same on both sides; trial i adds the same amount in all three
#   stop                   where does the robot stop walking: at the faucet or at the table edge?
#     c0_baseline          nothing: taken from the room sweep's report, as for the table preset
#     c4_1_faucet_x        the faucet moves 0-20 cm farther from the robot, deeper on the table; table and robot stay
#     c4_2_edge_x_visual   the table's near edge is drawn 0-20 cm closer (the table grows toward the robot; the robot
#                          collides only with the recorded table); faucet and robot stay. Trial i uses the same
#                          amount in both sets; both redraw their table the same way
#                          The report shows, where the robot's walk toward the table ended (its last forward walking
#                          run before a 2 s stand; side-steps and the start count as no walk), how far its pelvis
#                          was from the drawn table edge and from the faucet, against the change
#   full                   does the policy act on the robot-faucet relation alone, or take shortcuts from the table
#                          or the room? 15 sets, 1800 trials (COUNTS). A controlled design: each set changes one
#                          thing; trial i starts from the same scene in every set and is compared with the same
#                          trial of its baseline (run here, not imported; as many trials as the sets it controls);
#                          trials run index by index across the sets. No set has the table's clutter objects (a
#                          changed table could not hold them all), but c0_with_clutter. In the t sets the robot
#                          collides only with the recorded table and the drawn one keeps its texture scale; the _phys
#                          twins change the physics too
#     c0_baseline          nothing: the baseline of the r and f sets (200)
#     c0_redraw            nothing, the table redrawn as it is: the baseline of the t sets (100)
#     c0_with_clutter      nothing, with the clutter as in the training scenes: what taking it away does (100)
#     t1_edge_x            the near edge drawn 20 cm farther (the faucet then floats) to 20 cm closer (100)
#     t2_edge_y            one side edge drawn in to the faucet base or up to 100 cm out (100)
#     t3_scale_xy          depth and width drawn 0.4-2x, the near edge in front of the faucet kept (100)
#     t4_rotate            the table drawn turned 5-45 degrees about the edge in front of the faucet (100)
#     t4_rotate_phys       the same turn in the physics too (100)
#     t5_shape             round, half-round, triangle, trapezoid, hexagon or pedestal, drawn (100)
#     t5_shape_phys        the same shapes in the physics too (100)
#     r1_room_shift        as c1_3_room_shift (100)
#     r2_room_rotation     as c1_4_room_rotation (100)
#     r3_room_other        another HSSD room, one of those on disk the table fits in, in turn (100)
#     f1_faucet_x          only the faucet moves, 0-20 cm farther (200)
#     f2_faucet_y          only the faucet moves, up to 25 cm left or right (200)
#                          The report adds, per set, where the robot stopped walking relative to the faucet (pelvis
#                          short of it and beside it) against its baseline, and against each set's change
# The faucet always stays on the table, and the table and the robot clear of the room's furniture and walls. Success
# is SIMPLE's task test: a hand (not an arm or the body) turns the faucet handle past 0.7 rad the way the demos turn
# it. Output: $OUT/report/index.html (per set: success rate with its 95% interval and the trial-by-trial comparison
# with c0; a top-down map of every trial's faucet, and its table; success against the change; every trial's video;
# refreshed every REPORT_EVERY s while it runs) and $OUT/report/results.json (with the robot-table contacts of every
# trial run here; imported c0 trials have none).
# Once every trial has finished, the report folder is all OUT keeps.
#
# One command, four stages:
#   0. prepare  Psi0 venv:   the checkpoint and the eval pack (downloaded if missing), the faucet's training range
#               SIMPLE venv: a top-down map of the room's furniture (room_map.py; Isaac Sim's USD libraries, no Isaac
#               process) and every trial's scene file (make_scene_trials.py, CPU, about a minute), with a top-down
#               picture per set ($OUT/scenes/<set>.png); for the table and stop presets, c0_baseline's finished
#               trials from the room sweep (import_baseline.py)
#   1. serve    Psi0 venv:   serve_psi0_simple with the checkpoint (port $PORT, log server.log)
#   2. eval     SIMPLE venv: SIMPLE's Teleop eval loop, unchanged, driven by eval_scene_sweep.py (the target-pose
#               sweep's driver with the scene files): one episode per trial, CHUNK trials per Isaac process, trials
#               ordered index by index (every set's trial 0, then 1, ...); a crash is resumed, and re-running the
#               command continues
#   3. report   Psi0 venv:   report/ (plot_scene_sweep.py); then, if every trial has finished, the rest of OUT is deleted
#
# Usage:  run_scene_sweep.sh [TASK] [room|table|stop|full] [TRIALS] [--tmux]   (the preset follows SETS when not given)
#   scripts/viz/scene_sweep/run_scene_sweep.sh --tmux            # OpenFaucet, room preset, 5 sets x 50, in tmux
#   .../run_scene_sweep.sh table --tmux                          # the table preset: 5 new sets x 20 (c0 imported)
#   .../run_scene_sweep.sh stop --tmux                           # the stop preset: 2 new sets x 50 (c0 imported)
#   .../run_scene_sweep.sh full --tmux                           # the full preset: 15 sets, 1800 trials
#   .../run_scene_sweep.sh full 2                                # the full preset, 2 trials per set
#   .../run_scene_sweep.sh table 2                               # 2 trials per set, for a quick look
#   BASELINE=none .../run_scene_sweep.sh table                   # run c0_baseline here (a new OUT) as well
#   BASELINE=<report dir> .../run_scene_sweep.sh table           # c0_baseline from another sweep
#   RUN_DIR=<run dir> CKPT_STEP=40000 .../run_scene_sweep.sh     # another checkpoint
#   SETS=c0_baseline,c1_3_room_shift .../run_scene_sweep.sh      # only some of the sets
#   SEED=1 .../run_scene_sweep.sh                                # other scenes
#   KEEP=1 .../run_scene_sweep.sh               # keep the rest of OUT too (scene files, raw videos, logs)
#   SKIP_EVAL=1 OUT=<existing dir> .../run_scene_sweep.sh        # rebuild the report only (KEEP=1 runs)
#   REUSE_SERVER=1 .../run_scene_sweep.sh       # a server for this checkpoint already answers on $PORT
#   OPEN=1 .../run_scene_sweep.sh               # open the report at the end
# A trial takes 4-8 min (a failure runs the full budget), so the room preset's 250 take about 21 h, the table
# preset's 100 about 10 h, the stop preset's 100 about 11 h and the full preset's 1800 about 8 days. Only one
# Isaac process fits in this machine's memory: don't run another sweep or replay at the same time. Delete OUT to
# start over.
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
    room|table|stop|full) PRESET="$a" ;;
    *[!0-9]*) TASK="$a" ;;
    *) TRIALS="$a" ;;
  esac
done

TASK="${TASK:-G1WholebodyOpenFaucetTeleop-v0}"
TASK="${TASK#simple/}"
if [[ -z "${PRESET:-}" ]]; then
  if [[ ",${SETS:-}," =~ ,(c0_redraw|[tfr][0-9]_) ]]; then PRESET=full; elif [[ ",${SETS:-}," =~ ,c4_ ]]; then PRESET=stop
  elif [[ ",${SETS:-}," =~ ,c[23]_ ]]; then PRESET=table; else PRESET=room; fi
fi
case "$PRESET" in
  room) preset_sets="c0_baseline,c1_1_table_y,c1_2_table_width,c1_3_room_shift,c1_4_room_rotation"; tag="" ;;
  table) preset_sets="c0_baseline,c2_1_table_x,c2_2_table_x_visual,c3_1_width_left,c3_2_width_right,c3_3_width_both"
         tag="_table" ;;
  stop) preset_sets="c0_baseline,c4_1_faucet_x,c4_2_edge_x_visual"; tag="_stop" ;;
  full) preset_sets="c0_baseline,c0_redraw,c0_with_clutter,t1_edge_x,t2_edge_y,t3_scale_xy,t4_rotate,t4_rotate_phys,t5_shape"
        preset_sets+=",t5_shape_phys,r1_room_shift,r2_room_rotation,r3_room_other,f1_faucet_x,f2_faucet_y"; tag="_full" ;;
  *) echo "unknown preset $PRESET (room, table, stop or full)" >&2; exit 1 ;;
esac
SETS="${SETS:-$preset_sets}"
full_counts="c0_baseline=200,c0_redraw=100,c0_with_clutter=100,t1_edge_x=100,t2_edge_y=100,t3_scale_xy=100"
full_counts+=",t4_rotate=100,t4_rotate_phys=100,t5_shape=100,t5_shape_phys=100,r1_room_shift=100"
full_counts+=",r2_room_rotation=100,r3_room_other=100,f1_faucet_x=200,f2_faucet_y=200"
if [[ "$PRESET" == "full" ]]; then  # trials per set: COUNTS, or TRIALS for every set when given
  if [[ -z "${TRIALS:-}" ]]; then COUNTS="${COUNTS:-$full_counts}"; TRIALS=200; trials_tag=""
  else COUNTS="${COUNTS:-}"; trials_tag="_${TRIALS}x"; fi
  clutter=none; strict=1  # (and clear table materials re-drawn by their file name too)
else
  clutter=keep; strict=""
  [[ "$PRESET" == "table" ]] && TRIALS="${TRIALS:-20}" || TRIALS="${TRIALS:-50}"  # trials per set
  COUNTS="${COUNTS:-}"; trials_tag="_${TRIALS}x"
fi
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-}"          # empty = the task's released checkpoint (downloaded if missing)
CKPT_STEP="${CKPT_STEP:-40000}"
BUDGET_STEPS="${BUDGET_STEPS:-0}"  # 0 = the task's max_episode_steps
SAVE_VIDEO="${SAVE_VIDEO:-1}"
INSTRUCTION="${INSTRUCTION:-}"  # empty = the instruction SIMPLE's eval sends (the scene's language)
PORT="${PORT:-8015}"
ACTION_EXEC_HORIZON="${ACTION_EXEC_HORIZON:-24}"
RTC="${RTC:-1}"
ATTEMPTS="${ATTEMPTS:-3}"       # crashes in a row without a finished trial before giving up
CHUNK="${CHUNK:-40}"            # trials per Isaac process (0 = all in one)
REPORT_EVERY="${REPORT_EVERY:-1200}"  # s between report rebuilds while it runs (0 = only at the end)
ROOM_MAPS="${ROOM_MAPS:-$PSI0_DIR/.runs/scene_sweep/room_maps}"  # the other rooms' maps (r3_room_other), kept between sweeps
name="${TASK}${tag}${trials_tag}_seed${SEED}"
[[ "$SETS" != "$preset_sets" ]] && name+="_$(echo "$SETS" | tr ',' '+')"
[[ -n "$RUN_DIR" ]] && name+="_$(basename "$RUN_DIR")"
OUT="$(realpath -m "${OUT:-$PSI0_DIR/.runs/scene_sweep/$name}")"
# where c0_baseline comes from instead of running it: the room sweep's report (same task, seed and checkpoint)
room_report="$PSI0_DIR/.runs/scene_sweep/${TASK}_50x_seed${SEED}${RUN_DIR:+_$(basename "$RUN_DIR")}/report"
if [[ -n "${BASELINE:-}" && "$BASELINE" != "none" ]]; then
  BASELINE="$(realpath -m "$BASELINE")"
  [[ -f "$BASELINE/results.json" || ! -f "$BASELINE/report/results.json" ]] || BASELINE="$BASELINE/report"
  [[ -f "$BASELINE/results.json" ]] || { echo "BASELINE: no report (results.json) in $BASELINE" >&2; exit 1; }
fi
if [[ -f "$OUT/report/results.json" && ! -f "$OUT/plan.json" ]]; then
  echo "$OUT holds only the report of a finished sweep; set OUT to another folder, or delete it to run it again" >&2
  exit 1
fi

if [[ "$use_tmux" == "1" ]]; then
  session="scene_sweep_$(date +%H%M%S)"
  # every setting passed explicitly, the unset ones unset: a tmux server keeps the environment it was started from
  unset_args=() set_args=()
  for v in TASK PRESET TRIALS COUNTS SETS SEED RUN_DIR CKPT_STEP BUDGET_STEPS SAVE_VIDEO INSTRUCTION PORT ACTION_EXEC_HORIZON \
           RTC ATTEMPTS CHUNK REPORT_EVERY ROOM_MAPS OUT BASELINE SKIP_EVAL KEEP REUSE_SERVER OPEN HEADLESS SIMPLE_DIR \
           CUDA_VISIBLE_DEVICES; do
    if [[ -n "${!v:-}" ]]; then set_args+=("$v=${!v}"); else unset_args+=(-u "$v"); fi
  done
  tmux new-session -d -s "$session" "env $(printf '%q ' "${unset_args[@]}" "${set_args[@]}")bash \
$(printf '%q' "${BASH_SOURCE[0]}"); echo; echo '[done - press Enter to close]'; read"
  echo "started tmux session '$session'   attach: tmux attach -t $session"
  echo "report will be: $OUT/report/index.html (rebuilt every $((REPORT_EVERY / 60)) min while it runs)"
  exit 0
fi

[[ "$PRESET" == "table" || "$PRESET" == "stop" ]] && BASELINE="${BASELINE:-$room_report}" || BASELINE="${BASELINE:-none}"
[[ -x "$SIMPLE_DIR/.venv/bin/python" ]] || { echo "SIMPLE venv missing: $SIMPLE_DIR/.venv" >&2; exit 1; }
[[ -x "$PSI0_DIR/.venv-psi/bin/python" ]] || { echo "Psi0 venv missing: $PSI0_DIR/.venv-psi" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
echo "task: $TASK   output: $OUT"
settings="TASK=$TASK TRIALS=$TRIALS SETS=$SETS SEED=$SEED RUN_DIR=$RUN_DIR CKPT_STEP=$CKPT_STEP"
[[ -n "$COUNTS" ]] && settings+=" COUNTS=$COUNTS"
settings+=" ACTION_EXEC_HORIZON=$ACTION_EXEC_HORIZON RTC=$RTC BUDGET_STEPS=$BUDGET_STEPS INSTRUCTION=$INSTRUCTION"
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
    (cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/prepare_scene_sweep.py" --task "$TASK" \
        --psi0-dir "$PSI0_DIR" --simple-dir "$SIMPLE_DIR" --out "$OUT" --run-dir "$RUN_DIR" --ckpt-step "$CKPT_STEP") \
      2>&1 | tee -a "$OUT/prepare.log"
    read -r room yaw ox oy < <("$SIMPLE_DIR/.venv/bin/python" -c \
      'import json, sys; r = json.load(open(sys.argv[1])); print(r["room"], r["yaw"], *r["offset"])' "$OUT/room.json")
    usd="$(ls -d "$SIMPLE_DIR"/.venv/lib/python3*/site-packages/isaacsim/extscache/omni.usd.libs-* | head -n 1)"
    plan_settings="$(python3 -c 'import json, sys; k = sys.argv[1::2]; print(json.dumps(dict(zip(k, sys.argv[2::2]))))' \
      run_dir "$(cat "$OUT/run_dir.txt")" ckpt_step "$CKPT_STEP" preset "$PRESET" action_exec_horizon "$ACTION_EXEC_HORIZON" \
      rtc "$RTC" budget_steps "$BUDGET_STEPS" instruction "$INSTRUCTION")"
    rooms_args=()  # r3_room_other: every other room on disk, mapped too (once: ROOM_MAPS keeps them, ~2 h the first time)
    [[ ",$SETS," == *",r3_room_other,"* ]] && rooms_args=(--all-rooms "$ROOM_MAPS")
    (cd "$SIMPLE_DIR" && PYTHONPATH="$usd" LD_LIBRARY_PATH="$usd/bin:${LD_LIBRARY_PATH:-}" .venv/bin/python -u \
        "$SCRIPT_DIR/room_map.py" --simple-dir "$SIMPLE_DIR" --room "$room" --yaw "$yaw" --offset "$ox" "$oy" \
        --out "$OUT/room_map.npz" --png "$OUT/room_map.png" "${rooms_args[@]}") >> "$OUT/prepare.log" 2>&1 \
      || { echo "room_map.py failed, see $OUT/prepare.log" >&2; exit 1; }
    grep "^\[room_map\]" "$OUT/prepare.log" | tail -n 1 || true
    (cd "$SIMPLE_DIR" && .venv/bin/python -u "$SCRIPT_DIR/make_scene_trials.py" --out "$OUT" --task "$TASK" \
        --simple-dir "$SIMPLE_DIR" --base-scenes "$(cat "$OUT/base_scenes.txt")" --room-map "$OUT/room_map.npz" \
        --trials "$TRIALS" --counts "$COUNTS" --sets "$SETS" --seed "$SEED" --clutter "$clutter" ${strict:+--strict-materials} \
        ${rooms_args:+--room-maps "$ROOM_MAPS"} \
        --settings "$plan_settings") >> "$OUT/prepare.log" 2>&1 \
      || { rm -f "$OUT/plan.json"; echo "make_scene_trials.py failed, see $OUT/prepare.log" >&2; exit 1; }
    grep -E "^\[trials\] [0-9]+ trials|WARNING" "$OUT/prepare.log" || true
    echo "scenes, top-down per set: $OUT/scenes/<set>.png"
  fi
  if [[ "$BASELINE" != "none" && ",$SETS," == *",c0_baseline,"* ]]; then
    if [[ -f "$BASELINE/results.json" ]]; then
      "$PSI0_DIR/.venv-psi/bin/python" -u "$SCRIPT_DIR/import_baseline.py" --out "$OUT" --source "$BASELINE" 2>&1 \
        | tee -a "$OUT/prepare.log" || { echo "c0_baseline not imported; BASELINE=none runs it here" >&2; exit 1; }
    else
      echo "no earlier c0_baseline in $BASELINE: running it here"
    fi
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
  if (( REPORT_EVERY > 0 )); then  # a fresh report every REPORT_EVERY s, so a long sweep can be looked at as it goes
    (trap 'kill $(jobs -p) 2>/dev/null; exit 0' TERM
     while true; do
       sleep "$REPORT_EVERY" & wait $!
       [[ "$OUT/results.jsonl" -nt "$OUT/report/index.html" ]] || continue
       (cd "$PSI0_DIR" && exec nice -n 10 .venv-psi/bin/python -u "$SCRIPT_DIR/plot_scene_sweep.py" --results "$OUT") \
         >> "$OUT/report.log" 2>&1 &
       wait $! && echo "[report] rebuilt at $(date +%H:%M) ($(n_done)/$n_trials done): $OUT/report/index.html" || true
     done) &
    REPORT_PID=$!
  fi
  eval_args=(--out "$OUT" --trials "$n_trials" --max-new "$CHUNK" --budget-steps "$BUDGET_STEPS" --host 127.0.0.1
             --port "$PORT")
  [[ "$SAVE_VIDEO" == "1" ]] && eval_args+=(--save-video) || eval_args+=(--no-save-video)
  [[ "${HEADLESS:-1}" == "1" ]] && eval_args+=(--headless) || eval_args+=(--no-headless)
  [[ -n "$INSTRUCTION" ]] && eval_args+=(--instruction "$INSTRUCTION")
  fails=0
  while (( $(n_done) < n_trials )); do
    before=$(n_done)
    (cd "$SIMPLE_DIR" && MUJOCO_GL="${MUJOCO_GL:-egl}" exec .venv/bin/python -u "$SCRIPT_DIR/eval_scene_sweep.py" \
        "${eval_args[@]}") >> "$OUT/eval.log" 2>&1 &
    pid=$!
    # Isaac prints a lot; show only the sweep's progress lines
    tail -n 0 -f "$OUT/eval.log" --pid "$pid" 2>/dev/null \
      | grep --line-buffered -E "^\[sweep\]|Traceback|Error:" || true
    rc=0; wait "$pid" || rc=$?
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
  (( $(n_done) >= n_trials )) || echo "WARNING: the sweep is incomplete ($(n_done)/$n_trials); reporting what finished (re-run to resume)" >&2
fi

echo "=== [3/3] report"
(cd "$PSI0_DIR" && .venv-psi/bin/python -u "$SCRIPT_DIR/plot_scene_sweep.py" --results "$OUT")

# The report (with results.json) is self-contained; the rest is only needed to resume, so it goes once every trial is in
n_done="$(grep -c . "$OUT/results.jsonl" 2>/dev/null || true)"
n_trials="$(cat "$OUT/n_trials.txt" 2>/dev/null || echo 1000000)"
if [[ "${KEEP:-0}" != "1" && -f "$OUT/report/index.html" && -f "$OUT/report/results.json" ]] \
    && (( ${n_done:-0} >= n_trials )); then
  find "$OUT" -mindepth 1 -maxdepth 1 ! -name report -exec rm -rf {} +
  echo "kept only the report ($(du -sh "$OUT/report" | cut -f1)); KEEP=1 keeps everything"
fi

echo
echo "report: $OUT/report/index.html"
if [[ "${OPEN:-0}" == "1" ]] && command -v xdg-open >/dev/null; then
  xdg-open "$OUT/report/index.html" >/dev/null 2>&1 || true
fi
