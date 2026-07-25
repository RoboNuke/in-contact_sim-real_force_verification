#!/usr/bin/env bash
# Drive the scaled-hole correctness test one variant per process.
#
# Isaac Lab deadlocks when a second gym env is built in the same process, so we
# launch a fresh python process for each of the 5 variants (idx 0..4).
set -u

PY="${PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${1:-/tmp/claude-1000/-home-hunter-in-contact-sim-real-force-verification/950cd73b-0a40-4d1e-a6d9-c122c8ca1630/scratchpad/scaled_hole}"
mkdir -p "$LOGDIR"

declare -a RESULTS
for idx in 0 1 2 3 4; do
    log="$LOGDIR/variant_${idx}.log"
    echo "[driver] === variant idx=$idx -> $log ===" | tee -a "$LOGDIR/summary.log"
    "$PY" "$ROOT/scripts/test_scaled_hole.py" --headless --num_envs 2 --idx "$idx" \
        > "$log" 2>&1
    code=$?
    line="$(grep -E 'VARIANT_RESULT' "$log" | tail -1)"
    if [[ -z "$line" ]]; then line="VARIANT_RESULT idx=$idx result=NO_OUTPUT (exit $code)"; fi
    echo "[driver] $line" | tee -a "$LOGDIR/summary.log"
    RESULTS[$idx]="$line"
done

echo "[driver] ================= SUMMARY =================" | tee -a "$LOGDIR/summary.log"
for idx in 0 1 2 3 4; do echo "[driver] ${RESULTS[$idx]}" | tee -a "$LOGDIR/summary.log"; done
echo "[driver] done." | tee -a "$LOGDIR/summary.log"
