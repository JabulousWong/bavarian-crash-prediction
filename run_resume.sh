#!/bin/bash
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
run() { echo "=== $(date '+%F %T') START: $*"; local t0=$SECONDS; "$@" || { echo "=== FAILED ($?): $*"; return 1; }; echo "=== $(date '+%F %T') DONE in $((SECONDS-t0))s: $*"; }
run $PY analysis/phase6_minute_dataset.py --workers 8 &&
run $PY analysis/phase6_minute_pu.py &&
run $PY analysis/phase7_frequency_prior.py &&
run $PY analysis/phase7_two_stage.py &&
run $PY analysis/phase8_pu_value.py &&
run $PY analysis/v2_pipeline.py --workers 10 --steps network cohort traffic phase2 phase5 phase6 phase7 phase8 &&
echo "=== RESUME_COMPLETE"
