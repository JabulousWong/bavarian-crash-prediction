#!/bin/bash
# Full analysis rerun with Basis-DLM network membership (dlm_assign.py). Scripts are the
# original ones except phase1_dlm_assignment.py (new), dlm_assign.py (new) and the assignment
# functions in phase2_label_cohort, phase6_minute_dataset, phase7_frequency_prior, v2_pipeline.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
run() { echo "=== $(date '+%F %T') START: $*"; local t0=$SECONDS; "$@" || { echo "=== FAILED ($?): $*"; return 1; }; echo "=== $(date '+%F %T') DONE in $((SECONDS-t0))s: $*"; }

stage1() {
  run $PY analysis/phase1_dlm_assignment.py &&
  run $PY analysis/phase2_label_cohort.py &&
  run $PY analysis/phase2_build_dataset.py &&
  run $PY analysis/phase2_baseline.py &&
  run $PY analysis/phase2_label_comparison.py &&
  run $PY analysis/phase2_bootstrap.py
}
stage2() {  # GPU 1
  run $PY analysis/phase5_weather.py --workers 10 &&
  run $PY analysis/phase5_weather_eval.py --replicates 1000 &&
  run $PY analysis/phase6_minute_dataset.py --workers 8 &&
  run $PY analysis/phase6_minute_pu.py &&
  run $PY analysis/phase7_frequency_prior.py &&
  run $PY analysis/phase7_two_stage.py &&
  run $PY analysis/phase8_pu_value.py &&
  run $PY analysis/v2_pipeline.py --workers 10 --steps network cohort traffic phase2 phase5 phase6 phase7 phase8
}
stage3() {  # CPU; class_prior cannot run (monthly h1101c workbooks not on this server), copied from the original
  run $PY analysis/phase3_population.py --workers 8
  run $PY analysis/phase3_network_sensitivity.py --replicates 1000 --threads 4
  run $PY analysis/phase3_early_stopping.py --replicates 1000
  run $PY analysis/phase3_early_stopping.py --replicates 1000 --strict-cohort
  run $PY analysis/phase4_severity_pu.py --repetitions 30
  run $PY analysis/phase3_pu_simulation.py
  run $PY analysis/phase3_pu_simulation.py --repetitions 30 --n-train 150000 --n-u 50000 --n-test 150000 --priors 0.001 --out analysis/results/phase3/pu_simulation_rare
  run $PY analysis/phase3_pu_simulation.py --repetitions 30 --n-train 500000 --n-u 50000 --n-test 250000 --priors 0.0002 0.0005 0.0015 --out analysis/results/phase3/pu_simulation_rare_grid
  run $PY analysis/phase3_pu_geometry_audit.py
  run $PY analysis/phase3_pu_guided_oversampling.py --repetitions 30
  run $PY analysis/phase3_continuation_report.py
  run $PY analysis/phase3_report.py
}
stage1 > run_stage1.log 2>&1 || { echo STAGE1_FAILED; exit 1; }
stage2 > run_stage2.log 2>&1 &
p2=$!
stage3 > run_stage3.log 2>&1 &
p3=$!
wait $p2; s2=$?; wait $p3; s3=$?
echo "=== ALL DONE stage2=$s2 stage3=$s3"
