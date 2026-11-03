#!/bin/bash
# re-run Hamba with extra outputs (HO-Cap first), then reproduce AUROC tables in separate folders
B=${HAND_PILOT_ROOT:-$HOME/hand_pilot}
for s in stage_b_hocap stage_b; do
  D=diagnostics/repro_$s
  cd $B/hamba_run && PILOT_SET=$D PYTHONNOUSERSITE=1 $B/hamba_env/bin/python -u $B/code/stage_b_infer.py 2>&1 | grep -v -i "warn\|downsampling\|lightning\|selective_scan_cuda_core\|^>>>\|so far"
  echo "infer done $s $(date +%T)"
  (cd $B/code && PILOT_SET=$D PYTHONNOUSERSITE=1 $B/env/bin/python -I stage_b_score.py > $B/results/$D/score.log 2>&1; echo "score done $s $(date +%T)") &
done
wait
