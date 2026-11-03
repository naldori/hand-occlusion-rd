#!/bin/bash
B=${HAND_PILOT_ROOT:-$HOME/hand_pilot}; cd $B/code
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
until [ -f $B/results/diagnostics2/c_weights_dexycb.json ]; do sleep 10; done
echo "grid done $(date +%T)"; grep -A4 '"chosen"' $B/results/diagnostics2/c_weights_dexycb.json
for s in stage_b_hocap stage_b; do
  DIAG_SET=$s C_MODE=apply ../env/bin/python -I diagnostics2_c.py 2>&1 | grep -vi "warn\|\.r)" | tail -20
  echo "apply done $s $(date +%T)"
done
# AUROC re-scoring of corrected B2 predictions (6 sets in parallel)
for s in stage_b_hocap stage_b; do for v in none proxy oracle; do
  (OMP_NUM_THREADS=3 PILOT_SET=diagnostics2/rescore_${s}_$v ../env/bin/python -I stage_b_score.py > $B/results/diagnostics2/rescore_${s}_$v/score.log 2>&1; echo "rescore done $s $v $(date +%T)") &
done; done
# post-hoc variant C* (5 mm penetration tolerance): grid on DexYCB, apply to both, depth/tip only
(PEN_TOL_MM=5 DIAG_SET=stage_b C_MODE=grid ../env/bin/python -I diagnostics2_c.py > $B/results/diagnostics2/posthoc_grid.log 2>&1
 for s in stage_b_hocap stage_b; do PEN_TOL_MM=5 DIAG_SET=$s C_MODE=apply ../env/bin/python -I diagnostics2_c.py 2>&1 | grep -vi "warn\|\.r)" | tail -12; done
 echo "posthoc done $(date +%T)") &
wait
echo "all done $(date +%T)"
