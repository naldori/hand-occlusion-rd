#!/bin/bash
B=${HAND_PILOT_ROOT:-$HOME/hand_pilot}; cd $B/code; export PYTHONNOUSERSITE=1
for s in stage_b_hocap stage_b; do
  W=$B/results/wilor_compare/$s
  OMP_NUM_THREADS=2 DIAG_SET=$s PRED_PATH=$W/pred.npz DIAG_OUT=$W/diag ../env/bin/python -I diagnostics.py > $W/diag.log 2>&1; echo "D1 done $s $(date +%T)"
  OMP_NUM_THREADS=1 DIAG_SET=$s C_MODE=apply C_VARIANTS=proxy PRED_PATH=$W/pred.npz C_OUT=$W/c C_WEIGHTS=$B/results/wilor_compare/c_weights_from_hamba.json \
    ../env/bin/python -I diagnostics2_c.py > $W/c.log 2>&1; echo "C-proxy done $s $(date +%T)"
done
