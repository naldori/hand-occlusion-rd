#!/bin/bash
# wait for HO-Cap labels, then Hamba inference (GPU env) and scoring (CPU env)
B=${HAND_PILOT_ROOT:-$HOME/hand_pilot}
while ps -eo args | awk '$1 ~ /env\/bin\/python$/' | grep -q hocap_labels; do sleep 20; done
test -f $B/results/stage_b_hocap/labels.json || { echo "labels missing"; exit 1; }
cd $B/hamba_run && PILOT_SET=stage_b_hocap PYTHONNOUSERSITE=1 $B/hamba_env/bin/python -u $B/code/stage_b_infer.py 2>&1 | grep -v -i "warn\|downsampling\|lightning\|selective_scan_cuda_core\|^>>>"
cd $B/code && PILOT_SET=stage_b_hocap PYTHONNOUSERSITE=1 $B/env/bin/python -I -u stage_b_score.py 2>&1 | grep -v -i "warn\|\.r)"
