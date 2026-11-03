"""Continuous score: minimal max-normal-force t needed for equilibrium (lower = easier grasp).
Compares correct power grasp vs. one wrongly estimated (occluded) finger, mu=0.5."""
import json, sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, ".")
import check_equilibrium as ce

def min_peak_force(P, N, g, mu=0.5):
    cols = []
    for p, n in zip(P, N):
        for d in ce.cone_edges(n, mu):
            cols.append(np.concatenate([d, np.cross(p, d) / ce.R]))
    A = np.array(cols).T; nv = A.shape[1]; k = len(P)
    A_eq = np.hstack([A, np.zeros((6, 1))])
    b_eq = -np.concatenate([ce.MASS * ce.G * g, np.zeros(3)])
    A_ub = np.zeros((k, nv + 1))
    for i in range(k):
        A_ub[i, i*ce.N_EDGES:(i+1)*ce.N_EDGES] = 1; A_ub[i, -1] = -1
    c = np.zeros(nv + 1); c[-1] = 1
    r = linprog(c, A_ub=A_ub, b_ub=np.zeros(k), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return r.fun if r.status == 0 else np.inf

G = ce.GRAVS[:100]
P0, N0 = ce.power_grasp()
base = np.array([min_peak_force(P0, N0, g) for g in G])
out = {"correct_grasp_median_N": round(float(np.median(base)), 2)}
for kind, d in [("slide", .005), ("slide", .01), ("slide", .02), ("slide", .04), ("drop", 0), ("wrong_side", 0)]:
    ratios = []
    for i in range(1, 5):
        for _ in range(3):
            P, N = ce.perturb(P0, N0, i, kind, d)
            s = np.array([min_peak_force(P, N, g) for g in G])
            ratios.append(np.median(s / base))
    out[f"{kind}_{int(d*1000)}mm_median_ratio"] = round(float(np.median(ratios)), 3)
    out[f"{kind}_{int(d*1000)}mm_max_ratio"] = round(float(np.max(ratios)), 3)
print(out); json.dump(out, open("score_sensitivity.json", "w"), indent=2)
