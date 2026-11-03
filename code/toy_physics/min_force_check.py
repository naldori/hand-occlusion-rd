"""Force check only: does a minimum normal force per touching finger (fmin) make
wrong-but-touching finger placements detectable? Floating/penetrating fingers are left
to the geometric check and not tested here. mu=0.5, Fmax=15N, 100 gravity dirs."""
import json, sys
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, ".")
import check_equilibrium as ce
from cause_check import grasp

G = ce.GRAVS[:100]

def feasible(P, N, g, mu=0.5, fmin=0.0, fmax=15.0):
    cols = []
    for p, n in zip(P, N):
        for d in ce.cone_edges(n, mu):
            cols.append(np.concatenate([d, np.cross(p, d) / ce.R]))
    A = np.array(cols).T; k = len(P); nv = A.shape[1]
    S = np.zeros((k, nv))
    for i in range(k):
        S[i, i*ce.N_EDGES:(i+1)*ce.N_EDGES] = 1      # normal force of contact i
    r = linprog(np.zeros(nv), A_ub=np.vstack([S, -S]), b_ub=np.r_[np.full(k, fmax), np.full(k, -fmin)],
                A_eq=A, b_eq=-np.r_[ce.MASS*ce.G*g, np.zeros(3)], bounds=(0, None), method="highs")
    return r.status == 0

def wrong_place(P0, N0, i, kind, rng):
    P, N = list(P0), list(N0)
    p = P[i]; th = np.arctan2(p[1], p[0])
    if kind == "slide30":          # another valid-looking grasp: inherent ambiguity
        a = rng.uniform(0, 2*np.pi)
        P[i], N[i] = ce.surface_point(th + .03*np.cos(a)/ce.R, np.clip(p[2] + .03*np.sin(a), -ce.H/2, ce.H/2))
    elif kind == "side90":         # touching, but on the side face (90 deg away)
        P[i], N[i] = ce.surface_point(th + rng.choice([-1, 1]) * np.pi/2, p[2])
    elif kind == "thumb_side":     # touching, next to the thumb (pushes same way as thumb)
        P[i], N[i] = ce.surface_point(np.pi + rng.uniform(-.5, .5), p[2])
    return P, N

out = {}
for nf in [2, 4]:
    P0, N0 = grasp(nf)
    for fmin in [0.0, 0.5, 1.0, 2.0]:
        ok = [g for g in G if feasible(P0, N0, g, fmin=fmin)]
        row = {"correct_feasible": len(ok) / len(G)}
        for kind in ["slide30", "side90", "thumb_side"]:
            rng = np.random.default_rng(1); r = []
            for i in range(1, nf + 1):
                for _ in range(3):
                    P, N = wrong_place(P0, N0, i, kind, rng)
                    r += [not feasible(P, N, g, fmin=fmin) for g in ok]
            row[kind] = round(float(np.mean(r)), 2) if r else None
        key = f"contacts={nf+1},fmin={fmin}"; out[key] = row; print(key, row, flush=True)
json.dump(out, open("min_force_check.json", "w"), indent=2)
