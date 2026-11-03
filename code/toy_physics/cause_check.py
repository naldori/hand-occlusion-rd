"""Is the null result due to contact redundancy, or to loose settings (Fmax)?
Detection rate = P(equilibrium infeasible | one non-thumb finger wrong), averaged over gravity dirs
where the correct grasp IS feasible."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import check_equilibrium as ce

G = ce.GRAVS[:100]
def grasp(n_fingers):
    P, N = [], []
    p, n = ce.surface_point(np.pi, 0.0); P.append(p); N.append(n)
    th = [-0.35, -0.1, 0.15, 0.4][:n_fingers]; z = [0.03, 0.01, -0.01, -0.03][:n_fingers]
    if n_fingers == 1: th, z = [0.0], [0.0]
    if n_fingers == 2: th, z = [-0.2, 0.2], [0.015, -0.015]
    for a, b in zip(th, z):
        p, n = ce.surface_point(a, b); P.append(p); N.append(n)
    return P, N

out = {}
for nf in [1, 2, 4]:
    P0, N0 = grasp(nf)
    for fmax in [15.0, 5.0, 3.0]:
        ok = [g for g in G if ce.min_residual(P0, N0, g, 0.5, fmax) < 1e-6]
        if not ok:
            out[f"fingers={nf},Fmax={fmax}"] = "correct grasp infeasible"; continue
        det = {}
        for kind, d in [("slide", .01), ("slide", .03), ("drop", 0), ("wrong_side", 0)]:
            r = []
            for i in range(1, nf + 1):
                for _ in range(3):
                    P, N = ce.perturb(P0, N0, i, kind, d)
                    r += [ce.min_residual(P, N, g, 0.5, fmax) > 1e-6 for g in ok]
            det[f"{kind}{int(d*1000) or ''}"] = round(float(np.mean(r)), 2)
        out[f"fingers={nf},Fmax={fmax}"] = {"correct_feasible_dirs": len(ok), **det}
        print(f"fingers={nf},Fmax={fmax}", out[f"fingers={nf},Fmax={fmax}"], flush=True)
json.dump(out, open("cause_check.json", "w"), indent=2)
