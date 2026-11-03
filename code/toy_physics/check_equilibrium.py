"""Quick numerical check of the 'simulation-free static equilibrium' idea.

E1  LP solve time for ~5 contacts.
E2  How often does a feasible equilibrium exist vs. number of contacts?
E3  If one (occluded) finger is estimated wrongly, does the equilibrium check notice?

Object: cylinder (YCB-can-like), contacts as point contacts with linearized
Coulomb friction cones. Gravity direction is sampled because the hand/object
orientation w.r.t. gravity is unknown from a single image.
All settings are fixed below before running.
"""
import json
import time

import numpy as np
from scipy.optimize import linprog

RNG = np.random.default_rng(0)
R, H = 0.035, 0.12          # cylinder radius / height [m]
MASS, G = 0.3, 9.81         # object mass [kg]
MU_LIST = [0.3, 0.5, 0.8]
FMAX = 15.0                 # max normal force per finger [N]
N_EDGES = 8                 # friction pyramid edges
N_GRAV = 200                # gravity directions per configuration
CONTACT_TOL = 0.002         # a finger counts as touching if within 2 mm


def surface_point(theta, z):
    p = np.array([R * np.cos(theta), R * np.sin(theta), z])
    n_in = -np.array([np.cos(theta), np.sin(theta), 0.0])   # inward normal
    return p, n_in


def cone_edges(n, mu):
    t1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.cross(n, [1.0, 0, 0])
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    ang = np.linspace(0, 2 * np.pi, N_EDGES, endpoint=False)
    return np.stack([n + mu * (np.cos(a) * t1 + np.sin(a) * t2) for a in ang])


def min_residual(P, N, g_dir, mu, fmax=FMAX):
    """LP: min ||wrench residual||_1 s.t. forces in friction cones, normal force <= fmax.
    Returns residual (0 => feasible equilibrium) in units of N / N*m(scaled by R)."""
    k = len(P)
    if k == 0:
        return np.inf
    cols = []
    for p, n in zip(P, N):
        for d in cone_edges(n, mu):
            cols.append(np.concatenate([d, np.cross(p, d) / R]))  # torque scaled to N
    A = np.array(cols).T                      # 6 x (k*E)
    w_ext = np.concatenate([MASS * G * g_dir, np.zeros(3)])
    nv = A.shape[1]
    # variables: lambda (nv), s+ (6), s- (6);  A lam + s+ - s- = -w_ext
    A_eq = np.hstack([A, np.eye(6), -np.eye(6)])
    b_eq = -w_ext
    # normal force bound: sum_j lambda_ij * (d_ij . n_i) = sum_j lambda_ij <= fmax
    A_ub = np.zeros((k, nv + 12))
    for i in range(k):
        A_ub[i, i * N_EDGES:(i + 1) * N_EDGES] = 1.0
    b_ub = np.full(k, fmax)
    c = np.concatenate([np.zeros(nv), np.ones(12)])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    return res.fun


def random_gravity(n):
    v = RNG.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


GRAVS = random_gravity(N_GRAV)


def feasible_rate(P, N, mu):
    return float(np.mean([min_residual(P, N, g, mu) < 1e-6 for g in GRAVS]))


def power_grasp():
    """thumb on one side, 4 fingers wrapped on the opposite side (index..pinky)."""
    P, N = [], []
    p, n = surface_point(np.pi, 0.0)            # thumb
    P.append(p); N.append(n)
    for th, z in zip([-0.35, -0.1, 0.15, 0.4], [0.03, 0.01, -0.01, -0.03]):
        p, n = surface_point(th, z)
        P.append(p); N.append(n)
    return P, N


def e1_timing():
    P, N = power_grasp()
    g = GRAVS[0]
    min_residual(P, N, g, 0.5)
    t0 = time.perf_counter()
    for _ in range(500):
        min_residual(P, N, g, 0.5)
    return (time.perf_counter() - t0) / 500 * 1e3


def e2_contact_count():
    out = {}
    for mu in MU_LIST:
        for k in [2, 3, 4, 5]:
            rates = []
            for _ in range(30):                   # random contact sets on side surface
                th = RNG.uniform(0, 2 * np.pi, k)
                z = RNG.uniform(-H / 2 * 0.8, H / 2 * 0.8, k)
                P, N = zip(*[surface_point(a, b) for a, b in zip(th, z)])
                rates.append(feasible_rate(P, N, mu))
            out[f"mu={mu},k={k}"] = round(float(np.mean(rates)), 3)
    return out


def perturb(P, N, i, kind, d):
    P, N = list(P), list(N)
    p = P[i]
    th = np.arctan2(p[1], p[0])
    if kind == "slide":                        # stays on surface, wrong location
        a = RNG.uniform(0, 2 * np.pi)
        th2, z2 = th + d * np.cos(a) / R, np.clip(p[2] + d * np.sin(a), -H / 2, H / 2)
        P[i], N[i] = surface_point(th2, z2)
    elif kind == "lift":                       # finger floats off surface by d
        if d > CONTACT_TOL:
            del P[i]; del N[i]
    elif kind == "drop":                       # occluded finger entirely missing
        del P[i]; del N[i]
    elif kind == "wrong_side":                 # finger placed next to the thumb
        P[i], N[i] = surface_point(np.pi + RNG.uniform(-0.5, 0.5), p[2])
    return P, N


def e3_sensitivity(mu=0.5):
    P0, N0 = power_grasp()
    base = feasible_rate(P0, N0, mu)
    out = {"baseline_feasible_rate": base}
    for kind, ds in [("slide", [0.005, 0.01, 0.02, 0.04]),
                     ("lift", [0.001, 0.005, 0.02]),
                     ("drop", [0]), ("wrong_side", [0])]:
        for d in ds:
            rates = []
            for i in range(1, 5):              # each non-thumb finger
                for _ in range(3):
                    P, N = perturb(P0, N0, i, kind, d)
                    rates.append(feasible_rate(P, N, mu))
            out[f"{kind}_{int(d*1000)}mm"] = round(float(np.mean(rates)), 3)
    # thumb missing (largest single-contact error)
    out["thumb_drop"] = feasible_rate(P0[1:], N0[1:], mu)
    return out


if __name__ == "__main__":
    res = {"settings": dict(R=R, H=H, mass=MASS, fmax=FMAX, mu=MU_LIST,
                            n_edges=N_EDGES, n_gravity_dirs=N_GRAV)}
    res["E1_ms_per_LP"] = round(e1_timing(), 3)
    print("E1", res["E1_ms_per_LP"], "ms", flush=True)
    res["E2_feasible_rate_random_contacts"] = e2_contact_count()
    print("E2", res["E2_feasible_rate_random_contacts"], flush=True)
    res["E3_feasible_rate_after_error_mu0.5"] = e3_sensitivity()
    print("E3", res["E3_feasible_rate_after_error_mu0.5"], flush=True)
    json.dump(res, open("results.json", "w"), indent=2)
