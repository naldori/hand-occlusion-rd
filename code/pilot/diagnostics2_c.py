"""Diagnostics 2, part C (E2): fix only the global hand translation using the (GT) object as a ruler.

Usage:
  DIAG_SET=stage_b C_MODE=grid  python diagnostics2_c.py   # weight search on DexYCB (all variants)
  DIAG_SET=<set> C_MODE=apply   python diagnostics2_c.py   # chosen weights -> corrected preds + error summary
Definitions: PROTOCOL.md 'Diagnostics 2' C.
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D  # noqa: E402
import stage_a as A  # noqa: E402
from stage_b_infer_utils import to_local  # noqa: E402

SET = os.environ.get("DIAG_SET", "stage_b_hocap")
MODE = os.environ.get("C_MODE", "apply")
PEN_TOL = float(os.environ.get("PEN_TOL_MM", "0")) / 1000      # 0 = pre-registered; 5 = post-hoc variant C*
OUT = os.environ.get("C_OUT", f"{D.RES}/diagnostics2" + ("" if PEN_TOL == 0 else f"/posthoc_pen_tol_{os.environ['PEN_TOL_MM']}mm"))
WEIGHTS = os.environ.get("C_WEIGHTS", f"{OUT}/c_weights_dexycb.json")
PRED_PATH = os.environ.get("PRED_PATH", f"{D.RES}/diagnostics/repro_{SET}/pred.npz")
VARIANTS = os.environ.get('C_VARIANTS', 'none,proxy,oracle').split(',')
GRID = [(wc, wp) for wc in (0.1, 1.0, 10.0) for wp in (1.0, 10.0)]
SIG_PX, SIG_M, K_NEAR = 5.0, 0.005, 20
N_SAMPLES = 50000
PALM = [0, 2, 5, 9, 13, 17]
DISTAL_PART = {'index': 3, 'middle': 6, 'pinky': 9, 'ring': 12, 'thumb': 15}
FINGER_JOINTS = {f: [4 * i + 1, 4 * i + 2, 4 * i + 3, 4 * i + 4] for i, f in enumerate(D.FINGERS)}
_G = {}


def prepare():
    labels = json.load(open(f"{D.RES}/{SET}/labels.json"))
    pred = np.load(PRED_PATH)
    idx = {p: i for i, p in enumerate(pred['image'])}
    info = D.gt_objects_and_betas(labels)
    layer = A.mano_layer()
    part_of_v = layer.th_weights.numpy().argmax(1)
    distal = {f: np.where(part_of_v == DISTAL_PART[f])[0] for f in D.FINGERS}
    by_img = {}
    for r in labels:
        by_img.setdefault(r['image'], []).append(r)
    samplers, items = {}, []
    for img, rs in by_img.items():
        mesh, pose, _, gt = info[img]
        key = id(mesh)
        if key not in samplers:
            pts, fidx = trimesh.sample.sample_surface(mesh, N_SAMPLES, seed=0)
            samplers[key] = (pts, mesh)
        i = idx[img]
        occ = {r['finger']: r['occlusion'] for r in rs}
        vis = sorted(set(PALM) | {j for f in D.FINGERS if occ[f] != 'object' for j in FINGER_JOINTS[f]})
        items.append(dict(
            image=img, seq=rs[0]['seq'], key=key, pose=pose, gt=gt, K=np.array(rs[0]['K']),
            Vh=pred['v_b2'][i] - pred['t_b2'][i], Jh=pred['j_b2'][i] - pred['t_b2'][i], t0=pred['t_b2'][i],
            uv=pred['kp2d_full'][i], vis=vis, occ=occ,
            proxy=[f for f in D.FINGERS if occ[f] == 'object'],
            oracle=[f for r in rs for f in [r['finger']] if r['gt_contact_3mm']]))
    return items, samplers, distal, labels, pred, idx


def _init(samplers, distal):
    _G['trees'] = {k: (cKDTree(p), m) for k, (p, m) in samplers.items()}
    _G['distal'] = distal


def signed(item, pts):
    """|distance| from surface samples (KD-tree), sign from the exact ray-based inside test.
    (The first version took the sign from the nearest sample's normal, which flipped ~20 far vertices
    per hand to 'inside' and inflated the penetration sum 15-80x; caught by the GT-hand unit test.)"""
    tree, mesh = _G['trees'][item['key']]
    q = to_local(pts, item['pose'])
    d, _ = tree.query(q)
    return np.where(mesh.contains(q), -d, d)


def cost(t, item, variant, wc, wp):
    J = item['Jh'] + t
    V = item['Vh'] + t
    K = item['K']
    pj = J[item['vis']] @ K.T
    uv = pj[:, :2] / pj[:, 2:]
    c2d = np.mean(np.sum((uv - item['uv'][item['vis']]) ** 2, axis=1)) / SIG_PX ** 2
    sd = signed(item, V)
    fingers = item['proxy'] if variant == 'proxy' else item['oracle'] if variant == 'oracle' else []
    if fingers:
        c = np.mean([np.abs(sd[_G['distal'][f]]).min() for f in fingers])
    else:
        c = np.sort(np.abs(sd))[:K_NEAR].mean()
    pen = np.maximum(-sd - PEN_TOL, 0).sum()
    return c2d + wc * (c / SIG_M) ** 2 + wp * (pen / SIG_M) ** 2


def optimize(args):
    item, variant, wc, wp = args
    x0 = item['t0']
    simplex = np.vstack([x0, x0 + [0.02, 0, 0], x0 + [0, 0.02, 0], x0 + [0, 0, 0.02]])
    res = minimize(cost, x0, args=(item, variant, wc, wp), method='Nelder-Mead',
                   options=dict(initial_simplex=simplex, maxfev=300, xatol=1e-4, fatol=1e-4))
    return res.x


def depth_err(item, t):
    return abs((item['Jh'] + t)[0, 2] - item['gt'][0, 2]) * 1000


def run(items, combos, samplers, distal):
    jobs = [(it, v, wc, wp) for (v, wc, wp) in combos for it in items]
    light = [({k: it[k] for k in it}, v, wc, wp) for it, v, wc, wp in jobs]
    with Pool(16, initializer=_init, initargs=(samplers, distal)) as pool:
        ts = pool.map(optimize, light, chunksize=8)
    out = {}
    for (it, v, wc, wp), t in zip(jobs, ts):
        out.setdefault((v, wc, wp), []).append(t)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    items, samplers, distal, labels, pred, idx = prepare()
    if MODE == 'grid':
        res = run(items, [(v, wc, wp) for v in VARIANTS for wc, wp in GRID], samplers, distal)
        table = {f"{v}|wc={wc}|wp={wp}": float(np.median([depth_err(it, t) for it, t in zip(items, ts)]))
                 for (v, wc, wp), ts in res.items()}
        table['b2_baseline'] = float(np.median([depth_err(it, it['t0']) for it in items]))
        chosen = {v: min(((wc, wp) for wc, wp in GRID), key=lambda w: table[f"{v}|wc={w[0]}|wp={w[1]}"]) for v in VARIANTS}
        json.dump(dict(grid_median_wrist_depth_err_mm=table, chosen={v: dict(wc=w[0], wp=w[1]) for v, w in chosen.items()}),
                  open(f"{OUT}/c_weights_dexycb.json", "w"), indent=1)
        print(json.dumps(table, indent=1)); print('chosen', chosen)
        return
    chosen = json.load(open(WEIGHTS))['chosen']
    res = run(items, [(v, chosen[v]['wc'], chosen[v]['wp']) for v in VARIANTS], samplers, distal)
    os.makedirs(f"{OUT}/{SET}", exist_ok=True)
    rows = []
    tips = {f: 4 * i + 4 for i, f in enumerate(D.FINGERS)}
    for v in VARIANTS:
        ts = res[(v, chosen[v]['wc'], chosen[v]['wp'])]
        # corrected predictions for re-scoring (B1 untouched, B2 replaced)
        d = {k: pred[k].copy() for k in pred.files}
        for it, t in zip(items, ts):
            i = idx[it['image']]
            d['v_b2'][i] = it['Vh'] + t
            d['j_b2'][i] = it['Jh'] + t
        rd = f"{OUT}/rescore_{SET}_{v}"
        os.makedirs(rd, exist_ok=True)
        np.savez_compressed(f"{rd}/pred.npz", **d)
        if not os.path.exists(f"{rd}/labels.json"):
            os.symlink(f"{D.RES}/{SET}/labels.json", f"{rd}/labels.json")
        for it, t in zip(items, ts):
            J0, J1 = it['Jh'] + it['t0'], it['Jh'] + t
            for f in D.FINGERS:
                rows.append(dict(seq=it['seq'], image=it['image'], variant=v, finger=f, occlusion=it['occ'][f],
                                 depth_b2=depth_err(it, it['t0']), depth_c=depth_err(it, t),
                                 tip_b2=float(np.linalg.norm(J0[tips[f]] - it['gt'][tips[f]])) * 1000,
                                 tip_c=float(np.linalg.norm(J1[tips[f]] - it['gt'][tips[f]])) * 1000,
                                 shift_mm=float(np.linalg.norm(t - it['t0'])) * 1000))
    json.dump(rows, open(f"{OUT}/{SET}/c_rows.json", "w"))
    bt = D.Boot(rows)
    summ = {}
    for v in VARIANTS:
        img_rows = [r for r in rows if r['variant'] == v and r['finger'] == 'thumb']     # one per image
        summ[v] = dict(weights=chosen[v],
                       wrist_depth_b2_mm=bt.ci(img_rows, D.med('depth_b2')),
                       wrist_depth_c_mm=bt.ci(img_rows, D.med('depth_c')),
                       wrist_depth_change_mm=bt.ci(img_rows, lambda rr: float(np.median([r['depth_c'] for r in rr]) - np.median([r['depth_b2'] for r in rr])) if rr else np.nan),
                       wrist_depth_reduction=bt.ci(img_rows, lambda rr: float(1 - np.median([r['depth_c'] for r in rr]) / np.median([r['depth_b2'] for r in rr])) if rr else np.nan),
                       translation_shift_mm_median=bt.ci(img_rows, D.med('shift_mm')))
        for occ in ['visible', 'object']:
            S = [r for r in rows if r['variant'] == v and r['occlusion'] == occ]
            summ[v][f'{occ}_tip_b2_mm'] = bt.ci(S, D.med('tip_b2'))
            summ[v][f'{occ}_tip_c_mm'] = bt.ci(S, D.med('tip_c'))
            summ[v][f'{occ}_tip_change_mm'] = bt.ci(S, lambda rr: float(np.median([r['tip_c'] for r in rr]) - np.median([r['tip_b2'] for r in rr])) if rr else np.nan)
    met = {v: bool(summ[v]['wrist_depth_reduction']['value'] >= 0.30 and summ[v]['wrist_depth_change_mm']['ci95'][1] < 0)
           for v in ['none', 'proxy'] if v in VARIANTS}
    summ['criterion_object_tells_depth'] = dict(per_variant=met, met=bool(any(met.values())),
                                                applies_to='HO-Cap only (DexYCB reported for reference; weights tuned on DexYCB)')
    json.dump(summ, open(f"{OUT}/{SET}/c_object_ruler.json", "w"), indent=1)
    print(json.dumps({v: dict(depth_b2=summ[v]['wrist_depth_b2_mm']['value'], depth_c=summ[v]['wrist_depth_c_mm']['value'],
                              red=summ[v]['wrist_depth_reduction']) for v in VARIANTS}, indent=1))
    print(summ['criterion_object_tells_depth'])


if __name__ == "__main__":
    main()
