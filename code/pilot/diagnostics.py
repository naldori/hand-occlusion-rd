"""Diagnostics D1-D3 (PROTOCOL.md 'Diagnostics', post-hoc; does not change the H1 verdict).

Usage: DIAG_SET=stage_b_hocap|stage_b python diagnostics.py
Reads the re-run predictions (results/diagnostics/repro_<set>/pred.npz) and existing labels.
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import trimesh
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_a as A  # noqa: E402
from stage_b_infer_utils import kabsch, to_local  # noqa: E402

SET = os.environ.get("DIAG_SET", "stage_b_hocap")
RES = f"{A.ROOT}/results"
OUT = os.environ.get("DIAG_OUT", f"{RES}/diagnostics/{SET}")
PRED_PATH = os.environ.get("PRED_PATH", f"{RES}/diagnostics/repro_{SET}/pred.npz")
LARGE_MM, NEAR_MM, MIN_DEN_DEG = 15.0, 5.0, 5.0
N_BOOT, SEED = 2000, 0
FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
FI = {f: i for i, f in enumerate(FINGERS)}
ALIGN_IDX = [0, 2, 5, 9, 13, 17]
PALM_IDX = [0, 5, 9, 13, 17]
HOCAP = f"{A.ROOT}/data_hocap"


def flexion(J):
    """total flexion (deg) per finger: sum of the 3 angles between neighbouring bone vectors."""
    out = np.zeros(5)
    for i in range(5):
        idx = [0, 4 * i + 1, 4 * i + 2, 4 * i + 3, 4 * i + 4]
        b = [J[idx[k + 1]] - J[idx[k]] for k in range(4)]
        for k in range(1, 4):
            c = np.dot(b[k - 1], b[k]) / (np.linalg.norm(b[k - 1]) * np.linalg.norm(b[k]))
            out[i] += np.degrees(np.arccos(np.clip(c, -1, 1)))
    return out


def project(p, K):
    q = K @ p
    return q[:2] / q[2]


def mean_pose_joints(layer, betas):
    p = torch.zeros(1, 48)                 # PCA coefficients 0 + flat_hand_mean=False -> hands_mean pose
    _, j = layer(p, torch.tensor(betas, dtype=torch.float32).reshape(1, 10), torch.zeros(1, 3))
    return j[0].numpy() / 1000


def gt_objects_and_betas(labels):
    """per image: (object mesh, 4x4 pose) of the grasped object, and subject betas."""
    info, meshes = {}, {}
    for r in labels:
        img = r['image']
        if img in info:
            continue
        if 'obj_mesh' in r:
            if r['obj_mesh'] not in meshes:
                meshes[r['obj_mesh']] = trimesh.load(r['obj_mesh'], force='mesh')
            betas = yaml.safe_load(open(f"{HOCAP}/calibration/mano/subject_5.yaml"))['betas']
            info[img] = (meshes[r['obj_mesh']], np.array(r['obj_pose']), betas,
                         np.array(r['gt_j21']))
        else:
            seq = f"{A.DATA}/{A.SUBJECT}/{r['seq']}"
            meta = yaml.load(open(f"{seq}/meta.yml"), Loader=yaml.FullLoader)
            oid = meta['ycb_ids'][meta['ycb_grasp_ind']]
            if oid not in meshes:
                meshes[oid] = A.load_object(oid)
            lab = np.load(img.replace('color_', 'labels_').replace('.jpg', '.npz'))
            betas = yaml.load(open(f"{A.DATA}/calibration/mano_{meta['mano_calib'][0]}/mano.yml"),
                              Loader=yaml.FullLoader)['betas']
            info[img] = (meshes[oid], np.vstack([lab['pose_y'][meta['ycb_grasp_ind']], [0, 0, 0, 1]]),
                         betas, lab['joint_3d'][0].astype(np.float64))
    return info


def signed_dist(mesh, pose, pts):
    pl = to_local(pts, pose)
    _, d, _ = trimesh.proximity.closest_point(mesh, pl)
    return np.where(mesh.contains(pl), -d, d)


def build_rows():
    labels = json.load(open(f"{RES}/{SET}/labels.json"))
    pred = np.load(PRED_PATH)
    idx = {p: i for i, p in enumerate(pred['image'])}
    info = gt_objects_and_betas(labels)
    layer = A.mano_layer()
    mean_cache, per_image = {}, {}
    rows = []
    for r in labels:
        img, f = r['image'], r['finger']
        mesh, pose, betas, gt = info[img]
        i = idx[img]
        if img not in per_image:
            key = tuple(np.round(betas, 6))
            if key not in mean_cache:
                mean_cache[key] = mean_pose_joints(layer, betas)
            jm = kabsch(mean_cache[key][ALIGN_IDX], gt[ALIGN_IDX])(mean_cache[key])   # aligned mean-pose hand
            tips = [4 * k + 4 for k in range(5)]
            per_image[img] = dict(
                flex_pred=flexion(pred['j_b1'][i]), flex_gt=flexion(gt), flex_mean=flexion(jm),
                sd_gt=signed_dist(mesh, pose, gt[tips]),
                sd={al: signed_dist(mesh, pose, pred[f'j_{al}'][i][tips]) for al in ['b1', 'b2']})
        P = per_image[img]
        K = np.array(r['K'])
        t = 4 * FI[f] + 4
        gtip = gt[t]
        u = gtip / np.linalg.norm(gtip)
        uw = gt[0] / np.linalg.norm(gt[0])
        for al in ['b1', 'b2']:
            J = pred[f'j_{al}'][i]
            e = J[t] - gtip
            w = J[0] - gt[0]
            en = np.linalg.norm(e)
            rows.append(dict(
                seq=r['seq'], image=img, finger=f, occlusion=r['occlusion'], align=al,
                err_mm=en * 1000, large=bool(en * 1000 > LARGE_MM),
                los_signed_mm=float(e @ u) * 1000, plane_mm=float(np.linalg.norm(e - (e @ u) * u)) * 1000,
                los_frac=float(abs(e @ u) / en) if en > 0 else np.nan,
                e2=float(e @ e), r2=float((e - w) @ (e - w)), w_los_frac=float(abs(w @ uw) / np.linalg.norm(w)),
                reproj_px=float(np.linalg.norm(project(J[t], K) - project(gtip, K))),
                theta_pred=float(P['flex_pred'][FI[f]]), theta_gt=float(P['flex_gt'][FI[f]]),
                theta_mean=float(P['flex_mean'][FI[f]]),
                sd_pred_mm=float(P['sd'][al][FI[f]]) * 1000, sd_gt_mm=float(P['sd_gt'][FI[f]]) * 1000,
                palm_tip_pred_mm=float(np.linalg.norm(J[t] - J[PALM_IDX].mean(0))) * 1000,
                palm_tip_gt_mm=float(np.linalg.norm(gtip - gt[PALM_IDX].mean(0))) * 1000))
    return rows


class Boot:
    """cluster bootstrap over sequences/episodes; stat(rows_subset) -> float."""

    def __init__(self, rows):
        self.seqs = sorted({r['seq'] for r in rows})
        self.rng = np.random.default_rng(SEED)
        self.picks = [self.rng.choice(len(self.seqs), len(self.seqs)) for _ in range(N_BOOT)]

    def ci(self, rows, stat):
        by = {s: [] for s in self.seqs}
        for r in rows:
            by[r['seq']].append(r)
        point = stat(rows)
        vals = []
        for p in self.picks:
            rr = [r for k in p for r in by[self.seqs[k]]]
            v = stat(rr) if rr else np.nan
            vals.append(v)
        vals = np.array(vals, float)
        vals = vals[~np.isnan(vals)]
        return dict(value=None if np.isnan(point) else round(float(point), 4),
                    ci95=[round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
                    if len(vals) else [None, None], n=len(rows))


def med(key):
    return lambda rr: float(np.nanmedian([r[key] for r in rr])) if rr else np.nan


def shrink(r):
    den = abs(r['theta_gt'] - r['theta_mean'])
    return abs(r['theta_pred'] - r['theta_mean']) / den if den >= MIN_DEN_DEG else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = build_rows()
    for r in rows:
        r['shrink'] = shrink(r)
        r['flex_err'] = r['theta_pred'] - r['theta_gt']
        r['palm_tip_diff_mm'] = r['palm_tip_pred_mm'] - r['palm_tip_gt_mm']
    json.dump(rows, open(f"{OUT}/rows.json", "w"))
    bt = Boot(rows)
    sel = lambda al, occ=None, large=None: [r for r in rows if r['align'] == al and (occ is None or r['occlusion'] == occ)
                                            and (large is None or r['large'] == large)]
    # ---- D1
    d1 = {}
    explained = lambda rr: 1 - sum(r['r2'] for r in rr) / sum(r['e2'] for r in rr) if rr else np.nan
    for al in ['b1', 'b2']:
        for occ in ['visible', 'object', 'self']:
            for large in [True, False]:
                S = sel(al, occ, large)
                k = f"{al}|{occ}|{'large' if large else 'small'}"
                d1[k] = dict(los_frac_median=bt.ci(S, med('los_frac')), reproj_px_median=bt.ci(S, med('reproj_px')),
                             los_signed_mm_median=bt.ci(S, med('los_signed_mm')),
                             plane_mm_median=bt.ci(S, med('plane_mm')))
                if al == 'b2':
                    d1[k].update(wrist_translation_explained=bt.ci(S, explained),
                                 wrist_los_frac_median=bt.ci(S, med('w_los_frac')))
    tgt_b2, tgt_b1 = d1['b2|object|large'], d1['b1|object|large']
    d1['criterion_a_hand_depth'] = dict(
        explained=tgt_b2['wrist_translation_explained']['value'], wrist_los_frac=tgt_b2['wrist_los_frac_median']['value'],
        met=bool(tgt_b2['wrist_translation_explained']['value'] >= 0.70 and tgt_b2['wrist_los_frac_median']['value'] >= 0.7))
    d1['criterion_b_finger_depth'] = dict(
        los_frac=tgt_b1['los_frac_median']['value'], reproj_px=tgt_b1['reproj_px_median']['value'],
        met=bool(tgt_b1['los_frac_median']['value'] >= 0.6 and tgt_b1['reproj_px_median']['value'] < 10))
    json.dump(d1, open(f"{OUT}/d1_error_direction.json", "w"), indent=1)
    # ---- D2 (angles are the same for B1/B2; 'large' is per alignment)
    d2 = {}
    for al in ['b1', 'b2']:
        for occ in ['visible', 'object', 'self']:
            for large in [None, True, False]:
                S = sel(al, occ, large)
                k = f"{al}|{occ}|{ {None: 'all', True: 'large', False: 'small'}[large] }"
                d2[k] = dict(shrink_median=bt.ci(S, med('shrink')),
                             n_excluded_small_denominator=int(sum(np.isnan(r['shrink']) for r in S)),
                             flex_err_median_deg=bt.ci(S, med('flex_err')))
    diff = lambda rr: (float(np.nanmedian([r['shrink'] for r in rr if r['occlusion'] == 'object']))
                       - float(np.nanmedian([r['shrink'] for r in rr if r['occlusion'] == 'visible'])))
    d2['b1|object_minus_visible_shrink_median'] = bt.ci([r for r in sel('b1') if r['occlusion'] in ('object', 'visible')], diff)
    o = d2['b1|object|all']['shrink_median']
    dd = d2['b1|object_minus_visible_shrink_median']
    d2['criterion_c_regress_to_mean'] = dict(object_shrink_median=o['value'], diff=dd['value'], diff_ci=dd['ci95'],
                                             met=bool(o['value'] < 0.8 and (dd['ci95'][0] > 0 or dd['ci95'][1] < 0)))
    fe = d2['b1|object|large']['flex_err_median_deg']
    d2['criterion_more_extended'] = dict(median=fe['value'], ci=fe['ci95'], met=bool(fe['value'] < 0 and fe['ci95'][1] < 0))
    json.dump(d2, open(f"{OUT}/d2_flexion.json", "w"), indent=1)
    # ---- D3
    d3 = {}
    for al in ['b1', 'b2']:
        S = sel(al, 'object', True)
        for name, fn in [('floating_gt5mm', lambda r: r['sd_pred_mm'] > NEAR_MM),
                         ('near_surface_pm5mm', lambda r: abs(r['sd_pred_mm']) <= NEAR_MM),
                         ('penetrating_lt_m5mm', lambda r: r['sd_pred_mm'] < -NEAR_MM)]:
            d3[f"{al}|object|large|pred_{name}"] = bt.ci(S, lambda rr, fn=fn: float(np.mean([fn(r) for r in rr])) if rr else np.nan)
        d3[f"{al}|object|large|gt_near_surface_pm5mm"] = bt.ci(
            S, lambda rr: float(np.mean([abs(r['sd_gt_mm']) <= NEAR_MM for r in rr])) if rr else np.nan)
        d3[f"{al}|object|large|sd_pred_mm_median"] = bt.ci(S, med('sd_pred_mm'))
        d3[f"{al}|object|large|sd_gt_mm_median"] = bt.ci(S, med('sd_gt_mm'))
        for occ in ['visible', 'object', 'self']:
            for large in [True, False]:
                d3[f"{al}|{occ}|{'large' if large else 'small'}|palm_tip_pred_minus_gt_mm_median"] = \
                    bt.ci(sel(al, occ, large), med('palm_tip_diff_mm'))
    json.dump(d3, open(f"{OUT}/d3_object_relation.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in {**d1, **d2}.items() if k.startswith('criterion')}, indent=1))


if __name__ == "__main__":
    main()
