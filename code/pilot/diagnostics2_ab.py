"""Diagnostics 2, parts A ('floating' causes, B1) and B (hand-depth error causes, B2).
Definitions and criteria: PROTOCOL.md 'Diagnostics 2'. Usage: DIAG_SET=stage_b_hocap|stage_b python diagnostics2_ab.py
"""
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D  # noqa: E402
from stage_b_infer_utils import solve_translation, to_local  # noqa: E402

SET = os.environ.get("DIAG_SET", "stage_b_hocap")
OUT = f"{D.RES}/diagnostics2/{SET}"
FLOAT_MM = 5.0
STRAIGHT_DEG = 10.0
HAMBA_FOCAL = 5000.0 / 256.0          # x max(W, H), demo convention
IMG_W, IMG_H = 640, 480


def chain(i):
    return [0, 4 * i + 1, 4 * i + 2, 4 * i + 3, 4 * i + 4]


def length_displacement(Jp, Jg, i):
    c = chain(i)
    p = Jp[0].copy()
    for k in range(4):
        b = Jp[c[k + 1]] - Jp[c[k]]
        p = p + b / np.linalg.norm(b) * np.linalg.norm(Jg[c[k + 1]] - Jg[c[k]])
    return float(np.linalg.norm(Jp[c[-1]] - p))


def signed_angle(a, b, axis):
    return float(np.degrees(np.arctan2(axis @ np.cross(a, b), a @ b)))


def abd_twist(J, i):
    n = np.cross(J[5] - J[0], J[17] - J[0]); n /= np.linalg.norm(n)
    proj = lambda v: v - (v @ n) * n
    c = chain(i)
    prox = J[c[2]] - J[c[1]]
    mid = J[c[3]] - J[c[2]]
    abd = signed_angle(proj(J[9] - J[0]), proj(prox), n)
    a = prox / np.linalg.norm(prox)
    bend = np.degrees(np.arccos(np.clip(a @ (mid / np.linalg.norm(mid)), -1, 1)))
    if bend < STRAIGHT_DEG:
        return abd, None
    m = np.cross(prox, mid); m /= np.linalg.norm(m)
    ref = n - (n @ a) * a; ref /= np.linalg.norm(ref)
    return abd, signed_angle(ref, m, a)


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ beta
    return 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))


def main():
    os.makedirs(OUT, exist_ok=True)
    labels = json.load(open(f"{D.RES}/{SET}/labels.json"))
    pred = np.load(f"{D.RES}/diagnostics/repro_{SET}/pred.npz")
    idx = {p: i for i, p in enumerate(pred['image'])}
    info = D.gt_objects_and_betas(labels)
    drow = {(r['image'], r['finger'], r['align']): r for r in json.load(open(f"{D.RES}/diagnostics/{SET}/rows.json"))}
    rows, img_rows, n_straight = [], {}, 0
    for r in labels:
        img, f = r['image'], r['finger']
        mesh, pose, _, gt = info[img]
        i, fi = idx[img], D.FI[f]
        Jp = pred['j_b1'][i]
        base = drow[(img, f, 'b1')]
        t = 4 * fi + 4
        e = Jp[t] - gt[t]
        # A3 angles
        abd_p, tw_p = abd_twist(Jp, fi)
        abd_g, tw_g = abd_twist(gt, fi)
        tw_err = None if tw_p is None or tw_g is None else abs(wrap(tw_p - tw_g))
        n_straight += tw_err is None
        # A4 surface frame at GT tip
        tl = to_local(gt[t][None], pose)
        cp, _, tri = trimesh.proximity.closest_point(mesh, tl)
        n_s = pose[:3, :3] @ mesh.face_normals[tri[0]]
        en = np.linalg.norm(e)
        rows.append(dict(
            seq=r['seq'], image=img, finger=f, occlusion=r['occlusion'], large=base['large'],
            gt_contact=r['gt_contact_3mm'], err_mm=base['err_mm'],
            pred_float=base['sd_pred_mm'] > FLOAT_MM, gt_float=base['sd_gt_mm'] > FLOAT_MM,
            bone_ratio=[float(np.linalg.norm(Jp[chain(fi)[k + 1]] - Jp[chain(fi)[k]]) /
                              np.linalg.norm(gt[chain(fi)[k + 1]] - gt[chain(fi)[k]])) for k in range(4)],
            hand_size_ratio=float(np.linalg.norm(Jp[9] - Jp[0]) / np.linalg.norm(gt[9] - gt[0])),
            len_disp_mm=length_displacement(Jp, gt, fi) * 1000,
            flex_err=abs(base['theta_pred'] - base['theta_gt']), abd_err=abs(wrap(abd_p - abd_g)),
            twist_err=0.0 if tw_err is None else tw_err,
            normal_mm=float(e @ n_s) * 1000, normal_frac=float(abs(e @ n_s) / en)))
        if img not in img_rows:   # B: one row per image
            Jh = pred['j_b2'][i] - pred['t_b2'][i]
            K = np.array(r['K'])
            uv = pred['kp2d_full'][i]
            sp, sg = np.linalg.norm(Jh[9] - Jh[0]), np.linalg.norm(gt[9] - gt[0])
            zb2 = pred['j_b2'][i][0, 2]
            Js = Jh * (sg / sp)
            ts = solve_translation(Js, uv, K)
            fdef = HAMBA_FOCAL * max(IMG_W, IMG_H)
            Kd = np.array([[fdef, 0, IMG_W / 2], [0, fdef, IMG_H / 2], [0, 0, 1.0]])
            td = solve_translation(Jh, uv, Kd)
            tips = [4, 8, 12, 16, 20]
            img_rows[img] = dict(
                seq=r['seq'], image=img, s=float(sp / sg - 1), d=float((zb2 - gt[0, 2]) / gt[0, 2]),
                depth_err_b2_mm=abs(zb2 - gt[0, 2]) * 1000, depth_err_size_mm=abs((Js + ts)[0, 2] - gt[0, 2]) * 1000,
                depth_err_hamba_focal_mm=abs((Jh + td)[0, 2] - gt[0, 2]) * 1000,
                tip_err_b2_mm=float(np.median(np.linalg.norm(pred['j_b2'][i][tips] - gt[tips], axis=1))) * 1000,
                tip_err_size_mm=float(np.median(np.linalg.norm((Js + ts)[tips] - gt[tips], axis=1))) * 1000)
    json.dump(rows, open(f"{OUT}/a_rows.json", "w"))
    json.dump(list(img_rows.values()), open(f"{OUT}/b_rows.json", "w"))
    bt = D.Boot(rows)
    G = [r for r in rows if r['occlusion'] == 'object' and r['large']]
    med = D.med
    rate = lambda key: (lambda rr: float(np.mean([r[key] for r in rr])) if rr else np.nan)
    a = {}
    a['A1_gt_contact_fingers_gt_floating_rate'] = bt.ci([r for r in rows if r['gt_contact']], rate('gt_float'))
    a['A1_group_pred_floating_rate'] = bt.ci(G, rate('pred_float'))
    a['A1_group_gt_floating_rate_baseline'] = bt.ci(G, rate('gt_float'))
    a['A1_group_pred_minus_baseline'] = bt.ci(G, lambda rr: np.mean([r['pred_float'] for r in rr]) - np.mean([r['gt_float'] for r in rr]) if rr else np.nan)
    a['A2_group_hand_size_ratio_median'] = bt.ci(G, med('hand_size_ratio'))
    for k in range(4):
        a[f'A2_group_bone{k}_ratio_median'] = bt.ci(G, lambda rr, k=k: float(np.median([r['bone_ratio'][k] for r in rr])) if rr else np.nan)
    a['A2_group_length_displacement_mm_median'] = bt.ci(G, med('len_disp_mm'))
    a['A2_group_tip_error_mm_median'] = bt.ci(G, med('err_mm'))
    a['A2_group_length_share'] = bt.ci(G, lambda rr: float(np.median([r['len_disp_mm'] for r in rr]) / np.median([r['err_mm'] for r in rr])) if rr else np.nan)

    def partial(which):
        def f(rr):
            if len(rr) < 10:
                return np.nan
            y = np.array([r['err_mm'] for r in rr])
            F = np.array([[r['flex_err'], r['abd_err'], r['twist_err']] for r in rr])
            full = r2(F, y)
            return full - r2(F[:, 1:], y) if which == 'flex' else full - r2(F[:, :1], y)
        return f
    a['A3_group_partial_R2_flexion'] = bt.ci(G, partial('flex'))
    a['A3_group_partial_R2_abduction_twist'] = bt.ci(G, partial('abd'))
    a['A3_group_median_abs_err_deg'] = {k: float(np.median([r[k] for r in G])) for k in ['flex_err', 'abd_err', 'twist_err']}
    a['A3_n_twist_undefined_straight_fingers_all_rows'] = int(n_straight)
    Gc = [r for r in G if r['gt_contact']]
    a['A4_group_gt_contact_normal_frac_median'] = bt.ci(Gc, med('normal_frac'))
    a['A4_group_gt_contact_normal_mm_median'] = bt.ci(Gc, med('normal_mm'))
    ok = lambda v: v is not None
    p, bl = a['A1_group_pred_floating_rate']['value'], a['A1_group_gt_floating_rate_baseline']['value']
    pf, pa = a['A3_group_partial_R2_flexion'], a['A3_group_partial_R2_abduction_twist']
    a['criteria'] = dict(
        annotation_noise=bool(bl >= 0.5 * p),
        finger_length=bool(a['A2_group_length_displacement_mm_median']['value'] >= 0.5 * a['A2_group_tip_error_mm_median']['value']),
        abduction_twist=bool(pa['value'] > pf['value'] and pa['ci95'][0] > pf['ci95'][1]),
        object_unaware=bool(a['A4_group_gt_contact_normal_frac_median']['value'] >= 0.6 and a['A4_group_gt_contact_normal_mm_median']['value'] > 0))
    json.dump(a, open(f"{OUT}/a_floating.json", "w"), indent=1)
    # ---- B
    I = list(img_rows.values())
    bi = D.Boot(I)
    corr = lambda rr: float(np.corrcoef([r['s'] for r in rr], [r['d'] for r in rr])[0, 1]) if len(rr) > 3 else np.nan
    slope = lambda rr: float(np.polyfit([r['s'] for r in rr], [r['d'] for r in rr], 1)[0]) if len(rr) > 3 else np.nan
    red = lambda k0, k1: (lambda rr: 1 - np.median([r[k1] for r in rr]) / np.median([r[k0] for r in rr]) if rr else np.nan)
    b = dict(n_images=len(I), corr_s_d=bi.ci(I, corr), slope_d_on_s=bi.ci(I, slope),
             s_median=bi.ci(I, med('s')), d_median=bi.ci(I, med('d')),
             depth_err_b2_mm_median=bi.ci(I, med('depth_err_b2_mm')),
             depth_err_true_size_mm_median=bi.ci(I, med('depth_err_size_mm')),
             depth_err_reduction_true_size=bi.ci(I, red('depth_err_b2_mm', 'depth_err_size_mm')),
             tip_err_b2_mm_median=bi.ci(I, med('tip_err_b2_mm')), tip_err_true_size_mm_median=bi.ci(I, med('tip_err_size_mm')),
             tip_err_reduction_true_size=bi.ci(I, red('tip_err_b2_mm', 'tip_err_size_mm')),
             depth_err_hamba_focal_mm_median=bi.ci(I, med('depth_err_hamba_focal_mm')))
    c, sl, rd = b['corr_s_d'], b['slope_d_on_s'], b['depth_err_reduction_true_size']
    b['criterion_hand_size_main_cause'] = dict(corr=c['value'], slope_ci=sl['ci95'], depth_reduction=rd['value'],
                                               met=bool(c['value'] >= 0.7 and sl['ci95'][0] <= 1 <= sl['ci95'][1] and rd['value'] >= 0.5))
    json.dump(b, open(f"{OUT}/b_depth_cause.json", "w"), indent=1)
    print(json.dumps(dict(A=a['criteria'], B=b['criterion_hand_size_main_cause']), indent=1))


if __name__ == "__main__":
    main()
