"""Stage B scoring (PROTOCOL v3): geometric check only.

Per (image, finger, alignment B1/B2):
  err_mm      fingertip 3D error after alignment
  S_geo       deployable: max(penetration depth of finger vertices,
                              hover = min distance of distal-segment vertices if 0 < d <= 10 mm else 0)
  S_geo_oracle  upper bound: if GT contact finger, max(penetration, min finger-to-surface distance), else penetration
AUROC for "large error" (> 15 mm; secondary: top 25 %) per stratum, sequence-level cluster bootstrap 95% CI.
Strata with < 50 positives are reported as undecidable.
"""
import json
import os
import sys

import numpy as np
import trimesh
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_a as A  # noqa: E402
from stage_b_infer_utils import to_local  # noqa: E402

OUT = f"{A.ROOT}/results/" + os.environ.get("PILOT_SET", "stage_b")
LARGE_MM = 15.0
HOVER_M = 0.010
MIN_POS = 50
N_BOOT = 2000
FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
TIP = {f: 4 * i + 4 for i, f in enumerate(FINGERS)}
DISTAL_PART = {'index': 3, 'middle': 6, 'pinky': 9, 'ring': 12, 'thumb': 15}


def auroc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind='mergesort'), kind='mergesort') + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def boot_ci(df, score_key, label_key):
    rng = np.random.default_rng(A.SEED)
    seqs = sorted({r['seq'] for r in df})
    by = {s: [r for r in df if r['seq'] == s] for s in seqs}
    vals = []
    for _ in range(N_BOOT):
        rr = [r for s in rng.choice(seqs, len(seqs)) for r in by[s]]
        vals.append(auroc(np.array([r[score_key] for r in rr]), np.array([r[label_key] for r in rr])))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)] if len(vals) else [None, None]


def main():
    layer = A.mano_layer()
    part_of_v = layer.th_weights.numpy().argmax(1)
    finger_of_v = np.array([A.FINGER_OF_PART[p] or 'palm' for p in part_of_v])
    labels = json.load(open(f"{OUT}/labels.json"))
    pred = np.load(f"{OUT}/pred.npz")
    idx = {p: i for i, p in enumerate(pred['image'])}
    meshes, posed, rows = {}, {}, []
    for r in labels:
        img = r['image']
        if img not in posed and 'obj_mesh' in r:
            if r['obj_mesh'] not in meshes:
                meshes[r['obj_mesh']] = trimesh.load(r['obj_mesh'], force='mesh')
            m, pose = meshes[r['obj_mesh']], np.array(r['obj_pose'])
        elif img not in posed:
            seq = f"{A.DATA}/{A.SUBJECT}/{r['seq']}"
            meta = yaml.load(open(f"{seq}/meta.yml"), Loader=yaml.FullLoader)
            oid = meta['ycb_ids'][meta['ycb_grasp_ind']]
            if oid not in meshes:
                meshes[oid] = A.load_object(oid)
            lab = np.load(img.replace('color_', 'labels_').replace('.jpg', '.npz'))
            m, pose = meshes[oid], np.vstack([lab['pose_y'][meta['ycb_grasp_ind']], [0, 0, 0, 1]])
        if img not in posed:
            geo = {}
            for al in ['b1', 'b2']:
                v = pred[f'v_{al}'][idx[img]].astype(np.float64)
                vl = to_local(v, pose)            # object frame: mesh is never copied
                _, dist, _ = trimesh.proximity.closest_point(m, vl)
                inside = m.contains(vl)
                geo[al] = (v, dist, inside)
            posed[img] = geo
        f = r['finger']
        fv = finger_of_v == f
        dv = part_of_v == DISTAL_PART[f]
        for al in ['b1', 'b2']:
            v, dist, inside = posed[img][al]
            j = pred[f'j_{al}'][idx[img]]
            err = float(np.linalg.norm(j[TIP[f]] - np.array(r['gt_tip'])) * 1000)
            pen = float(np.where(inside & fv, dist, 0).max())
            dmin = float(np.where(inside[dv], 0.0, dist[dv]).min())
            hover = dmin if 0 < dmin <= HOVER_M else 0.0
            dmin_f = float(np.where(inside[fv], 0.0, dist[fv]).min())
            row = dict(seq=r['seq'], image=img, finger=f, occlusion=r['occlusion'], align=al, err_mm=err,
                       S_geo=max(pen, hover), pen_mm=pen * 1000, hover_mm=hover * 1000)
            for t in [2, 3, 5]:
                c = r[f'gt_contact_{t}mm']
                row[f'gt_contact_{t}mm'] = c
                row[f'S_geo_oracle_{t}mm'] = max(pen, dmin_f) if c else pen
            rows.append(row)
    json.dump(rows, open(f"{OUT}/finger_scores.json", "w"))
    report(rows)


def report(rows):
    res = {}
    for al in ['b1', 'b2']:
        R = [r for r in rows if r['align'] == al]
        q75 = np.percentile([r['err_mm'] for r in R], 75)
        for r in R:
            r['large'] = r['err_mm'] > LARGE_MM
            r['top25'] = r['err_mm'] > q75
        res[f'{al}_err_mm_median'] = round(float(np.median([r['err_mm'] for r in R])), 2)
        res[f'{al}_top25_threshold_mm'] = round(float(q75), 2)
        for t in [2, 3, 5]:
            for occ in ['visible', 'object', 'self']:
                for contact in [True, False, None]:
                    S = [r for r in R if r['occlusion'] == occ and (contact is None or r[f'gt_contact_{t}mm'] == contact)]
                    if t != 3 and contact is None:
                        continue          # pooled-contact strata only once
                    key = f"{al}|{t}mm|{occ}|{'contact' if contact else 'noncontact' if contact is False else 'all'}"
                    entry = dict(n=len(S))
                    for lab in ['large', 'top25']:
                        npos = int(sum(r[lab] for r in S))
                        e = dict(n_pos=npos)
                        if npos < MIN_POS or npos == len(S):
                            e['verdict'] = 'undecidable (<50 positives)' if npos < MIN_POS else 'undecidable'
                        else:
                            for sk in ['S_geo', f'S_geo_oracle_{t}mm']:
                                sc = np.array([r[sk] for r in S]); lb = np.array([r[lab] for r in S])
                                e[sk] = dict(auroc=round(float(auroc(sc, lb)), 3), ci95=boot_ci(S, sk, lab))
                        entry[lab] = e
                    res[key] = entry
    json.dump(res, open(f"{OUT}/auroc_summary.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict) and k.split('|')[1] == '3mm':
            l = v['large']
            s = l.get('S_geo', {}); o = l.get('S_geo_oracle_3mm', {})
            print(f"{k:32s} n={v['n']:4d} pos15={l['n_pos']:4d} S_geo={s.get('auroc')} {s.get('ci95')} "
                  f"oracle={o.get('auroc')} {o.get('ci95')} {l.get('verdict', '')}")
        elif not isinstance(v, dict):
            print(k, v)


if __name__ == "__main__":
    main()
