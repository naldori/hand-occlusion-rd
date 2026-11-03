"""C-learned stage 3: error detection with learned (HACO) contact instead of GT contact.

S_geo-learned = max(penetration, finger-to-object min distance) if HACO says the finger is in contact, else penetration
(same as S_geo-oracle with GT contact replaced by HACO finger contact > 0.5).
Evaluated on B1, B2 and B2 after C-learned correction. Writes results/c_learned/detection.json.
"""
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D  # noqa: E402
import stage_a as A  # noqa: E402
from stage_b_infer_utils import to_local  # noqa: E402
from stage_b_score import auroc  # noqa: E402

R = f"{D.RES}/c_learned"
LARGE_MM = 15.0
SETS = ["stage_b_hocap", "stage_b"]
TIP = {f: 4 * i + 4 for i, f in enumerate(D.FINGERS)}


MODEL = os.environ.get("HAND_MODEL", "hamba")            # hamba | wilor
C_MAIN = f"{R}/c_main" if MODEL == "hamba" else f"{R}/wilor/c_main"
OUT_JSON = f"{R}/detection.json" if MODEL == "hamba" else f"{R}/wilor/detection.json"


def sources(s):
    base = np.load(f"{D.RES}/diagnostics/repro_{s}/pred.npz" if MODEL == "hamba" else f"{D.RES}/wilor_compare/{s}/pred.npz")
    corr = np.load(f"{C_MAIN}/rescore_{s}_learned/pred.npz")
    return {'B1': (base, 'b1'), 'B2': (base, 'b2'), 'C-learned': (corr, 'b2')}


def main():
    layer = A.mano_layer()
    part_of_v = layer.th_weights.numpy().argmax(1)
    finger_of_v = np.array([A.FINGER_OF_PART[p] or 'palm' for p in part_of_v])
    out = {}
    for s in SETS:
        labels = json.load(open(f"{D.RES}/{s}/labels.json"))
        info = D.gt_objects_and_betas(labels)
        fc = {(r['image'], r['finger']): r for r in json.load(open(f"{R}/{s}/finger_contact.json"))}
        rows = []
        for src, (pred, al) in sources(s).items():
            idx = {p: i for i, p in enumerate(pred['image'])}
            geo = {}
            for r in labels:
                img, f = r['image'], r['finger']
                mesh, pose, _, gt = info[img]
                if img not in geo:
                    v = pred[f'v_{al}'][idx[img]].astype(np.float64)
                    vl = to_local(v, pose)
                    _, dist, _ = trimesh.proximity.closest_point(mesh, vl)
                    geo[img] = (dist, mesh.contains(vl))
                dist, inside = geo[img]
                fv = finger_of_v == f
                pen = float(np.where(inside & fv, dist, 0).max()) * 1000
                dmin = float(np.where(inside[fv], 0.0, dist[fv]).min()) * 1000
                c = fc[(img, f)]
                learned = c['p_learned'] > 0.5
                err = float(np.linalg.norm(pred[f'j_{al}'][idx[img]][TIP[f]] - gt[TIP[f]])) * 1000
                rows.append(dict(seq=r['seq'], image=img, finger=f, occlusion=r['occlusion'], source=src,
                                 err_mm=err, large=err > LARGE_MM, learned=learned, gt_contact=c['gt'],
                                 S_geo_learned=max(pen, dmin) if learned else pen,
                                 S_geo_oracle=max(pen, dmin) if c['gt'] else pen))
        bt = D.Boot(rows)
        res = {}
        for src in ['B1', 'B2', 'C-learned']:
            for occ in ['object', 'visible', 'self']:
                S = [r for r in rows if r['source'] == src and r['occlusion'] == occ]
                npos = int(sum(r['large'] for r in S))
                e = dict(n=len(S), n_large=npos, tip_err_median_mm=round(float(np.median([r['err_mm'] for r in S])), 2))
                if npos >= 50 and npos < len(S):
                    for k in ['S_geo_learned', 'S_geo_oracle']:
                        e[k] = dict(auroc=round(auroc(np.array([r[k] for r in S]), np.array([r['large'] for r in S])), 4),
                                    ci95=bt.ci(S, lambda rr, k=k: auroc(np.array([r[k] for r in rr]), np.array([r['large'] for r in rr])))['ci95'])
                else:
                    e['verdict'] = 'undecidable (<50 positives)'
                res[f"{src}|{occ}"] = e
                print(s, src, occ, e, flush=True)
        obj = [res[f"{src}|object"].get('S_geo_learned') for src in ['B1', 'B2', 'C-learned']]
        res['criterion3_learned_contact_rescues_detection'] = dict(
            met=bool(any(x and x['ci95'][0] is not None and x['ci95'][0] > 0.6 for x in obj)),
            note='post-hoc re-test of H1 with a score that uses no GT information')
        out[s] = res
    json.dump(out, open(OUT_JSON, "w"), indent=1)


if __name__ == "__main__":
    main()
