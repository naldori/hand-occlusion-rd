"""C-learned stage 1: accuracy of HACO finger contact vs the proxy rule 'object-occluded = contact'.
Writes results/c_learned/contact_accuracy.json (+ finger_contact.json per set for later stages).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D  # noqa: E402
from stage_b_score import auroc  # noqa: E402

R = f"{D.RES}/c_learned"
SETS = [("stage_b_hocap", "HO-Cap (not in HACO/Hamba training)"), ("stage_b", "DexYCB (seen by HACO and Hamba)")]
THRS = [0.5, 0.52]


def finger_rows(s):
    pad = json.load(open(f"{R}/pad_vertices.json"))['pad_vertices']
    h = np.load(f"{R}/{s}/haco_contact.npz")
    idx = {p: i for i, p in enumerate(h['image'])}
    rows = []
    for r in json.load(open(f"{D.RES}/{s}/labels.json")):
        p = float(h['prob'][idx[r['image']]][pad[r['finger']]].max())
        rows.append(dict(seq=r['seq'], image=r['image'], finger=r['finger'], occlusion=r['occlusion'],
                         gt=bool(r['gt_contact_3mm']), p_learned=p, proxy=float(r['occlusion'] == 'object')))
    json.dump(rows, open(f"{R}/{s}/finger_contact.json", "w"))
    return rows


def metrics(rr, key, thr):
    y = np.array([r['gt'] for r in rr]); s = np.array([r[key] for r in rr]); yh = s > thr
    tp, fp, fn = (yh & y).sum(), (yh & ~y).sum(), (~yh & y).sum()
    return dict(auroc=auroc(s, y), accuracy=float((yh == y).mean()),
                recall=float(tp / max(tp + fn, 1)), precision=float(tp / max(tp + fp, 1)), pred_pos_rate=float(yh.mean()))


def main():
    out = {}
    for s, name in SETS:
        rows = finger_rows(s)
        bt = D.Boot(rows)
        res = dict(name=name, n_fingers=len(rows), gt_contact_rate=float(np.mean([r['gt'] for r in rows])))
        for occ in ['all', 'visible', 'object', 'self']:
            S = rows if occ == 'all' else [r for r in rows if r['occlusion'] == occ]
            e = dict(n=len(S), gt_contact_rate=float(np.mean([r['gt'] for r in S])))
            for key, thr in [('p_learned', 0.5), ('proxy', 0.5)]:
                m = metrics(S, key, thr)
                e[key] = {k: dict(value=round(v, 4), ci95=bt.ci(S, lambda rr, k=k, key=key, thr=thr: metrics(rr, key, thr)[k])['ci95'])
                          for k, v in m.items()}
            e['p_learned_thr0.52'] = {k: round(v, 4) for k, v in metrics(S, 'p_learned', 0.52).items()}
            for k in ['auroc', 'accuracy', 'recall', 'precision']:
                e[f'diff_learned_minus_proxy_{k}'] = bt.ci(S, lambda rr, k=k: metrics(rr, 'p_learned', 0.5)[k] - metrics(rr, 'proxy', 0.5)[k])
            res[occ] = e
            print(s, occ, 'n', len(S), 'gt rate %.2f' % e['gt_contact_rate'],
                  '| learned AUROC', e['p_learned']['auroc'], 'acc', e['p_learned']['accuracy']['value'],
                  'rec', e['p_learned']['recall']['value'], 'prec', e['p_learned']['precision']['value'],
                  '| proxy AUROC', e['proxy']['auroc']['value'], 'acc', e['proxy']['accuracy']['value'],
                  'rec', e['proxy']['recall']['value'], 'prec', e['proxy']['precision']['value'],
                  '| dAUROC', e['diff_learned_minus_proxy_auroc']['value'], e['diff_learned_minus_proxy_auroc']['ci95'], flush=True)
        out[s] = res
    json.dump(out, open(f"{R}/contact_accuracy.json", "w"), indent=1)


if __name__ == "__main__":
    main()
