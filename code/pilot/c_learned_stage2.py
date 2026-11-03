"""C-learned stage 2 table: B2 vs C-none / C-proxy / C-learned / C-oracle (main) and C* (post-hoc), paired CIs.
none/oracle come from Diagnostics 2 (same B2 start, same weights); proxy/learned from the joint run in results/c_learned.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D  # noqa: E402

R = f"{D.RES}/c_learned"
SETS = ["stage_b_hocap", "stage_b"]


def load(s, cstar):
    d2 = f"{D.RES}/diagnostics2/posthoc_pen_tol_5mm" if cstar else f"{D.RES}/diagnostics2"
    cl = f"{R}/c_cstar" if cstar else f"{R}/c_main"
    rows = [r for r in json.load(open(f"{d2}/{s}/c_rows.json")) if r['variant'] in ('none', 'oracle')]
    rows += json.load(open(f"{cl}/{s}/c_rows.json"))
    by = {}
    for r in rows:      # one record per (image, finger) holding every variant
        k = (r['image'], r['finger'])
        b = by.setdefault(k, dict(seq=r['seq'], image=r['image'], finger=r['finger'], occlusion=r['occlusion'],
                                  depth_b2=r['depth_b2'], tip_b2=r['tip_b2']))
        b[f"depth_{r['variant']}"] = r['depth_c']; b[f"tip_{r['variant']}"] = r['tip_c']
    return list(by.values())


def med(key, occ=None, per_image=False):
    def f(rr):
        S = [r for r in rr if (occ is None or r['occlusion'] == occ) and (not per_image or r['finger'] == 'thumb')]
        return float(np.median([r[key] for r in S])) if S else np.nan
    return f


def red(v):
    return lambda rr: 1 - med(f'depth_{v}', per_image=True)(rr) / med('depth_b2', per_image=True)(rr)


def main():
    out = {}
    for cstar in [False, True]:
        tag = "C*_posthoc_pen_tol_5mm" if cstar else "main_preregistered"
        out[tag] = {}
        for s in SETS:
            rows = load(s, cstar)
            bt = D.Boot(rows)
            t = {}
            t['B2'] = dict(wrist_depth_mm=bt.ci(rows, med('depth_b2', per_image=True)),
                           visible_tip_mm=bt.ci(rows, med('tip_b2', 'visible')), occluded_tip_mm=bt.ci(rows, med('tip_b2', 'object')))
            for v in ['none', 'proxy', 'learned', 'oracle']:
                t[f'C-{v}'] = dict(wrist_depth_mm=bt.ci(rows, med(f'depth_{v}', per_image=True)),
                                   wrist_depth_reduction=bt.ci(rows, red(v)),
                                   wrist_depth_change_mm=bt.ci(rows, lambda rr, v=v: med(f'depth_{v}', per_image=True)(rr) - med('depth_b2', per_image=True)(rr)),
                                   visible_tip_mm=bt.ci(rows, med(f'tip_{v}', 'visible')),
                                   occluded_tip_mm=bt.ci(rows, med(f'tip_{v}', 'object')))
            t['learned_minus_proxy_reduction'] = bt.ci(rows, lambda rr: red('learned')(rr) - red('proxy')(rr))
            if s == 'stage_b_hocap' and not cstar:
                L = t['C-learned']
                t['criterion1_learned_depth_correction'] = dict(
                    reduction=L['wrist_depth_reduction'], change_ci=L['wrist_depth_change_mm']['ci95'],
                    met=bool(L['wrist_depth_reduction']['value'] >= 0.30 and L['wrist_depth_change_mm']['ci95'][1] < 0))
                d = t['learned_minus_proxy_reduction']
                t['criterion2_replaces_gt_occlusion_labels'] = dict(diff=d, met=bool(d['ci95'][0] > -0.10))
            out[tag][s] = t
            print(tag, s, {k: (v['wrist_depth_mm']['value'], v.get('wrist_depth_reduction', {}).get('value')) for k, v in t.items() if k.startswith(('B2', 'C-'))},
                  'learned-proxy', t['learned_minus_proxy_reduction'], flush=True)
    json.dump(out, open(f"{R}/depth_correction.json", "w"), indent=1)
    h = out['main_preregistered']['stage_b_hocap']
    print('criterion1', h['criterion1_learned_depth_correction']['met'], '| criterion2', h['criterion2_replaces_gt_occlusion_labels'])


if __name__ == "__main__":
    main()
