"""Side-by-side Hamba vs WiLoR table (existing protocol and criteria) + hand-size figure.
Writes results/wilor_compare/compare.json, compare.md, fig_hand_size.png. Hamba files are only read.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_a as A  # noqa: E402

R = f"{A.ROOT}/results"
W = f"{R}/wilor_compare"
SETS = [("stage_b_hocap", "HO-Cap subject_5"), ("stage_b", "DexYCB subject-01")]
TIPS = [4, 8, 12, 16, 20]


def paths(model, s):
    if model == 'Hamba':
        return dict(pred=f"{R}/diagnostics/repro_{s}/pred.npz", auroc=f"{R}/{s}/auroc_summary.json",
                    d1=f"{R}/diagnostics/{s}/d1_error_direction.json", c=f"{R}/diagnostics2/{s}/c_object_ruler.json")
    return dict(pred=f"{W}/{s}/pred.npz", auroc=f"{W}/{s}/auroc_summary.json",
                d1=f"{W}/{s}/diag/d1_error_direction.json", c=f"{W}/{s}/c/{s}/c_object_ruler.json")


def gt_size(s):
    f = f"{A.ROOT}/data_hocap/calibration/mano/subject_5.yaml" if s == 'stage_b_hocap' else \
        f"{A.DATA}/calibration/mano_20200709_140042_subject-01_right/mano.yml"
    b = yaml.safe_load(open(f))['betas']
    _, j = A.mano_layer()(torch.zeros(1, 48), torch.tensor(b, dtype=torch.float32)[None], torch.zeros(1, 3))
    j = j[0].numpy() / 1000
    return float(np.linalg.norm(j[9] - j[0]) * 1000)


def gt_joints(s, images):
    labels = json.load(open(f"{R}/{s}/labels.json"))
    first = {}
    for r in labels:
        first.setdefault(r['image'], r)
    out = []
    for img in images:
        r = first[img]
        out.append(np.array(r['gt_j21']) if 'gt_j21' in r else
                   np.load(img.replace('color_', 'labels_').replace('.jpg', '.npz'))['joint_3d'][0].astype(np.float64))
    return np.array(out)


def ci(x):
    return {"value": x.get('value', x.get('auroc')), "ci95": x['ci95']}


def main():
    table, sizes = {}, {}
    for s, sname in SETS:
        gs = gt_size(s)
        for model in ['Hamba', 'WiLoR']:
            p = paths(model, s)
            pred = np.load(p['pred'])
            G = gt_joints(s, pred['image'])
            Jh = pred['j_b2'] - pred['t_b2'][:, None]
            size = np.linalg.norm(Jh[:, 9] - Jh[:, 0], axis=1) * 1000
            sizes[(model, s)] = size
            au = json.load(open(p['auroc'])); d1 = json.load(open(p['d1'])); c = json.load(open(p['c']))
            tip = lambda al: float(np.median(np.linalg.norm(pred[f'j_{al}'][:, TIPS] - G[:, TIPS], axis=2)) * 1000)
            row = {
                "PA-MPJPE median (mm)": float(np.median(pred['pa_mpjpe_mm'])),
                "tip error median B1 (mm)": tip('b1'), "tip error median B2 (mm)": tip('b2'),
                "pred hand size median (mm)": float(np.median(size)),
                "pred hand size p5-p95 (mm)": [float(np.percentile(size, 5)), float(np.percentile(size, 95))],
                "GT hand size (mm)": gs, "pred vs GT size (%)": float((np.median(size) / gs - 1) * 100),
                "B2 wrist depth error median (mm)": float(np.median(np.abs(pred['j_b2'][:, 0, 2] - G[:, 0, 2])) * 1000),
                "B2 wrist-translation line-of-sight share (object-occl. large)": d1['b2|object|large']['wrist_los_frac_median']['value'],
                "B2 wrist translation explains tip error (object-occl. large)": d1['b2|object|large']['wrist_translation_explained']['value'],
            }
            for al in ['b1', 'b2']:
                e = au[f'{al}|3mm|object|all']['large']
                row[f"AUROC S_geo {al.upper()} object-occluded"] = ci(e['S_geo']) if 'S_geo' in e else e.get('verdict')
                row[f"AUROC S_geo-oracle {al.upper()} object-occluded"] = ci(e['S_geo_oracle_3mm']) if 'S_geo_oracle_3mm' in e else e.get('verdict')
                row[f"n large errors {al.upper()} object-occluded"] = e['n_pos']
            pr = c['proxy']
            row["C-proxy wrist depth (mm) before -> after"] = [pr['wrist_depth_b2_mm']['value'], pr['wrist_depth_c_mm']['value']]
            row["C-proxy wrist depth reduction"] = ci(pr['wrist_depth_reduction'])
            row["C-proxy wrist depth change CI (mm)"] = pr['wrist_depth_change_mm']['ci95']
            h1 = [row[f"AUROC S_geo {a} object-occluded"] for a in ['B1', 'B2']]
            row["H1 met (S_geo CI lower > 0.6, B1 or B2)"] = bool(any(isinstance(x, dict) and x['ci95'][0] > 0.6 for x in h1))
            if s == 'stage_b_hocap':
                row["object tells depth met (C-proxy >=30% and CI excludes 0)"] = bool(
                    pr['wrist_depth_reduction']['value'] >= 0.30 and pr['wrist_depth_change_mm']['ci95'][1] < 0)
            table[f"{model} · {sname}"] = row
    json.dump(table, open(f"{W}/compare.json", "w"), indent=1)
    # markdown table
    keys = list(next(iter(table.values())).keys()) + ["object tells depth met (C-proxy >=30% and CI excludes 0)"]
    keys = list(dict.fromkeys(keys))
    cols = list(table)
    fmt = lambda v, k="": ("" if v is None else f"{v['value']:.3f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" if isinstance(v, dict)
                     else (f"{v:.2f}" if abs(v) < 2 else f"{v:.1f}") if isinstance(v, float) else
                     (" → " if "before" in k else " – ").join(f"{x:.1f}" for x in v) if isinstance(v, list) else str(v))
    lines = ["| item | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(fmt(table[c].get(k), k) for c in cols) + " |")
    open(f"{W}/compare.md", "w").write("\n".join(lines) + "\n")
    # figure: predicted hand-size distributions
    plt.rcParams.update({"font.size": 9, "axes.facecolor": "#fcfcfb", "figure.facecolor": "#fcfcfb",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3dd"})
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6), sharex=True)
    bins = np.arange(84, 100.5, 0.5)
    for a, (s, sname) in zip(ax, SETS):
        for model, col in [('Hamba', '#2a78d6'), ('WiLoR', '#eb6834')]:
            a.hist(np.clip(sizes[(model, s)], bins[0], bins[-1] - 0.01), bins=bins, color=col, alpha=0.6, label=f"{model} prediction",
                   edgecolor="#fcfcfb", linewidth=0.8)
        a.axvline(gt_size(s), color="#21211f", lw=1.6, label="this subject (GT)")
        a.axvline(94.7, color="#6b6a63", lw=1, ls="--", label="MANO mean shape (β=0)")
        a.set_title(sname, fontsize=9); a.set_xlabel("hand size: wrist – middle MCP (mm)")
    ax[0].set_ylabel("images"); ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Predicted hand size per image vs the subject's true size (DexYCB 10-subject mean 96.8, HO-Cap 9-subject mean 93.3 mm)",
                 fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{W}/fig_hand_size.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
