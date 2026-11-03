"""C-learned figures: (1) per-stratum finger-contact AUROC, HACO vs proxy rule; (2) wrist-depth error by variant."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as D  # noqa: E402
from c_learned_stage2 import load  # noqa: E402

R = f"{D.RES}/c_learned"
SETS = [("stage_b_hocap", "HO-Cap (unseen by HACO and Hamba)"), ("stage_b", "DexYCB (seen by HACO and Hamba)")]
INK, MUTED, GRID, SURF = "#21211f", "#6b6a63", "#e4e3dd", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig_auroc():
    acc = json.load(open(f"{R}/contact_accuracy.json"))
    strata = [("all", "all"), ("visible", "visible"), ("object", "object-\nocclud."), ("self", "hand-\nocclud.")]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=True)
    for a, (s, name) in zip(ax, SETS):
        x = np.arange(len(strata)); w = 0.36
        for k, (key, lab, col) in enumerate([("p_learned", "HACO (learned)", "#2a78d6"), ("proxy", "proxy rule: object-occluded = contact", "#eb6834")]):
            v = np.array([acc[s][st][key]['auroc']['value'] for st, _ in strata])
            lo = np.array([acc[s][st][key]['auroc']['ci95'][0] for st, _ in strata])
            hi = np.array([acc[s][st][key]['auroc']['ci95'][1] for st, _ in strata])
            pos = x + (k - 0.5) * w
            a.bar(pos, v, width=w - 0.04, color=col, label=lab)
            a.errorbar(pos, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK, lw=1, capsize=3)
        a.axhline(0.5, color=MUTED, lw=1, ls="--")
        a.set_xticks(x, [f"{lab}\n(n={acc[s][st]['n']})" for st, lab in strata]); a.set_ylim(0, 1.0)
        a.set_title(name, fontsize=9, color=INK)
    ax[0].set_ylabel("finger-contact AUROC vs GT contact (≤3 mm)"); ax[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("C-learned stage 1  Finger contact prediction (pad-vertex max, 95% CI; proxy is binary so its AUROC is balanced accuracy)",
                 fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_contact_auroc.png", dpi=150)


def fig_depth():
    order = [("b2", "B2 (before)", "#9a9990"), ("none", "C-none", "#2a78d6"), ("proxy", "C-proxy\n(GT occlusion)", "#eb6834"),
             ("learned", "C-learned\n(HACO)", "#1baf7a"), ("oracle", "C-oracle\n(GT contact)", "#eda100")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.9))
    for a, (s, name) in zip(ax, SETS):
        rows = [r for r in load(s, False) if r['finger'] == 'thumb']
        data = [[r[f'depth_{k}'] for r in rows] for k, _, _ in order]
        bp = a.boxplot(data, widths=0.5, whis=(5, 95), patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
        for patch, (_, _, col) in zip(bp["boxes"], order):
            patch.set_facecolor(col); patch.set_alpha(0.6); patch.set_edgecolor(col)
        a.set_xticks(range(1, len(order) + 1), [lab for _, lab, _ in order], fontsize=8)
        a.set_title(f"{name}  (n={len(rows)} images)", fontsize=9, color=INK)
    ax[0].set_ylabel("wrist depth error (mm)")
    fig.suptitle("C-learned stage 2  Wrist depth error by contact source (pre-registered cost, w_c=10, w_p=1; GT object; whiskers 5–95%)",
                 fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_depth_by_variant.png", dpi=150)


if __name__ == "__main__":
    fig_auroc(); fig_depth(); print("saved")
