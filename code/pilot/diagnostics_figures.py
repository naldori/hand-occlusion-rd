"""Figures for Diagnostics D1-D3 (reads results/diagnostics/<set>/rows.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = os.environ.get("HAND_PILOT_ROOT", os.path.expanduser("~/hand_pilot")) + "/results/diagnostics"
SETS = [("stage_b_hocap", "HO-Cap (unseen by Hamba)"), ("stage_b", "DexYCB (seen in training)")]
OCC = [("visible", "visible"), ("object", "object-occluded"), ("self", "hand-occluded")]
C = {"visible": "#2a78d6", "object": "#eb6834", "self": "#1baf7a"}     # categorical slots 1-3, fixed order
INK, MUTED, GRID, SURF = "#21211f", "#6b6a63", "#e4e3dd", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
rows = {s: json.load(open(f"{R}/{s}/rows.json")) for s, _ in SETS}


def d1():
    fig, ax = plt.subplots(2, 2, figsize=(8, 7), sharex="col", sharey=True)
    for i, (s, name) in enumerate(SETS):
        for j, al in enumerate(["b1", "b2"]):
            S = [r for r in rows[s] if r["align"] == al and r["occlusion"] == "object" and r["large"]]
            x = np.array([r["los_signed_mm"] for r in S]); y = np.array([r["plane_mm"] for r in S])
            a = ax[i, j]
            a.scatter(x, y, s=8, color=C["object"], alpha=0.35, linewidths=0)
            lim = 80 if al == "b1" else 200
            a.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--"); a.plot([0, -lim], [0, lim], color=MUTED, lw=1, ls="--")
            a.set_xlim(-lim, lim); a.set_ylim(0, 90)
            frac = np.nanmedian([r["los_frac"] for r in S])
            a.set_title(f"{name.split(' (')[0]} · {al.upper()} · n={len(S)} · line-of-sight share {frac:.2f}", fontsize=8.5, color=INK)
            if j == 0: a.set_ylabel(f"{name.split(' (')[0]}\nimage-plane component (mm)")
    fig.suptitle("D1  Tip error direction, object-occluded fingers with error > 15 mm\n"
                 "dashed = equal line-of-sight and image-plane parts; B1 = palm aligned to GT, B2 = camera re-alignment",
                 fontsize=10, color=INK)
    fig.supxlabel("line-of-sight component of tip error (mm, + = farther from camera)", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_d1_error_direction.png", dpi=150)


def d2():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for a, (s, name) in zip(ax, SETS):
        data = []
        for occ, _ in OCC:
            v = np.array([r["shrink"] for r in rows[s] if r["align"] == "b1" and r["occlusion"] == occ])
            data.append(v[~np.isnan(v)])
        bp = a.boxplot(data, widths=0.5, whis=(5, 95), patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
        for patch, (occ, _) in zip(bp["boxes"], OCC):
            patch.set_facecolor(C[occ]); patch.set_alpha(0.55); patch.set_edgecolor(C[occ])
        a.axhline(1.0, color=MUTED, lw=1); a.axhline(0.8, color=MUTED, lw=1, ls="--")
        a.set_xticks([1, 2, 3], [f"{lab}\n(n={len(d)})" for (_, lab), d in zip(OCC, data)])
        a.set_yscale("log"); a.set_title(name, fontsize=9, color=INK)
    ax[0].set_ylabel("shrinkage ratio |θpred−θmean| / |θgt−θmean|  (log)")
    fig.suptitle("D2  Regression toward the mean pose?  (<1 = toward mean; dashed 0.8 = criterion; whiskers 5–95%)",
                 fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_d2_shrinkage.png", dpi=150)


def d3():
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6), sharey=False)
    bins = np.arange(-30, 121, 5)
    for a, (s, name) in zip(ax, SETS):
        S = [r for r in rows[s] if r["align"] == "b1" and r["occlusion"] == "object" and r["large"]]
        for key, col, lab in [("sd_gt_mm", "#2a78d6", "ground truth"), ("sd_pred_mm", "#eb6834", "Hamba (B1)")]:
            v = np.clip([r[key] for r in S], bins[0], bins[-1] - 0.01)
            a.hist(v, bins=bins, color=col, alpha=0.55, label=lab, edgecolor=SURF, linewidth=1)
        a.axvspan(-5, 5, color=GRID, alpha=0.8, zorder=0)
        a.set_title(f"{name}  (n={len(S)})", fontsize=9, color=INK)
    ax[0].set_ylabel("fingertips"); ax[1].legend(frameon=False)
    fig.suptitle("D3  Where object-occluded large-error fingertips end up (B1; grey band = ±5 mm)", fontsize=10, color=INK)
    fig.supxlabel("signed fingertip-to-object distance (mm; − inside, + outside; clipped at 120)", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_d3_signed_distance.png", dpi=150)


if __name__ == "__main__":
    d1(); d2(); d3()
    print("saved")
