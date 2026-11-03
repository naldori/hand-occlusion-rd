"""Figures for Diagnostics 2 (A shares, B s-vs-d, C depth error by variant)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = os.environ.get("HAND_PILOT_ROOT", os.path.expanduser("~/hand_pilot")) + "/results/diagnostics2"
SETS = [("stage_b_hocap", "HO-Cap (unseen by Hamba)"), ("stage_b", "DexYCB (seen in training)")]
SC = {"stage_b_hocap": "#2a78d6", "stage_b": "#eb6834"}           # categorical slots 1-2, fixed order
VC = {"B2": "#9a9990", "none": "#2a78d6", "proxy": "#eb6834", "oracle": "#1baf7a"}
INK, MUTED, GRID, SURF = "#21211f", "#6b6a63", "#e4e3dd", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})


def fig_a():
    items = [("annotation noise\nGT ÷ pred floating", lambda a: (a['A1_group_gt_floating_rate_baseline'], a['A1_group_pred_floating_rate'])),
             ("finger length\nlength shift ÷ tip error", lambda a: a['A2_group_length_share']),
             ("flexion\npartial R²", lambda a: a['A3_group_partial_R2_flexion']),
             ("abduction + twist\npartial R²", lambda a: a['A3_group_partial_R2_abduction_twist']),
             ("object-unaware\nnormal share of error", lambda a: a['A4_group_gt_contact_normal_frac_median'])]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(len(items)); w = 0.36
    for k, (s, name) in enumerate(SETS):
        a = json.load(open(f"{R}/{s}/a_floating.json"))
        vals, lo, hi = [], [], []
        for _, get in items:
            v = get(a)
            if isinstance(v, tuple):          # ratio of two rates; CI not computed for the ratio -> point only
                vals.append(v[0]['value'] / v[1]['value']); lo.append(np.nan); hi.append(np.nan)
            else:
                vals.append(v['value']); lo.append(v['ci95'][0]); hi.append(v['ci95'][1])
        vals, lo, hi = map(np.array, (vals, lo, hi))
        pos = x + (k - 0.5) * w
        ax.bar(pos, vals, width=w - 0.04, color=SC[s], label=name)
        m = ~np.isnan(lo)
        ax.errorbar(pos[m], vals[m], yerr=[vals[m] - lo[m], hi[m] - vals[m]], fmt="none", ecolor=INK, lw=1, capsize=3)
    for xi, thr in zip(x, [0.5, 0.5, None, None, 0.6]):
        if thr is not None:
            ax.plot([xi - 0.42, xi + 0.42], [thr, thr], color=INK, lw=1, ls="--")
    ax.set_xticks(x, [lab for lab, _ in items]); ax.set_ylim(0, 1.05); ax.set_ylabel("share (0–1)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("A  Candidate causes of 'floating' (object-occluded fingers with error > 15 mm, B1)\n"
                 "dashed = pre-set criterion; bars 3–4 are compared with each other; 95% CI where available",
                 fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_a_floating_causes.png", dpi=150)


def fig_b():
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8), sharex=True, sharey=True)
    for a, (s, name) in zip(ax, SETS):
        rows = json.load(open(f"{R}/{s}/b_rows.json")); b = json.load(open(f"{R}/{s}/b_depth_cause.json"))
        x = np.array([r['s'] for r in rows]); y = np.array([r['d'] for r in rows])
        a.scatter(x, y, s=8, color=SC[s], alpha=0.4, linewidths=0)
        g = np.linspace(-0.4, 0.4, 2)
        a.plot(g, g, color=MUTED, lw=1, ls="--")
        k, c0 = np.polyfit(x, y, 1)
        a.plot(g, k * g + c0, color=INK, lw=1.5)
        a.set_xlim(-0.4, 0.4); a.set_ylim(-0.4, 0.4)
        a.set_title(f"{name}\nr = {b['corr_s_d']['value']:.2f}, slope = {b['slope_d_on_s']['value']:.2f} "
                    f"{b['slope_d_on_s']['ci95']}", fontsize=8.5, color=INK)
        a.set_xlabel("hand-size error s = pred/GT − 1")
    ax[0].set_ylabel("relative wrist-depth error d")
    fig.suptitle("B  Does hand-size error explain depth error?  (dashed = d = s, solid = fit; B2, one dot per image)",
                 fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_b_size_vs_depth.png", dpi=150)


def fig_c():
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=False)
    for a, (s, name) in zip(ax, SETS):
        rows = [r for r in json.load(open(f"{R}/{s}/c_rows.json")) if r['finger'] == 'thumb']
        data = [[r['depth_b2'] for r in rows if r['variant'] == 'none']] + \
               [[r['depth_c'] for r in rows if r['variant'] == v] for v in ['none', 'proxy', 'oracle']]
        labs = ["B2 (before)", "C-none", "C-proxy", "C-oracle"]
        bp = a.boxplot(data, widths=0.5, whis=(5, 95), patch_artist=True, showfliers=False,
                       medianprops=dict(color=INK, lw=1.5))
        for patch, key in zip(bp["boxes"], ["B2", "none", "proxy", "oracle"]):
            patch.set_facecolor(VC[key]); patch.set_alpha(0.6); patch.set_edgecolor(VC[key])
        a.set_xticks([1, 2, 3, 4], labs)
        a.set_title(f"{name}  (n={len(data[0])} images)", fontsize=9, color=INK)
    ax[0].set_ylabel("wrist depth error (mm)")
    fig.suptitle("C  Translation-only correction using the object (GT object pose: upper-bound setting; whiskers 5–95%)",
                 fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(f"{R}/fig_c_depth_by_variant.png", dpi=150)


if __name__ == "__main__":
    fig_a(); fig_b(); fig_c(); print("saved")
