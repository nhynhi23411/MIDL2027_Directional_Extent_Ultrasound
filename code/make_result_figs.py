import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
R = "E:/MIDL_2/results"; O = "E:/MIDL_2/paper/figures"; import os; os.makedirs(O, exist_ok=True)
d = pd.read_csv(f"{R}/final_per_case_results.csv"); d = d[(d.pos == 1) & (d.split == "test")]
comp = json.load(open(f"{R}/frozen_comparators.json"))
a = d.groupby(["dataset", "model", "loss", "case_id"])[["rel_err", "dice", "signed"]].mean().reset_index()
col = dict(dice_bce="#555555", boundary="#2F6DB5", bbox="#3E8E5A", directional="#D9822B"); nm = dict(dice_bce="Dice+BCE", boundary="+Boundary", bbox="+BBox", directional="Directional")
# Fig: overlap vs measurement (U-Net, both datasets)
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
for k, ds in enumerate(("mmotu", "busi")):
    for loss in ("dice_bce", "directional"):
        x = a[(a.dataset == ds) & (a.model == "unet") & (a.loss == loss)]
        ax[k].scatter(100 * x.dice, x.rel_err.clip(upper=1.0), s=5, alpha=.45, c=col[loss], label=nm[loss], lw=0)
    ax[k].axhline(.2, c="k", lw=.6, ls=":"); ax[k].set_xlabel("Dice (%)"); ax[k].set_title(["MMOTU (benchmark replication)", "BUSI"][k], fontsize=8)
ax[0].set_ylabel("Relative diameter error"); ax[0].legend(frameon=False, fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig(f"{O}/overlap_vs_measurement.pdf"); fig.savefig(f"{O}/overlap_vs_measurement.png", dpi=250)
# Fig: paired diameter errors, directional vs frozen strongest comparator (both models)
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
for k, (ds, m) in enumerate([(ds, m) for ds in ("mmotu", "busi") for m in ("unet", "dlv3p")]):
    c = comp[f"{ds}|{m}"]; p = a[(a.dataset == ds) & (a.model == m)].pivot(index="case_id", columns="loss", values="rel_err")
    ax[k].scatter(p[c].clip(upper=1), p["directional"].clip(upper=1), s=4, alpha=.4, c=col["directional"], lw=0); ax[k].plot([0, 1], [0, 1], "k", lw=.6)
    imp = 100 * (p["directional"] < p[c]).mean(); ax[k].set_title(f"{ds.upper()} {'U-Net' if m=='unet' else 'DLv3+'}\n{imp:.0f}% below diagonal", fontsize=7)
    ax[k].set_xlabel(f"{nm[c]} error", fontsize=7)
    if k == 0: ax[k].set_ylabel("Directional error", fontsize=7)
fig.tight_layout(); fig.savefig(f"{O}/paired_errors.pdf"); fig.savefig(f"{O}/paired_errors.png", dpi=250)
# Fig: error-size relationship (signed bias vs ref diameter)
print("figs ok")
