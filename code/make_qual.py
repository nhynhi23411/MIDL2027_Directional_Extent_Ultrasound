import pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metrics import largest_component
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42, "font.size": 7})
R = "../results/"; O = "../paper/figures/"
V = pickle.load(open(R + "viz_probs.pkl", "rb"))
fig, axs = plt.subplots(2, 3, figsize=(6.4, 4.5)); stats = []
for i, ds in enumerate(("busi", "mmotu")):
    recs = {r["id"]: r for r in pickle.load(open(f"../prep/{ds}.pkl", "rb"))}
    A = pd.DataFrame(V[(ds, "dice_bce")]["rows"]); B = pd.DataFrame(V[(ds, "directional")]["rows"])
    A = A[(A.split == "test") & (A.pos == 1)].set_index("case_id"); B = B[(B.split == "test") & (B.pos == 1)].set_index("case_id"); ix = A.index.intersection(B.index)
    diff = (B.loc[ix].rel_err - A.loc[ix].rel_err); med = diff.abs().sort_values().index[len(diff) // 4]
    picks = [("typical", med), ("largest benefit", diff.idxmin()), ("largest harm", diff.idxmax())]
    for j, (lab, cid) in enumerate(picks):
        r = recs[cid]; ax = axs[i, j]; ax.imshow(r["img"], cmap="gray", vmin=0, vmax=255); M = r["mask"] > 0
        for k, c, ls in (("dice_bce", "#FF4FA3", "--"), ("directional", "#F5A623", "-")):
            h = V[(ds, k)]["probs"][cid].astype(float) / 255 > .5
            if h.any(): ax.contour(largest_component(h).astype(float), [0.5], colors=c, linewidths=1.0, linestyles=ls)
        ax.contour(M.astype(float), [0.5], colors="#3EE07A", linewidths=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{'BUSI' if ds == 'busi' else 'MMOTU'}, {lab}", fontsize=7)
        ax.set_xlabel("NAE: Dice+BCE %.2f | directional %.2f" % (A.loc[cid, "rel_err"], B.loc[cid, "rel_err"]), fontsize=6)
        stats.append((ds, lab, cid, float(A.loc[cid, "rel_err"]), float(B.loc[cid, "rel_err"])))
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([0], [0], c="#3EE07A", lw=1.2, label="reference $M$"), Line2D([0], [0], c="#FF4FA3", lw=1.2, ls="--", label="Dice+BCE"), Line2D([0], [0], c="#F5A623", lw=1.2, label="directional (v1)")],
           loc="lower center", ncol=3, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.0); fig.savefig(O + "qualitative.pdf"); fig.savefig(O + "qualitative.png", dpi=250); print(stats)
