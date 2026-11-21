import pickle, numpy as np, pandas as pd, cv2, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from metrics import largest_component
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 6})
C = dict(blue="#2F6DB5", lblue="#DCE8F6", green="#3E8E5A", lgreen="#E4F2E9", orange="#D9822B", lor="#FBEBD8", red="#C0392B", gray="#555555", lgray="#F4F4F4", pur="#8E44AD")
R = "../results/"; O = "../paper/figures/"
V = pickle.load(open(R + "viz_probs.pkl", "rb")); recs = {r["id"]: r for r in pickle.load(open("../prep/busi.pkl", "rb"))}
Rb = pd.DataFrame(V[("busi", "dice_bce")]["rows"]); Rd = pd.DataFrame(V[("busi", "directional")]["rows"])
Rb = Rb[(Rb.split == "test") & (Rb.pos == 1)].set_index("case_id"); Rd = Rd[(Rd.split == "test") & (Rd.pos == 1)].set_index("case_id")
cand = Rb[(Rb.rel_err < 0.12) & (Rb.rel_err_all - Rb.rel_err > 0.15)].copy(); cand["gap"] = cand.rel_err_all - cand.rel_err
cid = cand.sort_values("gap", ascending=False).index[0]; print("case", cid, Rb.loc[cid, ["rel_err", "rel_err_all", "dice"]].tolist(), Rd.loc[cid, ["rel_err", "rel_err_all"]].tolist())
r = recs[cid]; img = r["img"]; M = r["mask"] > 0
def prob(k): return V[("busi", k)]["probs"][cid].astype(float) / 255
Pb, Pd = prob("dice_bce"), prob("directional"); Hb = Pb > .5; Hd = Pd > .5; Lb = largest_component(Hb) if Hb.any() else Hb
def fer(m):
    ys, xs = np.nonzero(m); pts = np.stack([xs, ys], 1).astype(float)
    h = pts[ConvexHull(pts).vertices] if len(pts) > 3 else pts; D = squareform(pdist(h)); i, j = np.unravel_index(D.argmax(), D.shape); return h[i], h[j], D.max()
fig = plt.figure(figsize=(7.2, 5.0)); bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 100); bg.set_ylim(0, 70); bg.axis("off")
def box(x, y, w, h, fc="white", ec=C["gray"], ls="--", lw=0.7, r=1.0): bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))
def arr(p, q, c=C["gray"], lw=0.8, ls="-"): bg.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, color=c, lw=lw, ls=ls))
def T(x, y, s, fs=6, ha="center", c="black", w="normal", st="normal"): bg.text(x, y, s, fontsize=fs, ha=ha, va="center", color=c, fontweight=w, fontstyle=st)
def im(x, y, w, h, a, cmap="gray"):
    ax = fig.add_axes([x / 100, y / 70, w / 100, h / 70]); ax.imshow(a, cmap=cmap, vmin=0, vmax=1 if a.dtype != np.uint8 else 255, interpolation="bilinear"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(C["gray"]); s.set_linewidth(0.6)
    return ax
def cont(ax, m, c, lw=0.9, ls="-"):
    if m.any(): ax.contour(m.astype(float), [0.5], colors=c, linewidths=lw, linestyles=ls)
# ---------- top panel
box(0.5, 37, 99, 32.5); T(2, 67.6, "(a) Training and exact measurement pipeline (BUSI example, U-Net, seed 0)", 6.8, ha="left", w="bold")
a1 = im(2, 47, 11.5, 11.5, img); cont(a1, M, C["green"]); T(7.75, 44.8, r"$\tilde{x}$ ($256^2$), mask $M$", 5.6); T(7.75, 42.6, "released label", 5.2, c=C["gray"], st="italic")
arr((14, 52.7), (17, 52.7))
bx = 17.5
for i, (h, col) in enumerate([(12, C["lblue"]), (9, C["lblue"]), (6.5, C["lblue"]), (4.5, "#B9D0EC"), (6.5, C["lor"]), (9, C["lor"]), (12, C["lor"])]):
    xx = bx + i * 2.6; bg.add_patch(Rectangle((xx, 52.7 - h / 2), 2.0, h, fc=col, ec=C["blue"] if i < 4 else C["orange"], lw=0.6))
bg.plot([bx + 1, bx + 15.6], [60.3, 60.3], color=C["gray"], lw=0.5, ls=":"); T(bx + 9, 62.2, "skip connections", 5, c=C["gray"], st="italic")
T(bx + 9, 45.0, r"$f_\theta$: U-Net / DeepLabV3+ (ResNet-34)", 5.4); T(bx + 9, 43.0, "3 seeds per setting", 5.2, c=C["gray"], st="italic")
arr((bx + 19, 52.7), (bx + 22, 52.7))
a2 = im(41.5, 47, 11.5, 11.5, Pd, cmap="viridis"); T(47.25, 44.8, r"probability map $P$", 5.6)
arr((53.5, 52.7), (56.2, 52.7)); T(61.2, 66.0, r"$\tau{=}0.5$, component rule", 5.4, c=C["gray"])
a3 = im(56.5, 53.0, 9, 9, img); cont(a3, Hb, C["pur"], 0.8); T(61, 51.0, r"$\mathcal{C}_{\mathrm{all}}$", 5.8)
a4 = im(56.5, 40.2, 9, 9, img); cont(a4, Lb, C["pur"], 0.8); T(61, 38.4, r"$\mathcal{C}_{\max}$ (largest)", 5.6)
arr((66, 52.0), (69.5, 52.7)); arr((66, 45), (69.5, 50.5))
p0, p1, dp = fer(Lb); g0, g1, dg = fer(M)
a5 = im(70, 47, 11.5, 11.5, img); cont(a5, M, C["green"], 0.7); cont(a5, Lb, C["pur"], 0.7)
a5.plot([g0[0], g1[0]], [g0[1], g1[1]], color=C["green"], lw=1.2); a5.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#FF4FA3", lw=1.2, ls="--")
T(75.75, 44.8, r"max Feret $D$ (hard mask)", 5.5); T(75.75, 42.8, "original grid; green: $M$, pink: $\\hat{M}$", 5.0, c=C["gray"], st="italic")
arr((82, 52.7), (84.2, 52.7))
box(84.5, 43, 14.5, 18, fc=C["lgray"], ec=C["gray"], ls="-", lw=0.8, r=1.4)
T(91.75, 58.3, r"$\mathrm{NAE}=\frac{|D_{\hat M}-D_M|}{D_M}$", 6.6); T(91.75, 53.2, "empty = failure", 5.4, c=C["gray"])
T(91.75, 49.3, "$\\mathcal{C}_{\\max}$ NAE %.2f" % Rb.loc[cid, "rel_err"], 5.8, c=C["pur"]); T(91.75, 46.6, "$\\mathcal{C}_{\\mathrm{all}}$ NAE %.2f" % Rb.loc[cid, "rel_err_all"], 5.8, c=C["red"])
# ---------- bottom left: loss profile
box(0.5, 1, 48, 34); T(2, 33.2, "(b) Directional extent loss (training only)", 6.8, ha="left", w="bold")
ax = fig.add_axes([0.085, 0.085, 0.19, 0.25]); K = 16; ang = np.arange(K) * np.pi / K
def wid(m):
    yy, xx = np.nonzero(m); return np.array([((xx * np.cos(a) + yy * np.sin(a)).max() - (xx * np.cos(a) + yy * np.sin(a)).min()) for a in ang])
wM, wB, wD = wid(M), wid(Hb), wid(Hd); deg = np.degrees(ang)
ax.plot(deg, wM, "-o", c=C["green"], ms=2.2, lw=1, label=r"$w_k(M)$ reference"); ax.plot(deg, wB, "-s", c=C["gray"], ms=2.2, lw=.9, label="Dice+BCE"); ax.plot(deg, wD, "-^", c=C["orange"], ms=2.2, lw=.9, label="+ directional")
ax.set_xlabel(r"direction $\phi_k$ (deg)", fontsize=5.8); ax.set_ylabel("width (px, 256 grid)", fontsize=5.8); ax.tick_params(labelsize=5.2); ax.legend(frameon=False, fontsize=5, loc="best", handlelength=1.4)
for s in ("top", "right"): ax.spines[s].set_visible(False)
X0 = 30.5
T(X0, 28.0, r"project $P$, $M$ on $K{=}16$ directions $u_k$", 5.2, ha="left", c="#7A4410"); T(X0, 24.6, r"soft supports $s_k^{\pm}$; width $w_k=s_k^{+}-s_k^{-}$", 5.2, ha="left")
T(X0, 19.2, r"$\mathcal{L}_{\mathrm{ext}}=\frac{1}{K}\sum_k\rho\!\left(\frac{w_k(P)-w_k(M)}{\max(w_k(M),0.05)}\right)$", 5.6, ha="left")
T(X0, 13.6, r"$\mathcal{L}=\mathcal{L}_{\mathrm{Dice+BCE}}+\lambda\,\mathcal{L}_{\mathrm{ext}}$", 6.0, ha="left")
T(X0, 9.6, r"v2: + soft-max over $k$", 5.0, ha="left", c=C["gray"]); T(X0, 7.2, r"v3: + negative-image penalty", 5.0, ha="left", c=C["gray"])
T(X0, 4.0, "hard-mask widths plotted", 4.8, ha="left", c=C["gray"], st="italic")
# ---------- bottom right
box(50.5, 1, 49, 34); T(52, 33.2, "(c) Frozen evaluation: paired contrasts and lesion-free behaviour", 6.8, ha="left", w="bold")
allp = pd.read_csv(R + "all_contrasts.csv"); sets = [("MMOTU", "U-Net"), ("MMOTU", "DeepLabV3+"), ("BUSI", "U-Net"), ("BUSI", "DeepLabV3+")]
vers = [("v1", "v1", C["orange"]), ("v2", "v2", C["red"]), ("v3", "v3", C["blue"])]
ax = fig.add_axes([0.665, 0.075, 0.17, 0.265]); y = 0; yt = []; yl = []
for ds, m in sets:
    ys_ = []
    for v, lab, c in vers:
        q = allp[(allp.dataset == ds) & (allp.model == m) & (allp.ver == v)]
        if not len(q): continue
        q = q.iloc[0]; ax.errorbar(q.delta, y, xerr=[[q.delta - q.lo], [q.hi - q.delta]], fmt="o", ms=2.5, c=c, capsize=1.2, lw=.8); ys_.append(y); y -= 1
    yt.append(np.mean(ys_)); yl.append(ds + " " + ("U-Net" if m == "U-Net" else "DLv3+")); y -= .7
ax.axvline(0, c="k", lw=.5); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5); ax.tick_params(axis="x", labelsize=5); ax.set_xlabel(r"$\Delta$ NAE vs frozen comparator", fontsize=5.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for i, (v, lab, c) in enumerate(vers): bg.plot([52 + i * 5.5], [29.5], "o", ms=2.5, c=c); T(53.2 + i * 5.5, 29.5, lab, 5.2, ha="left")
T(72, 29.5, "(<0 favours the extent loss)", 4.8, c=C["gray"], st="italic")
d = pd.concat([pd.read_csv(R + "v2final_per_case_results.csv"), pd.read_csv(R + "v3final_per_case_results.csv")]); d = d[(d.dataset == "busi") & (d.split == "test") & (d.pos == 0)]
ax = fig.add_axes([0.905, 0.075, 0.075, 0.265]); names = [("dice_bce", "D+B", C["gray"]), ("directional", "v1", C["orange"]), ("directional2", "v2", C["red"]), ("directional3", "v3", C["blue"])]
for j, m in enumerate(("unet", "dlv3p")):
    for i, (l, n, c) in enumerate(names):
        v = d[(d.model == m) & (d.loss == l)].groupby("case_id").pred_any.mean().mean() * 100; ax.bar(i + j * 5, v, color=c, width=.8)
ax.set_xticks([1.5, 6.5]); ax.set_xticklabels(["U-Net", "DLv3+"], fontsize=5); ax.tick_params(axis="y", labelsize=5); ax.set_ylabel("lesion-free images\nwith foreground (%)", fontsize=5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.savefig(O + "pipeline.pdf"); fig.savefig(O + "pipeline.png", dpi=250); print("saved")
