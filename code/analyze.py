import pandas as pd, numpy as np, json, sys, os
R = "E:/MIDL_2/results"; T = "E:/MIDL_2/paper/tables"; os.makedirs(T, exist_ok=True)
rng = np.random.default_rng(0)
d = pd.read_csv(f"{R}/final_per_case_results.csv")
LOSSES = ["dice_bce", "boundary", "bbox", "directional"]; NAME = dict(dice_bce="Dice+BCE", boundary="+ Boundary", bbox="+ BBox extent", directional="+ Directional (ours)")
DS = dict(mmotu="MMOTU", busi="BUSI"); MD = dict(unet="U-Net", dlv3p="DeepLabV3+")
pos = d[d.pos == 1]
# seed-averaged per-case table
agg = pos.groupby(["dataset", "model", "loss", "split", "case_id", "group"]).agg(
    rel_err=("rel_err", "mean"), rel_err_all=("rel_err_all", "mean"), signed=("signed", "mean"), dice=("dice", "mean"), bf1=("bf1", "mean"),
    ge20=("rel_err", lambda x: (x >= .2).mean()), missed=("missed", "sum")).reset_index()
# strongest comparator frozen on dev (mean seed-averaged rel_err), per dataset x model
dev = agg[agg.split == "dev"].groupby(["dataset", "model", "loss"]).rel_err.mean().reset_index()
comp = {}
for (ds, m), g in dev.groupby(["dataset", "model"]):
    comp[(ds, m)] = g[g.loss != "directional"].sort_values("rel_err").iloc[0].loss
json.dump({f"{k[0]}|{k[1]}": v for k, v in comp.items()}, open(f"{R}/frozen_comparators.json", "w"), indent=1)
def boot(delta, groups, B=5000):
    df = pd.DataFrame(dict(d=delta, g=groups)); gm = df.groupby("g").d.agg(["sum", "count"]); s, c = gm["sum"].values, gm["count"].values; n = len(s)
    ix = rng.integers(0, n, (B, n)); v = s[ix].sum(1) / c[ix].sum(1); return np.percentile(v, [2.5, 97.5])
test = agg[agg.split == "test"]; rows = []; paired = []
for ds in DS:
    for m in MD:
        for loss in LOSSES:
            x = test[(test.dataset == ds) & (test.model == m) & (test.loss == loss)]
            seeds = pos[(pos.dataset == ds) & (pos.model == m) & (pos.loss == loss) & (pos.split == "test")]
            ps = seeds.groupby("seed").rel_err.mean()
            rows.append(dict(dataset=DS[ds], model=MD[m], loss=NAME[loss], n=len(x), mean=x.rel_err.mean(), median=x.rel_err.median(), ge20=100 * x.ge20.mean(), bias=100 * x.signed.mean(),
                             dice=100 * x.dice.mean(), bf1=100 * x.bf1.mean(), missed=int(x.missed.sum() / seeds.seed.nunique()), seed_sd=ps.std(), mean_all=x.rel_err_all.mean()))
        c = comp[(ds, m)]; a = test[(test.dataset == ds) & (test.model == m) & (test.loss == "directional")].set_index("case_id"); b = test[(test.dataset == ds) & (test.model == m) & (test.loss == c)].set_index("case_id")
        ix = a.index.intersection(b.index); dl = (a.loc[ix].rel_err - b.loc[ix].rel_err).values; lo, hi = boot(dl, a.loc[ix].group.values)
        dd = (a.loc[ix].dice - b.loc[ix].dice).values; dlo, dhi = boot(dd, a.loc[ix].group.values)
        paired.append(dict(dataset=DS[ds], model=MD[m], comparator=NAME[c], delta_relerr=dl.mean(), lo=lo, hi=hi, delta_dice=100 * dd.mean(), dlo=100 * dlo, dhi=100 * dhi, frac_improved=100 * (dl < 0).mean()))
R1 = pd.DataFrame(rows); P = pd.DataFrame(paired); R1.to_csv(f"{R}/main_results.csv", index=False); P.to_csv(f"{R}/paired_improvement.csv", index=False)
print(R1.round(3).to_string(index=False)); print(P.round(3).to_string(index=False))
# BUSI normal images
neg = d[(d.dataset == "busi") & (d.cls == "normal") & (d.split.isin(["dev", "test"]))]
if len(neg):
    N = neg[neg.split == "test"].groupby(["model", "loss"]).agg(n=("case_id", "nunique"), fp_rate=("pred_any", "mean"), fp_area=("pred_area_frac", "mean")).reset_index(); N.to_csv(f"{R}/busi_normal_fp.csv", index=False); print(N.round(4))
# macro average over datasets (equal weight)
M = R1.groupby(["model", "loss"])[["mean", "median", "ge20", "dice"]].mean().reset_index(); print(M.round(3))
# ablation K
if os.path.exists(f"{R}/ablation_per_case_results.csv"):
    ab = pd.read_csv(f"{R}/ablation_per_case_results.csv"); ab = ab[(ab.pos == 1)]
    f = pos[(pos.loss == "directional") & (pos.model == "unet") & (pos.seed == 0)]
    K = pd.concat([ab, f]).groupby(["dataset", "split", "K"]).agg(rel_err=("rel_err", "mean"), dice=("dice", "mean")).reset_index(); K.to_csv(f"{R}/ablation_K.csv", index=False); print(K.round(4))
# LaTeX
def f(x, n=3): return f"{x:.{n}f}"
L = []
for ds in DS:
    for m in MD:
        sub = R1[(R1.dataset == DS[ds]) & (R1.model == MD[m])]; best = sub["mean"].min()
        for i, r in sub.iterrows():
            ds_c = f"\multirow{{8}}{{*}}{{{r.dataset}}}" if r.loss == NAME["dice_bce"] and m == "unet" else ""
            mm = f"\multirow{{4}}{{*}}{{{r.model}}}" if r.loss == NAME["dice_bce"] else ""
            val = f"\textbf{{{f(r['mean'])}}}" if r["mean"] == best else f(r["mean"])
            L.append(f"{ds_c} & {mm} & {r.loss} & {val} & {f(r['median'])} & {f(r.ge20,1)} & {f(r.bias,1)} & {f(r.dice,1)} & {f(r.bf1,1)} & {int(r.missed)} \\\\")
        L.append("\midrule" if not (ds == "busi" and m == "dlv3p") else "")
open(f"{T}/main.tex", "w").write(chr(10).join(l for l in L if l).replace(chr(9) + "extbf", chr(92) + "textbf"))
Lp = [f"{r.dataset} & {r.model} & {r.comparator} & {r.delta_relerr:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] & {r.frac_improved:.0f} & {r.delta_dice:+.1f} [{r.dlo:+.1f}, {r.dhi:+.1f}] \\\\" for r in P.itertuples()]
open(f"{T}/paired.tex", "w").write("\n".join(Lp))
