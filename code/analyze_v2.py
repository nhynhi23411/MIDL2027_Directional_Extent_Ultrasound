import pandas as pd, numpy as np, json
R = "E:/MIDL_2/results"; T = "E:/MIDL_2/paper/tables"; rng = np.random.default_rng(1)
v1 = pd.read_csv(f"{R}/final_per_case_results.csv"); v2 = pd.read_csv(f"{R}/v2final_per_case_results.csv")
v1 = v1[v1.dataset == "mmotu"]
d = pd.concat([v1, v2], ignore_index=True)
LOSS = ["dice_bce", "boundary", "bbox", "directional", "directional2"]
NAME = dict(dice_bce="Dice+BCE", boundary="+ Boundary", bbox="+ BBox extent", directional="+ Directional (v1)", directional2="+ Feret-aligned (v2)")
DS = dict(mmotu="MMOTU", busi="BUSI"); MD = dict(unet="U-Net", dlv3p="DeepLabV3+")
pos = d[d.pos == 1]
agg = pos.groupby(["dataset", "model", "loss", "split", "case_id", "group"]).agg(rel_err=("rel_err", "mean"), rel_err_all=("rel_err_all", "mean"), signed=("signed", "mean"),
      dice=("dice", "mean"), bf1=("bf1", "mean"), ge20=("rel_err", lambda x: (x >= .2).mean()), missed=("missed", "sum")).reset_index()
dev = agg[agg.split == "dev"].groupby(["dataset", "model", "loss"]).rel_err.mean().reset_index()
comp = {}
for (ds, m), g in dev.groupby(["dataset", "model"]):
    comp[(ds, m)] = g[g.loss.isin(["dice_bce", "boundary", "bbox"])].sort_values("rel_err").iloc[0].loss
json.dump({f"{k[0]}|{k[1]}": v for k, v in comp.items()}, open(f"{R}/frozen_comparators_v2.json", "w"), indent=1)
def boot(delta, groups, B=5000):
    df = pd.DataFrame(dict(d=delta, g=groups)); gm = df.groupby("g").d.agg(["sum", "count"]); s, c = gm["sum"].values, gm["count"].values; n = len(s)
    ix = rng.integers(0, n, (B, n)); return np.percentile(s[ix].sum(1) / c[ix].sum(1), [2.5, 97.5])
test = agg[agg.split == "test"]; rows = []; paired = []
for ds in DS:
    for m in MD:
        for loss in LOSS:
            x = test[(test.dataset == ds) & (test.model == m) & (test.loss == loss)]
            if not len(x): continue
            sd = pos[(pos.dataset == ds) & (pos.model == m) & (pos.loss == loss) & (pos.split == "test")].groupby("seed").rel_err.mean()
            rows.append(dict(dataset=DS[ds], model=MD[m], loss=NAME[loss], n=len(x), mean=x.rel_err.mean(), median=x.rel_err.median(), ge20=100 * x.ge20.mean(), bias=100 * x.signed.mean(),
                             dice=100 * x.dice.mean(), bf1=100 * x.bf1.mean(), missed=int(round(x.missed.sum() / pos[(pos.dataset == ds) & (pos.model == m) & (pos.loss == loss)].seed.nunique())), seed_sd=sd.std(), mean_all=x.rel_err_all.mean()))
        c = comp[(ds, m)]
        for new in ("directional", "directional2"):
            a = test[(test.dataset == ds) & (test.model == m) & (test.loss == new)].set_index("case_id")
            if not len(a): continue
            b = test[(test.dataset == ds) & (test.model == m) & (test.loss == c)].set_index("case_id"); ix = a.index.intersection(b.index)
            dl = (a.loc[ix].rel_err - b.loc[ix].rel_err).values; lo, hi = boot(dl, a.loc[ix].group.values)
            dd = (a.loc[ix].dice - b.loc[ix].dice).values; dlo, dhi = boot(dd, a.loc[ix].group.values)
            paired.append(dict(dataset=DS[ds], model=MD[m], new=NAME[new], comparator=NAME[c], delta=dl.mean(), lo=lo, hi=hi, frac=100 * (dl < 0).mean(), ddice=100 * dd.mean(), dlo=100 * dlo, dhi=100 * dhi))
R1 = pd.DataFrame(rows); P = pd.DataFrame(paired); R1.to_csv(f"{R}/main_results_v2.csv", index=False); P.to_csv(f"{R}/paired_v2.csv", index=False)
print(R1.round(3).to_string(index=False)); print(P.round(3).to_string(index=False))
neg = d[(d.dataset == "busi") & (d.cls == "normal") & (d.split == "test")]
N = neg.groupby(["model", "loss"]).agg(n=("case_id", "nunique"), fp_rate=("pred_any", "mean"), fp_area=("pred_area_frac", "mean")).reset_index(); N.to_csv(f"{R}/busi_normal_fp_v2.csv", index=False); print(N.round(4))
# LaTeX (raw strings: this shell collapses backslashes)
b = chr(92); L = []
for ds in DS:
    for m in MD:
        sub = R1[(R1.dataset == DS[ds]) & (R1.model == MD[m])]; best = sub["mean"].min(); k = 0
        for r in sub.itertuples():
            dsc = f"{b}multirow{{{len(sub)*2}}}{{*}}{{{r.dataset}}}" if (m == "unet" and k == 0) else ""
            mm = f"{b}multirow{{{len(sub)}}}{{*}}{{{r.model}}}" if k == 0 else ""
            v = f"{b}textbf{{{r.mean:.3f}}}" if r.mean == best else f"{r.mean:.3f}"
            L.append(f"{dsc} & {mm} & {r.loss} & {v} & {r.median:.3f} & {r.ge20:.1f} & {r.bias:.1f} & {r.dice:.1f} & {r.bf1:.1f} & {r.missed} {b}{b}"); k += 1
        L.append(f"{b}midrule")
L = L[:-1]; open(f"{T}/main_v2.tex", "w").write("\n".join(L))
open(f"{T}/paired_v2.tex", "w").write("\n".join(f"{r.dataset} & {r.model} & {r.new} & {r.comparator} & {r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] & {r.frac:.0f} & {r.ddice:+.1f} [{r.dlo:+.1f}, {r.dhi:+.1f}] {b}{b}" for r in P.itertuples()))
