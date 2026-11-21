import pandas as pd, json
d = pd.read_csv("E:/MIDL_2/results/v2pilot_per_case_results.csv"); d = d[(d.split == "dev") & (d.pos == 1)]
g = d.groupby(["dataset", "loss", "lam"]).agg(rel_err=("rel_err", "mean"), dice=("dice", "mean"), ge20=("rel_err", lambda x: (x >= .2).mean())).reset_index()
print(g.to_string(index=False)); B = {}
for ds, gg in g.groupby("dataset"):
    B[ds] = {l: float(x.sort_values("rel_err").iloc[0].lam) for l, x in gg.groupby("loss") if l != "dice_bce"}
json.dump(B, open("E:/MIDL_2/results/best_hparams_v2.json", "w"), indent=1); print(B)
