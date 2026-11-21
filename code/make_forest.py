import pandas as pd, numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":8,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
R="../results/"; O="../paper/figures/"; rng=np.random.default_rng(3)
v1=pd.read_csv(R+"paired_improvement.csv"); v2=pd.read_csv(R+"paired_v2.csv"); print(v1.columns.tolist()); print(v1.head(3)); print(v2.head(3))
d3=pd.concat([pd.read_csv(R+"v2final_per_case_results.csv"),pd.read_csv(R+"v3final_per_case_results.csv")]); d3=d3[(d3.dataset=="busi")&(d3.split=="test")&(d3.pos==1)]
comp=json.load(open(R+"frozen_comparators_v2.json"))
a=d3.groupby(["model","loss","case_id"]).agg(re=("rel_err","mean"),g=("group","first")).reset_index()
def boot(dl,g,B=5000):
    df=pd.DataFrame(dict(d=dl,g=g)).groupby("g").d.agg(["sum","count"]); s,c=df["sum"].values,df["count"].values; ix=rng.integers(0,len(s),(B,len(s))); return np.percentile(s[ix].sum(1)/c[ix].sum(1),[2.5,97.5])
out=[]
for m in ("unet","dlv3p"):
    c=comp[f"busi|{m}"]; p=a[a.model==m].pivot(index="case_id",columns="loss",values="re"); g=a[(a.model==m)&(a.loss=="dice_bce")].set_index("case_id").g.loc[p.index].values
    for new,lab in (("directional3","v3"),("dice_neg","neg-only")):
        dl=(p[new]-p[c]).values; lo,hi=boot(dl,g); out.append(dict(dataset="BUSI",model="U-Net" if m=="unet" else "DeepLabV3+",ver=lab,delta=dl.mean(),lo=lo,hi=hi,comp=c))
    dl=(p["directional3"]-p["dice_neg"]).values; lo,hi=boot(dl,g); out.append(dict(dataset="BUSI",model="U-Net" if m=="unet" else "DeepLabV3+",ver="v3-vs-neg",delta=dl.mean(),lo=lo,hi=hi,comp="dice_neg"))
V3=pd.DataFrame(out); V3.to_csv(R+"paired_v3.csv",index=False); print(V3.round(3))
v1=v1.rename(columns={"delta_relerr":"delta"}); v1["ver"]="v1"; v2=v2.rename(columns={"new":"ver_"}); v2["ver"]=v2.ver_.map(lambda s:"v2" if "v2" in s else "v1neg")
allp=pd.concat([v1[["dataset","model","ver","delta","lo","hi"]],v2[["dataset","model","ver","delta","lo","hi"]],V3[V3.ver=="v3"][["dataset","model","ver","delta","lo","hi"]]])
allp.to_csv(R+"all_contrasts.csv",index=False)
sets=[("MMOTU","U-Net"),("MMOTU","DeepLabV3+"),("BUSI","U-Net"),("BUSI","DeepLabV3+")]
vers=[("v1","Directional (v1)","#D9822B"),("v1neg","v1, lesion-free in training","#8E6BBF"),("v2","Feret-aligned (v2)","#C0392B"),("v3","v2 + negative penalty (v3)","#2F6DB5")]
fig,ax=plt.subplots(figsize=(3.4,3.0)); y=0; yt=[]; yl=[]
for ds,m in sets:
    ys=[]
    for v,lab,c in vers:
        r=allp[(allp.dataset==ds)&(allp.model==m)&(allp.ver==v)]
        if not len(r): continue
        r=r.iloc[0]; ax.errorbar(r.delta,y,xerr=[[r.delta-r.lo],[r.hi-r.delta]],fmt="o",ms=3.5,c=c,capsize=2,lw=1,label=lab if (ds,m)==("BUSI","U-Net") or (v=="v1" and (ds,m)==("MMOTU","U-Net")) else None); ys.append(y); y-=1
    yt.append(np.mean(ys)); yl.append(f"{ds}\n{m}"); y-=0.8
ax.axvline(0,c="k",lw=.7); ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=7); ax.set_xlabel(r"$\Delta$ mean relative error vs frozen comparator"+"\n(negative favours the extent loss)",fontsize=7)
h,l=ax.get_legend_handles_labels(); u=dict(zip(l,h)); ax.legend(u.values(),u.keys(),frameon=False,fontsize=6,loc="upper left")
fig.tight_layout(); fig.savefig(O+"forest.pdf"); fig.savefig(O+"forest.png",dpi=250); print("ok")
