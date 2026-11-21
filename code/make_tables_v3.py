import pandas as pd, numpy as np
R='../results/'; T='../paper/tables/'
v2=pd.read_csv(R+'v2final_per_case_results.csv'); v3=pd.read_csv(R+'v3final_per_case_results.csv')
d=pd.concat([v2[v2.dataset=='busi'],v3]); d=d[d.split=='test']
name={'dice_bce':'Dice+BCE','boundary':'+ Boundary','bbox':'+ BBox extent','directional':'+ Directional (v1)','directional2':'+ Feret-aligned (v2)','dice_neg':'Dice+BCE + neg.\ penalty','directional3':'+ Feret-aligned + neg.\ (v3)'}
model={'unet':'U-Net','dlv3p':'DeepLabV3+'}
rows=[];neg=[]
for m in ('unet','dlv3p'):
    x=d[d.model==m]; first=True
    for l in ('dice_bce','directional2','dice_neg','directional3'):
        P=x[(x.loss==l)&(x.pos==1)]; a=P.groupby('case_id').agg(re=('rel_err','mean'),sg=('signed','mean'),dice=('dice','mean'),bf=('bf1','mean'))
        ms=P.groupby('seed').missed.sum().mean()
        N=x[(x.loss==l)&(x.pos==0)]; fp=N.groupby('case_id').pred_any.mean().mean()*100; ar=N.groupby('case_id').pred_area_frac.mean().mean()*100
        rows.append(f"{'U-Net' if m=='unet' else 'DeepLabV3+'} & {name[l]} & {a.re.mean():.3f} & {a.re.median():.3f} & {(a.re>=.2).mean()*100:.1f} & {a.sg.mean()*100:.1f} & {a.dice.mean()*100:.1f} & {a.bf.mean()*100:.1f} & {ms:.0f} & {fp:.0f} & {ar:.1f} \\\\")
    rows.append('\midrule')
open(T+'v3.tex','w').write('\n'.join(rows[:-1]))
# lesion-free + missed for ALL losses (appendix)
r2=[]
for m in ('unet','dlv3p'):
    for l in ('dice_bce','boundary','bbox','directional','directional2','dice_neg','directional3'):
        x=d[(d.model==m)&(d.loss==l)]
        if len(x)==0: continue
        N=x[x.pos==0]; P=x[x.pos==1]
        r2.append(f"{model[m]} & {name[l]} & {N.groupby('case_id').pred_any.mean().mean()*100:.0f} & {N.groupby('case_id').pred_area_frac.mean().mean()*100:.1f} & {P.groupby('seed').missed.sum().mean():.1f} & {P.groupby('case_id').rel_err.mean().mean():.3f} & {P.groupby('case_id').rel_err_all.mean().mean():.3f} \\\\")
open(T+'busi_neg.tex','w').write('\n'.join(r2))
# K ablation
ab=pd.read_csv(R+'ablation_per_case_results.csv'); fin=pd.read_csv(R+'final_per_case_results.csv'); ab=pd.concat([ab,fin[(fin.loss=='directional')&(fin.model=='unet')&(fin.seed==0)]])
r3=[]
for ds in ('mmotu','busi'):
    for K in (4,8,16,32):
        x=ab[(ab.dataset==ds)&(ab.loss=='directional')&(ab.K==K)&(ab.pos==1)]
        dv=x[x.split=='dev'].rel_err.mean(); ts=x[x.split=='test'].rel_err.mean()
        r3.append(f"{'MMOTU' if ds=='mmotu' else 'BUSI'} & {K} & {dv:.3f} & {ts:.3f} \\\\")
open(T+'ablK.tex','w').write('\n'.join(r3)); print(open(T+'v3.tex').read()); print('\n'.join(r3))
