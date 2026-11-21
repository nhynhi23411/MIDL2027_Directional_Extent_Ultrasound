import pandas as pd, numpy as np
r=np.random.default_rng(0)
v3=pd.read_csv('../results/v3final_per_case_results.csv'); v2=pd.read_csv('../results/v2final_per_case_results.csv')
d=pd.concat([v3,v2[(v2.dataset=='busi')]]); d=d[d.split=='test']
def boot(a,b,g,n=5000):
    df=pd.DataFrame(dict(x=a-b,g=g)).groupby('g').x.agg(['sum','count']); s,c=df['sum'].values,df['count'].values; k=len(s)
    ix=r.integers(0,k,(n,k)); m=s[ix].sum(1)/c[ix].sum(1); return s.sum()/c.sum(),*np.percentile(m,[2.5,97.5])
out=[]
for m in ('unet','dlv3p'):
    x=d[d.model==m]; P=x[x.pos==1]
    ag=lambda l:P[P.loss==l].groupby('case_id').agg(re=('rel_err','mean'),dice=('dice','mean'),g=('group','first'),ms=('missed','mean'))
    N=x[x.pos==0]; fp=lambda l:N[N.loss==l].groupby('case_id').pred_any.mean().mean()
    for l in ('dice_bce','directional2','dice_neg','directional3'):
        a=ag(l); print(m,l,'relerr %.3f dice %.3f miss %.1f FP %.2f'%(a.re.mean(),a.dice.mean(),a.ms.sum()*3/3,fp(l)))
    for a_,b_ in (('directional3','dice_neg'),('directional3','dice_bce'),('directional3','directional2'),('dice_neg','dice_bce')):
        A,B=ag(a_),ag(b_); e=boot(A.re,B.re,A.g); print(m,a_,'-',b_,'%+.3f [%+.3f,%+.3f]'%e)
