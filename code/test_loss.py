import torch, math
from losses import *
from metrics import feret
torch.manual_seed(0)
H=W=128
def ell(cx,cy,a,b,th=0):
    yy,xx=torch.meshgrid(torch.arange(H).float(),torch.arange(W).float(),indexing="ij")
    X=(xx-cx)*math.cos(th)+(yy-cy)*math.sin(th); Y=-(xx-cx)*math.sin(th)+(yy-cy)*math.cos(th)
    return ((X/a)**2+(Y/b)**2<1).float()[None,None]
L=DirectionalExtentLoss()
m=ell(64,64,40,12,0.6); lg=lambda x:(x*12-6)
def run(name,pred,tgt=m):
    z=lg(pred).requires_grad_(True); l=L(z,tgt); l.backward()
    print(f"{name:28s} loss={l.item():.4f} finite_grad={bool(torch.isfinite(z.grad).all())}")
run("perfect",m); run("translated",ell(80,50,40,12,0.6)); run("shorter ellipse",ell(64,64,25,12,0.6))
run("distant island",torch.clamp(m+ell(10,10,4,4),0,1)); run("empty output",torch.zeros_like(m))
run("uniform low prob",torch.full_like(m,0.01)); run("uniform 0.5",torch.full_like(m,.5))
run("empty target",m,torch.zeros_like(m))
print("feret ellipse a=40:",feret(m[0,0].numpy()>0.5),"expected ~80")
