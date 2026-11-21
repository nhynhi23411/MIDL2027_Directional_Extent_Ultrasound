import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, Ellipse
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
                     "mathtext.fontset":"stix","pdf.fonttype":42,"font.size":7})
C=dict(blue="#2F6DB5",lblue="#E6EEF8",green="#3E8E5A",lgreen="#E8F3EC",
       orange="#D9822B",lor="#FBEFE0",red="#C0392B",gray="#555555",lgray="#F3F3F3")
fig,ax=plt.subplots(figsize=(7.2,4.3)); ax.set_xlim(0,100); ax.set_ylim(0,64); ax.axis("off"); ax.set_aspect("equal")
def box(x,y,w,h,fc,ec,ls="-",lw=0.8,r=1.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0,rounding_size={r}",fc=fc,ec=ec,lw=lw,ls=ls))
def arrow(p,q,c=C["gray"],ls="-",rad=0,lw=0.9):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=7,color=c,lw=lw,ls=ls,connectionstyle=f"arc3,rad={rad}"))
def T(x,y,s,fs=7,ha="center",c="black",w="normal",st="normal"):
    ax.text(x,y,s,fontsize=fs,ha=ha,va="center",color=c,fontweight=w,fontstyle=st)
yy,xx=np.mgrid[0:64,0:64]
def lesion(cx=32,cy=32,a=17,b=10,th=0.5):
    X=(xx-cx)*np.cos(th)+(yy-cy)*np.sin(th); Y=-(xx-cx)*np.sin(th)+(yy-cy)*np.cos(th)
    return (X/a)**2+(Y/b)**2
rng=np.random.default_rng(0)
us=np.clip(0.35+0.18*rng.standard_normal((64,64)),0,1)
us=np.where(lesion()<1,us*0.35,us)
mask=(lesion()<1).astype(float)
prob=1/(1+np.exp(8*(lesion(33,31,16,9.5,0.45)-1)))
def img(a,ext,cmap="gray",vmin=0,vmax=1):
    ax.imshow(a,extent=ext,cmap=cmap,vmin=vmin,vmax=vmax,origin="lower",zorder=2,interpolation="bilinear"); 
    ax.add_patch(plt.Rectangle((ext[0],ext[2]),ext[1]-ext[0],ext[3]-ext[2],fc="none",ec=C["gray"],lw=0.5,zorder=3))
# headers
T(0.5,62.5,r"$\mathbf{(a)}$ Training: measurement-preserving directional extent supervision",8,ha="left",w="bold")
# A inputs
box(0.5,34,25,25,C["lgray"],C["gray"]); T(13,57,"Public ultrasound data",7,w="bold"); T(13,54.5,"MMOTU, BUSI (released masks)",6,c=C["gray"])
img(us,(2,12,38,48)); img(mask,(14,24,38,48))
T(7,35.8,r"image $x$",6.5); T(19,35.8,r"mask $M$",6.5)
# B network
box(29,34,19,25,C["lblue"],C["blue"]); T(38.5,57,"Segmentation net",7,w="bold"); T(38.5,54.5,r"$f_\theta$ (2 families)",6,c=C["gray"])
ax.add_patch(Polygon([(30.5,40),(30.5,48),(38,45.5),(38,42.5)],fc="#9DBDE3",ec=C["blue"],lw=0.7))
ax.add_patch(Polygon([(39,42.5),(39,45.5),(46.5,48),(46.5,40)],fc="#9DBDE3",ec=C["blue"],lw=0.7))
ax.add_patch(plt.Rectangle((37.5,42.8),2,2.4,fc=C["blue"],ec="none"))
T(38.5,36.3,"U-Net / enc.–dec.",6.2)
# P
img(prob,(51,61,38,48),cmap="viridis"); T(56,35.8,r"prob. map $P$",6.5)
arrow((25.7,46.5),(28.8,46.5)); arrow((48.2,46.5),(50.8,46.5))
# C loss panel
box(64,30,35.5,29.5,C["lor"],C["orange"]); T(81.7,57.2,r"Directional extent loss $\mathcal{L}_{\mathrm{ext}}$",7,w="bold",c="#7A4410")
# mini geometry
gx,gy=73,45
ell=Ellipse((gx,gy),15,9,angle=28,fc="none",ec=C["green"],lw=1.0,zorder=4)
ell2=Ellipse((gx+0.6,gy-0.4),13,8,angle=24,fc=(0.2,0.4,0.8,0.25),ec=C["blue"],lw=1.0,zorder=4,ls="--")
ax.add_patch(ell); ax.add_patch(ell2)
for ang,col in [(20,C["red"]),(70,"#8E44AD"),(125,"#8E44AD")]:
    a=np.deg2rad(ang); u=np.array([np.cos(a),np.sin(a)]); n=np.array([-u[1],u[0]])
    ax.plot(*zip(np.array([gx,gy])-u*8.5,np.array([gx,gy])+u*8.5),color=col,lw=0.6,zorder=3,alpha=.9)
# support ticks along u (red)
a=np.deg2rad(20); u=np.array([np.cos(a),np.sin(a)]); n=np.array([-u[1],u[0]])
for s in (-1,1):
    c0=np.array([gx,gy])+u*s*7.2
    ax.plot(*zip(c0-n*3.0,c0+n*3.0),color=C["red"],lw=1.0,zorder=5)
T(gx,gy-9.6,r"directions $u_k$, $k=1..K$",6,c=C["red"])
ax.plot([67,69],[32.5,32.5],color=C["blue"],lw=1,ls="--"); T(69.8,32.5,r"$P$",6.2,ha="left",c=C["gray"])
ax.plot([73,75],[32.5,32.5],color=C["green"],lw=1); T(75.8,32.5,r"$M$",6.2,ha="left",c=C["gray"])
# formula text
T(90.5,52.3,r"project onto $u_k$",6.3,c="#7A4410")
T(90.5,49.4,r"$s_k^{\pm}(P)$: soft max / min",6.3)
T(90.5,46.6,r"$w_k=s_k^{+}-s_k^{-}$",6.8)
T(90.5,42.6,r"$\mathcal{L}_{\mathrm{ext}}=\frac{1}{K}\sum_k \rho\!\left(\frac{w_k(P)-w_k(M)}{\bar w_k}\right)$",6.2)
T(90.5,37.8,r"$\rho$: robust loss, $K{=}16$",6.2,c=C["gray"])
T(90.5,34.8,"differentiable surrogate",6,c=C["gray"],st="italic")
arrow((61.2,46.5),(63.8,46.5),c=C["orange"])
# M to loss
ax.plot([13,13],[59.2,60.8],color=C["gray"],lw=0.9); ax.plot([13,81],[60.8,60.8],color=C["gray"],lw=0.9)
# (arrow hidden behind title: shorten) 
ax.add_patch(FancyArrowPatch((81,60.8),(81,59.7),arrowstyle="-|>",mutation_scale=7,color=C["gray"],lw=0.9))
T(47,60.0,r"reference $M$",6,c=C["gray"],st="italic")
# total loss
box(64,20.5,35.5,6.5,C["lgreen"],C["green"])
T(81.7,23.8,r"$\mathcal{L}=\mathcal{L}_{\mathrm{Dice+BCE}}+\lambda\,\mathcal{L}_{\mathrm{ext}}$",7.4)
arrow((81.7,30),(81.7,27.2),c=C["orange"])
arrow((64,23.8),(38.5,23.8)); ax.plot([38.5,38.5],[23.8,33.8],color=C["gray"],lw=0.9,ls=(0,(3,2)))
ax.add_patch(FancyArrowPatch((38.5,33.0),(38.5,33.9),arrowstyle="-|>",mutation_scale=7,color=C["gray"],lw=0.9))
T(51.5,25.6,"backprop.",6.2,c=C["gray"],st="italic")
# comparators
box(0.5,20.5,25,10.5,"white",C["gray"],ls=(0,(3,2)))
T(13,28.6,"Matched comparators",6.5,w="bold"); T(13,26.0,"Dice+BCE · +boundary loss",6)
T(13,23.6,"· bbox-extent penalty",6)
arrow((26,25.8),(29,25.8),c=C["gray"],ls=":",lw=.8) if False else None
# Eval
T(0.5,17.0,r"$\mathbf{(b)}$ Evaluation: exact hard-mask measurement (training surrogate not used)",8,ha="left",w="bold")
bx=[(0.5,"Test image",r"frozen $f_\theta$"),(20.5,"Threshold + component",r"fixed, prespecified rule"),(40.5,r"Hard mask $\hat{M}$","original resolution"),
    (60.5,"Max Feret diameter",r"$\hat{d}$ (exact, hard mask)"),(80.5,"Relative error",r"$|\hat{d}-d|\,/\,d$")]
for i,(x,t1,t2) in enumerate(bx):
    k=C["blue"] if i<3 else (C["green"] if i==3 else C["orange"])
    box(x,1.5,18.5,11.5,C["lblue"] if i<3 else (C["lgreen"] if i==3 else C["lor"]),k)
    T(x+9.25,10.0,t1,6.5,w="bold"); T(x+9.25,5.8,t2,6.2,c=C["gray"])
    if i<4: arrow((x+18.7,7.2),(x+20.3,7.2))
T(89.75,3.2,r"$d$: reference mask $M$",5.6,c=C["gray"])
T(89.5,-1.2,"",5)
fig.subplots_adjust(0.005,0.005,0.995,0.995)
for ext in ("pdf","png","svg"): fig.savefig(f"pipeline.{ext}",dpi=300)
