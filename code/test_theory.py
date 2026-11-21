import math, numpy as np, torch
from losses import soft_widths, _grid
torch.manual_seed(0); rng = np.random.default_rng(0)
H = W = 128; tau = 0.02; K = 16
ang = torch.arange(K) * math.pi / K; dirs = torch.stack([ang.cos(), ang.sin()], 1)
xx, yy = _grid(H, W, "cpu"); X = xx.numpy(); Y = yy.numpy(); N = H * W
def blob():
    cx, cy = rng.uniform(-.4, .4, 2); a, b = rng.uniform(.08, .5, 2); th = rng.uniform(0, math.pi)
    u = (X - cx) * math.cos(th) + (Y - cy) * math.sin(th); v = -(X - cx) * math.sin(th) + (Y - cy) * math.cos(th)
    return ((u / a) ** 2 + (v / b) ** 2 < 1)
worst_lo, worst_hi, bound_hi, ratio_min, tr_dev = 1e9, -1e9, 0, 1e9, 0
for _ in range(200):
    m = blob()
    if m.sum() < 5: continue
    S = m.sum(); t = np.stack([X * math.cos(a) + Y * math.sin(a) for a in ang.numpy()])  # (K,H,W)
    Wex = np.array([t[k][m].max() - t[k][m].min() for k in range(K)])
    w = soft_widths(torch.tensor(m, dtype=torch.float32)[None], dirs, tau, xx, yy)[0].numpy()
    worst_lo = min(worst_lo, (w - Wex).min()); worst_hi = max(worst_hi, ((w - Wex) - 2 * tau * math.log(S)).max())
    # direction discretisation: D from convex-hull pair of pixel centres vs max_k W_k
    ys, xs = np.nonzero(m); pts = np.stack([X[ys, xs], Y[ys, xs]], 1)
    from scipy.spatial import ConvexHull; from scipy.spatial.distance import pdist
    h = pts[ConvexHull(pts).vertices]; D = pdist(h).max(); ratio_min = min(ratio_min, Wex.max() / D)
    # translation invariance of soft width (shift mask by 6 px inside grid)
    m2 = np.roll(m, (3, 5), (0, 1));
    if m2.sum() == S and m[:3].sum() + m[-3:].sum() + m[:, :5].sum() + m[:, -5:].sum() == 0:
        w2 = soft_widths(torch.tensor(m2, dtype=torch.float32)[None], dirs, tau, xx, yy)[0].numpy(); tr_dev = max(tr_dev, np.abs(w2 - w).max())
c = torch.full((1, H, W), 0.01); wc = soft_widths(c, dirs, tau, xx, yy)[0].numpy()
t = np.stack([X * math.cos(a) + Y * math.sin(a) for a in ang.numpy()]); Wg = np.array([t[k].max() - t[k].min() for k in range(K)])
print("min (w - exact width)  [>=0 expected]: %.2e" % worst_lo)
print("max (w - exact - 2 tau log|S|) [<=0 expected]: %.3f" % worst_hi)
print("min max_k W_k / D  [>= cos(pi/2K)=%.4f]: %.4f" % (math.cos(math.pi / (2 * K)), ratio_min))
print("max |w(shifted) - w| : %.2e" % tr_dev)
print("uniform p=0.01: widths/ full-grid exact widths range: %.3f - %.3f ; grid W_k mean %.3f" % ((wc / Wg).min(), (wc / Wg).max(), Wg.mean()))
print("cos(pi/2K) K=4,8,16,32:", [round(math.cos(math.pi / (2 * k)), 4) for k in (4, 8, 16, 32)])
d, r, ell = 2 * 40, 40, 20; A = math.pi * r * r; print("needle: Dice %.4f rel err %.3f" % (2 * A / (2 * A + ell), ell / d))
