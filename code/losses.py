import math, torch, torch.nn as nn, torch.nn.functional as F

def _grid(h, w, device):
    ys = torch.linspace(-1, 1, h, device=device) * (h - 1) / max(h, w)
    xs = torch.linspace(-1, 1, w, device=device) * (w - 1) / max(h, w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy

def soft_widths(p, dirs, tau, xx, yy, eps=1e-5, sharp=20.0):
    """p: (B,H,W) mass in [0,1]; dirs: (K,2). Returns widths (B,K) = s+ - s-
    with s+ = tau*log sum p exp(t/tau) (soft max), s- = -tau*log sum p exp(-t/tau)."""
    t = xx[None] * dirs[:, 0, None, None] + yy[None] * dirs[:, 1, None, None]   # (K,H,W)
    logp = ((p - 1.0) * (3.0 / tau)).flatten(1)                              # (B,HW)
    tf = t.flatten(1)                                                          # (K,HW)
    sp = tau * torch.logsumexp(logp[:, None, :] + tf[None] / tau, dim=-1)
    sm = -tau * torch.logsumexp(logp[:, None, :] - tf[None] / tau, dim=-1)
    return sp - sm

class DirectionalExtentLoss(nn.Module):
    def __init__(self, K=16, tau=0.02, delta=0.25):
        super().__init__()
        a = torch.arange(K) * math.pi / K
        self.register_buffer("dirs", torch.stack([a.cos(), a.sin()], 1))
        self.tau, self.delta = tau, delta
    def forward(self, logits, target):
        p = torch.sigmoid(logits)[:, 0]; m = target[:, 0]
        B, H, W = p.shape
        xx, yy = _grid(H, W, p.device)
        wp = soft_widths(p, self.dirs, self.tau, xx, yy)
        wm = soft_widths(m, self.dirs, self.tau, xx, yy)
        valid = (m.flatten(1).sum(1) > 0).float()
        d = (wp - wm) / wm.detach().clamp_min(0.05)
        l = F.huber_loss(d, torch.zeros_like(d), delta=self.delta, reduction="none").mean(1)
        return (l * valid).sum() / valid.sum().clamp_min(1)

class BBoxExtentLoss(nn.Module):
    """Axis-aligned bounding-box extent penalty (soft, K=2 directions) as simple geometric control."""
    def __init__(self, tau=0.02, delta=0.25):
        super().__init__()
        self.register_buffer("dirs", torch.tensor([[1., 0.], [0., 1.]])); self.tau, self.delta = tau, delta
    def forward(self, logits, target):
        p = torch.sigmoid(logits)[:, 0]; m = target[:, 0]; B, H, W = p.shape
        xx, yy = _grid(H, W, p.device)
        wp = soft_widths(p, self.dirs, self.tau, xx, yy); wm = soft_widths(m, self.dirs, self.tau, xx, yy)
        valid = (m.flatten(1).sum(1) > 0).float()
        d = (wp - wm) / wm.detach().clamp_min(0.05)
        l = F.huber_loss(d, torch.zeros_like(d), delta=self.delta, reduction="none").mean(1)
        return (l * valid).sum() / valid.sum().clamp_min(1)

class FeretAlignedLoss(nn.Module):
    """v2: mean-width term (as v1) + term on the soft maximum over directions of the width,
    which is the quantity the evaluation metric (max Feret) actually measures. Normalised by the reference max width;
    the same soft maximum is applied to the reference so its smoothing bias cancels."""
    def __init__(self, K=16, tau=0.02, delta=0.25, tau_k=0.05):
        super().__init__()
        a = torch.arange(K) * math.pi / K
        self.register_buffer("dirs", torch.stack([a.cos(), a.sin()], 1)); self.tau, self.delta, self.tau_k = tau, delta, tau_k
    def forward(self, logits, target):
        p = torch.sigmoid(logits)[:, 0]; m = target[:, 0]; B, H, W = p.shape
        xx, yy = _grid(H, W, p.device)
        wp = soft_widths(p, self.dirs, self.tau, xx, yy); wm = soft_widths(m, self.dirs, self.tau, xx, yy)
        valid = (m.flatten(1).sum(1) > 0).float()
        d = (wp - wm) / wm.detach().clamp_min(0.05)
        l1 = F.huber_loss(d, torch.zeros_like(d), delta=self.delta, reduction="none").mean(1)
        mx = wm.detach().max(1, keepdim=True).values.clamp_min(0.05)
        sp = self.tau_k * torch.logsumexp(wp / mx / self.tau_k, 1); sm = self.tau_k * torch.logsumexp(wm.detach() / mx / self.tau_k, 1)
        l2 = F.huber_loss(sp, sm, delta=self.delta, reduction="none")
        return (((l1 + l2) * valid).sum() / valid.sum().clamp_min(1))

def dice_loss(logits, target, eps=1e-6):
    p = torch.sigmoid(logits); i = (p * target).sum((1, 2, 3)); u = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return (1 - (2 * i + eps) / (u + eps)).mean()

class BoundaryLoss(nn.Module):
    """Distance-transform boundary loss (Kervadec et al.): mean(p * signed_dist(target))."""
    def forward(self, logits, target):
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt
        t = target.detach().cpu().numpy()[:, 0] > 0.5
        sd = np.stack([(edt(~x) - edt(x)) if x.any() else np.zeros_like(x, float) for x in t])
        sd = torch.from_numpy(sd).float().to(logits.device)[:, None]
        sd = sd / max(logits.shape[-2:])
        return (torch.sigmoid(logits) * sd).mean()

def neg_penalty(logits, target, frac=0.01):
    """v3: on lesion-free images, mean of the top 1% foreground probabilities (pushes the maximum response to zero)."""
    p = torch.sigmoid(logits)[:, 0].flatten(1); neg = (target[:, 0].flatten(1).sum(1) == 0).float()
    k = max(1, int(frac * p.shape[1])); t = p.topk(k, dim=1).values.mean(1)
    return (t * neg).sum() / neg.sum().clamp_min(1)

class Combined(nn.Module):
    def __init__(self, kind="dice_bce", lam=1.0, K=16, tau=0.02, boundary_w=0.5, lam_n=0.0):
        super().__init__()
        self.kind, self.lam, self.bw, self.lam_n = kind, lam, boundary_w, lam_n
        if kind == "directional3": kind = "directional2"
        if kind == "dice_neg": kind = "dice_bce"
        self.kind0 = kind
        self.ext = DirectionalExtentLoss(K, tau) if kind == "directional" else None
        self.bbox = BBoxExtentLoss(tau) if kind == "bbox" else None
        self.ext2 = FeretAlignedLoss(K, tau) if kind == "directional2" else None
        self.bnd = BoundaryLoss() if kind == "boundary" else None
    def forward(self, logits, target):
        l = dice_loss(logits, target) + F.binary_cross_entropy_with_logits(logits, target)
        if self.kind == "directional": l = l + self.lam * self.ext(logits, target)
        elif self.kind0 == "directional2": l = l + self.lam * self.ext2(logits, target)
        elif self.kind == "bbox": l = l + self.lam * self.bbox(logits, target)
        elif self.kind == "boundary": l = l + self.bw * self.bnd(logits, target)
        if self.lam_n > 0: l = l + self.lam_n * neg_penalty(logits, target)
        return l
