import numpy as np, cv2
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

def feret(mask):
    """Max Feret diameter (pixels): max distance between foreground pixel corners. mask: HxW bool."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0: return 0.0
    pts = np.concatenate([np.stack([xs + dx, ys + dy], 1) for dx in (0, 1) for dy in (0, 1)]).astype(float)
    if len(pts) > 3:
        try: pts = pts[ConvexHull(pts).vertices]
        except Exception: pass
    return float(pdist(pts).max())

def largest_component(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1: return mask.astype(bool)
    return lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))

def dice(a, b):
    s = a.sum() + b.sum(); return 1.0 if s == 0 else 2 * (a & b).sum() / s

def boundary_f1(a, b, tol=2):
    if a.sum() == 0 or b.sum() == 0: return 0.0
    k = np.ones((3, 3), np.uint8)
    ea = a.astype(np.uint8) - cv2.erode(a.astype(np.uint8), k); eb = b.astype(np.uint8) - cv2.erode(b.astype(np.uint8), k)
    da = cv2.distanceTransform(1 - ea, cv2.DIST_L2, 3); db = cv2.distanceTransform(1 - eb, cv2.DIST_L2, 3)
    prec = (db[ea > 0] <= tol).mean(); rec = (da[eb > 0] <= tol).mean()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
