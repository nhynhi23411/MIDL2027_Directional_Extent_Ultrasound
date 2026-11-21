import os, re, hashlib, glob, numpy as np, pandas as pd, cv2
from PIL import Image
R = "E:/MIDL_2/data"; rows = []; issues = []
def comps(m, tiny=20):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    a = st[1:, cv2.CC_STAT_AREA]; return len(a), int((a < tiny).sum())
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
def log(ds, i, kind, d=""): issues.append(dict(dataset=ds, case_id=i, issue=kind, detail=d))
# MMOTU
M = f"{R}/MMOTU/MMOTU/OTU_2d"
split = {i: s for s in ("train", "val") for i in open(f"{M}/{s}.txt").read().split()}
for f in sorted(os.listdir(f"{M}/images"), key=lambda x: int(x.split(".")[0])):
    i = f.split(".")[0]; ip = f"{M}/images/{f}"; mp = f"{M}/annotations/{i}_binary.PNG"
    im = Image.open(ip); mk = np.array(Image.open(mp)); full = np.array(Image.open(f"{M}/annotations/{i}.PNG"))
    ok = mk.shape[:2] == im.size[::-1]
    if not ok: log("MMOTU", i, "size_mismatch", f"{mk.shape} vs {im.size}")
    vals = np.unique(mk); 
    if not set(vals) <= {0, 1}: log("MMOTU", i, "nonbinary_mask", str(vals))
    m = mk > 0; n, t = comps(m)
    if m.sum() == 0: log("MMOTU", i, "empty_mask")
    if t: log("MMOTU", i, "tiny_components", str(t))
    if n > 1: log("MMOTU", i, "multi_component", str(n))
    cl = np.unique(full.reshape(-1, full.shape[-1] if full.ndim == 3 else 1), axis=0)
    rows.append(dict(dataset="MMOTU", case_id=i, image=ip, mask=mp, orig_split=split.get(i, "none"), h=mk.shape[0], w=mk.shape[1],
                     area=int(m.sum()), n_comp=n, tiny=t, empty=int(m.sum() == 0), img_md5=md5(ip), n_classes_full=len(cl)))
# BUSI
B = f"{R}/BUSI/Dataset_BUSI_with_GT"
for c in ("benign", "malignant", "normal"):
    for ip in sorted(glob.glob(f"{B}/{c}/*.png")):
        if "_mask" in ip: continue
        base = os.path.basename(ip)[:-4]; i = f"{c}_{re.sub(r'\D', '', base) or base}"
        mps = sorted(glob.glob(f"{B}/{c}/{base}_mask*.png"))
        im = Image.open(ip); u = None; shape_ok = True; nm = len(mps)
        for mp in mps:
            mk = np.array(Image.open(mp).convert("L")) > 0
            if mk.shape != im.size[::-1]: shape_ok = False
            else: u = mk if u is None else (u | mk)
        if not shape_ok: log("BUSI", i, "size_mismatch")
        if nm > 1: log("BUSI", i, "multiple_masks_union", str(nm))
        if nm == 0: log("BUSI", i, "no_mask")
        m = u if u is not None else np.zeros(im.size[::-1], bool); n, t = comps(m)
        if c != "normal" and m.sum() == 0: log("BUSI", i, "empty_mask_on_lesion_class")
        if t: log("BUSI", i, "tiny_components", str(t))
        rows.append(dict(dataset="BUSI", case_id=i, image=ip, mask=";".join(mps), orig_split=c, h=m.shape[0], w=m.shape[1],
                         area=int(m.sum()), n_comp=n, tiny=t, empty=int(m.sum() == 0), img_md5=md5(ip), n_classes_full=nm))
df = pd.DataFrame(rows)
d = df[df.duplicated("img_md5", keep=False)]
for _, r in d.iterrows(): log(r.dataset, r.case_id, "exact_duplicate_image", r.img_md5[:8])
df.to_csv("E:/MIDL_2/manifests/dataset_manifest.csv", index=False)
pd.DataFrame(issues).to_csv("E:/MIDL_2/manifests/annotation_issue_log.csv", index=False)
print(df.groupby("dataset").agg(n=("case_id", "size"), empty=("empty", "sum"), multi=("n_comp", lambda x: (x > 1).sum()), tiny=("tiny", lambda x: (x > 0).sum())))
print(pd.DataFrame(issues).groupby(["dataset", "issue"]).size())
print(df.groupby(["dataset", "orig_split"]).size())
