"""Build grouped splits + letterboxed 256px arrays. Duplicate clusters: dHash hamming<=4 (conservative, reduces but cannot exclude leakage)."""
import os, re, glob, pickle, numpy as np, pandas as pd, cv2
from PIL import Image
S = 256; R = "E:/MIDL_2/data"; rng = np.random.default_rng(2027)
def dhash(g):
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA); return (s[:, 1:] > s[:, :-1]).flatten()
def clusters(h):
    n = len(h); par = list(range(n))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    H = np.array(h)
    for i in range(n):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.nonzero(d <= 4)[0]: par[f(i)] = f(i + 1 + j)
    return [f(i) for i in range(n)]
def letter(a, interp):
    h, w = a.shape[:2]; s = S / max(h, w); nh, nw = max(1, round(h * s)), max(1, round(w * s))
    r = cv2.resize(a, (nw, nh), interpolation=interp); out = np.zeros((S, S), a.dtype)
    t, l = (S - nh) // 2, (S - nw) // 2; out[t:t + nh, l:l + nw] = r; return out, (t, l, nh, nw)
def load_mmotu():
    M = f"{R}/MMOTU/MMOTU/OTU_2d"; off = {}
    for s in ("train", "val"):
        for i in open(f"{M}/{s}.txt").read().split(): off[i] = s
    out = []
    for i in sorted(off, key=int):
        im = np.array(Image.open(f"{M}/images/{i}.JPG").convert("L")); m = (np.array(Image.open(f"{M}/annotations/{i}_binary.PNG")) > 0)
        out.append(dict(id=f"mmotu_{i}", img=im, mask=m, off=off[i], cls="lesion"))
    return out
def load_busi():
    B = f"{R}/BUSI/Dataset_BUSI_with_GT"; out = []
    for c in ("benign", "malignant", "normal"):
        for ip in sorted(glob.glob(f"{B}/{c}/*.png")):
            if "_mask" in ip: continue
            base = os.path.basename(ip)[:-4]; im = np.array(Image.open(ip).convert("L")); m = np.zeros(im.shape, bool)
            for mp in sorted(glob.glob(f"{B}/{c}/{base}_mask*.png")):
                mk = np.array(Image.open(mp).convert("L")) > 0
                if mk.shape == m.shape: m |= mk
            out.append(dict(id=f"busi_{c}_{re.sub(r'\D', '', base)}", img=im, mask=m, off=c, cls=c))
    return out
def build(name, items, fracs):
    cl = clusters([dhash(x["img"]) for x in items])
    for x, c in zip(items, cl): x["group"] = f"{name}_g{c}"
    groups = sorted(set(x["group"] for x in items)); rng.shuffle(groups)
    if name == "mmotu":  # official val = reserved benchmark; dev carved from official train by group
        tr_groups = [g for g in groups if all(x["off"] == "train" for x in items if x["group"] == g)]
        nd = int(round(0.15 * len(tr_groups))); dev = set(tr_groups[:nd])
        for x in items: x["split"] = "test" if x["off"] == "val" else ("dev" if x["group"] in dev else "train")
        # groups spanning official train and val: keep with test side (no leakage into train)
        tg = {x["group"] for x in items if x["split"] == "test"}
        for x in items:
            if x["group"] in tg and x["off"] == "train": x["split"] = "excluded"  # near-duplicate of a reserved image: dropped, reserved set stays the official 469
    else:
        for cls in ("benign", "malignant", "normal"):
            g = sorted({x["group"] for x in items if x["cls"] == cls}); rng.shuffle(g)
            a = int(len(g) * fracs[0]); b = a + int(len(g) * fracs[1]); sp = {}
            for k, gg in enumerate(g): sp[gg] = "train" if k < a else ("dev" if k < b else "test")
            for x in items:
                if x["cls"] == cls: x["split"] = sp[x["group"]]
        # a group present in several classes: send to the most held-out split
        rank = {"train": 0, "dev": 1, "test": 2}; best = {}
        for x in items: best[x["group"]] = max(best.get(x["group"], "train"), x["split"], key=rank.get)
        for x in items: x["split"] = best[x["group"]]
    recs = []
    for x in items:
        if x["split"] == "excluded": continue
        li, pi = letter(x["img"], cv2.INTER_AREA); lm, _ = letter(x["mask"].astype(np.uint8), cv2.INTER_NEAREST)
        recs.append(dict(id=x["id"], group=x["group"], split=x["split"], cls=x["cls"], img=li, mask=lm, pad=pi, shape=x["img"].shape, orig_mask=x["mask"]))
    return recs
if __name__ == "__main__":
    os.makedirs("E:/MIDL_2/prep", exist_ok=True); man = []
    for name, loader, fr in (("mmotu", load_mmotu, None), ("busi", load_busi, (0.7, 0.1))):
        recs = build(name, loader(), fr)
        pickle.dump(recs, open(f"E:/MIDL_2/prep/{name}.pkl", "wb"))
        for r in recs: man.append(dict(dataset=name, case_id=r["id"], group=r["group"], split=r["split"], cls=r["cls"], h=r["shape"][0], w=r["shape"][1], lesion_pos=int(r["orig_mask"].any())))
    df = pd.DataFrame(man); df.to_csv("E:/MIDL_2/manifests/grouped_split_manifest.csv", index=False)
    print(df.groupby(["dataset", "split"]).agg(n=("case_id", "size"), groups=("group", "nunique"), pos=("lesion_pos", "sum")))
    print("multi-image groups:", df.groupby("dataset").apply(lambda d: int((d.groupby("group").size() > 1).sum())).to_dict())
