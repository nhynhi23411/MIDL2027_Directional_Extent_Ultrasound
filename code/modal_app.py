import modal, sys, json
app = modal.App("midl2027-extent")
vol = modal.Volume.from_name("midl-data", create_if_missing=True)
img = (modal.Image.debian_slim(python_version="3.12")
       .pip_install("torch", "torchvision", "segmentation-models-pytorch", "albumentations", "opencv-python-headless", "scipy", "pandas", "numpy")
       .add_local_file("E:/MIDL_2/code/losses.py", "/root/losses.py").add_local_file("E:/MIDL_2/code/metrics.py", "/root/metrics.py"))

@app.function(image=img, gpu=["L4", "T4", "A10G"], volumes={"/data": vol}, timeout=7200, cpu=4, memory=16384)
def run(cfg: dict):
    import pickle, time, numpy as np, pandas as pd, torch, cv2, albumentations as A, segmentation_models_pytorch as smp
    from losses import Combined
    from metrics import feret, largest_component, dice, boundary_f1
    seed = cfg["seed"]; torch.manual_seed(seed); np.random.seed(seed)
    recs = pickle.load(open(f"/data/{cfg['dataset']}.pkl", "rb")); dev = "cuda"
    train = [r for r in recs if r["split"] == "train" and (r["orig_mask"].any() or cfg.get("neg"))]
    aug = A.Compose([A.HorizontalFlip(), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.7),
                     A.RandomBrightnessContrast(0.2, 0.2, p=0.5)])
    class DS(torch.utils.data.Dataset):
        def __len__(s): return len(train)
        def __getitem__(s, i):
            o = aug(image=train[i]["img"], mask=train[i]["mask"]); return torch.from_numpy(o["image"]).float()[None] / 255, torch.from_numpy(o["mask"]).float()[None]
    dl = torch.utils.data.DataLoader(DS(), batch_size=16, shuffle=True, num_workers=4, drop_last=True, worker_init_fn=lambda w: np.random.seed(seed * 100 + w))
    mean, std = 0.449, 0.226
    kw = dict(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    net = (smp.Unet(**kw) if cfg["model"] == "unet" else smp.DeepLabV3Plus(**kw)).to(dev)
    crit = Combined(cfg["loss"], lam=cfg["lam"], K=cfg["K"], tau=0.02, boundary_w=cfg["lam"], lam_n=cfg.get("lam_n", 0.0)).to(dev)
    E = cfg["epochs"]; opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=E * len(dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(); t0 = time.time()
    fwd = lambda x: net(((x - mean) / std).repeat(1, 3, 1, 1))
    for ep in range(E):
        net.train(); te = time.time()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", dtype=torch.float16): lg = fwd(x)
            loss = crit(lg.float(), y); opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        if ep % 10 == 0 or ep == E - 1: print(cfg['tag'], 'ep', ep, 'loss', round(float(loss), 4), 'ep_secs', round(time.time() - te, 1), flush=True)
    net.eval(); rows = []; probs = {}
    for r in recs:
        if r["split"] not in ("dev", "test"): continue
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(fwd(torch.from_numpy(r["img"]).float()[None, None].to(dev) / 255).float())[0, 0].cpu().numpy()
        if cfg.get("dump") and r["split"] == "test": probs[r["id"]] = (p * 255).round().astype("uint8")
        t, l, nh, nw = r["pad"]; H, W = r["shape"]; p = cv2.resize(p[t:t + nh, l:l + nw], (W, H), interpolation=cv2.INTER_LINEAR)
        pm = p > 0.5; lc = largest_component(pm) if pm.any() else pm; ref = r["orig_mask"]
        row = dict(dataset=cfg["dataset"], model=cfg["model"], loss=cfg["loss"], lam=cfg["lam"], lam_n=cfg.get("lam_n", 0.0), K=cfg["K"], seed=seed, split=r["split"], case_id=r["id"], group=r["group"], cls=r["cls"],
                   pos=int(ref.any()), pred_area_frac=float(pm.mean()), pred_any=int(pm.any()), h=H, w=W)
        if ref.any():
            d = feret(ref); da, dl_ = feret(pm), feret(lc)
            row.update(ref_feret=d, feret_all=da, feret_lc=dl_, rel_err=abs(dl_ - d) / d, rel_err_all=abs(da - d) / d, signed=(dl_ - d) / d, missed=int(not pm.any()),
                       dice=dice(lc, ref), bf1=boundary_f1(lc, ref))
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(f"/data/results/{cfg['tag']}.csv", index=False) if False else None
    return dict(cfg=cfg, secs=time.time() - t0, rows=rows, probs=probs)

def configs(stage):
    return _configs(stage)
def _configs(stage):
    C = []
    def add(ds, m, loss, lam, K, seed, ep=60): C.append(dict(dataset=ds, model=m, loss=loss, lam=lam, K=K, seed=seed, epochs=ep, tag=f"{stage}_{ds}_{m}_{loss}_l{lam}_K{K}_s{seed}"))
    return C, add

@app.local_entrypoint()
def main(stage: str = "smoke", best: str = ""):
    import pandas as pd, os
    C, add = configs(stage); os.makedirs("E:/MIDL_2/results", exist_ok=True)
    if stage == "smoke": add("busi", "unet", "directional", 1.0, 16, 0, ep=2)
    elif stage == "pilot":
        for ds in ("mmotu", "busi"):
            add(ds, "unet", "dice_bce", 0, 16, 0)
            for lam in (0.1, 0.5, 2.0): add(ds, "unet", "boundary", lam, 16, 0)
            for lam in (0.25, 1.0, 4.0): add(ds, "unet", "bbox", lam, 2, 0); add(ds, "unet", "directional", lam, 16, 0)
    elif stage == "v2pilot":
        for ds in ("mmotu", "busi"):
            ng = ds == "busi"
            if ng:
                add(ds, "unet", "dice_bce", 0, 16, 0)
                for lam in (0.1, 0.5, 2.0): add(ds, "unet", "boundary", lam, 16, 0)
                for lam in (0.25, 1.0, 4.0): add(ds, "unet", "bbox", lam, 2, 0); add(ds, "unet", "directional", lam, 16, 0)
            for lam in (0.25, 1.0, 4.0): add(ds, "unet", "directional2", lam, 16, 0)
        for c in C: c["neg"] = c["dataset"] == "busi"
    elif stage == "v2final":
        B = json.load(open("E:/MIDL_2/results/best_hparams_v2.json"))
        for ds in ("mmotu", "busi"):
            for m in ("unet", "dlv3p"):
                for loss in (("dice_bce", "boundary", "bbox", "directional", "directional2") if ds == "busi" else ("directional2",)):
                    for seed in (0, 1, 2):
                        add(ds, m, loss, 0 if loss == "dice_bce" else B[ds][loss], 2 if loss == "bbox" else 16, seed)
        for c in C: c["neg"] = c["dataset"] == "busi"
    elif stage == "v3pilot":
        for ln in (0.5, 2.0):
            add("busi", "unet", "directional3", 4.0, 16, 0); C[-1]["lam_n"] = ln; C[-1]["tag"] += f"_n{ln}"
            add("busi", "unet", "dice_neg", 0, 16, 0); C[-1]["lam_n"] = ln; C[-1]["tag"] += f"_n{ln}"
        for c in C: c["neg"] = True
    elif stage == "v3final":
        B = json.load(open("E:/MIDL_2/results/best_hparams_v3.json"))
        for m in ("unet", "dlv3p"):
            for seed in (0, 1, 2):
                for loss in ("directional3", "dice_neg"):
                    add("busi", m, loss, 4.0 if loss == "directional3" else 0, 16, seed); C[-1]["lam_n"] = B[loss]; C[-1]["tag"] += f"_n{B[loss]}"
        for c in C: c["neg"] = True
    elif stage == "viz":
        import pickle
        for ds, lam in (("mmotu", 1.0), ("busi", 4.0)):
            add(ds, "unet", "dice_bce", 0, 16, 0); add(ds, "unet", "directional", lam, 16, 0)
        for c in C: c["dump"] = True
        calls = [run.spawn(c) for c in C]; V = {}
        for c, call in zip(C, calls):
            res = call.get(); V[(c["dataset"], c["loss"])] = dict(rows=res["rows"], probs=res["probs"]); print("DONE", c["tag"], flush=True)
        pickle.dump(V, open("E:/MIDL_2/results/viz_probs.pkl", "wb")); return
    elif stage == "ablation":
        B = json.load(open("E:/MIDL_2/results/best_hparams.json"))
        for ds in ("mmotu", "busi"):
            for K in (4, 8, 32): add(ds, "unet", "directional", B[ds]["directional"], K, 0)
    elif stage == "final":
        B = json.load(open("E:/MIDL_2/results/best_hparams.json"))
        for ds in ("mmotu", "busi"):
            for m in ("unet", "dlv3p"):
                for loss in ("dice_bce", "boundary", "bbox", "directional"):
                    for seed in (0, 1, 2):
                        add(ds, m, loss, 0 if loss == "dice_bce" else B[ds][loss], 2 if loss == "bbox" else 16, seed)
    out = []
    calls = [run.spawn(c) for c in C]
    for c, call in zip(C, calls):
        res = call.get(); out += res["rows"]; print("DONE", res["cfg"]["tag"], round(res["secs"]), flush=True)
        pd.DataFrame(out).to_csv(f"E:/MIDL_2/results/{stage}_per_case_results.csv", index=False)
