"""Train the Redline timing model from a YAML config.

Single GPU. bf16 autocast, cosine LR + warmup (Muon on the 2-D trunk, AdamW on heads),
periodic validation on a held-out set. Step-based loop over the streaming .osz dataset.

  python -m redline.train configs/redline_v1.yaml
  python -m redline.train configs/redline_v1.yaml --set steps=600 eval_interval=150

Any config key can be overridden on the command line with --set key=value.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from redline.muon_utils import Muon          # noqa: E402
from redline.dataset_osz import OszTimingDataset, list_osz, FRAME_MS  # noqa: E402
from redline.model import TimingModel                 # noqa: E402
from redline.losses import timing_loss, beat_f1       # noqa: E402
from redline.targets import LOG_MPB_CENTER, BPM_MIN, N_TEMPO_BINS  # noqa: E402


def infinite(dl):
    while True:
        for b in dl:
            yield b


def build_optimizer(args, muon_params, adamw_params):
    """muon (default): Muon on the 2-D trunk weights + AdamW on heads/frontend.
    adamw: a single AdamW over all params at args.adamw_lr."""
    name = args.optimizer.lower()
    if name == "muon":
        return Muon(muon_params=muon_params, lr=args.muon_lr, adamw_params=adamw_params,
                    adamw_lr=args.adamw_lr, adamw_wd=args.wd)
    if name == "adamw":
        all_params = list(muon_params) + list(adamw_params)
        return torch.optim.AdamW(all_params, lr=args.adamw_lr, weight_decay=args.wd)
    raise ValueError(f"unknown optimizer {args.optimizer}")


def lr_scale(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def evaluate(model, val_dl, dev, n_batches=12):
    model.eval()
    fb = fd = mae = cf1 = tmae = oct_ok = rf1 = tot = 0.0
    for i, b in enumerate(val_dl):
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["mel"])
        out = {k: v.float() for k, v in out.items()}
        fb += beat_f1(out["beat"], b["beat"], None)
        fd += beat_f1(out["down"], b["down"], None)
        om = b["mask"]
        mae += float((((out["off"] - b["off"]).abs() * om).sum() / om.sum().clamp_min(1)) * FRAME_MS)
        if "red" in out and "red" in b:
            rf1 += beat_f1(out["red"], b["red"], None)
        if "change" in out and "change" in b:
            cf1 += beat_f1(out["change"], b["change"], None)
            tm = b["tmask"]
            p = torch.softmax(out["tempo"], dim=-1)                       # [B,F,N_BINS]
            bins = torch.arange(N_TEMPO_BINS, device=p.device).float()
            pbpm = BPM_MIN + (p * bins).sum(-1)                           # soft-argmax -> BPM
            tbpm = 60000.0 / torch.exp(b["tempo"] + LOG_MPB_CENTER)
            tmae += float((((pbpm - tbpm).abs() * tm).sum() / tm.sum().clamp_min(1)))
            abpm = BPM_MIN + p.argmax(-1).float()                         # hard-argmax BPM
            ok = ((abpm - tbpm).abs() <= 0.04 * tbpm).float()
            hard = tm * ((tbpm < 130) | (tbpm > 200)).float()             # sparse-band frames
            oct_ok += float((ok * hard).sum() / hard.sum().clamp_min(1))
        tot += 1
        if i + 1 >= n_batches:
            break
    model.train()
    return fb / tot, fd / tot, mae / tot, cf1 / tot, tmae / tot, oct_ok / tot, rf1 / tot


def build_dense_index(root, paths):
    """{osz_path: redline count} for the corpus — used by hard-oversampling. Built once by
    scanning each mapset's .osu timing points; cached in ~/.cache/redline keyed by corpus."""
    import hashlib, json, zipfile
    from redline.osu_format import _gt_from_text
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "redline")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"dense_index_{hashlib.md5(root.encode()).hexdigest()[:12]}.json")
    if os.path.exists(cache):
        idx = json.load(open(cache))
        if len(idx) >= 0.9 * len(paths):          # corpus unchanged enough to reuse
            return idx
    print(f"building dense-song index for {len(paths)} mapsets (one-time, cached)...", flush=True)
    idx = {}
    for i, p in enumerate(paths):
        try:
            best = 0
            with zipfile.ZipFile(p) as z:
                for n in (m for m in z.namelist() if m.lower().endswith(".osu")):
                    R, *_ = _gt_from_text(z.read(n).decode("utf-8-sig", "ignore"))
                    best = max(best, len(R))
            idx[p] = best
        except Exception:
            pass
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    json.dump(idx, open(cache, "w"))
    print(f"dense index cached -> {cache}", flush=True)
    return idx


DEFAULTS = dict(
    # data
    root="data/osz",                        # training corpus: a directory of .osz archives
    val_root="",                            # held-out val dir (song-deduped vs root)
    val_sets=400,
    min_year=0,
    genre_balanced=False,                   # genre-balanced sampling (needs genre_json)
    genre_json="",
    genre_alpha=0.5,
    hard_oversample=1,                      # oversample sets with >= hard_min_redlines
    hard_min_redlines=4,
    hard_oversample2=1,                     # second tier for live-dense sets
    hard_min_redlines2=16,
    length_aware=False,
    sample_alpha=0.65,
    unique_crops=False,
    crops=6,                                # crops per song per epoch
    chunk_frames=1500,                      # crop length in 20ms frames (1500 = 30s)
    # augmentation
    speed_aug=False,
    speed_factors=[0.8, 0.9, 1.1, 1.2],
    speed_aug_prob=0.5,
    degrade_aug=False,                      # mel-domain onset degradation (soft-onset robustness)
    degrade_prob=0.5,
    # representation
    sr=22050, n_fft=1024, n_mels=128, fmax=0.0,
    onset=True,                             # append the onset-flux channel to the mel
    # model
    d_model=512, depth=14, n_heads=8, ffn=2048,
    sum_head=False,
    red_head=False,                         # notation head (ALL-redline supervision)
    conv_v2=True, change_head_v2=True,      # fixed architecture (kept in saved args)
    # loss
    w_down=1.0, w_tempo=1.0, w_change=1.0, w_sub=0.0,
    w_red=1.0, pos_weight_red=15.0,
    # optimization
    optimizer="muon", muon_lr=8e-3, adamw_lr=1.5e-3, wd=0.01,
    warmup=1000, steps=40000, batch=16, accum=1, compile=False,
    # run
    ckpt="checkpoints/run",                 # output dir (ckpts, metrics.csv, tb/)
    init="",                                # warm-start weights (strict=False)
    resume="",                              # resume model+optimizer+step
    snapshot_every=0,
    eval_interval=1000, log_interval=50,
    workers=8, prefetch=1,
    wandb=False, wandb_project="redline", wandb_entity="", wandb_name="",
)


def load_config(path, overrides):
    """YAML config over DEFAULTS, then --set key=value overrides (typed by the default)."""
    import yaml
    cfg = dict(DEFAULTS)
    user = yaml.safe_load(open(path)) or {}
    unknown = set(user) - set(cfg)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    cfg.update(user)
    for ov in overrides:
        k, _, v = ov.partition("=")
        if k not in cfg:
            raise SystemExit(f"unknown --set key: {k}")
        d = DEFAULTS[k]
        if isinstance(d, bool):
            cfg[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(d, int):
            cfg[k] = int(v)
        elif isinstance(d, float):
            cfg[k] = float(v)
        elif isinstance(d, list):
            cfg[k] = [float(x) for x in v.split(",") if x.strip()]
        else:
            cfg[k] = v
    import types
    return types.SimpleNamespace(**cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML training config (see configs/)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override config keys, e.g. --set steps=600 batch=8")
    cli = ap.parse_args()
    args = load_config(cli.config, cli.set)

    dev = "cuda"
    os.makedirs(args.ckpt, exist_ok=True)

    # tensor-core matmul precision (speeds up compiled training on Ampere+)
    if args.compile:
        torch.set_float32_matmul_precision('high')

    import redline.dataset_osz as _ds
    _ds.SR = args.sr; _ds.N_FFT = args.n_fft; _ds.N_MELS = args.n_mels
    _ds.FMAX = (args.fmax or None); _ds.HOP = args.sr // 50; _ds.ONSET = args.onset
    _ds.FRAME_MS = _ds.HOP / _ds.SR * 1000.0
    model_n_mels = args.n_mels * (2 if args.onset else 1)   
    print(f"REP: sr={args.sr} n_fft={args.n_fft} n_mels={args.n_mels} fmax={args.fmax or 'sr/2'} "
          f"hop={_ds.HOP} frame_ms={_ds.FRAME_MS:.1f}", flush=True)

    paths = list_osz(args.root)
    if args.val_root:
        import glob as _glob, re as _re2, random as _rnd
        def _songkey(p):
            nm = _re2.sub(r"\.osz$", "", os.path.basename(p), flags=_re2.I)
            m = _re2.match(r"^\s*\d+\s+(.*)$", nm)
            return (m.group(1) if m else nm).strip().lower()
        train_keys = set(_songkey(p) for p in paths)
        vp = sorted(_glob.glob(os.path.join(args.val_root, "**", "*.osz"), recursive=True))
        vp_dd = [p for p in vp if _songkey(p) not in train_keys]
        _rnd.Random(1234).shuffle(vp_dd)
        val_paths = vp_dd[:args.val_sets] if args.val_sets else vp_dd
        train_paths = paths  # all of --root is training; val is the held-out set
        print(f"VAL from {args.val_root}: {len(vp)} maps, {len(vp)-len(vp_dd)} song-dupes excl vs "
              f"--root, using {len(val_paths)}; train = all {len(paths)} of --root", flush=True)
    else:
        val_paths = paths[-args.val_sets:]
        train_paths = paths[:-args.val_sets]
    if getattr(args, "min_year", 0):
        import re as _re
        def _yr(p):
            m = _re.search(r"/(20\d\d) \(osu", p)
            return int(m.group(1)) if m else 0
        before = len(train_paths)
        train_paths = [p for p in train_paths if _yr(p) >= args.min_year]
        print(f"min_year>={args.min_year}: {before} -> {len(train_paths)} train paths", flush=True)
    if args.hard_oversample > 1 or args.hard_oversample2 > 1:
        dense = build_dense_index(args.root, paths)
        valset = set(val_paths)
        if args.hard_oversample > 1:
            hard = [p for p, n in dense.items() if n >= args.hard_min_redlines and p not in valset]
            train_paths = train_paths + hard * (args.hard_oversample - 1)
            print(f"hard-oversample x{args.hard_oversample}: +{len(hard)*(args.hard_oversample-1)} "
                  f"dense sets (>={args.hard_min_redlines} redlines)")
        if args.hard_oversample2 > 1:
            hard2 = [p for p, n in dense.items() if n >= args.hard_min_redlines2 and p not in valset]
            train_paths = train_paths + hard2 * (args.hard_oversample2 - 1)
            print(f"hard-oversample2 x{args.hard_oversample2}: +{len(hard2)*(args.hard_oversample2-1)} "
                  f"LIVE-dense sets (>={args.hard_min_redlines2} redlines)")
    print(f"{len(train_paths)} train sets, {len(val_paths)} val sets")

    # genre-balanced sampling: {genre: [train paths]} built from the setid->genre map.
    genre_buckets = None
    if args.genre_balanced:
        import json as _json, re as _re
        gmap = {int(k): v for k, v in _json.load(open(args.genre_json)).items()}
        def _sid(p):
            m = _re.match(r"^\s*(\d+)\s", os.path.basename(p))
            return int(m.group(1)) if m else None
        from collections import defaultdict as _dd
        genre_buckets = _dd(list)
        miss = 0
        for p in train_paths:
            g = gmap.get(_sid(p) or -1)
            (genre_buckets[g].append(p) if g else None)
            miss += g is None
        genre_buckets = dict(genre_buckets)
        sizes = {g: len(v) for g, v in sorted(genre_buckets.items(), key=lambda x: -len(x[1]))}
        print(f"genre-balanced ON (alpha={args.genre_alpha}): {sizes}  [no-genre dropped: {miss}]", flush=True)

    speed_factors = [float(x) for x in args.speed_factors] if args.speed_aug else []
    if args.speed_aug:
        print(f"speed-aug ON: factors={speed_factors} prob={args.speed_aug_prob}", flush=True)
    if args.length_aware:
        print(f"length-aware sampling ON: k=round({args.sample_alpha}*dur/30s)", flush=True)

    _ds.CHUNK_FRAMES = args.chunk_frames
    train_ds = OszTimingDataset(train_paths, crops_per_song=args.crops, seed=1, gain_aug=True,
                                chunk_frames=args.chunk_frames,
                                speed_aug=args.speed_aug, speed_factors=speed_factors,
                                speed_aug_prob=args.speed_aug_prob,
                                degrade_aug=args.degrade_aug, degrade_prob=args.degrade_prob,
                                genre_buckets=genre_buckets, genre_alpha=args.genre_alpha,
                                length_aware=args.length_aware, sample_alpha=args.sample_alpha,
                                unique_crops=args.unique_crops)
    val_ds = OszTimingDataset(val_paths, crops_per_song=1, seed=999, shuffle=False, gain_aug=False,
                              speed_aug=False, chunk_frames=args.chunk_frames)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch, num_workers=args.workers,
                          persistent_workers=True, prefetch_factor=args.prefetch, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, num_workers=2)

    model = TimingModel(n_mels=model_n_mels, d_model=args.d_model, depth=args.depth,
                        n_heads=args.n_heads, ffn=args.ffn, sum_head=args.sum_head,
                        red_head=args.red_head).to(dev)
    print(model)
    if args.init:
        ck = torch.load(args.init, map_location=dev)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"warm-started from {args.init}: {len(missing)} new params (heads) kept random, {len(unexpected)} unexpected")
        
    muon, adamw = model.param_groups()
    opt = build_optimizer(args, muon, adamw)
    print(f"model {sum(p.numel() for p in model.parameters())/1e6:.1f}M | "
          f"Muon/2D {sum(p.numel() for p in muon)/1e6:.1f}M / AdamW {sum(p.numel() for p in adamw)/1e6:.1f}M")

    csv_path = os.path.join(args.ckpt, "metrics.csv")
    new = not os.path.exists(csv_path)
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if new:
        csv_w.writerow(["step", "type", "lr", "loss", "beat", "down", "off",
                        "val_beat_f1", "val_down_f1", "val_off_mae", "wall_s"])
        csv_f.flush()
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(args.ckpt, "tb"))
    except Exception as ex:
        print(f"(tensorboard unavailable: {ex})")
    wb = None
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=(args.wandb_name or os.path.basename(os.path.normpath(args.ckpt))),
                       config=vars(args))
            wb = wandb
            print(f"wandb logging -> {args.wandb_entity}/{args.wandb_project}", flush=True)
        except Exception as ex:
            print(f"(wandb unavailable, continuing without it: {ex})", flush=True)

    model.train()
    it = infinite(train_dl)
    run = {}
    best_f1 = -1.0
    best_oct = -1.0
    best_down = -1.0
    best_off = 1e9
    best_comp = -1.0
    best_red = -1.0
    start_step = 0
    
    # 1. LOAD RESUME WEIGHTS INTO THE RAW MODEL FIRST
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            try:
                opt.load_state_dict(ck["opt"])
                print("restored optimizer state")
            except Exception as e:
                print(f"(optimizer state not restored, fresh momentum: {e})")
        start_step = int(ck.get("step", 0))
        if ck.get("val"):
            best_f1 = ck["val"].get("beat_f1", -1.0)
        print(f"RESUMED from {args.resume}: step {start_step}, best_f1 {best_f1:.3f}")

    # --- ADDED: 2. COMPILE AFTER ALL WEIGHTS ARE LOADED ---
    # We assign it to `train_model`. We will use `train_model` for forward passes, 
    # but `model.state_dict()` for saving, preventing '_orig_mod' key corruption.
    if args.compile:
        print("Compiling model (this might take a minute on the first step)...", flush=True)
        train_model = torch.compile(model)
    else:
        train_model = model

    t0 = time.time()
    # AMUSE owns its LR schedule (linear warmup + implicit z-averaging decay) and requires
    # explicit train/eval mode calls (it keeps params in y-space while training and swaps to
    # x-space for eval/save). Muon/AdamW use the external cosine schedule + per-step lr set.
    for step in range(start_step + 1, args.steps + 1):
        sc = lr_scale(step, args.warmup, args.steps)
        for g in opt.param_groups:
            if "lr_base" not in g:
                g["lr_base"] = g["lr"]      # cache the base LR from the optimizer ctor
            g["lr"] = g["lr_base"] * sc
        else:
            sc = 1.0  # placeholder for log lines; AMUSE reports its own lr below
        opt.zero_grad(set_to_none=True)
        loss = None
        
        for _ in range(args.accum):
            b = next(it)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # --- MODIFIED: Use train_model for the forward pass ---
                out = train_model(b["mel"])
                
            mloss, parts = timing_loss({k: v.float() for k, v in out.items()}, b,
                                       w_tempo=args.w_tempo, w_change=args.w_change, w_sub=args.w_sub,
                                       w_down=args.w_down,
                                       w_red=args.w_red, pos_weight_red=args.pos_weight_red)
            (mloss / args.accum).backward()
            loss = mloss if loss is None else loss
            for k, v in parts.items():
                run[k] = 0.9 * run.get(k, float(v)) + 0.1 * float(v)
                
        # --- NOTE: Raw model.parameters() is fully safe here because compilation 
        #           doesn't detach or replace the physical parameter tensors. ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        cur_lr = float(opt.param_groups[0]["lr"])   # accurate for all opts (AMUSE sets its own)

        if step % args.log_interval == 0:
            wall = time.time() - t0
            sps = (step - start_step) / wall
            change_val = run.get("change", 0.0)
            red_str = f" red {run['red']:.3f}" if "red" in run else ""
            print(f"step {step:>6} | lr {cur_lr:.2e} | loss {float(loss):.3f} "
                f"(beat {run['beat']:.3f} down {run['down']:.3f} off {run['off']:.3f} change {change_val:.3f}{red_str}) | {sps:.1f} it/s",
                flush=True)
            csv_w.writerow([step, "train", f"{cur_lr:.3e}", f"{float(loss):.4f}",
                            f"{run['beat']:.4f}", f"{run['down']:.4f}", f"{run['off']:.4f}",
                            "", "", "", f"{wall:.0f}"])
            csv_f.flush()
            if tb:
                tb.add_scalar("train/loss", float(loss), step)
                for k in ("beat", "down", "off", "red"):
                    if k in run:
                        tb.add_scalar(f"train/{k}", run[k], step)
                tb.add_scalar("lr", cur_lr, step)
            if wb:
                wb_dict = {
                    "train/loss": float(loss),
                    "train/beat": run["beat"],
                    "train/down": run["down"],
                    "train/off": run["off"],
                    "lr": cur_lr
                }
                if "change" in run:
                    wb_dict["train/change"] = run["change"]
                if "red" in run:
                    wb_dict["train/red"] = run["red"]
                wb.log(wb_dict, step=step)

        if step % args.eval_interval == 0:
            # AMUSE keeps params in y-space while training; swap to x-space for eval + save
            # so the checkpoint and the eval forward-pass see the averaged weights.
            fb, fd, mae, cf1, tmae, oct_ok, rf1 = evaluate(train_model, val_dl, dev)
            red_str = f"  red_F1 {rf1:.3f}" if args.red_head else ""
            print(f"  ── VAL step {step}: beat_F1 {fb:.3f}  down_F1 {fd:.3f}  off_MAE {mae:.1f}ms"
                  f"  change_F1 {cf1:.3f}  tempo_MAE {tmae:.1f}BPM  octave {oct_ok:.3f}{red_str}", flush=True)
            csv_w.writerow([step, "val", "", "", "", "", "", f"{fb:.4f}", f"{fd:.4f}", f"{mae:.2f}",
                            f"{time.time()-t0:.0f}"])
            csv_f.flush()
            if tb:
                tb.add_scalar("val/beat_f1", fb, step)
                tb.add_scalar("val/down_f1", fd, step)
                tb.add_scalar("val/off_mae_ms", mae, step)
                tb.add_scalar("val/change_f1", cf1, step)
                if args.red_head:
                    tb.add_scalar("val/red_f1", rf1, step)
                tb.add_scalar("val/tempo_mae_bpm", tmae, step)
                tb.add_scalar("val/octave_ok", oct_ok, step)
            if wb:
                wb_val = {"val/beat_f1": fb, "val/down_f1": fd, "val/off_mae_ms": mae,
                          "val/change_f1": cf1, "val/tempo_mae_bpm": tmae,
                          "val/octave_ok": oct_ok}
                if args.red_head:
                    wb_val["val/red_f1"] = rf1
                wb.log(wb_val, step=step)
                
            # --- NOTE: Saving `model.state_dict()` (the uncompiled reference). 
            #           This completely avoids the `_orig_mod.` prefix bug! ---
            ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "args": vars(args), "val": {"beat_f1": fb, "down_f1": fd, "off_mae": mae,
                                                "octave_ok": oct_ok, "red_f1": rf1}}
            torch.save(ckpt, os.path.join(args.ckpt, "last.pt"))
            if fb > best_f1:
                best_f1 = fb
                torch.save(ckpt, os.path.join(args.ckpt, "best.pt"))
                print(f"     (new best beat_F1 {fb:.3f} -> best.pt)", flush=True)
            if oct_ok > best_oct:
                best_oct = oct_ok
                torch.save(ckpt, os.path.join(args.ckpt, "best_octave.pt"))
                print(f"     (new best octave {oct_ok:.3f} -> best_octave.pt)", flush=True)
            if fd > best_down:
                best_down = fd
                torch.save(ckpt, os.path.join(args.ckpt, "best_down.pt"))
                print(f"     (new best down_F1 {fd:.3f} -> best_down.pt)", flush=True)
            if args.red_head and rf1 > best_red:
                best_red = rf1
                torch.save(ckpt, os.path.join(args.ckpt, "best_red.pt"))
                print(f"     (new best red_F1 {rf1:.3f} -> best_red.pt)", flush=True)
            if mae < best_off:
                best_off = mae
                torch.save(ckpt, os.path.join(args.ckpt, "best_off.pt"))
                print(f"     (new best off_MAE {mae:.2f}ms -> best_off.pt)", flush=True)
            # composite "all-around best" = beat + down + octave (offset saturated, low-variance).
            # Guards against the trap where beat_f1-only best.pt selects a beat-saturated checkpoint
            # while down/octave are past their sweet spot.
            comp = fb + fd + oct_ok
            if comp > best_comp:
                best_comp = comp
                torch.save(ckpt, os.path.join(args.ckpt, "best_composite.pt"))
                print(f"     (new best composite {comp:.3f} -> best_composite.pt)", flush=True)
            if args.snapshot_every > 0 and step % args.snapshot_every == 0:
                torch.save(ckpt, os.path.join(args.ckpt, f"step{step}.pt"))


if __name__ == "__main__":
    main()