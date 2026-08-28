"""
Train ParamCNN on CAMELS Mtot maps.

Sanity check first (train loss should collapse toward 0; if it can't, stop and
debug -- the bug is in the data or the loss, not the architecture):
    python train.py --overfit 10 --epochs 200 --downsample 4 \
                    --no_augment --dropout 0

Fast iteration on 64x64:
    python train.py --downsample 4 --widths 32 64 128 256 --epochs 60

Full run:
    python train.py --epochs 150 --batch_size 64

Then the uncertainty-aware version:
    python train.py --head moment --epochs 150 --batch_size 128
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from camels_data import make_loaders
from model import ParamCNN, get_loss
from preprocess import PARAM_NAMES, denormalise_params

# The 8 dihedral symmetries, applied at test time and averaged (test-time
# augmentation).  Free accuracy: the parameters are invariant under all of them.
TTA = [(k, f) for k in range(4) for f in (False, True)]


class ConstantLR:
    """Minimal stand-in with the two methods the loop uses."""

    def __init__(self, opt, lr):
        self.lr = lr

    def step(self):
        pass

    def get_last_lr(self):
        return [self.lr]


def pick_device(arg):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def metrics(y_true, y_pred):
    """
    Per-parameter RMSE and R^2, in normalised [0,1] units.

    R^2 is measured against the spread of the true labels in THIS set.  On a
    small pilot that spread can be tiny -- with one validation simulation all
    15 maps carry the same label, the denominator is 0, and R^2 is meaningless
    rather than merely bad.  Return NaN there instead of a huge negative
    number that looks like a catastrophic model.
    """
    err = y_pred - y_true
    rmse = np.sqrt((err ** 2).mean(axis=0))
    ss_res = (err ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 1e-8 * max(len(y_true), 1),
                      1.0 - ss_res / ss_tot, np.nan)
    return rmse, r2


def fmt_table(rmse, r2, rmse_phys=None):
    head = f"  {'param':9s} {'RMSE':>8s} {'R^2':>8s}"
    if rmse_phys is not None:
        head += f" {'RMSE(phys)':>11s}"
    rows = [head]
    for i, n in enumerate(PARAM_NAMES):
        r = f"  {n:9s} {rmse[i]:8.4f} {r2[i]:8.4f}"
        if rmse_phys is not None:
            r += f" {rmse_phys[i]:11.4f}"
        rows.append(r)
    return "\n".join(rows)


@torch.no_grad()
def evaluate(net, loader, loss_fn, device, tta=False):
    net.eval()
    tot, n = 0.0, 0
    mus, sigs, ys = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = net(x)
        tot += loss_fn(out, y).item() * x.shape[0]
        n += x.shape[0]
        if tta:
            acc_mu, acc_var = 0.0, 0.0
            for k, flip in TTA:
                xa = torch.rot90(x, k, dims=(2, 3))
                if flip:
                    xa = torch.flip(xa, dims=(3,))
                mu, sig = net.predict(xa)
                acc_mu = acc_mu + mu
                acc_var = acc_var + sig ** 2
            mu, sig = acc_mu / len(TTA), torch.sqrt(acc_var / len(TTA))
        else:
            mu, sig = net.predict(x)
        mus.append(mu.float().cpu().numpy())
        sigs.append(sig.float().cpu().numpy())
        ys.append(y.cpu().numpy())
    return (tot / max(n, 1), np.concatenate(mus), np.concatenate(sigs),
            np.concatenate(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default="preprocessing_Mtot_IllustrisTNG.npz")
    ap.add_argument("--head", choices=["mse", "moment"], default="mse")
    ap.add_argument("--widths", type=int, nargs="+",
                    default=[32, 64, 128, 256, 256, 256])
    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--downsample", type=int, default=1)
    ap.add_argument("--roll", action="store_true")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--overfit", type=int, default=0,
                    help="train on only N simulations (memorisation sanity check)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: a handful of sims per split, "
                         "3 maps each, 64px, 3 epochs.  Proves the wiring, "
                         "tells you nothing about accuracy.")
    ap.add_argument("--fraction", type=float, default=0.0,
                    help="PILOT RUN: use this fraction of every split, e.g. "
                         "0.1 for a tenth of the data.  Same normalisation, "
                         "same split boundaries, nested subsets -- so results "
                         "are directly comparable as you scale it up.")
    ap.add_argument("--limit_sims", type=int, nargs=3, default=None,
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="keep only N sims of each split (0 = all)")
    ap.add_argument("--max_maps_per_sim", type=int, default=0,
                    help="keep only the first M of the 15 maps per simulation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.smoke:
        # Everything small enough that the whole run is a couple of minutes on
        # a CPU.  Overridden by anything you pass explicitly after --smoke.
        given = set(a.lstrip("-") for a in __import__("sys").argv if a.startswith("--"))
        preset = {"limit_sims": [8, 4, 4], "max_maps_per_sim": 3, "epochs": 3,
                  "downsample": 4, "widths": [32, 64, 128, 256],
                  "batch_size": 8, "num_workers": 0, "patience": 10 ** 6}
        for k, v in preset.items():
            if k not in given:
                setattr(args, k, v)
        args.tag = args.tag or "smoke"
        print("*** SMOKE TEST: tiny subset, few epochs.  This checks that the "
              "pipeline RUNS, not that it works. ***")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    tag = args.tag or f"{args.head}_d{args.downsample}_{time.strftime('%m%d_%H%M')}"
    run = os.path.join(args.out_dir, tag)
    os.makedirs(run, exist_ok=True)
    print(f"device={device}  run={run}")

    # ------------------------------------------------------------- data
    limit = args.limit_sims
    if args.fraction:
        assert 0 < args.fraction <= 1, "--fraction must be in (0, 1]"
        n_sim = [len(np.asarray(np.load(args.prep, allow_pickle=True)[f"{s}_sims"]))
                 for s in ("train", "val", "test")]
        limit = [max(1, int(round(args.fraction * n))) for n in n_sim]
        print(f"pilot run: {args.fraction:.0%} of each split -> "
              f"{limit[0]}/{limit[1]}/{limit[2]} sims "
              f"(of {n_sim[0]}/{n_sim[1]}/{n_sim[2]})")
    if args.overfit:                       # cap val/test too, they are noise here
        limit = [args.overfit, args.overfit, args.overfit]
    train_ld, val_ld, test_ld = make_loaders(
        args.prep, batch_size=args.batch_size, num_workers=args.num_workers,
        downsample=args.downsample, roll=args.roll,
        augment_train=not args.no_augment,
        pin_memory=(device.type == "cuda"),
        limit_sims=limit, max_maps_per_sim=args.max_maps_per_sim)
    n_sims = {s: len(np.unique(ld.dataset.index[:, 0]))
              for s, ld in (("train", train_ld), ("val", val_ld), ("test", test_ld))}
    print(f"maps  train={len(train_ld.dataset)}  val={len(val_ld.dataset)}  "
          f"test={len(test_ld.dataset)}   "
          f"(sims {n_sims['train']}/{n_sims['val']}/{n_sims['test']})")
    for s in ("val", "test"):
        if n_sims[s] < 10:
            print(f"  WARNING: only {n_sims[s]} {s} simulations. R^2 is measured "
                  f"against the label spread in that set, so it will be very "
                  f"noisy (NaN if there is only one). Trust RMSE here, or "
                  f"raise --fraction.")
    if args.overfit:
        args.patience = args.epochs + 1     # watch TRAIN loss here, not val
        print("overfit mode: early stopping disabled -- watch the TRAIN loss "
              "collapse toward 0.  Val/test numbers here are meaningless.")
        if args.dropout or not args.no_augment:
            print("  tip: use --dropout 0 --no_augment so nothing fights the "
                  "memorisation you are testing for.")

    npx = 256 // args.downsample
    if npx // 2 ** len(args.widths) < 1:
        raise SystemExit(f"{len(args.widths)} stride-2 blocks is too many for "
                         f"{npx}px input -- pass fewer --widths")

    # ------------------------------------------------------------ model
    net = ParamCNN(in_channels=args.in_channels, head=args.head,
                   widths=tuple(args.widths), dropout=args.dropout).to(device)
    loss_fn = get_loss(args.head)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"model: {len(args.widths)} blocks, {npx}px -> "
          f"{npx // 2 ** len(args.widths)}px, {n_par / 1e6:.2f}M params")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * max(len(train_ld), 1)
    if total_steps >= 40:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=total_steps,
            pct_start=min(0.3, max(0.1, 5.0 / total_steps)))
    else:
        # OneCycle needs a warmup phase at least a couple of steps long; on a
        # tiny run (smoke test, --overfit with few maps) just hold the LR flat.
        sched = ConstantLR(opt, args.lr)

    # ------------------------------------------------------------ train
    best, best_ep, hist = np.inf, -1, []
    ckpt = os.path.join(run, "best.pt")
    for ep in range(args.epochs):
        net.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        tr = tot / max(n, 1)

        vl, mu, _, yv = evaluate(net, val_ld, loss_fn, device)
        rmse, r2 = metrics(yv, mu)
        hist.append({"epoch": ep, "train": tr, "val": vl,
                     "val_rmse": rmse.tolist(), "val_r2": r2.tolist(),
                     "lr": sched.get_last_lr()[0]})

        flag = ""
        if vl < best:
            best, best_ep, flag = vl, ep, "  *"
            torch.save({"model": net.state_dict(), "args": vars(args),
                        "epoch": ep, "val_loss": vl}, ckpt)
        print(f"ep {ep:3d}  train {tr:10.5f}  val {vl:10.5f}  "
              f"R2(Om,s8)={r2[0]:.3f},{r2[1]:.3f}  {time.time()-t0:5.1f}s{flag}")

        if ep - best_ep >= args.patience:
            print(f"early stop (no val improvement for {args.patience} epochs)")
            break

    with open(os.path.join(run, "history.json"), "w") as f:
        json.dump(hist, f, indent=1)

    # ------------------------------------------------------------- test
    net.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    print(f"\nbest epoch {best_ep} (val {best:.5f})")

    log_astro = bool(np.load(args.prep, allow_pickle=True)["log_astro"])
    for use_tta in (False, True):
        tl, mu, sig, yt = evaluate(net, test_ld, loss_fn, device, tta=use_tta)
        rmse, r2 = metrics(yt, mu)
        phys = np.sqrt(((denormalise_params(mu, log_astro)
                         - denormalise_params(yt, log_astro)) ** 2).mean(axis=0))
        print(f"\nTEST per map{' + TTA(x8)' if use_tta else ''}  loss={tl:.5f}")
        print(fmt_table(rmse, r2, phys))
        if use_tta:
            # One prediction per simulation: average the 15 maps.  This is the
            # number to quote if the question is "given a simulation", and it
            # should beat the per-map result noticeably.
            sims = test_ld.dataset.index[:, 0]
            uniq = np.unique(sims)
            mu_s = np.stack([mu[sims == s].mean(0) for s in uniq])
            y_s = np.stack([yt[sims == s][0] for s in uniq])
            r_s, r2_s = metrics(y_s, mu_s)
            print(f"\nTEST per simulation (mean of 15 maps, n={len(uniq)})")
            print(fmt_table(r_s, r2_s))
            np.savez(os.path.join(run, "test_predictions.npz"),
                     mu=mu, sigma=sig, y=yt, sims=sims,
                     mu_sim=mu_s, y_sim=y_s, param_names=np.array(PARAM_NAMES))

    if args.head == "moment":
        z = (mu - yt) / np.maximum(sig, 1e-8)
        print("\ncalibration: std of (pred - true)/sigma per param, "
              "should be ~1.0")
        print("  " + "  ".join(f"{n}={z[:, i].std():.2f}"
                               for i, n in enumerate(PARAM_NAMES)))

    print(f"\nsaved -> {run}/  (best.pt, history.json, test_predictions.npz)")


if __name__ == "__main__":
    main()
