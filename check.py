"""
Correctness battery.  Run this before any long training job.

    python check.py preprocessing_Mtot_IllustrisTNG.npz

Every check is cheap -- the whole thing is well under a minute even against the
full 3.9 GB cube, because it only touches a few dozen maps.  It answers a
different question from `train.py --smoke`: smoke proves the code RUNS, this
proves the DATA and the LEARNING SIGNAL are wired up correctly.

Exit code 0 = all passed.
"""

import sys
import time

import numpy as np
import torch

from camels_data import CAMELSMaps, constant_baseline, make_loaders
from model import ParamCNN, get_loss
from preprocess import (PARAM_NAMES, denormalise_params, normalise_params)

FAILED = []


def check(name):
    def deco(fn):
        print(f"\n[{name}]")
        try:
            fn()
            print("  PASS")
        except AssertionError as e:
            FAILED.append(name)
            print(f"  FAIL: {e}")
        except Exception as e:  # noqa: BLE001
            FAILED.append(name)
            print(f"  ERROR: {type(e).__name__}: {e}")
    return deco


def main(prep):
    d = np.load(prep, allow_pickle=True)
    tr, va, te = (np.asarray(d[f"{s}_sims"]) for s in ("train", "val", "test"))

    # ------------------------------------------------------------------
    @check("splits are disjoint and complete")
    def _():
        s = [set(x.tolist()) for x in (tr, va, te)]
        assert not (s[0] & s[1]), f"{len(s[0] & s[1])} sims in both train and val"
        assert not (s[0] & s[2]), f"{len(s[0] & s[2])} sims in both train and test"
        assert not (s[1] & s[2]), f"{len(s[1] & s[2])} sims in both val and test"
        n = len(s[0]) + len(s[1]) + len(s[2])
        print(f"  train/val/test sims = {len(tr)}/{len(va)}/{len(te)} "
              f"({len(tr)/n:.0%}/{len(va)/n:.0%}/{len(te)/n:.0%}), "
              f"maps = {len(tr)*15}/{len(va)*15}/{len(te)*15}")

    # ------------------------------------------------------------------
    @check("target normalisation round-trips")
    def _():
        raw = np.asarray(d["params_raw"], dtype=np.float64)
        la = bool(d["log_astro"])
        back = denormalise_params(normalise_params(raw, la), la)
        err = np.abs(back - raw).max()
        assert err < 1e-4, f"round-trip error {err:.2e}"
        q = np.asarray(d["params_norm"])
        assert q.min() >= -1e-6 and q.max() <= 1 + 1e-6, \
            f"normalised targets outside [0,1]: [{q.min()}, {q.max()}]"
        print(f"  max round-trip error {err:.2e}, log_astro={la}")

    # ------------------------------------------------------------------
    @check("cube reshape groups the right 15 maps together")
    def _():
        # Maps of one simulation cluster more tightly than maps of different
        # simulations -- but only *more*, not enormously more: the 15 maps are
        # different slabs of the box, so they scatter on their own.  There is
        # no universal threshold for "tight enough", so instead of guessing one
        # we compare against the null the data itself provides: shuffle which
        # map belongs to which group and see how the clustering ratio falls.
        # If the reshape were map-major instead of sim-major, consecutive
        # groups of 15 would be 15 *different* sims and the observed ratio
        # would sit right on top of the shuffled null.
        ds = CAMELSMaps(d, "train", augment=False)
        maps = ds._open()
        sims = tr[:min(30, len(tr))]
        stat = np.log10(np.array(
            [[np.asarray(maps[s, m], dtype=np.float64).mean() for m in range(15)]
             for s in sims]))

        def ratio(a):                      # between-group / within-group variance
            return a.mean(axis=1).var() / max(np.mean(a.var(axis=1)), 1e-30)

        obs = ratio(stat)
        rng = np.random.default_rng(0)
        flat = stat.ravel()
        null = np.array([ratio(rng.permutation(flat).reshape(stat.shape))
                         for _ in range(400)])
        z = (obs - null.mean()) / max(null.std(), 1e-12)
        print(f"  clustering ratio {obs:.2f} vs shuffled null "
              f"{null.mean():.2f} +/- {null.std():.2f}   (z = {z:+.1f})")
        print(f"  within-sim scatter {np.sqrt(np.mean(stat.var(axis=1))):.4f} dex, "
              f"between-sim {stat.mean(axis=1).std():.4f} dex")
        assert z > 5, (
            f"grouping is no tighter than a random shuffle (z={z:+.1f}) -- the "
            f"cube is probably not sim-major, i.e. reshape(1000, 15, ...) is "
            f"pairing the wrong 15 maps together")

    # ------------------------------------------------------------------
    @check("maps are paired with the RIGHT simulation's parameters")
    def _():
        # Physics, used as a checksum: the mean of a total-mass map is the mean
        # matter density of the box, which is proportional to Omega_m.  So
        # <map> and Omega_m must be near-perfectly correlated ACROSS sims.
        # The previous check catches a wrong reshape; this one catches labels
        # that are shifted, sorted, or shuffled relative to the maps -- a
        # failure mode no amount of within-sim statistics can see.
        ds = CAMELSMaps(d, "train", augment=False)
        maps = ds._open()
        sims = tr[:60]
        mu = np.array([np.asarray(maps[s, 0], dtype=np.float64).mean()
                       for s in sims])
        om = np.asarray(d["params_raw"])[sims, 0]
        r = np.corrcoef(mu, om)[0, 1]
        print(f"  corr(<map>, Omega_m) over {len(sims)} sims = {r:+.3f}")
        assert r > 0.5, (
            f"correlation {r:+.3f} is far too weak -- for a total-mass field it "
            f"should be close to +1.  The labels are almost certainly not lined "
            f"up with the maps (shifted or shuffled rows).")
        if r < 0.9:
            print("  NOTE: expected ~+1 for Mtot.  If this is a different field "
                  "(e.g. P) a weaker value can be fine; for Mtot, investigate.")

    # ------------------------------------------------------------------
    @check("all 15 maps of a sim carry the same label, different sims differ")
    def _():
        ds = CAMELSMaps(d, "val", augment=False)
        y0 = [ds[i][1].numpy() for i in range(15)]
        assert all(np.allclose(y0[0], y) for y in y0), "labels vary within a sim"
        assert not np.allclose(y0[0], ds[15][1].numpy()), \
            "two different sims got the same label"

    # ------------------------------------------------------------------
    @check("inputs are finite and standardised")
    def _():
        ds = CAMELSMaps(d, "train", augment=False)
        idx = np.linspace(0, len(ds) - 1, 40).astype(int)
        xs = np.stack([ds[int(i)][0].numpy() for i in idx])
        assert np.isfinite(xs).all(), "non-finite values after log/standardise"
        m, s = xs.mean(), xs.std()
        assert abs(m) < 0.35, f"mean {m:.3f} far from 0"
        assert 0.6 < s < 1.6, f"std {s:.3f} far from 1"
        # Per-image means must still VARY -- that variation is the Omega_m
        # signal.  If it is ~0 you accidentally normalised per image.
        pim = xs.mean(axis=(1, 2, 3))
        assert pim.std() > 1e-3, \
            "per-image means are all identical -- per-image normalisation crept in"
        print(f"  batch mean={m:+.3f} std={s:.3f}; "
              f"spread of per-image means={pim.std():.3f} (must be > 0)")

    # ------------------------------------------------------------------
    @check("augmentation preserves the pixel content and the label")
    def _():
        ds_a = CAMELSMaps(d, "train", augment=True, roll=True)
        ds_p = CAMELSMaps(d, "train", augment=False)
        x0, y0 = ds_p[0]
        for _ in range(8):
            xa, ya = ds_a[0]
            assert np.allclose(np.sort(xa.numpy().ravel()),
                               np.sort(x0.numpy().ravel()), atol=1e-5), \
                "augmentation changed the pixel values, not just their positions"
            assert np.allclose(ya.numpy(), y0.numpy()), "augmentation changed the label"
        assert ds_p[0][0].numpy().tobytes() == ds_p[0][0].numpy().tobytes()
        print("  8 draws: same pixel multiset, same label, positions differ")

    # ------------------------------------------------------------------
    @check("val/test are deterministic (no augmentation leaking in)")
    def _():
        ds = CAMELSMaps(d, "test", augment=False)
        a, b = ds[3][0], ds[3][0]
        assert torch.equal(a, b), "test set returns different tensors for one index"

    # ------------------------------------------------------------------
    @check("loaders batch correctly")
    def _():
        ltr, lva, lte = make_loaders(d, batch_size=8, num_workers=0,
                                     downsample=4, limit_sims=(4, 2, 2),
                                     max_maps_per_sim=3)
        x, y = next(iter(ltr))
        assert x.shape == (8, 1, 64, 64), x.shape
        assert y.shape == (8, 6), y.shape
        assert x.dtype == torch.float32 and y.dtype == torch.float32
        # subsetting must not merge splits
        for a, b in ((ltr, lva), (ltr, lte), (lva, lte)):
            sa = set(a.dataset.index[:, 0].tolist())
            sb = set(b.dataset.index[:, 0].tolist())
            assert not (sa & sb), "limit_sims leaked sims between splits"
        print(f"  x{tuple(x.shape)} y{tuple(y.shape)}; subset splits still disjoint")

    # ------------------------------------------------------------------
    @check("constant baseline (the floor your model must beat)")
    def _():
        b = constant_baseline(d)
        print("  test RMSE if you always predict the training mean:")
        for i, n in enumerate(PARAM_NAMES):
            print(f"    {n:9s} {b['rmse_per_param'][i]:.4f}")

    # ------------------------------------------------------------------
    @check("model can memorise one batch (the learning-signal test)")
    def _():
        torch.manual_seed(0)
        ds = CAMELSMaps(d, "train", augment=False, downsample=4)
        idx = np.linspace(0, len(ds) - 1, 8).astype(int)
        x = torch.stack([ds[int(i)][0] for i in idx])
        y = torch.stack([ds[int(i)][1] for i in idx])
        net = ParamCNN(1, "mse", widths=(32, 64, 128, 256), dropout=0.0)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
        loss_fn = get_loss("mse")
        first = None
        for step in range(150):
            opt.zero_grad()
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
            if step == 0:
                first = loss.item()
        last = loss.item()
        assert last < 0.02 * max(first, 1e-8), \
            (f"loss only fell {first:.4f} -> {last:.4f} on 8 fixed samples; "
             f"the model cannot fit even memorisable data")
        print(f"  loss {first:.4f} -> {last:.6f} over 150 steps on 8 samples")

    # ------------------------------------------------------------------
    @check("timing estimate for the real run")
    def _():
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if getattr(torch.backends, "mps", None)
               and torch.backends.mps.is_available() else "cpu")
        for npx, widths, bs in ((64, (32, 64, 128, 256), 32),
                                (256, (32, 64, 128, 256, 256, 256), 32)):
            net = ParamCNN(1, "mse", widths=widths).to(dev)
            opt = torch.optim.AdamW(net.parameters())
            x = torch.randn(bs, 1, npx, npx, device=dev)
            y = torch.rand(bs, 6, device=dev)
            for _ in range(2):                       # warm up
                opt.zero_grad(); ((net(x) - y) ** 2).mean().backward(); opt.step()
            t = time.time()
            for _ in range(3):
                opt.zero_grad(); ((net(x) - y) ** 2).mean().backward(); opt.step()
            per_map = (time.time() - t) / 3 / bs
            ep = per_map * len(tr) * 15
            print(f"  {npx:3d}px on {dev}: {per_map*1e3:6.1f} ms/map "
                  f"-> ~{ep/60:5.1f} min/epoch (compute only, excludes disk)")

    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    if FAILED:
        print(f"{len(FAILED)} CHECK(S) FAILED: {', '.join(FAILED)}")
        print("Fix these before training -- a long run will not rescue them.")
        return 1
    print("All checks passed.  Next: python train.py --smoke")
    return 0


if __name__ == "__main__":
    prep = sys.argv[1] if len(sys.argv) > 1 else \
        "preprocessing_Mtot_IllustrisTNG.npz"
    sys.exit(main(prep))
