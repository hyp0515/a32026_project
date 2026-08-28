"""
CAMELS Multifield -> parameter inference: one-off preprocessing / diagnostics.

What this does NOT do: it does not write a transformed copy of the 4 GB map cube.
The log + normalise transform is ~free per image, so it is applied on the fly in
the Dataset (see camels_data.py). This script only:

  1. sanity-checks the raw maps and the parameter file,
  2. splits by SIMULATION (never by map) into train / val / test,
  3. computes the log-space mean and std over the TRAINING simulations only,
  4. saves everything small into preprocessing_Mtot.npz.

Run once:
    python preprocess.py --field Mtot --simcode IllustrisTNG
"""

import argparse
import json
import os

import numpy as np

# Prior ranges of the CAMELS LH latin hypercube (fixed by design, not measured
# from the file -- keeps the target scaling identical across any subset).
PARAM_NAMES = ["Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2"]
PARAM_MIN = np.array([0.1, 0.6, 0.25, 0.25, 0.5, 0.5])
PARAM_MAX = np.array([0.5, 1.0, 4.00, 4.00, 2.0, 2.0])
LOG_PARAM = np.array([False, False, True, True, True, True])  # log-uniform sampled

N_SIM, N_MAP, NPIX = 1000, 15, 256


def normalise_params(p, use_log_for_astro=False):
    """(N, 6) raw parameters -> (N, 6) in [0, 1]."""
    p = np.asarray(p, dtype=np.float64).copy()
    lo, hi = PARAM_MIN.copy(), PARAM_MAX.copy()
    if use_log_for_astro:
        m = LOG_PARAM
        p[:, m] = np.log10(p[:, m])
        lo[m], hi[m] = np.log10(lo[m]), np.log10(hi[m])
    return ((p - lo) / (hi - lo)).astype(np.float32)


def denormalise_params(q, use_log_for_astro=False):
    """Inverse of normalise_params -- use this on model predictions."""
    q = np.asarray(q, dtype=np.float64)
    lo, hi = PARAM_MIN.copy(), PARAM_MAX.copy()
    if use_log_for_astro:
        m = LOG_PARAM
        lo[m], hi[m] = np.log10(lo[m]), np.log10(hi[m])
    p = q * (hi - lo) + lo
    if use_log_for_astro:
        p[:, LOG_PARAM] = 10.0 ** p[:, LOG_PARAM]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/Users/hyp0515/data/a3/CAMELS_multifield/")
    ap.add_argument("--field", default="Mtot")
    ap.add_argument("--simcode", default="IllustrisTNG")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="fractions of the 1000 SIMULATIONS (not maps)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_astro", action="store_true",
                    help="min-max the 4 astro params in log10 space instead of linear")
    ap.add_argument("--n_sim", type=int, default=0,
                    help="SMOKE TEST: use only the first N simulations, so the "
                         "stats pass reads a fraction of the 3.9 GB.  Writes a "
                         "separate _smokeN npz -- never overwrites the real one.")
    args = ap.parse_args()

    global N_SIM
    if args.n_sim:
        N_SIM = min(args.n_sim, N_SIM)
        print(f"*** SMOKE TEST: using the first {N_SIM} simulations only. ***\n"
              f"*** The mean/std below are NOT the real ones -- rerun without "
              f"--n_sim before any run you intend to keep. ***\n")

    maps_path = os.path.join(
        args.data_dir, f"Maps_{args.field}_{args.simcode}_LH_z=0.00.npy")
    params_path = os.path.join(args.data_dir, f"params_LH_{args.simcode}.txt")

    # ---------------------------------------------------------------- params
    params = np.loadtxt(params_path)
    assert params.ndim == 2 and params.shape[1] == 6, params.shape
    assert params.shape[0] >= N_SIM, f"only {params.shape[0]} rows in {params_path}"
    params = params[:N_SIM]
    print("params:", params.shape)
    for i, n in enumerate(PARAM_NAMES):
        print(f"  {n:8s} [{params[:, i].min():.4f}, {params[:, i].max():.4f}]"
              f"  prior [{PARAM_MIN[i]}, {PARAM_MAX[i]}]")

    # Is the astro sampling uniform in linear or in log space?  Compare the
    # median to the midpoint of each: log-uniform => median ~ geometric mean.
    print("\nsampling check (median vs linear mid vs geometric mid):")
    for i, n in enumerate(PARAM_NAMES):
        med = np.median(params[:, i])
        lin = 0.5 * (PARAM_MIN[i] + PARAM_MAX[i])
        geo = np.sqrt(PARAM_MIN[i] * PARAM_MAX[i])
        print(f"  {n:8s} med={med:.4f}  lin={lin:.4f}  geo={geo:.4f}"
              f"  -> {'log-uniform' if abs(med-geo) < abs(med-lin) else 'uniform'}")

    y = normalise_params(params, use_log_for_astro=args.log_astro)
    print(f"\nnormalised targets: min={y.min():.4f} max={y.max():.4f} "
          f"(should be ~[0,1])")

    # ------------------------------------------------------------------ maps
    maps = np.load(maps_path, mmap_mode="r")
    print(f"\nmaps on disk: shape={maps.shape} dtype={maps.dtype} "
          f"({maps.nbytes / 1e9:.2f} GB)")
    maps = maps.reshape(-1, N_MAP, NPIX, NPIX)[:N_SIM]

    # The raw file is (15000, 256, 256) ordered as sim-major, i.e. row
    # s*15 + m belongs to simulation s.  The reshape above therefore aligns
    # maps[s] with params[s].  Spot-check: maps of one sim should look far
    # more alike than maps of different sims.
    j = N_SIM // 2
    a = np.log10(np.asarray(maps[0], dtype=np.float64)).mean(axis=(1, 2))
    b = np.log10(np.asarray(maps[j], dtype=np.float64)).mean(axis=(1, 2))
    print(f"alignment spot-check: within-sim spread of <log10 map> "
          f"sim0={a.std():.4f} sim{j}={b.std():.4f} | between-sim gap="
          f"{abs(a.mean() - b.mean()):.4f}")

    # ----------------------------------------------------------------- split
    # Split the SIMULATIONS, then take all 15 maps of each -- maps of one
    # simulation are correlated siblings sharing a label, so splitting maps
    # would leak the test set into training.
    frac = np.asarray(args.split, dtype=float)
    frac = frac / frac.sum()
    n_train = int(round(frac[0] * N_SIM))
    n_val = int(round(frac[1] * N_SIM))
    n_test = N_SIM - n_train - n_val          # absorbs the rounding
    assert min(n_train, n_val, n_test) > 0, f"degenerate split {args.split}"

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N_SIM)
    train_sims = np.sort(perm[:n_train])
    val_sims = np.sort(perm[n_train: n_train + n_val])
    test_sims = np.sort(perm[n_train + n_val:])
    print(f"\nsplit by simulation ({frac[0]:.0%}/{frac[1]:.0%}/{frac[2]:.0%}): "
          f"train={len(train_sims)} val={len(val_sims)} test={len(test_sims)}"
          f"  -> {len(train_sims)*N_MAP}/{len(val_sims)*N_MAP}/"
          f"{len(test_sims)*N_MAP} maps")
    s_tr, s_va, s_te = set(train_sims.tolist()), set(val_sims.tolist()), set(test_sims.tolist())
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te)
    assert len(s_tr | s_va | s_te) == N_SIM

    # ------------------------------------- streaming log-space mean / std
    # Training simulations only.  A single scalar mean/std for the whole
    # dataset -- NOT per image (see PREPROCESSING.md).
    n = 0
    s1 = 0.0
    s2 = 0.0
    vmin, vmax = np.inf, -np.inf
    nonpos = 0
    for k, sim in enumerate(train_sims):
        x = np.asarray(maps[sim], dtype=np.float64)      # (15, 256, 256)
        nonpos += int((x <= 0).sum())
        x = np.log10(np.clip(x, 1e-30, None))
        n += x.size
        s1 += x.sum()
        s2 += (x ** 2).sum()
        vmin = min(vmin, x.min())
        vmax = max(vmax, x.max())
        if (k + 1) % 200 == 0:
            print(f"  ...{k + 1}/{len(train_sims)} sims")
    mean = s1 / n
    std = np.sqrt(max(s2 / n - mean ** 2, 0.0))
    print(f"\nlog10({args.field}) over training sims: mean={mean:.6f} "
          f"std={std:.6f} min={vmin:.4f} max={vmax:.4f}")
    print(f"non-positive pixels found: {nonpos}"
          + ("  <-- clipped before log" if nonpos else ""))
    print(f"after standardising, range is roughly "
          f"[{(vmin-mean)/std:.2f}, {(vmax-mean)/std:.2f}] sigma")

    # ------------------------------------------------------------------ save
    suffix = f"_smoke{N_SIM}" if args.n_sim else ""
    out = os.path.join(
        args.out_dir, f"preprocessing_{args.field}_{args.simcode}{suffix}.npz")
    np.savez(
        out,
        maps_path=maps_path,
        params_raw=params.astype(np.float32),
        params_norm=y,
        log_astro=args.log_astro,
        train_sims=train_sims,
        val_sims=val_sims,
        test_sims=test_sims,
        log_mean=mean,
        log_std=std,
        log_min=vmin,
        log_max=vmax,
        param_names=np.array(PARAM_NAMES),
    )
    print(f"\nsaved -> {out}")
    print(json.dumps({"log_mean": float(mean), "log_std": float(std)}, indent=2))


if __name__ == "__main__":
    main()
