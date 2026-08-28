"""
PyTorch Dataset / DataLoaders for CAMELS Multifield maps -> 6 parameters.

Sample  = one 256x256 map.
Input   = float32 tensor (1, 256, 256), log10 then standardised.
Target  = float32 tensor (6,), each parameter min-max scaled to [0, 1].

The 4 GB cube is memory-mapped, never loaded whole.  Run preprocess.py first.

    from camels_data import make_loaders
    train, val, test = make_loaders("preprocessing_Mtot_IllustrisTNG.npz",
                                    batch_size=64, num_workers=4)
    x, y = next(iter(train))          # (64, 1, 256, 256), (64, 6)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

N_MAP, NPIX = 15, 256


class CAMELSMaps(Dataset):
    def __init__(self, prep, split, augment=False, downsample=1, roll=False):
        """
        prep       : path to preprocessing_*.npz (or the loaded dict)
        split      : 'train' | 'val' | 'test'
        augment    : random 90-deg rotation + flip (the 8 dihedral symmetries)
        downsample : integer factor; 4 -> 64x64 maps, useful for fast iteration
        roll       : also random periodic translation (valid: the box is periodic)
        """
        d = np.load(prep, allow_pickle=True) if isinstance(prep, str) else prep
        self.maps_path = str(d["maps_path"])
        self.mean = float(d["log_mean"])
        self.std = float(d["log_std"])
        self.y = np.asarray(d["params_norm"], dtype=np.float32)       # (1000, 6)
        self.sims = np.asarray(d[f"{split}_sims"])
        self.augment = augment
        self.downsample = int(downsample)
        self.roll = roll
        self._maps = None                       # opened lazily, per worker

        # flat index: every (simulation, map) pair in this split
        self.index = np.array([(s, m) for s in self.sims for m in range(N_MAP)],
                              dtype=np.int64)

    def _open(self):
        if self._maps is None:
            self._maps = np.load(self.maps_path, mmap_mode="r").reshape(
                -1, N_MAP, NPIX, NPIX)
        return self._maps

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sim, m = self.index[i]
        x = np.array(self._open()[sim, m], dtype=np.float32)   # copy: memmap is RO

        # ---- field transform (cheap, so done on the fly) ----------------
        np.clip(x, 1e-30, None, out=x)
        np.log10(x, out=x)
        x = (x - self.mean) / self.std

        # ---- augmentation: symmetries of the map, label unchanged -------
        if self.augment:
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k)
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
            if self.roll:
                x = np.roll(x, np.random.randint(NPIX, size=2), axis=(0, 1))
            x = np.ascontiguousarray(x)         # undo negative strides

        if self.downsample > 1:
            f = self.downsample
            x = x.reshape(NPIX // f, f, NPIX // f, f).mean(axis=(1, 3))

        return torch.from_numpy(x)[None], torch.from_numpy(self.y[sim])


def make_loaders(prep, batch_size=64, num_workers=4, downsample=1, roll=False,
                 augment_train=True, pin_memory=False, limit_sims=None,
                 max_maps_per_sim=0):
    """
    limit_sims       : (n_train, n_val, n_test) -- keep only N simulations of
                       each split.  0 or None = keep all.  Splits stay
                       disjoint, because it subsets each split's own list.

                       The subset is drawn from a FIXED permutation of each
                       split, so it is (a) an unbiased sample of parameter
                       space rather than the lowest sim indices, and (b)
                       NESTED: the 70 sims you get at n=70 are all inside the
                       175 you get at n=175.  That is what makes a learning
                       curve across sizes interpretable -- each larger run is
                       the previous run plus more data, not a different draw.
    max_maps_per_sim : keep only the first M of the 15 maps per simulation.
    Neither changes any transform, so results scale up cleanly.
    """
    d = np.load(prep, allow_pickle=True) if isinstance(prep, str) else prep
    kw = dict(downsample=downsample, roll=roll)
    ds = {
        "train": CAMELSMaps(d, "train", augment=augment_train, **kw),
        "val": CAMELSMaps(d, "val", augment=False, **kw),
        "test": CAMELSMaps(d, "test", augment=False, **kw),
    }
    lim = dict(zip(("train", "val", "test"), limit_sims or (0, 0, 0)))
    for s, dset in ds.items():
        if lim[s] and lim[s] > len(dset.sims):
            raise ValueError(f"asked for {lim[s]} '{s}' sims, only "
                             f"{len(dset.sims)} exist")
        order = np.random.default_rng(1234).permutation(len(dset.sims))
        keep_sim = set(dset.sims[order[:lim[s]]].tolist()) if lim[s] else None
        if keep_sim is not None or max_maps_per_sim:
            dset.index = np.array(
                [p for p in dset.index
                 if (keep_sim is None or int(p[0]) in keep_sim)
                 and (not max_maps_per_sim or int(p[1]) < max_maps_per_sim)],
                dtype=np.int64).reshape(-1, 2)
            if len(dset.index) == 0:
                raise ValueError(f"'{s}' split is empty after limiting")
    return tuple(
        DataLoader(ds[s], batch_size=batch_size, shuffle=(s == "train"),
                   num_workers=num_workers, pin_memory=pin_memory,
                   drop_last=(s == "train" and len(ds[s]) > batch_size),
                   persistent_workers=num_workers > 0)
        for s in ("train", "val", "test")
    )


# --------------------------------------------------------------------------
# Baseline you must beat: predict the training-set mean for every map.
# R^2 = 0 by construction.  If your CNN scores below this, something is wrong.
# --------------------------------------------------------------------------
def constant_baseline(prep):
    d = np.load(prep, allow_pickle=True) if isinstance(prep, str) else prep
    y = np.asarray(d["params_norm"])
    tr, te = np.asarray(d["train_sims"]), np.asarray(d["test_sims"])
    pred = y[tr].mean(axis=0)
    mse = ((y[te] - pred) ** 2).mean(axis=0)
    return {"mse_per_param": mse, "rmse_per_param": np.sqrt(mse)}


if __name__ == "__main__":
    import sys
    prep = sys.argv[1] if len(sys.argv) > 1 else \
        "preprocessing_Mtot_IllustrisTNG.npz"
    tr, va, te = make_loaders(prep, batch_size=8, num_workers=0)
    print(f"train/val/test maps: {len(tr.dataset)}/{len(va.dataset)}/{len(te.dataset)}")
    x, y = next(iter(tr))
    print("x", x.shape, x.dtype, f"mean={x.mean():.3f} std={x.std():.3f}")
    print("y", y.shape, y.dtype, f"min={y.min():.3f} max={y.max():.3f}")
    print("constant baseline (test RMSE, normalised units):",
          np.round(constant_baseline(prep)["rmse_per_param"], 4))
