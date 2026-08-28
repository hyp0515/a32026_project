# CAMELS maps → parameters: preprocessing before the network

Task type: **multi-output regression**, image in, 6 numbers out. Not classification.

## 0. What one training example actually is

Your cube is `(1000, 15, 256, 256)`, but the model never sees a simulation — it sees
*one map*. So:

| | shape | meaning |
|---|---|---|
| sample `x` | `(1, 256, 256)` | one map, channel-first (`C=1` because you're using Mtot only) |
| label `y` | `(6,)` | the parameters of the parent simulation |
| dataset | 15 000 samples | the same label repeated 15× per simulation |

Later, when you add P, the natural move is `C=2` — stack Mtot and P as two channels of
the *same* map index, since they're the same field of view. Don't concatenate them as
extra samples.

## 1. Split by simulation, before anything else

This is the one mistake that silently ruins the whole project. The 15 maps of a
simulation are 15 slices of the same box: same parameters, same large-scale structure,
highly correlated. If you shuffle 15 000 maps and split 80/20, nearly every test map has
a sibling in the training set, and your test error measures memorisation, not inference.
You'll get a beautiful R² and it will mean nothing.

Split the **1000 simulation indices**, then take all 15 maps of each. The default is
70 / 15 / 15 → 700 / 150 / 150 sims → 10 500 / 2250 / 2250 maps. Change it with
`--split`, which takes any three weights (`--split 7 1.5 1.5` and `--split 0.7 0.15 0.15`
are the same thing) and normalises them. Fix the seed and save the index arrays so every
experiment uses the same split.

## 2. Transform the field: log10

Mtot is log-normal and spans ~10 orders of magnitude per pixel. Feed that raw to a CNN
and the loss is dominated by a handful of the densest halo pixels. Take `log10` first —
this is what every CAMELS-CMD paper does. Check for zeros/negatives before the log
(`preprocess.py` reports the count and clips).

## 3. Normalise with ONE global mean/std — not per image

After the log, standardise: `x = (log10(x) - mean) / std`, where `mean` and `std` are
scalars computed over **the training simulations only**.

Do **not** normalise each image by its own mean and std. It's the reflex from natural-image
work and here it destroys the signal: the mean of a Mtot map is directly related to Ω_m,
and the scatter carries σ_8. Per-image standardisation subtracts exactly the information
you're trying to predict. (You can test this later — it's a nice ablation.)

Computing the stats on training sims only, and reusing them for val/test, is what keeps
the evaluation honest.

## 4. Normalise the targets too

The six parameters live on wildly different scales (Ω_m ~ 0.3, A_SN1 ~ 2). An MSE loss
on raw values would be almost entirely about the astrophysical parameters. Map each to
[0, 1] with the **prior bounds** of the latin hypercube, not the empirical min/max:

```
Ω_m ∈ [0.1, 0.5]   σ_8 ∈ [0.6, 1.0]
A_SN1, A_AGN1 ∈ [0.25, 4.0]   A_SN2, A_AGN2 ∈ [0.5, 2.0]
```

Keep the inverse transform next to the forward one (`denormalise_params`) — you'll need it
to report errors in physical units.

One subtlety: the four astrophysical parameters are sampled **log-uniformly**, so their
values are bunched near 1 in linear space. `preprocess.py` prints a check that confirms
this from your file. Linear min-max is what the published CAMELS baselines use, so it's
the right default if you want comparable numbers; `--log_astro` gives you the variant
that makes the target distribution actually uniform. Worth trying both.

## 5. Augmentation: use the symmetries, on the fly

The parameters are invariant under rotation and reflection of the map, so 4 rotations ×
2 flips = **8× more effective data for free**. Apply it randomly inside `__getitem__`,
never as a precomputed 8× copy on disk.

Bonus: the simulation box is periodic, so random wrap-around translations
(`np.roll`) are *also* label-preserving. Not standard in the literature, but physically
valid — `roll=True` in the Dataset. Try it as an ablation.

Nothing else. No random crops (changes the physical scale), no brightness/contrast
jitter (that's exactly the Ω_m signal), no ImageNet normalisation.

## 6. Memory: memory-map, don't load

`1000 × 15 × 256 × 256 × 4 bytes ≈ 3.9 GB` per field. On a laptop, loading that plus a
float32 copy will hurt. `np.load(path, mmap_mode='r')` and let the OS page it in; the
log+normalise per 256×256 image is negligible next to the disk read. That's why I don't
write a transformed copy of the cube to disk — it'd be another 4 GB for no gain.

If you do want it in RAM: after the log, float16 is plenty (0.001 dex precision) and
halves it to ~2 GB.

## 7. Sanity checks before you train anything

1. **Alignment.** The raw file is `(15000, 256, 256)` ordered sim-major, so
   `reshape(1000, 15, ...)` puts `maps[s]` with `params[s]`. `preprocess.py` verifies it:
   the 15 maps of a sim should have near-identical mean log-density, and two different
   sims should differ much more.
2. **Batch stats.** A training batch should come out with mean ≈ 0, std ≈ 1.
3. **Constant baseline.** Predict the training mean for everything → that's your zero.
   `constant_baseline()` prints its per-parameter test RMSE. If your CNN doesn't clearly
   beat that on Ω_m within a few epochs, the bug is in the pipeline, not the architecture.
4. **Overfit 10 simulations.** Turn off augmentation, train on 150 maps, drive the loss
   to ~0. If it can't, the model or the loss is wrong. Do this *before* the real run.

## Expectations, so you can calibrate

From Mtot maps, Ω_m is easy (R² > 0.95 is normal), σ_8 is moderate, and the four
astrophysical parameters are genuinely hard — A_SN1 is partially constrained, A_AGN2 is
close to hopeless from a single field. Weak astro performance is a property of the data,
not a bug in your code. Report per-parameter metrics, never a single averaged one.

## Running it

```bash
python preprocess.py --data_dir /Users/hyp0515/data/a3/CAMELS_multifield/ \
                     --field Mtot --simcode IllustrisTNG --split 7 1.5 1.5
python camels_data.py preprocessing_Mtot_IllustrisTNG.npz    # smoke test
```

Then in your training script:

```python
from camels_data import make_loaders
train, val, test = make_loaders("preprocessing_Mtot_IllustrisTNG.npz",
                                batch_size=64, num_workers=4,
                                downsample=4)   # 64x64 while you're iterating
```

`downsample=4` averages to 64×64 — a full epoch in a fraction of the time while you
debug the model. Move to `downsample=1` for the real runs.

## What comes next (not in these files)

A CNN of ~6 conv blocks with stride 2, going 256→128→…→4, then global pooling and a
linear head to 6 outputs. MSE on the normalised targets to start. Once that works, the
upgrade the CAMELS papers use is a *moment network*: output 12 numbers (6 means, 6
log-variances) and train with a likelihood-free loss so the model also reports its own
uncertainty per parameter. Worth doing — but get the pipeline above verified first.
