# The model and the training loop

`model.py` (architecture + losses), `train.py` (loop, evaluation, checkpointing) and
`check.py` (correctness battery). `camels_data.py` and `preprocess.py` were both updated
for subset running — re-copy all of them.

## Architecture

Six stride-2 blocks take 256 → 128 → 64 → 32 → 16 → 8 → 4 px, widths
32/64/128/256/256/256. Each block is a stride-2 conv followed by a stride-1 conv at the
new resolution, both with BatchNorm and LeakyReLU(0.2). Then a head, then 6 outputs.
About 4.9M parameters.

Two choices in there worth explaining:

**Pooling on mean *and* std.** The usual global-average-pool collapses each feature map
to its mean. But for this problem the *scatter* of the field is as informative as its
level — that's most of where σ_8 lives. So the head sees `concat(mean, std)` over the
spatial dimensions, `2×256` numbers. Cheap, and it gives the network somewhere to put
variance information instead of forcing it through the mean.

**BatchNorm is safe here**, even though normalisation was the thing to avoid in
preprocessing. It normalises across the batch, not within a sample, so differences in
overall amplitude between maps survive — which is exactly the Ω_m signal you need to keep.

Use `--widths 32 64 128 256` (4 blocks) when training on `--downsample 4` maps, otherwise
you run out of pixels. `train.py` checks this and tells you.

## Two heads

`--head mse` gives 6 outputs and plain MSE on the normalised targets. Start here.

`--head moment` gives 12 — six means and six log-variances — trained with the two-moment
loss:

```
L = Σ_i log⟨(θ_i − μ_i)²⟩ + Σ_i log⟨((θ_i − μ_i)² − σ_i²)²⟩
```

The first term drives μ to the posterior mean; the second drives σ² to the posterior
variance. The network then reports its own error bar per parameter per map, which for a
parameter-inference paper is the number you actually want. Note both averages are taken
over the batch, so this head needs a reasonably large batch (128 is a good target) to be
stable. `train.py` prints a calibration check at the end: the std of
`(pred − true)/σ` should come out near 1.0 per parameter. Meaningfully below 1 means the
network is overstating its uncertainty, above 1 means overconfident.

## Training loop

AdamW, one-cycle LR schedule, gradient clipping at 1.0, early stopping on validation loss
with `--patience 25`, best checkpoint saved by val loss. Device is picked automatically —
on your MacBook that means MPS if PyTorch sees it, otherwise CPU.

Evaluation reports, per parameter:

- RMSE and R² in normalised [0,1] units, plus RMSE back in physical units
- the same with **8× test-time augmentation** (average the prediction over the 4
  rotations and 2 flips). Free, and it always helps a little
- **per-simulation** results: average the 15 maps of each test sim into one prediction.
  If your framing is "given a simulation, infer its parameters", this is the honest
  headline number, and it should beat the per-map number clearly

Predictions land in `runs/<tag>/test_predictions.npz` (`mu`, `sigma`, `y`, plus the
per-sim versions) so you can make the usual true-vs-predicted scatter plots without
re-running anything.

## Order to run things

Rungs, cheapest first. Each answers a different question, so don't skip ahead — a
failure caught on rung 0 costs seconds, the same failure caught on rung 5 costs a night.

**0. Does the data make sense?** (~1 minute, and it barely touches the 3.9 GB)

```bash
python check.py preprocessing_Mtot_IllustrisTNG.npz
```

Twelve assertions covering the things that silently produce a plausible-looking but
worthless model: split overlap, label misalignment, non-finite pixels, augmentation that
changes the label, per-image normalisation creeping back in, target round-tripping. It
also prints the constant baseline and a timing estimate per epoch on your hardware.

Three are worth calling out.

**`maps are paired with the RIGHT simulation's parameters`** uses physics as a checksum:
the mean of a total-mass map *is* the mean matter density of the box, so
`corr(⟨map⟩, Ω_m)` across simulations must come out near +1. An off-by-one or a sorted
parameter file drops it to ~0.

**`cube reshape groups the right 15 maps together`** asks whether `reshape(1000, 15, …)`
really puts 15 maps of one simulation in each row. It has no fixed threshold, because
there isn't a defensible one — the 15 maps are different slabs of the box and scatter on
their own, by an amount nobody knows in advance. So it builds the null from your own data:
shuffle which map belongs to which group, recompute the clustering ratio 400 times, and
report how many sigma the real grouping sits above that null. Healthy data lands at
z ≈ +500; a map-major cube lands at z ≈ −2.

**`model can memorise one batch`** takes 8 fixed maps and drives the loss to ~1e-4 in 150
steps. If that fails, gradients aren't reaching the inputs and nothing downstream matters.

Each check is verified to fail when it should: injected shuffled labels, a one-row shift,
an accidental per-image normalisation, and a map-major cube are each caught by the
right assertion.

**1. Does the pipeline run end to end?** (~5 seconds)

```bash
python train.py --smoke
```

8 train sims / 4 val / 4 test, 3 maps each, 64 px, 3 epochs — through training,
validation, checkpointing, TTA, per-simulation aggregation and saving. Anything you pass
after `--smoke` overrides the preset, e.g. `--smoke --head moment --batch_size 32` to
smoke-test the other head. The accuracy numbers it prints are meaningless; you're
watching for it to reach the end without an exception.

For a fully independent subset, `--limit_sims 10 3 3 --max_maps_per_sim 5` gives you
manual control, and `preprocess.py --n_sim 50` builds the npz from only the first 50
simulations so even the stats pass is quick. That one writes to a `_smoke50` filename so
it can never clobber your real npz — but its mean/std are *not* the real ones, so don't
train anything you intend to keep on it.

**2. Can the model memorise a small subset?** (~2 minutes)

```bash
python train.py --overfit 10 --epochs 200 --downsample 4 \
                --widths 32 64 128 256 --no_augment --dropout 0
```

150 maps, no regularisation. Train loss must collapse toward zero (~6e-4 in my test). If
it plateaus, the problem is in the data or the loss, and no architecture tuning will fix
it. This is the last rung where a failure is cheap.

Note what rung 2 does *not* prove: memorisation works fine even on shuffled labels. That's
exactly why rung 0 exists — only the correlation check catches mispaired labels.

**3. Pilot on a tenth of the data.** (~20 minutes)

```bash
python train.py --fraction 0.1 --downsample 4 --widths 32 64 128 256 --epochs 60
```

70 / 15 / 15 simulations = 1050 / 225 / 225 maps. This is the first rung that produces
numbers worth reading, and it's where you tune LR, dropout and width.

`--fraction` deliberately does **not** rebuild the npz. It keeps the real normalisation
statistics and the real split boundaries and just uses fewer simulations from each split,
so a pilot result and a full result are directly comparable — scaling up changes the
amount of data and nothing else. (`preprocess.py --n_sim 100` is the other thing you could
do, and it's the wrong tool here: it recomputes mean/std from 100 sims and redraws the
split, so its numbers don't line up with anything. Use it only to test the preprocessing
script itself.) So run `preprocess.py` once over all 1000 simulations first — a few
minutes, once — and every subset run afterwards is instant.

The subsets are **nested**: the 70 training sims at `--fraction 0.1` are all inside the
175 at `--fraction 0.25`. That's what makes a learning curve meaningful, because each
larger run is the previous run *plus more data* rather than a different random draw. So:

```bash
for f in 0.1 0.25 0.5; do python train.py --fraction $f --downsample 4 \
    --widths 32 64 128 256 --epochs 60 --tag pilot_$f; done
```

Then plot R²(Ω_m) against fraction. Still climbing at 0.5 → the full dataset will buy you
real accuracy. Flat → you're architecture-limited, not data-limited, and a night of
256 px training won't rescue it. That's a few hours well spent before committing to the
full run.

What to expect at 10%: Ω_m should clear the constant baseline comfortably but land below
its full-data value; σ_8 marginal; the astro parameters near zero. **Don't conclude from a
pilot that the astro parameters are hopeless** — they're the most data-hungry of the six,
and a tenth of the training set is exactly where they look worst.

One caveat the script now warns about: R² is measured against the label spread *within the
evaluation set*, so a small val/test set makes it noisy — and with a single simulation the
denominator is zero and R² is undefined. `train.py` prints NaN there rather than a huge
negative number that looks like a broken model, and warns below 10 simulations. At
`--fraction 0.1` you have 15 val sims, which is fine; go much lower and read RMSE instead.

**4. Fast iteration on the full training set at 64 px.**

```bash
python train.py --downsample 4 --widths 32 64 128 256 --epochs 60
```

Compare against the constant baseline that `camels_data.py` prints.

**5. Full resolution.**

```bash
python train.py --epochs 150 --batch_size 64
```

Be realistic about cost: 10 500 maps at 256 px is an overnight job on a laptop even with
MPS. If the 64 px run already looks healthy and you need the full-resolution numbers, a
free Colab GPU will finish it in well under an hour.

**6. Uncertainties.**

```bash
python train.py --head moment --epochs 150 --batch_size 128
```

## What to check as it runs

- Train and val loss both falling → fine. Val flat while train falls → more augmentation
  (`--roll`), more dropout, or fewer parameters.
- R²(Ω_m) should climb fast and end high. If it's stuck near 0 you're at the constant
  baseline, i.e. the network found nothing — check the sanity test again.
- σ_8 improves slowly. The astro parameters may never get far above 0; that's the physics,
  not the code.
- Loss going NaN with `--head moment`: lower `--lr`, and make sure the batch is large
  enough for the per-batch moments.

## Then what

The obvious next moves, roughly in order of payoff:

1. **Add the P field as a second channel** — `--in_channels 2`. Stack Mtot and P for the
   same map index. Multifield is where the astrophysical parameters actually become
   constrainable, and it's the headline result of the CAMELS-CMD paper.
2. **Cross-simulator test.** Train on IllustrisTNG, test on SIMBA. Performance will drop
   hard. That gap is the interesting scientific result — it measures how much the network
   learned about *this* subgrid model rather than about cosmology.
3. **Saliency / ablation** — mask the densest pixels, or high-pass the maps, and see which
   parameter degrades. Tells you what the network is keying on.
