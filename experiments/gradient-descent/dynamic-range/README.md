# Direct third-moment optimisation

Scripts for the direct-optimisation experiments: every coefficient block,
including degrees `l = 0` and `l = 1`, is a free variable, and the loss

    L(A) = ||P3(A) - P3_hat||_F^2 / ||P3_hat||_F^2

is minimised by MATLAB's `fminunc` (quasi-Newton, BFGS, analytic gradient).
No band is supplied from the truth, so the minimiser is determined only up to a
global element of O(3) and every restart is scored after an explicit O(3)
alignment.

## Reproducing the noise sweep

Place the volumes listed in [`data/raw/README.md`](../../../data/raw/README.md),
then run one cell per SNR:

```matlab
for snr = [Inf 30 10 3 1 0.3 0.1 0.03]
    run_noise_sweep_unpinned(snr,'vol_80S_true.mrc',31);          % 80S
    run_noise_sweep_unpinned(snr,'vol_TRPV1_emd24084.mat',32);    % TRPV1
end
plot_unpinned_noise_sweep('80S','vol_80S_true','80S, EMD-2660');
plot_unpinned_noise_sweep('TRPV1','vol_TRPV1_emd24084','TRPV1, EMD-24084');
```

then `python3 plot_unpinned_robustness.py` for the combined figure. Each cell
takes about 80 minutes on an M3 Pro; the eight cells are independent and can be
run in parallel. Results land in
`results/gradient-descent/noise-sweep/<LABEL>_unpinned_noise sweep/`, which is
untracked.

Settings: `L = 5`, `K = 12` shells at every degree (so the decay-by-three
support condition is *not* satisfied), 432 real parameters, `n = 1e5`
projections per moment, 20 restarts per SNR from seeds 42..61, iteration cap
8000.

## Four conventions that are easy to get wrong

1. **The `i^l` twist.** `nu_expand_volume_shells` multiplies degree `l` by
   `i^l` so its output satisfies the reality condition
   `A_{l,-m} = (-1)^m conj(A_{l,m})` that the real parameterisation needs.
   `gd_precompute_model` is the forward model for whatever array it is handed
   — the Monte Carlo sampler reproduces it on any input, to sampling error —
   and `gd_physical_model(x)` equals `gd_precompute_model` evaluated at
   `(-i)^l x`, an identity that holds to 0 in double precision. Choosing the
   model is therefore exactly choosing which array counts as the volume's
   expansion, and the two choices give moments 10.4% apart for 80S and 27.9%
   apart for TRPV1.

   The noise sweep samples the observed moment from the untwisted array
   `(-i)^l .* a` with the default model and fits the twisted array with
   `gd_physical_model` — a matched pair. `run_single_restart` instead samples
   and fits the twisted array with the default model, also a matched pair, but
   a different inverse problem. Mixing the halves of either pair raises the
   floor loss from ~1e-6 to ~8e-2, which is the symptom to look for.

2. **SNR normalisation.** The noise variance per projected coefficient is
   `sigma^2 = P_ref / SNR`, where the reference power `P_ref` is the mean power
   of the equatorial slice of the volume **in its original orientation**,
   `P_ref = (1/D_proj) sum_{k,s} |c_k[s]|^2` with `D_proj = (2L+1)K = 132`.
   That is a fixed reference orientation, not an average over the sampled
   views, and it is the convention of the reported sweeps.

   `run_noise_sweep_unpinned(...,'haar')` instead averages the projected power
   over the viewing distribution. That average has a closed form: by Wigner-D
   orthogonality and the spherical-harmonic addition theorem it collapses to
   `||a||_F^2 / (4 pi)`, independent of the volume's shape, so
   `sigma = ||a||_F / sqrt(4 pi D_proj SNR)`. The two differ only through the
   volume's anisotropy — `'slice'` gives a power 1.6% below `'haar'` for 80S
   but 22.3% above it for TRPV1 (EMD-24084). Switching scales sigma by 1.008
   and 0.904 respectively, so it changes the results; `'slice'` is the default
   and reproduces the figures.

3. **Which rotations each SNR cell sees.** The sampler draws rotations and
   noise from one stream seeded at 42, and when `sigma = 0` it skips the noise
   draws. Every *finite*-SNR cell therefore sees the **same** viewing
   rotations, because the number of random draws per sample is identical;
   `SNR = Inf` is the one cell on a different draw. That is why the TRPV1
   error at `SNR = Inf` sits slightly above the one at `SNR = 30` rather than
   below it: the two are not paired. Separate `RandStream`s for rotations and
   noise would pair all eight cells.

4. **No noise debiasing is needed, and none is applied.** The noise is circular
   complex Gaussian, i.i.d. per projected coefficient, and the empirical third
   moment averages *unconjugated* triple products over blocks with
   `m1 + m2 + m3 = 0`. Every term containing noise factors has zero expectation
   (`E[n] = 0`, `E[n^2] = 0` for a circular variable), so the estimator is
   already unbiased for the signal-only moment.

## Initialisation

`gd_run_josh_baseline_restarts` with `initialization = 'real'` draws i.i.d.
Gaussian coefficients whose per-degree standard deviation is taken from the
truth, `s(l) = ||A_l^true||_F / sqrt(K (2l+1))`. The *scale* of each degree is
therefore informed by the ground truth even though no coefficient is; starting
from a single global scale instead changes how many restarts converge.

## Alignment

`gd_align_rotation_reflection` minimises `||A_hat - g . A_true||_F` over all of
O(3) — all degrees, not just degree one — by scoring 20000 Haar-random
rotations per parity and refining the best 25 with Nelder-Mead. An earlier
coarse version (1296-point Euler grid scored on degrees 1..3) put one TRPV1
restart per SNR cell into a near-symmetric wrong pose, reporting errors near
0.25 for solutions actually at 0.025-0.11. `rescore_unpinned_noise_sweep`
re-scores saved cells without re-optimising.

Checks worth repeating before trusting a sweep: the loss at the truth should
equal the squared relative moment error; the pre-alignment `l = 0` and `l = 1`
errors should be non-zero (proving nothing was pinned); the aligner should
score the truth against itself at ~1e-16; and no restart should hit the cap.

What the reported sweeps give on those checks: no restart hit the cap, final
gradient norms were at most 2.09e-8 with median 2.92e-9, and the longest run
took 3762 iterations. For 80S at every `SNR >= 1` the 20 restarts' aligned
all-degree errors span at most 4.4e-6 (standard deviation at most 1.1e-6), and
at `SNR = Inf` and `SNR = 30` the recovered coefficient arrays themselves
coincide to within 2.8e-5 after alignment. For TRPV1 the restarts settle on
several distinct stationary points of nearly equal loss, with the standard
deviation of the aligned error at most 6.2e-3 at every SNR.

## Other scripts here

| Script | What it does |
| --- | --- |
| `run_pinning_comparison.m` | Random restarts against the *analytic* moment, with `l = 0,1` pinned to the truth or free, to isolate the effect of pinning |
| `run_exact_symmetry_uniqueness.m` | Whether an exactly Cn-symmetrised volume admits a second, distant solution of the analytic moment |
| `run_basin_radius.m` | How far from the truth an initialisation can start and still return, using the analytic moment |
| `compare_shell_band_norms.m` | Where each molecule's coefficient energy sits, per degree, per shell, and per degree-shell cell |
| `compare_basis_representation.m` | Representation accuracy only: delta-shell quadrature vs Fourier-Bessel least squares, no recovery |
| `export_bandlimited_volume.m` | Writes the preprocessed map and its band-limited form as MRC, for side-by-side viewing in ChimeraX |
