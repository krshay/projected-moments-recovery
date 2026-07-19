# Frequency-marching experiments

This directory implements the numerical plan for parity-aware frequency
marching under the decay-by-3 condition.

## Experiments

1. Expand TRPV1 and 80S into the Fourier--Bessel basis, record that their native
   supports violate decay-by-3, and run population noiseless recovery with a
   theorem-compatible assumed support. The unmodified coefficients outside
   that support create model misspecification and a nonzero global error.
2. Zero outer coefficients to enforce `S(l)-S(l+2) >= 3`, then repeat the
   noiseless recovery.
3. For one normalized random signal with exact decay-by-3, plot recovery error
   against complex-noise standard deviation `sigma`.
4. For the same signal, plot recovery error against the number of observations.

The layout follows the two reference repositories: `run.m` reproduces all
figures, sweep logic lives in `run_experiment_*.m`, random seeds are fixed,
numeric outputs are saved as MAT files, and figures are exported as vector PDF.

## Running

First place the molecular volumes as described in
[`data/raw/README.md`](../../data/raw/README.md). Then, from MATLAB:

```matlab
run('experiments/frequency-marching/run.m')
```

Edit `frequency_marching_experiment_config.m` to change paper-scale parameters
or disable individual experiment groups. Generated coefficient caches go to
`data/processed/frequency-marching/`, numeric results to
`results/frequency-marching/`, and figures to `figures/frequency-marching/`.

The recovery code uses base MATLAB only. `exportgraphics` requires MATLAB
R2020a or newer.

The default real-volume demonstration uses `L_max = 5` and the canonical
support `[9 9 6 6 3 3]`. The driver asserts both sides of the experiment:
unmodified error must exceed `1e-2`, and the decay-enforced noiseless error
must be below `1e-8`.
