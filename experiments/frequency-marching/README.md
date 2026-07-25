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

The finite-sample curves focus on the regime where relative errors remain below
`0.5`; larger errors are treated as outside the useful numerical range for the
paper figures. Full-support and global errors remain in the MAT results, and the
MRC maps contain the complete current estimate.

The layout follows the two reference repositories: `run.m` reproduces all
figures, sweep logic lives in `run_experiment_*.m`, random seeds are fixed,
numeric outputs are saved as MAT files, and figures are exported as vector PDF.
The real-volume experiment also writes mode-2 MRC maps for the coefficient
truth and reconstruction in both the unmodified and decay-enforced cases, and
exports a CSV table containing the condition number of every marching and
backfill system.

## Running

First place the molecular volumes as described in
[`data/raw/README.md`](../../data/raw/README.md). Then, from MATLAB:

```matlab
run('experiments/frequency-marching/run.m')
```

Edit `frequency_marching_experiment_config.m` to change paper-scale parameters
or disable individual experiment groups. Each execution creates a timestamped
run folder under `results/frequency-marching/`, for example
`results/frequency-marching/2026-07-25_14-30-00/`. Numeric outputs go to that
run folder's `results/` subdirectory, PDF figures to `figures/`, and MRC maps
to `results/mrc/`. The run folder also contains `config/` with `config.mat`,
`config.txt`, and copies of the MATLAB experiment files used for that run.
Generated coefficient caches go to `data/processed/frequency-marching/`.

The recovery code uses base MATLAB only. `exportgraphics` requires MATLAB
R2020a or newer.

The real-volume experiments use `L_max = 9` and canonical support
`[15 15 12 12 9 9 6 6 3 3]`. The finite-sample noise and observation sweeps
use `L_max = 5` with support `[9 9 6 6 3 3]`, ten trials, and larger
observation counts than the initial exploratory run. The curves plot medians
with interquartile error bars. The driver asserts both
sides of the noiseless molecular experiment:
unmodified error must exceed `1e-2`, and the decay-enforced noiseless error
must be below `1e-8`.
