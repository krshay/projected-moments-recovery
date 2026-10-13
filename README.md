# Provable Method of Moments for Cryo-EM

Code accompanying the paper by **Amnon Balanov, Tamir Bendory, Dan Edidin,
Josh Katz, Shay Kreymer, and Nir Sharon**.

MATLAB implementations of constructive frequency marching and direct
third-moment optimization, with numerical experiments on synthetic signals,
TRPV1, and the 80S ribosome.

## Getting started

Use MATLAB R2020a or newer. Optimization Toolbox enables `fminunc`;
an internal L-BFGS fallback is available. The combined noise-sweep plot uses
Python with Matplotlib.

From the repository root in MATLAB:

```matlab
addpath(genpath('src'));
addpath(genpath('experiments'));
addpath('tests');
run_frequency_marching_smoke_tests;
```

## Reproducing experiments

Download the molecular maps following [data/raw/README.md](data/raw/README.md).
Then follow the instructions for:

- [Frequency marching](experiments/frequency-marching/README.md): constructive recovery, noise, and sample-count experiments.
- [Direct third-moment optimization](experiments/gradient-descent/dynamic-range/README.md): unpinned recovery and noise robustness.

Implementations are in `src/`, experiment drivers in `experiments/`, and
validation checks in `tests/`. Generated results are saved under `results/`;
raw maps and generated results are excluded from version control.
