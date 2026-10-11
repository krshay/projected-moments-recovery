# Raw molecular volumes

Raw volumes are intentionally not committed. Place one accepted file for each
molecule in this directory:

| Molecule | EMDB entry | Accepted filenames | Used by |
| --- | --- | --- | --- |
| TRPV1 (C4-symmetrised) | [EMD-8117](https://www.ebi.ac.uk/emdb/EMD-8117) | `TRPV1.mat` or `vol_TRPV1_true.mrc` | constructive recovery / frequency marching |
| TRPV1 (no imposed symmetry) | [EMD-24084](https://www.ebi.ac.uk/emdb/EMD-24084) | `vol_TRPV1_emd24084.mat` | direct third-moment optimisation |
| 80S ribosome | [EMD-2660](https://www.ebi.ac.uk/emdb/EMD-2660) | `emd_2660.mat` or `vol_80S_true.mrc` | both |

MAT files may use any variable name; the largest numeric 3-D array is loaded.
MRC files with modes 0, 1, 2, and 6 are supported directly.

EMD-8117 is a C4-symmetrised refinement of TRPV1: 94.5% of its degree
`l >= 2` coefficient energy sits in `m = 0 mod 4`. That near-degeneracy makes
the direct-optimisation landscape badly conditioned, so the direct experiments
use EMD-24084, a TRPV1 refinement without imposed symmetry, instead. The two
maps are different depositions and are not interchangeable.

The MRC filenames follow the conventions in
[`krshay/Non-Uniform-Second-Moment`](https://github.com/krshay/Non-Uniform-Second-Moment).
Do not commit downloaded molecular maps unless their redistribution terms are
known to permit it.
