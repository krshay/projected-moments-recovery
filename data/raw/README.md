# Raw molecular volumes

Raw volumes are intentionally not committed. Place one accepted file for each
molecule in this directory:

| Molecule | Accepted filenames |
| --- | --- |
| TRPV1 | `TRPV1.mat` or `vol_TRPV1_true.mrc` |
| 80S | `emd_2660.mat` or `vol_80S_true.mrc` |

MAT files may use any variable name; the largest numeric 3-D array is loaded.
MRC files with modes 0, 1, 2, and 6 are supported directly.

The MRC filenames follow the conventions in
[`krshay/Non-Uniform-Second-Moment`](https://github.com/krshay/Non-Uniform-Second-Moment).
Do not commit downloaded molecular maps unless their redistribution terms are
known to permit it.
