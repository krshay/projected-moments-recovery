function [coefficients,band_targets] = gd_impose_band_dynamic_range( ...
    coefficients,L_max,support,ratio,radial_basis)
%GD_IMPOSE_BAND_DYNAMIC_RANGE Rescale whole angular bands to a decay profile.
%   [A,TARGETS] = GD_IMPOSE_BAND_DYNAMIC_RANGE(A,L_MAX,SUPPORT,RATIO) sets
%   the Frobenius norm of each angular band to
%       ||A_0|| = 1,   ||A_l|| = RATIO^(-(l-1)/(L_max-1))   for l = 1..L_max,
%   so ||A_1|| = 1 and the top band satisfies ||A_L|| = 1/RATIO. RATIO is a
%   ratio between the strongest and weakest band, not a decade count;
%   RATIO = 1 leaves every band at unit norm (the equal-normalization
%   control case).
%
%   Each band is multiplied by a single positive real scalar. The standard
%   reality relation a_{l,-m} = (-1)^m conj(a_{l,m}) is therefore preserved
%   exactly, and the relative shell and channel structure inside each band
%   is untouched -- only the band-to-band energy balance changes.
%
%   With the optional RADIAL_BASIS (a 1-by-(L_max+1) cell of Phi_l), the
%   targets are applied to the PROFILE norms ||Phi_l A_l|| rather than to
%   the coefficient norms ||A_l||. Since u_l = Phi_l A_l is what enters the
%   projected third moment, this makes the observed moment independent of
%   which radial parameterization is being optimized, so a Fourier-Bessel
%   run and a delta-basis run can be compared against the identical P3.
%   For an identity radial basis the two forms coincide.
%
%   NOT interchangeable with impose_dynamic_range.m. That helper assigns a
%   geometric profile to every per-(band, shell) block, runs the progression
%   upward from 1 to KAPPA across the flattened (l, s) list, and leaves
%   bands 0 and 1 alone. This one fixes the total energy of each band,
%   includes bands 0 and 1, and decays with l. Using either in place of the
%   other changes the meaning of the sweep parameter.

validateattributes(L_max,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(ratio,{'numeric'},{'scalar','real','positive'});
assert(numel(support) == L_max+1,'support must have L_max+1 entries.');
assert(size(coefficients,2) == (L_max+1)^2, ...
    'coefficients must have (L_max+1)^2 columns.');

band_targets = ones(1,L_max+1);
for l = 1:L_max
    if L_max > 1
        band_targets(l+1) = ratio^(-(l-1)/(L_max-1));
    else
        band_targets(l+1) = 1;
    end
end

use_profile_norm = nargin >= 5 && ~isempty(radial_basis);
if use_profile_norm
    assert(numel(radial_basis) == L_max+1, ...
        'radial_basis must have L_max+1 entries.');
end

for l = 0:L_max
    rows = 1:support(l+1);
    columns = gd_band_indices(l);
    block = coefficients(rows,columns);
    if use_profile_norm
        current = norm(radial_basis{l+1}*block,'fro');
    else
        current = norm(block,'fro');
    end
    if current > 0
        coefficients(rows,columns) = block*(band_targets(l+1)/current);
    end
end
end
