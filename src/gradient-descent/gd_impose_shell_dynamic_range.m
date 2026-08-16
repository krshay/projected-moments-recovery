function [coefficients,shell_targets] = gd_impose_shell_dynamic_range( ...
    coefficients,L_max,support,ratio)
%GD_IMPOSE_SHELL_DYNAMIC_RANGE Radial decay within bands, equal band norms.
%   [A,TARGETS] = GD_IMPOSE_SHELL_DYNAMIC_RANGE(A,L_MAX,SUPPORT,RATIO) sets
%   the per-(band, shell) row norms of every band to a common geometric
%   profile
%       ||A_l[s,:]|| proportional to RATIO^(-(s-1)/(S_l-1)),   s = 1..S_l,
%   so the innermost shell of each band carries RATIO times the outermost,
%   and then rescales each band to unit Frobenius norm. Every band therefore
%   ends with the SAME total energy and the SAME radial decay, isolating
%   within-band radial structure from band-to-band imbalance.
%
%   This is the complement of GD_IMPOSE_BAND_DYNAMIC_RANGE, which varies the
%   energy across bands and leaves the radial structure inside each band
%   untouched. Real molecular volumes have far more of the former kind of
%   range than the latter: for the 80S ribosome in the delta basis the
%   band-to-band spread is 8.1 while the within-band shell spread is 178 to
%   2785, geometric mean 856. RATIO near 1e3 is therefore the realistic
%   setting.
%
%   Rows are scaled by positive real scalars, so the standard reality
%   relation a_{l,-m} = (-1)^m conj(a_{l,m}) is preserved exactly.
%
%   NOTE. Row scaling is not spectrally neutral: imposing a radial profile
%   also changes sigma_1/sigma_n of each band. The two knobs are coupled, so
%   callers should measure and report the realized within-band conditioning
%   rather than assuming it is unchanged.

validateattributes(L_max,{'numeric'},{'scalar','integer','nonnegative'});
validateattributes(ratio,{'numeric'},{'scalar','real','positive'});
assert(numel(support) == L_max+1,'support must have L_max+1 entries.');
assert(size(coefficients,2) == (L_max+1)^2, ...
    'coefficients must have (L_max+1)^2 columns.');

shell_targets = cell(1,L_max+1);
for l = 0:L_max
    count = support(l+1);
    if count > 1
        targets = ratio.^(-(0:count-1).'/(count-1));
    else
        targets = 1;
    end
    shell_targets{l+1} = targets;

    rows = 1:count;
    columns = gd_band_indices(l);
    block = coefficients(rows,columns);
    for s = 1:count
        current = norm(block(s,:));
        if current > 0
            block(s,:) = block(s,:)*(targets(s)/current);
        end
    end
    total = norm(block,'fro');
    if total > 0
        block = block/total;
    end
    coefficients(rows,columns) = block;
end
end
