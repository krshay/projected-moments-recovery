function errors = gd_coefficient_errors(estimate,problem,with_alignment)
%GD_COEFFICIENT_ERRORS Honest total/unknown-band errors up to one reflection.
%   GD_COEFFICIENT_ERRORS(ESTIMATE,PROBLEM,false) skips the axial-rotation scan
%   and returns NaN for the aligned fields. GD_MINIMIZE logs this metric at
%   every iteration and uses only the raw errors, so the scan there is pure
%   cost; the restart harness passes false during optimization and true once.
%
%   The axial-rotation gauge. The projected third moment is invariant under
%   rotation about z for EVERY molecule: the m1+m2+m3 = 0 selection cancels the
%   phases exactly (measured at 1e-18). What normally removes that freedom is
%   pinning bands 0 and 1 to truth -- the l = 1 band is a set of vectors, and
%   fixing them fixes the orientation. A molecule with a Cn axis along z
%   (n >= 2) has an l = 1 band that is purely m = 0, so the pinned bands do not
%   move under the rotation at all and the freedom survives as a real gauge:
%   a whole circle of volumes reproduces both the moment and the pinned bands.
%
%   Scoring such a solution against one arbitrary point on that circle reports
%   a correct answer as a failure. Measured on an exactly-C4 TRPV1: three of
%   six restarts read 0.11, 0.36 and 0.50 raw, and 0.008, 0.007 and 0.018 once
%   aligned -- they had recovered the molecule at offsets of 85, 287 and 24
%   degrees. The other three sat at 0.4259 both before and after alignment, so
%   they are a genuinely different solution, not a rotation.
%
%   OPTIMIZED_BAND_ERROR is left exactly as it was, so every existing result
%   stays reproducible. The gauge-aligned figure is reported separately as
%   OPTIMIZED_BAND_ERROR_ALIGNED, together with AXIAL_GAUGE_FREE saying whether
%   the rotation is actually unbroken for this molecule. Read the aligned
%   number ONLY when that flag is true: when the pinned bands do break the
%   rotation, minimising over it can flatter a wrong answer by matching the
%   free bands to a rotated truth while the pinned bands match the unrotated
%   one, which is not a consistent volume.

truth = problem.coefficients_true;
reflected = problem.coefficients_reflected;
truth_norm = norm(truth(:));
error_truth = norm(estimate(:)-truth(:))/max(truth_norm,eps);
error_reflected = norm(estimate(:)-reflected(:))/max(truth_norm,eps);
use_reflection = error_reflected < error_truth;
if use_reflection
    target = reflected;
else
    target = truth;
end

unknown_columns = [];
for l = problem.optimized_bands
    unknown_columns = [unknown_columns,gd_band_indices(l)]; %#ok<AGROW>
end
unknown_truth = target(:,unknown_columns);
unknown_error = norm(estimate(:,unknown_columns)-unknown_truth,'fro')/ ...
    max(norm(unknown_truth,'fro'),eps);

band_relative = nan(1,problem.L_max+1);
for l = 0:problem.L_max
    rows = 1:problem.support(l+1);
    columns = gd_band_indices(l);
    band_relative(l+1) = norm(estimate(rows,columns)-target(rows,columns),'fro')/ ...
        max(norm(target(rows,columns),'fro'),eps);
end

% ---- is the axial rotation actually a gauge for this molecule? ----
% It is exactly when every pinned band is purely m = 0, since only then does
% rotating about z leave the pinned bands untouched.
pinned_bands = setdiff(0:problem.L_max,problem.optimized_bands);
scale = max(truth_norm,eps);
axial_gauge_free = true;
for l = pinned_bands
    for m = -l:l
        if m ~= 0 && norm(truth(:,gd_sh_index(l,m))) > 1e-8*scale
            axial_gauge_free = false;
            break
        end
    end
    if ~axial_gauge_free, break; end
end

% ---- error after aligning on the rotation circle (and the reflection) ----
if nargin < 3 || isempty(with_alignment), with_alignment = true; end
aligned_error = NaN;
aligned_alpha = NaN;
if with_alignment
    denominator = max(norm(unknown_truth,'fro'),eps);
    alphas = linspace(0,2*pi,1441);
    aligned_error = inf;
    for candidate = 1:2
        if candidate == 1, base = truth; else, base = reflected; end
        for a = alphas
            rotated = rotate_about_z(base,a,problem.L_max);
            value = norm(estimate(:,unknown_columns)-rotated(:,unknown_columns), ...
                'fro')/denominator;
            if value < aligned_error
                aligned_error = value;
                aligned_alpha = a;
            end
        end
    end
end

errors = struct('relative_error',min(error_truth,error_reflected), ...
    'error_to_truth',error_truth,'error_to_reflection',error_reflected, ...
    'optimized_band_error',unknown_error, ...
    'optimized_band_error_aligned',aligned_error, ...
    'axial_alpha',aligned_alpha, ...
    'axial_gauge_free',axial_gauge_free, ...
    'selected_reflection',use_reflection, ...
    'band_relative_error',band_relative);
end

function rotated = rotate_about_z(coefficients,alpha,L_max)
rotated = coefficients;
for l = 0:L_max
    for m = -l:l
        column = gd_sh_index(l,m);
        rotated(:,column) = exp(-1i*m*alpha)*coefficients(:,column);
    end
end
end
