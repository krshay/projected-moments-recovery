function errors = gd_coefficient_errors(estimate,problem)
%GD_COEFFICIENT_ERRORS Honest total/unknown-band errors up to one reflection.

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
errors = struct('relative_error',min(error_truth,error_reflected), ...
    'error_to_truth',error_truth,'error_to_reflection',error_reflected, ...
    'optimized_band_error',unknown_error, ...
    'selected_reflection',use_reflection, ...
    'band_relative_error',band_relative);
end
