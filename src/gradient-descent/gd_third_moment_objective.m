function [value,gradient,details] = gd_third_moment_objective(x,problem)
%GD_THIRD_MOMENT_OBJECTIVE Josh's normalized projected-P3 objective alone.

coefficients = gd_decode_parameters(x,problem.parameterization,problem.base);
profiles = gd_apply_radial_basis(coefficients,problem.radial_basis, ...
    problem.support,problem.L_max);
[raw_loss,profile_gradient] = gd_third_moment_loss_gradient( ...
    profiles,problem.model,problem.observed_moment);
coefficient_gradient = gd_adjoint_radial_basis(profile_gradient, ...
    problem.radial_basis,problem.support,problem.L_max,size(coefficients,1));
value = raw_loss/problem.moment_scale;
gradient = gd_pullback_parameter_gradient( ...
    coefficient_gradient/problem.moment_scale,problem.parameterization);
if nargout > 2
    details = struct('coefficients',coefficients, ...
        'gradient_norm',norm(gradient),'normalized_p3_loss',value);
end
end
