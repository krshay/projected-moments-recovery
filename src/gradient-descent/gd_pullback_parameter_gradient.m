function gradient = gd_pullback_parameter_gradient(coefficient_gradient,parameterization)
%GD_PULLBACK_PARAMETER_GRADIENT Pull a Wirtinger-style gradient to real DOF.
%   COEFFICIENT_GRADIENT is defined by
%       dF = 2*real(sum(COEFFICIENT_GRADIENT .* dA)).
%   This is the convention used by Josh's analytic third-moment gradient.

gradient = zeros(size(parameterization,1),1);
for j = 1:size(parameterization,1)
    radial_index = parameterization(j,1);
    l = parameterization(j,2);
    m = parameterization(j,3);
    component = parameterization(j,4);
    if component == 0
        value = coefficient_gradient(radial_index,gd_sh_index(l,0));
    elseif component == 1
        value = coefficient_gradient(radial_index,gd_sh_index(l,m))+ ...
            (-1)^m*coefficient_gradient(radial_index,gd_sh_index(l,-m));
    else
        value = 1i*coefficient_gradient(radial_index,gd_sh_index(l,m))- ...
            (-1)^m*1i*coefficient_gradient(radial_index,gd_sh_index(l,-m));
    end
    gradient(j) = 2*real(value);
end
end
