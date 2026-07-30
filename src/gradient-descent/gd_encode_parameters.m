function x = gd_encode_parameters(coefficients,parameterization)
%GD_ENCODE_PARAMETERS Serialize standard-reality SH coefficients to real DOF.

x = zeros(size(parameterization,1),1);
for j = 1:size(parameterization,1)
    radial_index = parameterization(j,1);
    l = parameterization(j,2);
    m = parameterization(j,3);
    component = parameterization(j,4);
    value = coefficients(radial_index,gd_sh_index(l,m));
    if component == 2
        x(j) = imag(value);
    else
        x(j) = real(value);
    end
end
end
