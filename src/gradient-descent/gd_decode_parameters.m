function coefficients = gd_decode_parameters(x,parameterization,base)
%GD_DECODE_PARAMETERS Map real optimization variables to real-volume SH data.

assert(numel(x) == size(parameterization,1), ...
    'Parameter vector and parameterization sizes disagree.');
coefficients = base;
for j = 1:size(parameterization,1)
    radial_index = parameterization(j,1);
    l = parameterization(j,2);
    m = parameterization(j,3);
    component = parameterization(j,4);
    value = x(j);
    if component == 0
        coefficients(radial_index,gd_sh_index(l,0)) = value;
    elseif component == 1
        coefficients(radial_index,gd_sh_index(l,m)) = ...
            coefficients(radial_index,gd_sh_index(l,m))+value;
        coefficients(radial_index,gd_sh_index(l,-m)) = ...
            coefficients(radial_index,gd_sh_index(l,-m))+(-1)^m*value;
    else
        coefficients(radial_index,gd_sh_index(l,m)) = ...
            coefficients(radial_index,gd_sh_index(l,m))+1i*value;
        coefficients(radial_index,gd_sh_index(l,-m)) = ...
            coefficients(radial_index,gd_sh_index(l,-m))-(-1)^m*1i*value;
    end
end
end
