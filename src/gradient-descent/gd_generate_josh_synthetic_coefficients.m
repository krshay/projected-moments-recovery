function coefficients = gd_generate_josh_synthetic_coefficients( ...
    L_max,shell_count,seed)
%GD_GENERATE_JOSH_SYNTHETIC_COEFFICIENTS Josh's unnormalized Gaussian signal.

rng(seed);
coefficients = zeros(shell_count,(L_max+1)^2);
for l = 0:L_max
    for radial_index = 1:shell_count
        coefficients(radial_index,gd_sh_index(l,0)) = randn();
        for m = 1:l
            value = (randn()+1i*randn())/sqrt(2);
            coefficients(radial_index,gd_sh_index(l,m)) = value;
            coefficients(radial_index,gd_sh_index(l,-m)) = ...
                (-1)^m*conj(value);
        end
    end
end
end
