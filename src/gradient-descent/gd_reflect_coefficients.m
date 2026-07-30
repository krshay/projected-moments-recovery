function reflected = gd_reflect_coefficients(coefficients,L_max)
%GD_REFLECT_COEFFICIENTS Apply the global spatial reflection a_lm -> (-1)^l a_lm.

reflected = coefficients;
for l = 0:L_max
    reflected(:,gd_band_indices(l)) = (-1)^l* ...
        reflected(:,gd_band_indices(l));
end
end
