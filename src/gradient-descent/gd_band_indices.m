function idx = gd_band_indices(l)
%GD_BAND_INDICES Coefficient columns belonging to angular band l.

idx = (l^2+1):(l+1)^2;
end
