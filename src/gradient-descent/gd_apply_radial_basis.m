function profiles = gd_apply_radial_basis(coefficients,radial_basis,support,L_max)
%GD_APPLY_RADIAL_BASIS Evaluate each angular band's radial coefficient basis.

radial_count = size(radial_basis{1},1);
profiles = zeros(radial_count,(L_max+1)^2);
for l = 0:L_max
    columns = gd_band_indices(l);
    profiles(:,columns) = radial_basis{l+1}* ...
        coefficients(1:support(l+1),columns);
end
end
