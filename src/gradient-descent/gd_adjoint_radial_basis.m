function coefficient_gradient = gd_adjoint_radial_basis( ...
    profile_gradient,radial_basis,support,L_max,max_support)
%GD_ADJOINT_RADIAL_BASIS Algebraic transpose for Josh's gradient convention.

coefficient_gradient = zeros(max_support,(L_max+1)^2);
for l = 0:L_max
    columns = gd_band_indices(l);
    coefficient_gradient(1:support(l+1),columns) = ...
        radial_basis{l+1}.'*profile_gradient(:,columns);
end
end
