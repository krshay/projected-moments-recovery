function [radial_basis,rho_grid] = ...
    gd_fourier_bessel_radial_basis(metadata,radial_count)
%GD_FOURIER_BESSEL_RADIAL_BASIS Evaluate the repository's FB basis on radii.

L_max = metadata.L_max;
rho_max = metadata.rho_max;
support = metadata.available_shells;
rho_grid = linspace(1e-10,rho_max,radial_count).';
radial_basis = cell(1,L_max+1);
for l = 0:L_max
    radial_basis{l+1} = zeros(radial_count,support(l+1));
    for shell = 1:support(l+1)
        zero = metadata.bessel_zeros(l+1,shell);
        normalization = sqrt(2/rho_max^3)/ ...
            abs(spherical_bessel_j(l+1,zero));
        radial_basis{l+1}(:,shell) = normalization* ...
            spherical_bessel_j(l,zero*rho_grid/rho_max);
    end
end
end

function value = spherical_bessel_j(l,x)
value = zeros(size(x));
nonzero = abs(x) > 1e-12;
if l == 0
    value(nonzero) = sin(x(nonzero))./x(nonzero);
    value(~nonzero) = 1;
else
    value(nonzero) = sqrt(pi./(2*x(nonzero))).* ...
        besselj(l+0.5,x(nonzero));
end
end
