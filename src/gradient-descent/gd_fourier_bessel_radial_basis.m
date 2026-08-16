function [radial_basis,rho_grid] = ...
    gd_fourier_bessel_radial_basis(metadata,radial_count)
%GD_FOURIER_BESSEL_RADIAL_BASIS Evaluate the repository's FB basis on radii.
%   Samples at cell midpoints rho_max*((1:K)-0.5)/K. BOTH endpoints of the
%   interval are degenerate for this basis and must be avoided:
%     rho = 0       -- j_l(x) ~ x^l vanishes for every l >= 1, so the row is
%                      identically zero in all but the l = 0 band;
%     rho = rho_max -- k_{l,s} is by construction a zero of j_l, so
%                      j_l(k_{l,s} * rho_max/rho_max) = 0 for EVERY shell,
%                      making the row identically zero in all bands.
%   The previous grid linspace(1e-10, rho_max, K) included both, costing two
%   usable rows and capping rank(Phi_l) at K-2. At K = 12 against 12 shells
%   that alone forced rank deficiency (rank(Phi_5) = 10 of 12, cond ~ 1e17);
%   with midpoints the same K = 12 basis is full rank at cond <= 6.8e3.
%   At K = 16 the two dead rows were affordable, so this change leaves the
%   K = 16 conditioning essentially unaltered (cond(Phi_5) 5.35 -> 5.40).

L_max = metadata.L_max;
rho_max = metadata.rho_max;
support = metadata.available_shells;
rho_grid = rho_max*(((1:radial_count).'-0.5)/radial_count);
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
