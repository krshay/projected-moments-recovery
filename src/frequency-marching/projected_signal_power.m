function signal_power = projected_signal_power(a, S)
%PROJECTED_SIGNAL_POWER Mean clean coefficient power under Haar rotations.
%   SIGNAL_POWER = PROJECTED_SIGNAL_POWER(A,S) returns
%
%       E_R[||c(R)||_F^2] / SUM_l ((2*l+1)*S(l+1)),
%
%   using the number of coefficients in the degree-dependent expansion.

n_coeffs = size(a,2);
L_max = round(sqrt(n_coeffs)-1);
assert((L_max+1)^2 == n_coeffs, ...
    'A must have (L_max+1)^2 spherical-harmonic columns.');

assert(numel(S) == L_max+1, 'S must contain one shell count per degree.');
assert(all(S >= 0) && all(S <= size(a,1)), 'Invalid shell counts in S.');
total_power = 0;
for l = 0:L_max
    band_idx = (l^2+1):(l+1)^2;
    for m = -l:l
        projection_weight = spherical_harmonic_equator_weight(l,m);
        total_power = total_power + projection_weight^2 * ...
            norm(a(:,band_idx),'fro')^2/(2*l+1);
    end
end

num_active = sum((2*(0:L_max)+1).*S(:).');
signal_power = total_power/num_active;
end

function weight = spherical_harmonic_equator_weight(l,m)
% N_{l,m} P_l^m(0) in the convention used by the recovery code.
normalization = sqrt((2*l+1)*factorial(l-m)/(4*pi*factorial(l+m)));
if mod(l-m,2) ~= 0
    legendre_at_zero = 0;
elseif m >= 0
    legendre_at_zero = (-1)^((l-m)/2)*factorial(l+m)/ ...
        (2^l*factorial((l-m)/2)*factorial((l+m)/2));
else
    positive_m = -m;
    positive_value = (-1)^((l-positive_m)/2)*factorial(l+positive_m)/ ...
        (2^l*factorial((l-positive_m)/2)*factorial((l+positive_m)/2));
    legendre_at_zero = (-1)^positive_m * ...
        factorial(l-positive_m)/factorial(l+positive_m)*positive_value;
end
weight = normalization*legendre_at_zero;
end
