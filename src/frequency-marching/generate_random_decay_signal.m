function a = generate_random_decay_signal(S, rng_seed)
%GENERATE_RANDOM_DECAY_SIGNAL Draw a real signal with prescribed support.
%   A = GENERATE_RANDOM_DECAY_SIGNAL(S, SEED) returns an R-by-(L+1)^2
%   coefficient array with standard spherical-harmonic reality and exact
%   band supports S(l+1). Each angular band is normalized to unit Frobenius
%   norm so that error curves are not dominated by low-frequency energy.

if nargin < 2
    rng_seed = 1;
end
validateattributes(S, {'numeric'}, {'row', 'integer', 'nonnegative'});
assert(all(diff(S) <= 0), 'S must be nonincreasing.');

rng(rng_seed);
L_max = length(S) - 1;
R = max(S);
a = zeros(R, (L_max+1)^2);

for l = 0:L_max
    for r = 1:S(l+1)
        a(r, sh_index(l, 0)) = randn();
        for m = 1:l
            z = (randn() + 1i*randn()) / sqrt(2);
            a(r, sh_index(l, m)) = z;
            a(r, sh_index(l, -m)) = (-1)^m * conj(z);
        end
    end
    band_idx = (l^2+1):(l+1)^2;
    band_norm = norm(a(:, band_idx), 'fro');
    if band_norm > 0
        a(:, band_idx) = a(:, band_idx) / band_norm;
    end
end
end

function idx = sh_index(l, m)
idx = l^2 + m + l + 1;
end
