function [a_decay, S, diagnostic_before, diagnostic_after] = ...
    enforce_decay_by_three(a, available_shells, terminal_shells)
%ENFORCE_DECAY_BY_THREE Zero outer Fourier--Bessel coefficients by band.

if nargin < 3
    terminal_shells = 3;
end
L_max = numel(available_shells) - 1;
assert(size(a, 2) == (L_max+1)^2, ...
    'Coefficient columns and AVAILABLE_SHELLS disagree.');

diagnostic_before = check_decay_by_three(available_shells);
S = make_decay_by_three_profile(available_shells, terminal_shells);
a_decay = a;
for l = 0:L_max
    idx = (l^2+1):(l+1)^2;
    a_decay(S(l+1)+1:end, idx) = 0;
end
diagnostic_after = check_decay_by_three(S);
assert(diagnostic_after.satisfies_decay_by_three, ...
    'Internal error: enforced profile does not satisfy decay-by-3.');
end
