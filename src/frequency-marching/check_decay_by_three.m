function diagnostic = check_decay_by_three(S)
%CHECK_DECAY_BY_THREE Check the parity-aware radial support condition.
%   The equatorial projection kills l-m odd terms. Consequently the march
%   uses S(l)-S(l+2)>=3, with the final two active bands retaining at least
%   three shells. This is the decay-by-3 convention used by the numerical
%   experiments.

validateattributes(S, {'numeric'}, {'row', 'integer', 'nonnegative'});
gaps = S(1:end-2) - S(3:end);
nonincreasing = all(diff(S) <= 0);
terminal_ok = numel(S) < 2 || all(S(max(1,end-1):end) >= 3);

diagnostic = struct();
diagnostic.support = S;
diagnostic.parity_gaps = gaps;
diagnostic.nonincreasing = nonincreasing;
diagnostic.terminal_bands_have_three_shells = terminal_ok;
diagnostic.satisfies_decay_by_three = ...
    nonincreasing && all(gaps >= 3) && terminal_ok;
diagnostic.minimum_parity_gap = min([Inf, gaps]);
end
