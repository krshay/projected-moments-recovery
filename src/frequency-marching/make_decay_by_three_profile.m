function S = make_decay_by_three_profile(available_shells, terminal_shells)
%MAKE_DECAY_BY_THREE_PROFILE Largest canonical paired decay profile.
%   S is built backward with S(l)-S(l+2)=3. Only zeroing is allowed, so
%   S(l+1) never exceeds AVAILABLE_SHELLS(l+1).

if nargin < 2
    terminal_shells = 3;
end
validateattributes(available_shells, {'numeric'}, ...
    {'row', 'integer', 'positive'});
validateattributes(terminal_shells, {'numeric'}, ...
    {'scalar', 'integer', 'positive'});

n_bands = numel(available_shells);
S = zeros(size(available_shells));
S(max(1,n_bands-1):n_bands) = terminal_shells;
for idx = n_bands-2:-1:1
    S(idx) = S(idx+2) + 3;
end

if any(S > available_shells)
    bad = find(S > available_shells, 1, 'last');
    error('frequency_marching:insufficientShells', ...
        ['Band l=%d has %d available shells, but the canonical decay-by-3 ', ...
         'profile requires %d. Increase the volume grid or lower L_max.'], ...
        bad-1, available_shells(bad), S(bad));
end
end
