function clusters = gd_cluster_solutions(estimates,support,L_max,tolerance)
%GD_CLUSTER_SOLUTIONS Group restarts by the SOLUTION they found, not the loss.
%   CLUSTERS = GD_CLUSTER_SOLUTIONS(ESTIMATES,SUPPORT,L_MAX) takes a cell array
%   of recovered coefficient arrays, one per restart, and groups them by
%   pairwise relative distance over the optimized bands (l >= 2), up to the
%   reflection the third moment cannot resolve. CLUSTERS(1) is the largest
%   group. Each element carries the member indices and the group's radius.
%
%   WHY NOT CLUSTER ON THE LOSS. GD_CLUSTER_BASINS groups restarts whose
%   objective values differ by less than 1 percent. That is meaningful only
%   while the loss is dominated by how good the solution is. With an empirical
%   moment the loss is floor + excess, and as the sampling/noise floor grows
%   the excess becomes a vanishing fraction of it, so every restart falls
%   inside the relative window regardless of what it found. Measured on the
%   TRPV1 sweep, the spread of losses across 20 restarts fell from 7.7 percent
%   at SNR = Inf to 0.014 percent at SNR = 0.03, while the spread of
%   coefficient errors stayed at 19 percent -- twenty different answers, one
%   reported basin. The loss-based count therefore reads BEST exactly where
%   recovery is worst, reporting 20/20 in a single basin at SNR = 0.03 where
%   the median error is 1.56.
%
%   Distance is measured on the optimized bands only, since bands 0 and 1 are
%   pinned to truth and would otherwise dilute every comparison -- for TRPV1
%   they carry 93.5 percent of the energy.
%
%   TOLERANCE defaults to 0.05: two restarts are the same solution when their
%   relative separation is under 5 percent. This is a property of the answers,
%   so it does not drift with the noise level.
%
%   See also GD_CLUSTER_BASINS, GD_COEFFICIENT_ERRORS.

if nargin < 4 || isempty(tolerance)
    tolerance = 0.05;
end
n = numel(estimates);
assert(n > 0,'No estimates supplied.');

columns = [];
for l = 2:L_max
    columns = [columns,gd_band_indices(l)]; %#ok<AGROW>
end

vectors = cell(1,n);
for k = 1:n
    block = estimates{k}(1:max(support),columns);
    vectors{k} = block(:);
end

distance = zeros(n,n);
for a = 1:n
    for b = a+1:n
        u = vectors{a}; v = vectors{b};
        scale = max(norm(u),norm(v));
        if scale <= eps
            d = 0;
        else
            % up to the reflection P3 cannot resolve
            d = min(norm(u-v),norm(u+v))/scale;
        end
        distance(a,b) = d; distance(b,a) = d;
    end
end

% single-linkage grouping at the tolerance
label = zeros(1,n); current = 0;
for k = 1:n
    if label(k) > 0, continue; end
    current = current+1;
    queue = k; label(k) = current;
    while ~isempty(queue)
        node = queue(1); queue(1) = [];
        near = find(distance(node,:) <= tolerance & label == 0);
        label(near) = current;
        queue = [queue,near]; %#ok<AGROW>
    end
end

clusters = struct('members',{},'count',{},'radius',{});
for c = 1:current
    members = find(label == c);
    if numel(members) > 1
        sub = distance(members,members);
        radius = max(sub(:));
    else
        radius = 0;
    end
    clusters(c) = struct('members',members,'count',numel(members), ...
        'radius',radius); %#ok<AGROW>
end
[~,order] = sort([clusters.count],'descend');
clusters = clusters(order);
end
