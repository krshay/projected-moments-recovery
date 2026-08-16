function basins = gd_cluster_basins(losses,exitflags,zero_level)
%GD_CLUSTER_BASINS Group final losses into basins by single-linkage gap.
%   BASINS = GD_CLUSTER_BASINS(LOSSES,EXITFLAGS) sorts the final objective
%   values ascending and starts a new basin wherever consecutive values are
%   separated by more than 1 percent. BASINS(1) is therefore the lowest
%   cluster, and NUMEL(BASINS) is the number of distinct minima the restarts
%   found. Each element carries the mean loss of its members, the member
%   count, and how many of those members converged.
%
%   BASINS = GD_CLUSTER_BASINS(...,ZERO_LEVEL) sets the numerical-zero floor
%   (default 1e-12). Losses at or below it are treated as identical.
%
%   WHY THE FLOOR IS NEEDED. A purely relative gap test is meaningless once
%   the losses reach machine zero. With an exact population moment every
%   recovered restart lands somewhere around 1e-18, and neighbouring values
%   there routinely differ by far more than 1 percent of themselves -- 5.7e-18
%   and 1.1e-17 differ by 90 percent -- so the test splits one genuine basin
%   into several and undercounts the lowest cluster. Flooring first collapses
%   them into a single basin.
%
%   The objective is a relative squared residual, ||r||^2/||P3_observed||^2,
%   so the floor is scale free and 1e-12 is a relative residual of 1e-6. That
%   sits six orders below the 1e-4 coefficient-error success threshold and
%   nine orders below the smallest genuine spurious minimum seen in these
%   sweeps (1.5e-3), while machine-zero losses sit six orders below it again.
%   Nothing depends on the exact value inside that gap.
%
%   Empirical-moment runs have a sampling floor many orders above 1e-12, so
%   the floor never binds there and clustering is unchanged.
%
%   See also GD_RUN_JOSH_BASELINE_RESTARTS.

if nargin < 3 || isempty(zero_level)
    zero_level = 1e-12;
end
validateattributes(zero_level,{'numeric'},{'scalar','real','nonnegative'});
assert(numel(losses) == numel(exitflags), ...
    'losses and exitflags must have the same number of elements.');

losses = losses(:);
exitflags = exitflags(:);
floored = max(losses,zero_level);

tolerance = 0.01;
[sorted,order] = sort(floored);
labels = zeros(size(sorted));
index = 0;
for k = 1:numel(sorted)
    if k == 1 || abs(sorted(k)-sorted(k-1)) > tolerance*abs(sorted(k))
        index = index+1;
    end
    labels(k) = index;
end
assignment = zeros(size(floored));
assignment(order) = labels;

basins = struct('loss',{},'count',{},'converged',{});
for value = 1:index
    members = find(assignment == value);
    basins(value) = struct('loss',mean(losses(members)), ...
        'count',numel(members),'converged',sum(exitflags(members) > 0));
end
[~,order] = sort([basins.loss]);
basins = basins(order);
end
