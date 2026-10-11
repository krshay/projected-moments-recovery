function problem = gd_create_problem(coefficients_true,support,radial_basis,options)
%GD_CREATE_PROBLEM Assemble a reusable synthetic or molecular GD problem.
%   Bands A_0 and A_1 are copied from truth and are never optimization
%   variables. This is intentionally not an end-to-end recovery experiment.

if nargin < 4
    options = struct();
end
L_max = round(sqrt(size(coefficients_true,2))-1);
assert((L_max+1)^2 == size(coefficients_true,2));
assert(numel(support) == L_max+1);
assert(numel(radial_basis) == L_max+1);
radial_count = size(radial_basis{1},1);
for l = 0:L_max
    assert(isequal(size(radial_basis{l+1}), ...
        [radial_count,support(l+1)]), ...
        'Invalid radial basis dimensions for l=%d.',l);
end
assert(gd_reality_defect(coefficients_true,L_max,support) < 1e-10, ...
    'Truth coefficients do not satisfy standard spherical-harmonic reality.');

model = gd_precompute_model(L_max);
profiles_true = gd_apply_radial_basis(coefficients_true,radial_basis, ...
    support,L_max);
moment_source = get_option(options,'moment_source','exact');
switch lower(moment_source)
    case 'exact'
        observed_moment = gd_projected_third_moment(profiles_true,model);
    case 'empirical'
        empirical = get_option(options,'empirical',struct());
        observed_moment = gd_empirical_projected_third_moment( ...
            coefficients_true,support,radial_basis,model,empirical);
    otherwise
        error('Unknown moment source: %s',moment_source);
end
moment_scale = 0;
for block_index = 1:numel(observed_moment)
    moment_scale = moment_scale+sum(abs(observed_moment{block_index}(:)).^2);
end
assert(moment_scale > eps,'Observed projected third moment is numerically zero.');

% Bands pinned to truth. Default [0,1] is the historical convention. Pinning is
% a prefix 0:(k-1): [] optimises every band, [0] pins only the rotation-
% invariant band. Note that pinning l = 1 breaks the SO(3) gauge of the moment
% by fixing one representative of an orbit of exact solutions, which can leave
% spurious minima near the other representatives.
fixed_bands = get_option(options,'fixed_bands',[0,1]);
assert(isequal(fixed_bands(:).',0:numel(fixed_bands)-1), ...
    'fixed_bands must be a prefix 0:k-1 (e.g. [], [0] or [0,1]).');
first_free = numel(fixed_bands);
base = zeros(size(coefficients_true));
for l = fixed_bands
    base(:,gd_band_indices(l)) = coefficients_true(:,gd_band_indices(l));
end
parameterization = gd_make_parameterization(L_max,support,first_free);
second_moment = get_option(options,'second_moment', ...
    gd_second_moment_gram(coefficients_true,support,L_max));

problem = struct();
problem.name = get_option(options,'name','gradient-descent problem');
problem.kind = get_option(options,'kind','synthetic');
problem.L_max = L_max;
problem.support = support;
problem.radial_basis = radial_basis;
problem.coefficients_true = coefficients_true;
problem.profiles_true = profiles_true;
problem.coefficients_reflected = ...
    gd_reflect_coefficients(coefficients_true,L_max);
problem.base = base;
problem.parameterization = parameterization;
problem.x_true = gd_encode_parameters(coefficients_true,parameterization);
problem.model = model;
problem.observed_moment = observed_moment;
problem.moment_scale = moment_scale;
problem.moment_source = moment_source;
problem.second_moment = second_moment;
problem.regularizer_options = get_option(options,'regularizer_options',struct());
problem.success_threshold = get_option(options,'success_threshold',1e-5);
problem.fixed_bands = fixed_bands;
problem.optimized_bands = first_free:L_max;
if isempty(fixed_bands)
    problem.disclaimer = ['No bands are pinned; the solution is determined ', ...
        'only up to rotation and reflection.'];
else
    problem.disclaimer = sprintf(['Bands %s are fixed to their true values; ', ...
        'this is not fully end-to-end recovery.'],mat2str(fixed_bands));
end
end

function value = get_option(options,name,default)
if isfield(options,name)
    value = options.(name);
else
    value = default;
end
end
