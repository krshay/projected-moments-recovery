function summary = run_dynamic_range_sweep()
%RUN_DYNAMIC_RANGE_SWEEP Band-dynamic-range sweep on the Josh baseline.
%   Expands the configured volume once, then for each ratio D in
%   config.ratios rescales the band norms onto
%       ||A_0|| = ||A_1|| = 1,  ||A_l|| = D^(-(l-1)/(L-1)),
%   rebuilds the exact-population problem through GD_CREATE_PROBLEM, and
%   runs GD_RUN_JOSH_BASELINE_RESTARTS. P3, its gradient and the optimizer
%   are the existing machinery; nothing is reimplemented here.
%
%   Bands 0 and 1 stay pinned to truth (enforced by GD_CREATE_PROBLEM).

experiment_dir = fileparts(mfilename('fullpath'));
gradient_experiment_dir = fileparts(experiment_dir);
repo_root = fileparts(fileparts(gradient_experiment_dir));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','frequency-marching'));
addpath(fullfile(repo_root,'src','nonuniform'));
addpath(experiment_dir);

config = dynamic_range_config(repo_root);
create_directories(config);
diary(fullfile(config.logs_dir,'run.log'));
diary_cleanup = onCleanup(@() diary('off'));
total_timer = tic;

switch lower(config.source)
    case 'synthetic'
        source_name = sprintf('Synthetic L=%d, S_l=%d', ...
            config.synthetic.L_max,config.synthetic.shell_count);
        source_kind = 'synthetic';
    case 'real'
        source_name = config.volume.name;
        source_kind = 'real molecular volume';
    otherwise
        error('Unknown coefficient source: %s',config.source);
end

fprintf('============================================================\n');
fprintf(' Band dynamic-range sweep (Josh baseline objective)\n');
fprintf(' Source: %s   restarts: %d   ratios: [%s]\n', ...
    source_name,config.restart_count,num2str(config.ratios));
fprintf(' Initialization: %s   max iterations: %d\n', ...
    config.initialization,config.optimizer.max_iterations);
fprintf(' Exact population P3; A0,A1 pinned to truth\n');
fprintf(' Run directory: %s\n',config.run_dir);
fprintf('============================================================\n');

build_timer = tic;
if strcmpi(config.source,'synthetic')
    L_max = config.synthetic.L_max;
    shell_count = config.synthetic.shell_count;
    fprintf('\nGenerating synthetic coefficients ...\n');
    coefficients_raw = gd_generate_josh_synthetic_coefficients( ...
        L_max,shell_count,config.synthetic.seed);
    support = shell_count*ones(1,L_max+1);
    radial_basis = repmat({eye(shell_count)},1,L_max+1);
    metadata = struct('source','synthetic','L_max',L_max, ...
        'shell_count',shell_count,'seed',config.synthetic.seed, ...
        'native_reality_convention','standard (by construction)', ...
        'native_reality_defect',0);
    fprintf('  generation time: %.2fs\n',toc(build_timer));
else
    L_max = config.volume_options.L_max;
    assert(exist(config.volume.path,'file') == 2, ...
        'Missing volume: %s',config.volume.path);
    fprintf('\nExpanding %s ...\n',config.volume.name);
    % NDFT-exact shell projection, replacing the Fourier-Bessel least-squares
    % expansion. The FB path solves for all (l,s,m) jointly through normal
    % equations and truncates 25 of 432 directions, so it returns an
    % arbitrary member of a solution set that moves by 1.36 relative when the
    % shell count changes with the volume fixed. Each NDFT shell is an
    % independent quadrature projection, so the array is a well-defined
    % function of the volume (4.1e-17 under radial refinement).
    shell_count = config.shell_count;
    [coefficients_raw,support,metadata] = nu_expand_volume_shells( ...
        config.volume.path,struct('L_max',L_max, ...
            'grid_size',config.volume_options.grid_size, ...
            'rho_max',config.volume_options.rho_max, ...
            'shell_count',shell_count,'mode','exact'));
    % The shell samples ARE the coefficients: the radial basis is the identity
    % and there is no Phi layer, so 'fourier-bessel' coordinates no longer
    % exist for a real volume and the sweep runs in delta coordinates only.
    radial_basis = repmat({eye(shell_count)},1,L_max+1);
    metadata.gradient_descent_rho_grid = metadata.rho_grid;
    fprintf('  expansion time: %.2fs\n',toc(build_timer));
    fprintf('  standard reality defect: %.3e (NDFT, %s)\n', ...
        metadata.standard_reality_defect,metadata.sampling_mode);
end
support_native = support;
radial_basis_native = radial_basis;
modes = config.radial_basis_modes;
if ~iscell(modes)
    modes = {modes};
end
% Neither source has a Phi layer any more: the synthetic problem is defined
% on the identity radial basis, and the NDFT expansion returns shell samples
% directly. 'fourier-bessel' coordinates are therefore not available.
if any(strcmpi(modes,'fourier-bessel'))
    warning(['fourier-bessel coordinates are not available: the NDFT ', ...
        'expansion has no Phi layer. Running delta coordinates.']);
end
modes = {'delta'};
report_basis_conditioning(radial_basis_native,L_max);

results = cell(numel(config.ratios)*numel(modes),1);
entry = 0;
for ratio_index = 1:numel(config.ratios)
    ratio = config.ratios(ratio_index);
    % Scale bands so the PROFILE norms hit the targets. u_true, and therefore
    % the observed P3, is then identical across radial parameterizations, so
    % the modes below differ only in the coordinates the optimizer works in.
    [coefficients_scaled,band_targets] = gd_impose_band_dynamic_range( ...
        coefficients_raw,L_max,support_native,ratio,radial_basis_native);
    profiles_scaled = gd_apply_radial_basis(coefficients_scaled, ...
        radial_basis_native,support_native,L_max);
    radial_count = size(profiles_scaled,1);

    for mode_index = 1:numel(modes)
        mode = modes{mode_index};
        switch lower(mode)
            case 'fourier-bessel'
                coefficients = coefficients_scaled;
                support = support_native;
                radial_basis = radial_basis_native;
            case 'delta'
                coefficients = profiles_scaled;
                support = radial_count*ones(1,L_max+1);
                radial_basis = repmat({eye(radial_count)},1,L_max+1);
            otherwise
                error('Unknown radial basis mode: %s',mode);
        end
        entry = entry+1;

        fprintf('\n--- D = %g | radial basis: %s ---\n',ratio,mode);
        fprintf('  target profile norms: [%s]\n', ...
            num2str(band_targets,'%.4g '));
        report_band_norms(coefficients,radial_basis,L_max,support);
        fprintf('  real DOF over l>=2: %d\n', ...
            size(gd_make_parameterization(L_max,support),1));

        options = struct('name', ...
            sprintf('%s D=%g [%s]',source_name,ratio,mode), ...
            'kind',source_kind,'moment_source','exact', ...
            'success_threshold',config.success_threshold);
        problem = gd_create_problem(coefficients,support,radial_basis,options);
        round_trip = gd_third_moment_objective(problem.x_true,problem);
        fprintf('  round-trip loss at x_true: %.3e\n',round_trip);
        assert(round_trip < 1e-12, ...
            ['Truth does not round-trip through the parameterization ', ...
            '(loss %.3e).'],round_trip);

        restart_options = struct('restart_count',config.restart_count, ...
            'seed',config.seed,'optimizer',config.optimizer, ...
            'initialization',config.initialization);
        experiment = gd_run_josh_baseline_restarts(problem,restart_options);
        experiment.ratio = ratio;
        experiment.radial_basis_mode = mode;
        experiment.band_targets = band_targets;
        experiment.native_reality_convention = ...
            metadata.native_reality_convention;
        experiment.native_reality_defect = metadata.native_reality_defect;
        experiment.round_trip_loss = round_trip;
        if strcmpi(mode,'delta') && ~strcmpi(config.source,'synthetic')
            experiment.bessel_backsolve = backsolve_bessel_error(experiment, ...
                radial_basis_native,support_native,coefficients_scaled,L_max);
        end
        if strcmpi(mode,'fourier-bessel')
            experiment.profile_crosscheck = forward_profile_error( ...
                experiment,radial_basis,support,coefficients,L_max);
        end
        results{entry} = experiment;

        report_experiment(experiment,config);
        write_restart_csv(experiment,config.data_dir,ratio,mode);
        write_band_csv(experiment,config.data_dir,ratio,mode);
    end
end

report_sweep(results,config);
write_sweep_csv(results,config);

summary = struct();
summary.config = config;
summary.metadata = metadata;
summary.results = results;
summary.runtime_seconds = toc(total_timer);
summary.disclaimer = ['A0,A1 pinned to truth; band norms forced onto a ', ...
    'geometric profile. Not end-to-end recovery.'];
save(fullfile(config.data_dir,'summary.mat'),'summary','-v7.3');
save(fullfile(config.config_dir,'config.mat'),'config','-v7.3');
fprintf('\nFinished in %.2f seconds.\n',summary.runtime_seconds);
fprintf('Run directory: %s\n',config.run_dir);
clear diary_cleanup;
end

function report_band_norms(coefficients,radial_basis,L_max,support)
profile_values = zeros(1,L_max+1);
coefficient_values = zeros(1,L_max+1);
for l = 0:L_max
    block = coefficients(1:support(l+1),gd_band_indices(l));
    coefficient_values(l+1) = norm(block,'fro');
    profile_values(l+1) = norm(radial_basis{l+1}*block,'fro');
end
fprintf('  actual profile norms: [%s]\n',num2str(profile_values,'%.4g '));
fprintf('  actual coefficient norms: [%s]\n', ...
    num2str(coefficient_values,'%.4g '));
end

function report_basis_conditioning(radial_basis,L_max)
fprintf('\n  radial basis conditioning (cond of Phi_l):\n');
for l = 0:L_max
    fprintf('    l = %d: %.4e\n',l,cond(radial_basis{l+1}));
end
end

function result = forward_profile_error(experiment,radial_basis,support, ...
    coefficients_truth,L_max)
%FORWARD_PROFILE_ERROR Score a Bessel-coefficient estimate in profile space.
%   The Fourier-Bessel run recovers a_l[s,m] and is scored in coefficient
%   space. Mapping the estimate forward through Phi_l gives the radial
%   profile it implies, so the same restart can be scored in both coordinate
%   systems. If Phi is well conditioned the two errors should agree closely;
%   a large gap means the coefficients carry components the profile -- and
%   hence the moment -- cannot see.
restarts = experiment.restarts;
[~,best_index] = min([restarts.optimized_band_error]);
estimate = restarts(best_index).coefficients_estimate;

profile_estimate = gd_apply_radial_basis(estimate,radial_basis,support,L_max);
profile_truth = gd_apply_radial_basis(coefficients_truth,radial_basis, ...
    support,L_max);

unknown = [];
for l = 2:L_max
    unknown = [unknown,gd_band_indices(l)]; %#ok<AGROW>
end
result = struct();
result.best_restart = best_index;
result.coefficient_error = restarts(best_index).optimized_band_error;
result.profile_error = ...
    norm(profile_estimate(:,unknown)-profile_truth(:,unknown),'fro')/ ...
    max(norm(profile_truth(:,unknown),'fro'),eps);
result.band_profile_error = nan(1,L_max+1);
for l = 0:L_max
    columns = gd_band_indices(l);
    result.band_profile_error(l+1) = ...
        norm(profile_estimate(:,columns)-profile_truth(:,columns),'fro')/ ...
        max(norm(profile_truth(:,columns),'fro'),eps);
end
end

function result = backsolve_bessel_error(experiment,radial_basis_native, ...
    support_native,coefficients_truth,L_max)
%BACKSOLVE_BESSEL_ERROR Map a recovered radial profile back to Bessel space.
%   The delta-basis run recovers u_l[k,m] directly. Solving Phi_l a_l = u_l
%   in least squares recovers the Bessel coefficients implied by that
%   profile, so we can tell whether a small profile error corresponds to a
%   small coefficient error -- i.e. whether the difficulty lives in the
%   P3 -> u step or in the u -> a step.
restarts = experiment.restarts;
[~,best_index] = min([restarts.optimized_band_error]);
profile_estimate = restarts(best_index).coefficients_estimate;

estimate = zeros(size(coefficients_truth));
for l = 0:L_max
    columns = gd_band_indices(l);
    estimate(1:support_native(l+1),columns) = ...
        radial_basis_native{l+1}\profile_estimate(:,columns);
end

unknown = [];
for l = 2:L_max
    unknown = [unknown,gd_band_indices(l)]; %#ok<AGROW>
end
truth_unknown = coefficients_truth(:,unknown);
result = struct();
result.best_restart = best_index;
result.profile_error = restarts(best_index).optimized_band_error;
result.coefficient_error = ...
    norm(estimate(:,unknown)-truth_unknown,'fro')/ ...
    max(norm(truth_unknown,'fro'),eps);
result.band_coefficient_error = nan(1,L_max+1);
for l = 0:L_max
    columns = gd_band_indices(l);
    rows = 1:support_native(l+1);
    result.band_coefficient_error(l+1) = ...
        norm(estimate(rows,columns)-coefficients_truth(rows,columns),'fro')/ ...
        max(norm(coefficients_truth(rows,columns),'fro'),eps);
end
end

function report_experiment(experiment,config)
restarts = experiment.restarts;
optimized = [restarts.optimized_band_error];
losses = [restarts.objective];
total = [restarts.relative_error];
[best_optimized,best_index] = min(optimized);
best_loss = min(losses);
success_total = sum(total < config.success_threshold);
success_optimized = sum(optimized < config.success_threshold);

fprintf('\n  === D = %g | %s summary (%d restarts) ===\n', ...
    experiment.ratio,experiment.radial_basis_mode,numel(restarts));
fprintf('  success rate (total error < %.0e):      %d/%d (%.0f%%)\n', ...
    config.success_threshold,success_total,numel(restarts), ...
    100*success_total/numel(restarts));
fprintf('  success rate (l>=2 error < %.0e):       %d/%d (%.0f%%)\n', ...
    config.success_threshold,success_optimized,numel(restarts), ...
    100*success_optimized/numel(restarts));
fprintf('  best-of-%d relative error over l>=2:    %.4e (restart %d)\n', ...
    numel(restarts),best_optimized,best_index);
fprintf('  best-of-%d loss:                        %.4e\n', ...
    numel(restarts),best_loss);
fprintf('  loss at the best-error restart:         %.4e\n', ...
    losses(best_index));
fprintf('  median relative error over l>=2:        %.4e\n',median(optimized));
fprintf('  per-band relative error at restart %d:\n',best_index);
band_errors = restarts(best_index).band_relative_error;
for l = 0:numel(band_errors)-1
    marker = '';
    if l <= 1
        marker = '  (pinned)';
    end
    fprintf('    l = %d: %.4e%s\n',l,band_errors(l+1),marker);
end
if isfield(experiment,'profile_crosscheck') && ...
        ~isempty(experiment.profile_crosscheck)
    p = experiment.profile_crosscheck;
    fprintf(['  cross-checked in profile space (Phi_l a_est) at restart ', ...
        '%d:\n'],p.best_restart);
    fprintf('    coefficient error over l>=2: %.4e\n',p.coefficient_error);
    fprintf('    profile error over l>=2:     %.4e\n',p.profile_error);
    for l = 2:numel(p.band_profile_error)-1
        fprintf('      l = %d profile error: %.4e\n', ...
            l,p.band_profile_error(l+1));
    end
end
if isfield(experiment,'bessel_backsolve') && ...
        ~isempty(experiment.bessel_backsolve)
    b = experiment.bessel_backsolve;
    fprintf(['  back-solved to Bessel space (Phi_l \\ u_est) at restart ', ...
        '%d:\n'],b.best_restart);
    fprintf('    profile error over l>=2:     %.4e\n',b.profile_error);
    fprintf('    coefficient error over l>=2: %.4e\n',b.coefficient_error);
    for l = 2:numel(b.band_coefficient_error)-1
        fprintf('      l = %d coefficient error: %.4e\n', ...
            l,b.band_coefficient_error(l+1));
    end
end
end

function report_sweep(results,config)
fprintf('\n============================================================\n');
fprintf(' Sweep summary (%d restarts per D)\n',config.restart_count);
fprintf('============================================================\n');
fprintf(['      D   radial basis   success   best l>=2   median l>=2    ', ...
    'best loss   converged\n']);
for index = 1:numel(results)
    experiment = results{index};
    restarts = experiment.restarts;
    optimized = [restarts.optimized_band_error];
    successes = sum(optimized < config.success_threshold);
    converged = sum([restarts.exitflag] > 0);
    fprintf('%7g   %-12s   %3d/%-3d   %9.3e   %11.3e   %10.3e   %3d/%-3d\n', ...
        experiment.ratio,experiment.radial_basis_mode,successes, ...
        numel(restarts),min(optimized),median(optimized), ...
        min([restarts.objective]),converged,numel(restarts));
end
fprintf(['\n"converged" counts restarts with exitflag > 0 (a real ', ...
    'stopping criterion);\nexitflag 0 means MaxIterations was hit ', ...
    'instead.\n']);
end

function write_sweep_csv(results,config)
path = fullfile(config.data_dir,'sweep_summary.csv');
fid = fopen(path,'w');
assert(fid >= 0,'Could not open %s.',path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,['ratio,radial_basis_mode,restarts,successes,', ...
    'best_optimized_band_error,median_optimized_band_error,best_loss,', ...
    'converged_restarts,backsolved_coefficient_error\n']);
for index = 1:numel(results)
    experiment = results{index};
    restarts = experiment.restarts;
    optimized = [restarts.optimized_band_error];
    backsolved = NaN;
    if isfield(experiment,'bessel_backsolve') && ...
            ~isempty(experiment.bessel_backsolve)
        backsolved = experiment.bessel_backsolve.coefficient_error;
    end
    fprintf(fid,'%g,%s,%d,%d,%.12e,%.12e,%.12e,%d,%.12e\n', ...
        experiment.ratio,experiment.radial_basis_mode,numel(restarts), ...
        sum(optimized < config.success_threshold),min(optimized), ...
        median(optimized),min([restarts.objective]), ...
        sum([restarts.exitflag] > 0),backsolved);
end
clear cleanup;
end

function write_restart_csv(experiment,data_dir,ratio,mode)
path = fullfile(data_dir,sprintf('restarts_D_%g_%s.csv',ratio,mode));
fid = fopen(path,'w');
assert(fid >= 0,'Could not open %s.',path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,['restart,seed,loss,relative_error,optimized_band_error,', ...
    'error_to_truth,error_to_reflection,gradient_norm,iterations,', ...
    'exitflag,runtime_seconds,reality_defect\n']);
for index = 1:numel(experiment.restarts)
    r = experiment.restarts(index);
    fprintf(fid,'%d,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%d,%.4f,%.4e\n', ...
        r.restart,r.seed,r.objective,r.relative_error,r.optimized_band_error, ...
        r.error_to_truth,r.error_to_reflection,r.gradient_norm,r.iterations, ...
        r.exitflag,r.runtime_seconds,r.reality_defect);
end
clear cleanup;
end

function write_band_csv(experiment,data_dir,ratio,mode)
path = fullfile(data_dir,sprintf('band_errors_D_%g_%s.csv',ratio,mode));
fid = fopen(path,'w');
assert(fid >= 0,'Could not open %s.',path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'restart,band,target_norm,band_relative_error,pinned\n');
for index = 1:numel(experiment.restarts)
    r = experiment.restarts(index);
    for l = 0:numel(r.band_relative_error)-1
        fprintf(fid,'%d,%d,%.12e,%.12e,%d\n',r.restart,l, ...
            experiment.band_targets(l+1),r.band_relative_error(l+1),l <= 1);
    end
end
clear cleanup;
end

function create_directories(config)
directories = {config.config_dir,config.data_dir,config.logs_dir};
for index = 1:numel(directories)
    if exist(directories{index},'dir') ~= 7
        mkdir(directories{index});
    end
end
end
