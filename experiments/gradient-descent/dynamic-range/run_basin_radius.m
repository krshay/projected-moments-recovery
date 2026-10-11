function summary = run_basin_radius(experiment,distances,restart_count)
%RUN_BASIN_RADIUS How far from the truth can gradient descent start and return?
%   RUN_BASIN_RADIUS(EXPERIMENT) uses the EXACT population third moment -- no
%   sampling, no noise -- and initializes at a controlled relative distance
%   from the true coefficients, sweeping that distance from near zero outward.
%   It answers a question the random-restart experiments cannot: not how often
%   a random start succeeds, but how large the basin around the truth is.
%
%   EXPERIMENT is 'real' (default), 'synthetic' or 'shell', matching
%   RUN_NOISE_SWEEP so the signals are identical across the two studies.
%
%   RUN_BASIN_RADIUS(EXPERIMENT,DISTANCES,RESTARTS) overrides the grid and the
%   number of random directions per distance.
%
%   THE PERTURBATION. Each start is
%       x0 = x_true + d * ||x_true|| * v/||v||,     v ~ N(0,I),
%   so ||x0 - x_true|| / ||x_true|| = d exactly, in a uniformly random
%   direction. This differs from the 'near_truth' mode in
%   GD_RUN_JOSH_BASELINE_RESTARTS, which applies a multiplicative
%   x_true.*(1+p*randn): that scales each coordinate by its own magnitude, so
%   the realized distance depends on the coefficient distribution and is not
%   directly comparable across signals. Here d is the distance.
%
%   THE REFERENCE ROW. The sweep also runs the standard per-band truth-scaled
%   Gaussian start used by every random-restart experiment in this repository,
%   and reports the relative distance it actually lands at. That anchors the
%   basin radius against the existing results: if the random start sits far
%   outside the capture radius, its hit rate is explained by distance alone.
%
%   Because the moment is exact, the 1e-4 success criterion is meaningful here
%   and is used, unlike in RUN_NOISE_SWEEP where the sampling floor makes it
%   unreachable.

experiment_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(experiment_dir)));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','frequency-marching'));

if nargin < 1 || isempty(experiment)
    experiment = 'real';
end
if nargin < 2 || isempty(distances)
    distances = [0.01,0.03,0.1,0.2,0.35,0.5,0.75,1.0,1.5];
end
if nargin < 3 || isempty(restart_count)
    restart_count = 20;
end
distances = sort(distances(:).','ascend');

L_max = 5;
shell_count = 12;
radial_count = 16;
seed = 42;
shell_ratio = 1000;
threshold = 1e-4;
max_iterations = 3000;
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');
volume_file = 'vol_TRPV1_true.mrc';

output_dir = fullfile(repo_root,'results','gradient-descent','basin-radius');
if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end
diary(fullfile(output_dir,sprintf('basin_radius_%s.log',experiment)));
diary_cleanup = onCleanup(@() diary('off'));
total_timer = tic;

fprintf('============================================================\n');
fprintf(' Basin radius: %s\n',experiment);
fprintf(' EXACT population moment (no sampling, no noise)\n');
fprintf(' %d random directions per distance, %d iterations\n', ...
    restart_count,max_iterations);
fprintf(' distances: [%s]\n',num2str(distances));
fprintf('============================================================\n');

[coefficients,support,radial_basis,label] = build_signal(experiment, ...
    repo_root,L_max,shell_count,radial_count,shell_ratio,volume_file);
fprintf('\n  signal: %s\n',label);

problem = gd_create_problem(coefficients,support,radial_basis, ...
    struct('name',label,'kind','basin-radius','moment_source','exact', ...
           'success_threshold',threshold));
round_trip = gd_third_moment_objective(problem.x_true,problem);
fprintf('  round-trip loss at x_true: %.3e\n',round_trip);
assert(round_trip < 1e-12,'Truth does not round-trip (%.3e).',round_trip);
fprintf('  DOF: %d   ||x_true|| = %.4e\n', ...
    numel(problem.x_true),norm(problem.x_true));

records = cell(numel(distances)+1,1);
for index = 1:numel(distances)
    records{index} = run_distance(distances(index),problem,seed, ...
        restart_count,optimizer,max_iterations,threshold,output_dir,label);
end
records{end} = run_random_reference(problem,seed,restart_count,optimizer, ...
    max_iterations,threshold,output_dir,label);

report_summary(records,experiment,restart_count);
write_summary(records,output_dir,experiment,restart_count);
summary = struct('experiment',experiment,'distances',distances, ...
    'records',{records},'restart_count',restart_count, ...
    'runtime_seconds',toc(total_timer));
save(fullfile(output_dir,sprintf('basin_radius_%s.mat',experiment)), ...
    'summary','-v7.3');
fprintf('\nFinished in %.1f s (%.1f min). Written to %s\n', ...
    summary.runtime_seconds,summary.runtime_seconds/60,output_dir);
clear diary_cleanup;
end

%% ==================== one distance ====================
function record = run_distance(distance,problem,seed,restart_count, ...
    optimizer,max_iterations,threshold,output_dir,label)
fprintf('\n============================================================\n');
fprintf(' relative distance d = %g\n',distance);
fprintf('============================================================\n');
timer = tic;
x_true = problem.x_true;
true_norm = norm(x_true);
restarts = repmat(empty_restart(),restart_count,1);
realized = zeros(1,restart_count);
for index = 1:restart_count
    rng(seed+index-1);
    direction = randn(size(x_true));
    direction = direction/norm(direction);
    x0 = x_true+distance*true_norm*direction;
    realized(index) = norm(x0-x_true)/true_norm;
    restarts(index) = single_restart(x0,problem,optimizer,threshold, ...
        seed+index-1,index);
    r = restarts(index);
    fprintf('  dir %2d: loss=%.3e  l>=2 error=%.4e  iters=%d  %.1fs%s\n', ...
        index,r.objective,r.optimized_band_error,r.iterations, ...
        r.runtime_seconds,label_success(r.success));
end
record = summarize(restarts,realized,distance,'perturbed',restart_count, ...
    max_iterations,threshold,timer);
report_record(record,restart_count);
write_cell_csv(record,output_dir,label,sprintf('d_%g',distance));
end

%% ==================== random-start reference ====================
function record = run_random_reference(problem,seed,restart_count, ...
    optimizer,max_iterations,threshold,output_dir,label)
fprintf('\n============================================================\n');
fprintf(' reference: standard per-band truth-scaled random start\n');
fprintf('============================================================\n');
timer = tic;
x_true = problem.x_true;
true_norm = norm(x_true);
band_scale = zeros(1,problem.L_max+1);
for l = 2:problem.L_max
    band = problem.coefficients_true(1:problem.support(l+1), ...
        gd_band_indices(l));
    band_scale(l+1) = norm(band,'fro')/sqrt(problem.support(l+1)*(2*l+1));
end
scale = zeros(size(problem.parameterization,1),1);
for index = 1:numel(scale)
    scale(index) = band_scale(problem.parameterization(index,2)+1);
end

restarts = repmat(empty_restart(),restart_count,1);
realized = zeros(1,restart_count);
for index = 1:restart_count
    rng(seed+index-1);
    x0 = randn(size(scale)).*scale;
    realized(index) = norm(x0-x_true)/true_norm;
    restarts(index) = single_restart(x0,problem,optimizer,threshold, ...
        seed+index-1,index);
    r = restarts(index);
    fprintf('  start %2d: d=%.3f  loss=%.3e  l>=2 error=%.4e  %.1fs%s\n', ...
        index,realized(index),r.objective,r.optimized_band_error, ...
        r.runtime_seconds,label_success(r.success));
end
record = summarize(restarts,realized,NaN,'random',restart_count, ...
    max_iterations,threshold,timer);
report_record(record,restart_count);
write_cell_csv(record,output_dir,label,'random');
end

%% ==================== shared ====================
function record = single_restart(x0,problem,optimizer,threshold,seed,index)
timer = tic;
objective = @(value) gd_third_moment_objective(value,problem);
[x,loss,info] = gd_minimize(objective,x0,optimizer);
[~,gradient] = objective(x);
estimate = gd_decode_parameters(x,problem.parameterization,problem.base);
errors = gd_coefficient_errors(estimate,problem);
record = struct('restart',index,'seed',seed,'objective',loss, ...
    'relative_error',errors.relative_error, ...
    'optimized_band_error',errors.optimized_band_error, ...
    'gradient_norm',norm(gradient),'iterations',info.iterations, ...
    'exitflag',info.exitflag,'runtime_seconds',toc(timer), ...
    'success',errors.optimized_band_error < threshold);
end

function record = empty_restart()
record = struct('restart',[],'seed',[],'objective',[],'relative_error',[], ...
    'optimized_band_error',[],'gradient_norm',[],'iterations',[], ...
    'exitflag',[],'runtime_seconds',[],'success',[]);
end

function record = summarize(restarts,realized,distance,kind,restart_count, ...
    max_iterations,threshold,timer) %#ok<INUSD>
errors = [restarts.optimized_band_error];
losses = [restarts.objective];
basins = gd_cluster_basins(losses,[restarts.exitflag]);
record = struct('distance',distance,'kind',kind, ...
    'realized_mean',mean(realized),'realized_min',min(realized), ...
    'realized_max',max(realized),'captured',sum(errors < threshold), ...
    'best_error',min(errors),'median_error',median(errors), ...
    'best_loss',min(losses),'basins',numel(basins), ...
    'converged',sum([restarts.exitflag] > 0), ...
    'capped',sum([restarts.iterations] >= max_iterations), ...
    'restarts',restarts,'runtime_seconds',toc(timer));
end

function report_record(record,restart_count)
if strcmp(record.kind,'random')
    fprintf('\n  --- random start (realized d = %.3f, range %.3f-%.3f) ---\n', ...
        record.realized_mean,record.realized_min,record.realized_max);
else
    fprintf('\n  --- d = %g (realized %.4f) ---\n', ...
        record.distance,record.realized_mean);
end
fprintf('  returned to truth : %d/%d\n',record.captured,restart_count);
fprintf('  best l>=2 error   : %.4e\n',record.best_error);
fprintf('  median l>=2 error : %.4e\n',record.median_error);
fprintf('  basins %d, converged %d/%d, capped %d/%d, %.1f s\n', ...
    record.basins,record.converged,restart_count,record.capped, ...
    restart_count,record.runtime_seconds);
end

function text = label_success(success)
if success
    text = '  -> returned';
else
    text = '';
end
end


%% ==================== signal construction ====================
function [coefficients,support,radial_basis,label] = build_signal( ...
    experiment,repo_root,L_max,shell_count,radial_count,shell_ratio,volume_file)
support = shell_count*ones(1,L_max+1);
switch lower(experiment)
    case 'synthetic'
        coefficients = gd_generate_josh_synthetic_coefficients( ...
            L_max,shell_count,42);
        coefficients = gd_impose_band_dynamic_range(coefficients,L_max, ...
            support,1);
        radial_basis = repmat({eye(shell_count)},1,L_max+1);
        label = sprintf('synthetic L=%d R=%d, identity basis, D=1', ...
            L_max,shell_count);
    case 'shell'
        coefficients = gd_generate_josh_synthetic_coefficients( ...
            L_max,shell_count,42);
        coefficients = gd_impose_shell_dynamic_range(coefficients,L_max, ...
            support,shell_ratio);
        radial_basis = repmat({eye(shell_count)},1,L_max+1);
        label = sprintf('synthetic, shell profile S=%g, band norms equal', ...
            shell_ratio);
    case 'real'
        volume_options = struct('L_max',L_max,'grid_size',31,'rho_max',pi, ...
            'basis_mode','manual','manual_shells',support);
        [raw,~,metadata] = expand_real_volume_fourier_bessel( ...
            fullfile(repo_root,'data','raw',volume_file),volume_options);
        native_basis = gd_fourier_bessel_radial_basis(metadata,radial_count);
        raw = gd_impose_band_dynamic_range(raw,L_max,support,1,native_basis);
        coefficients = gd_apply_radial_basis(raw,native_basis,support,L_max);
        support = radial_count*ones(1,L_max+1);
        radial_basis = repmat({eye(radial_count)},1,L_max+1);
        label = sprintf('%s, delta coordinates, K=%d, D=1', ...
            volume_file,radial_count);
    otherwise
        error('Unknown experiment "%s". Use real, synthetic or shell.', ...
            experiment);
end
end

%% ==================== reporting ====================
function report_summary(records,experiment,restart_count)
fprintf('\n============================================================\n');
fprintf(' Basin radius summary: %s (%d directions each)\n', ...
    experiment,restart_count);
fprintf('============================================================\n');
fprintf('%12s %10s %12s %12s %8s %8s\n', ...
    'distance','returned','best e','median e','basins','capped');
for index = 1:numel(records)
    r = records{index};
    if strcmp(r.kind,'random')
        name = sprintf('random~%.2f',r.realized_mean);
    else
        name = sprintf('%g',r.distance);
    end
    fprintf('%12s %6d/%-3d %12.3e %12.3e %8d %5d/%-3d\n', ...
        name,r.captured,restart_count,r.best_error,r.median_error, ...
        r.basins,r.capped,restart_count);
end
fprintf(['\nExact population moment, so the 1e-4 criterion is meaningful.\n', ...
    '"returned" counts starts that came back to the truth. The random row\n', ...
    'is the standard per-band truth-scaled start used elsewhere, with the\n', ...
    'relative distance it actually lands at.\n']);
end

function write_cell_csv(record,output_dir,label,tag)
name = regexprep(lower(label),'[^a-z0-9]+','_');
path = fullfile(output_dir,sprintf('restarts_%s_%s.csv', ...
    name(1:min(30,numel(name))),tag));
fid = fopen(path,'w');
assert(fid >= 0,'Could not open %s.',path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,['restart,seed,loss,relative_error,optimized_band_error,', ...
    'gradient_norm,iterations,exitflag,runtime_seconds,success\n']);
for k = 1:numel(record.restarts)
    r = record.restarts(k);
    fprintf(fid,'%d,%d,%.12e,%.12e,%.12e,%.12e,%d,%d,%.4f,%d\n', ...
        r.restart,r.seed,r.objective,r.relative_error, ...
        r.optimized_band_error,r.gradient_norm,r.iterations,r.exitflag, ...
        r.runtime_seconds,r.success);
end
clear cleanup;
end

function write_summary(records,output_dir,experiment,restart_count)
path = fullfile(output_dir,sprintf('basin_radius_%s_summary.csv',experiment));
fid = fopen(path,'w');
assert(fid >= 0,'Could not open %s.',path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,['distance,kind,realized_mean,realized_min,realized_max,', ...
    'directions,returned,best_error,median_error,best_loss,', ...
    'distinct_basins,converged,capped\n']);
for index = 1:numel(records)
    r = records{index};
    fprintf(fid,'%g,%s,%.12e,%.12e,%.12e,%d,%d,%.12e,%.12e,%.12e,%d,%d,%d\n', ...
        r.distance,r.kind,r.realized_mean,r.realized_min,r.realized_max, ...
        restart_count,r.captured,r.best_error,r.median_error,r.best_loss, ...
        r.basins,r.converged,r.capped);
end
clear cleanup;
end
