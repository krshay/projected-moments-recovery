function summary = run_josh_baseline_experiments()
%RUN_JOSH_BASELINE_EXPERIMENTS Synthetic, TRPV1, and 80S Josh baselines.

experiment_dir = fileparts(mfilename('fullpath'));
gradient_experiment_dir = fileparts(experiment_dir);
repo_root = fileparts(fileparts(gradient_experiment_dir));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','frequency-marching'));
addpath(gradient_experiment_dir);
addpath(experiment_dir);
config = josh_baseline_config(repo_root);
create_directories(config);
save_config(config);
diary(fullfile(config.logs_dir,'run.log'));
diary_cleanup = onCleanup(@() diary('off'));
total_timer = tic;

fprintf('============================================================\n');
fprintf(' Josh baseline: normalized projected-P3 residual only\n');
fprintf(' Exact population P3; A0,A1 fixed to truth\n');
fprintf(' Run directory: %s\n',config.run_dir);
fprintf('============================================================\n');

synthetic_problem = build_synthetic_problem(config.synthetic);
assert(gd_third_moment_objective( ...
    synthetic_problem.x_true,synthetic_problem) < 1e-12);
synthetic_options = struct('restart_count',config.synthetic.restart_count, ...
    'seed',config.synthetic.seed,'optimizer',config.synthetic.optimizer, ...
    'initialization','synthetic');
synthetic = gd_run_josh_baseline_restarts( ...
    synthetic_problem,synthetic_options);
save(fullfile(config.data_dir,'synthetic_results.mat'), ...
    'synthetic','config','-v7.3');
write_baseline_csv(synthetic,config.data_dir,'synthetic');
plot_josh_baseline_results(synthetic,config.figures_dir,'synthetic');

real_results = cell(numel(config.real_volumes),1);
for volume_index = 1:numel(config.real_volumes)
    volume = config.real_volumes(volume_index);
    assert(exist(volume.path,'file') == 2,'Missing volume: %s',volume.path);
    fprintf('\nBuilding unmodified %s coefficients...\n',volume.name);
    build_timer = tic;
    [problem,metadata] = build_real_problem(volume,config.real);
    build_runtime_seconds = toc(build_timer);
    assert(gd_third_moment_objective(problem.x_true,problem) < 1e-12);
    options = struct('restart_count',config.real.restart_count, ...
        'seed',config.seed+100*volume_index, ...
        'optimizer',config.real.optimizer,'initialization','real');
    experiment = gd_run_josh_baseline_restarts(problem,options);
    experiment.coefficient_build_runtime_seconds = build_runtime_seconds;
    stem = lower(volume.name);
    save(fullfile(config.data_dir,[stem,'_results.mat']), ...
        'experiment','metadata','config','-v7.3');
    write_baseline_csv(experiment,config.data_dir,stem);
    plot_josh_baseline_results(experiment,config.figures_dir,stem);
    write_real_products(experiment,metadata,config,stem);
    real_results{volume_index} = experiment;
end

summary = struct();
summary.config = config;
summary.synthetic = synthetic;
summary.real_results = real_results;
summary.runtime_seconds = toc(total_timer);
summary.method = 'Josh baseline; no regularizer';
summary.disclaimer = 'A0,A1 fixed to truth; not end-to-end recovery.';
save(fullfile(config.data_dir,'summary.mat'),'summary','-v7.3');
fprintf('\nFinished Josh baseline in %.2f seconds.\n',summary.runtime_seconds);
fprintf('Run directory: %s\n',config.run_dir);
clear diary_cleanup;
end

function problem = build_synthetic_problem(config)
coefficients = gd_generate_josh_synthetic_coefficients( ...
    config.L_max,config.shell_count,config.seed);
support = config.shell_count*ones(1,config.L_max+1);
radial_basis = repmat({eye(config.shell_count)},1,config.L_max+1);
options = struct('name',sprintf('Synthetic L=%d, S_l=%d', ...
    config.L_max,config.shell_count),'kind','synthetic', ...
    'moment_source','exact','success_threshold',config.success_threshold);
problem = gd_create_problem(coefficients,support,radial_basis,options);
end

function [problem,metadata] = build_real_problem(volume,config)
[coefficients,support,metadata] = expand_real_volume_fourier_bessel( ...
    volume.path,config.volume_options);
[radial_basis,rho_grid] = gd_fourier_bessel_radial_basis( ...
    metadata,config.radial_count);
metadata.gradient_descent_rho_grid = rho_grid;
options = struct('name',volume.name,'kind','real molecular volume', ...
    'moment_source','exact','success_threshold',config.success_threshold);
problem = gd_create_problem(coefficients,support,radial_basis,options);
end

function write_real_products(experiment,metadata,config,stem)
output_dir = fullfile(config.data_dir,'mrc',stem);
if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end
true_volume = reconstruct_fourier_bessel_volume( ...
    experiment.coefficients_true,metadata);
best = experiment.restarts(experiment.best_restart);
estimate_volume = reconstruct_fourier_bessel_volume( ...
    best.coefficients_estimate,metadata);
write_mrc_volume(fullfile(output_dir,[stem,'_true.mrc']),true_volume);
write_mrc_volume(fullfile(output_dir,[stem,'_reconstruction.mrc']), ...
    estimate_volume);
plot_josh_baseline_volume(true_volume,estimate_volume, ...
    best.relative_error,config.figures_dir,stem);
end

function create_directories(config)
directories = {config.config_dir,config.data_dir, ...
    config.figures_dir,config.logs_dir};
for index = 1:numel(directories)
    if exist(directories{index},'dir') ~= 7
        mkdir(directories{index});
    end
end
end

function save_config(config)
save(fullfile(config.config_dir,'config.mat'),'config','-v7.3');
file_id = fopen(fullfile(config.config_dir,'config.json'),'w');
assert(file_id >= 0);
cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id,'%s\n',jsonencode(config,'PrettyPrint',true));
clear cleanup;
end
