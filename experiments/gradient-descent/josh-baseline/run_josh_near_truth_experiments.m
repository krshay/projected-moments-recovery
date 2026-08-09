function summary = run_josh_near_truth_experiments(perturbation)
%RUN_JOSH_NEAR_TRUTH_EXPERIMENTS Molecular basin sanity test for Josh baseline.

if nargin < 1
    perturbation = 0.01;
end
experiment_dir = fileparts(mfilename('fullpath'));
gradient_experiment_dir = fileparts(experiment_dir);
repo_root = fileparts(fileparts(gradient_experiment_dir));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','frequency-marching'));
addpath(gradient_experiment_dir);
addpath(experiment_dir);
config = josh_baseline_config(repo_root);
config.perturbation = perturbation;
config.real.optimizer.max_iterations = 1000;
config.output_root = fullfile(repo_root,'results','gradient-descent', ...
    'josh-near-truth');
config.run_dir = unique_run_dir(fullfile(config.output_root, ...
    datestr(now,'yyyy-mm-dd_HH-MM-SS')));
config.config_dir = fullfile(config.run_dir,'config');
config.data_dir = fullfile(config.run_dir,'data');
config.figures_dir = fullfile(config.run_dir,'figures');
config.logs_dir = fullfile(config.run_dir,'logs');
for directory = {config.config_dir,config.data_dir, ...
        config.figures_dir,config.logs_dir}
    mkdir(directory{1});
end
save(fullfile(config.config_dir,'config.mat'),'config','-v7.3');
file_id = fopen(fullfile(config.config_dir,'config.json'),'w');
assert(file_id >= 0);
file_cleanup = onCleanup(@() fclose(file_id));
fprintf(file_id,'%s\n',jsonencode(config,'PrettyPrint',true));
clear file_cleanup;
diary(fullfile(config.logs_dir,'run.log'));
diary_cleanup = onCleanup(@() diary('off'));
total_timer = tic;

fprintf('Josh near-truth test: L=5, R=12, perturbation=%.3g\n',perturbation);
results = cell(numel(config.real_volumes),1);
for volume_index = 1:numel(config.real_volumes)
    volume = config.real_volumes(volume_index);
    [coefficients,support,metadata] = expand_real_volume_fourier_bessel( ...
        volume.path,config.real.volume_options);
    [radial_basis,rho_grid] = gd_fourier_bessel_radial_basis( ...
        metadata,config.real.radial_count);
    metadata.gradient_descent_rho_grid = rho_grid;
    problem = gd_create_problem(coefficients,support,radial_basis,struct( ...
        'name',volume.name,'kind','real molecular volume', ...
        'moment_source','exact', ...
        'success_threshold',config.real.success_threshold));
    options = struct('restart_count',1, ...
        'seed',config.seed+100*volume_index, ...
        'optimizer',config.real.optimizer,'initialization','near_truth', ...
        'perturbation',perturbation);
    experiment = gd_run_josh_baseline_restarts(problem,options);
    experiment.coefficients_initial = gd_decode_parameters( ...
        experiment.restarts(1).x0,problem.parameterization,problem.base);
    stem = lower(volume.name);
    save(fullfile(config.data_dir,[stem,'_results.mat']), ...
        'experiment','metadata','config','-v7.3');
    write_baseline_csv(experiment,config.data_dir,stem);
    plot_josh_baseline_results(experiment,config.figures_dir,stem);
    write_products(experiment,metadata,config,stem);
    results{volume_index} = experiment;
end
summary = struct('config',config,'results',{results}, ...
    'runtime_seconds',toc(total_timer), ...
    'method',sprintf('Josh baseline initialized %.3g from truth',perturbation));
save(fullfile(config.data_dir,'summary.mat'),'summary','-v7.3');
fprintf('Near-truth test finished in %.2f seconds.\n',summary.runtime_seconds);
fprintf('Run directory: %s\n',config.run_dir);
clear diary_cleanup;
end

function write_products(experiment,metadata,config,stem)
output_dir = fullfile(config.data_dir,'mrc',stem);
mkdir(output_dir);
truth = reconstruct_fourier_bessel_volume( ...
    experiment.coefficients_true,metadata);
initial = reconstruct_fourier_bessel_volume( ...
    experiment.coefficients_initial,metadata);
estimate = reconstruct_fourier_bessel_volume( ...
    experiment.restarts(1).coefficients_estimate,metadata);
write_mrc_volume(fullfile(output_dir,[stem,'_true.mrc']),truth);
write_mrc_volume(fullfile(output_dir,[stem,'_initial.mrc']),initial);
write_mrc_volume(fullfile(output_dir,[stem,'_reconstruction.mrc']),estimate);
plot_near_truth_volume(truth,initial,estimate, ...
    experiment.restarts(1).relative_error,config.figures_dir,stem);
end

function plot_near_truth_volume(truth,initial,estimate,error_value,figure_dir,stem)
middle = ceil(size(truth,3)/2);
limits = [min(truth(:,:,middle),[],'all'),max(truth(:,:,middle),[],'all')];
figure_handle = figure('Visible','off');
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
volumes = {truth,initial,estimate};
titles = {'True','Initial','Reconstruction'};
for index = 1:3
    nexttile;
    imagesc(volumes{index}(:,:,middle));
    axis image off;
    clim(limits);
    title(titles{index});
end
colormap(gray(256));
sgtitle(sprintf('Near-truth baseline, final error %.2g',error_value));
apply_gradient_descent_style(figure_handle);
exportgraphics(figure_handle,fullfile(figure_dir,[stem,'_volume.pdf']), ...
    'ContentType','vector');
exportgraphics(figure_handle,fullfile(figure_dir,[stem,'_volume.png']), ...
    'Resolution',300);
close(figure_handle);
end

function directory = unique_run_dir(candidate)
directory = candidate;
suffix = 2;
while exist(directory,'dir') == 7
    directory = sprintf('%s_%02d',candidate,suffix);
    suffix = suffix+1;
end
end
