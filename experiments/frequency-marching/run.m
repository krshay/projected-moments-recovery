%RUN Reproduce all frequency-marching numerical experiments and figures.
% This mirrors the top-level run.m convention in recovery-high-order-spectra.

clear variables;
close all;
clc;

experiment_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(experiment_dir));
addpath(fullfile(repo_root,'src','frequency-marching'));
addpath(experiment_dir);

config = frequency_marching_experiment_config(repo_root);
ensure_output_directories(config);
save_run_metadata(config,experiment_dir);

fprintf('============================================================\n');
fprintf(' Frequency-marching experiments\n');
fprintf('============================================================\n');
fprintf('Run folder: %s\n',config.run_dir);

if config.run_real_volumes
    missing = {config.real_volumes(cellfun(@isempty,{config.real_volumes.path})).name};
    if ~isempty(missing)
        error(['Missing raw volume(s): %s. See data/raw/README.md for the ', ...
            'accepted filenames.'],strjoin(missing,', '));
    end
    run_experiment_real_volumes(config);
end
if config.run_noise_sweep
    run_experiment_noise(config);
end
if config.run_observation_sweep
    run_experiment_N_samples(config);
end

fprintf('\nAll requested frequency-marching experiments are complete.\n');
fprintf('Run folder: %s\n',config.run_dir);
fprintf('Results: %s\n',config.results_dir);
fprintf('Figures: %s\n',config.figures_dir);

function ensure_output_directories(config)
directories = {config.results_dir,config.figures_dir,config.processed_dir, ...
    config.volume_results_dir,config.config_snapshot_dir};
for idx = 1:numel(directories)
    if exist(directories{idx},'dir') ~= 7
        mkdir(directories{idx});
    end
end
end

function save_run_metadata(config,experiment_dir)
save(fullfile(config.config_snapshot_dir,'config.mat'),'config');

config_text = evalc('disp(config)');
fid = fopen(fullfile(config.config_snapshot_dir,'config.txt'),'w');
if fid < 0
    error('Could not write config snapshot.');
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'%s',config_text);
clear cleanup;

snapshot_files = {'run.m', ...
    'frequency_marching_experiment_config.m', ...
    'run_experiment_real_volumes.m', ...
    'run_experiment_noise.m', ...
    'run_experiment_N_samples.m'};
for idx = 1:numel(snapshot_files)
    source_path = fullfile(experiment_dir,snapshot_files{idx});
    if exist(source_path,'file') == 2
        copyfile(source_path,fullfile(config.config_snapshot_dir,snapshot_files{idx}));
    end
end
end
