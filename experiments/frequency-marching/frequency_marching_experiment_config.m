function config = frequency_marching_experiment_config(repo_root)
%FREQUENCY_MARCHING_EXPERIMENT_CONFIG Paper-scale experiment parameters.

config = struct();
config.repo_root = repo_root;
config.seed = 1;
config.num_trials = 5;
config.sampling_num_trials = 10;
config.run_real_volumes = true;
config.run_noise_sweep = true;
config.run_observation_sweep = true;

config.results_dir = fullfile(repo_root,'results','frequency-marching');
config.figures_dir = fullfile(repo_root,'figures','frequency-marching');
config.processed_dir = fullfile(repo_root,'data','processed','frequency-marching');
config.volume_results_dir = fullfile(config.results_dir,'mrc');

config.volume_options = struct( ...
    'L_max',9, ...
    'grid_size',31, ...
    'rho_max',pi, ...
    'basis_mode','kc');
config.terminal_shells = 3;
config.noiseless_success_tolerance = 1e-8;
config.noiseless_failure_floor = 1e-2;

config.real_volumes = struct( ...
    'name',{'TRPV1','80S'}, ...
    'path',{find_volume(repo_root,{'TRPV1.mat','vol_TRPV1_true.mrc'}), ...
            find_volume(repo_root,{'emd_2660.mat','vol_80S_true.mrc'})});

% Sampling sweeps use ell_max=5 to keep finite-sample runs tractable.
config.synthetic_support = [9,9,6,6,3,3];
config.error_plot_ceiling = 0.5;
config.noise_sigma_values = logspace(log10(5e-3),log10(1.2e-1),8);
config.noise_sweep_n_samples = 200000;
config.observation_values = round(logspace(log10(2e4),log10(1e6),9));
config.observation_sweep_sigma = 0.05;
end

function path = find_volume(repo_root,candidates)
path = '';
for idx = 1:numel(candidates)
    candidate = fullfile(repo_root,'data','raw',candidates{idx});
    if exist(candidate,'file') == 2
        path = candidate;
        return;
    end
end
end
