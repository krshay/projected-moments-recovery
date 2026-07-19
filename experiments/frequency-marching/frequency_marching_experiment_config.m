function config = frequency_marching_experiment_config(repo_root)
%FREQUENCY_MARCHING_EXPERIMENT_CONFIG Paper-scale experiment parameters.

config = struct();
config.repo_root = repo_root;
config.seed = 1;
config.num_trials = 5;
config.run_real_volumes = true;
config.run_noise_sweep = true;
config.run_observation_sweep = true;

config.results_dir = fullfile(repo_root,'results','frequency-marching');
config.figures_dir = fullfile(repo_root,'figures','frequency-marching');
config.processed_dir = fullfile(repo_root,'data','processed','frequency-marching');

config.volume_options = struct( ...
    'L_max',5, ...
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

% Paired profile [9 9 6 6 3 3], satisfying S(l)-S(l+2)=3.
config.synthetic_support = [9,9,6,6,3,3];
config.noise_sigma_values = logspace(-3,0,8);
config.noise_sweep_n_samples = 10000;
config.observation_values = round(logspace(3,5,8));
config.observation_sweep_sigma = 0.1;
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
