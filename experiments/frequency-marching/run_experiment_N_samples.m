function results = run_experiment_N_samples(config)
%RUN_EXPERIMENT_N_SAMPLES Recovery error versus number of observations.

S = config.synthetic_support;
a_true = generate_random_decay_signal(S,config.seed);
N_values = config.observation_values;
errors = nan(numel(N_values),config.num_trials);

fprintf('\n=== OBSERVATION SWEEP (sigma = %g) ===\n', ...
    config.observation_sweep_sigma);
for N_idx = 1:numel(N_values)
    for trial = 1:config.num_trials
        options = struct('use_analytical_p3',false, ...
            'n_samples',N_values(N_idx), ...
            'noise_sigma',config.observation_sweep_sigma, ...
            'rng_seed',config.seed+1000*N_idx+trial, ...
            'enable_backfill',true,'verbose',false); %#ok<NASGU>
        [~,recovered] = evalc('frequency_marching_recovery(a_true,S,options)');
        errors(N_idx,trial) = recovered.global_relative_error;
    end
    fprintf('N=%d: mean error %.4e\n',N_values(N_idx),mean(errors(N_idx,:)));
end

results = struct('support',S,'a_true',a_true,'N_values',N_values, ...
    'noise_sigma',config.observation_sweep_sigma,'errors',errors, ...
    'mean_error',mean(errors,2),'std_error',std(errors,0,2));
save(fullfile(config.results_dir,'results_N_samples_sweep.mat'), ...
    'results','config','-v7.3');

figure('Color','w');
errorbar(N_values,results.mean_error,results.std_error,'-o', ...
    'LineWidth',2,'MarkerSize',8);
set(gca,'XScale','log','YScale','log');
xlabel('Number of observations');
ylabel('Relative coefficient error');
grid on;
apply_publication_style(gcf);
exportgraphics(gcf,fullfile(config.figures_dir, ...
    'frequency_marching_error_vs_observations.pdf'),'ContentType','vector');
end
