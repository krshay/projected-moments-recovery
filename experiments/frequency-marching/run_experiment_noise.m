function results = run_experiment_noise(config)
%RUN_EXPERIMENT_NOISE Recovery error versus complex-noise standard deviation.

S = config.synthetic_support;
a_true = generate_random_decay_signal(S,config.seed);
sigma_values = config.noise_sigma_values;
errors = nan(numel(sigma_values),config.num_trials);

fprintf('\n=== NOISE SWEEP (N = %d) ===\n',config.noise_sweep_n_samples);
for sigma_idx = 1:numel(sigma_values)
    for trial = 1:config.num_trials
        options = struct('use_analytical_p3',false, ...
            'n_samples',config.noise_sweep_n_samples, ...
            'noise_sigma',sigma_values(sigma_idx), ...
            'rng_seed',config.seed+1000*sigma_idx+trial, ...
            'enable_backfill',true,'verbose',false); %#ok<NASGU>
        [~,recovered] = evalc('frequency_marching_recovery(a_true,S,options)');
        errors(sigma_idx,trial) = recovered.global_relative_error;
    end
    fprintf('sigma=%g: mean error %.4e\n',sigma_values(sigma_idx), ...
        mean(errors(sigma_idx,:)));
end

results = struct('support',S,'a_true',a_true,'sigma_values',sigma_values, ...
    'n_samples',config.noise_sweep_n_samples,'errors',errors, ...
    'mean_error',mean(errors,2),'std_error',std(errors,0,2));
save(fullfile(config.results_dir,'results_noise_sweep.mat'),'results','config','-v7.3');

figure('Color','w');
errorbar(sigma_values,results.mean_error,results.std_error,'-o', ...
    'LineWidth',2,'MarkerSize',8);
set(gca,'XScale','log','YScale','log');
xlabel('Noise standard deviation \sigma');
ylabel('Relative coefficient error');
grid on;
apply_publication_style(gcf);
exportgraphics(gcf,fullfile(config.figures_dir, ...
    'frequency_marching_error_vs_noise.pdf'),'ContentType','vector');
end
