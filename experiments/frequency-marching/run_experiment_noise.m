function results = run_experiment_noise(config)
%RUN_EXPERIMENT_NOISE Recovery error versus complex-noise standard deviation.

S = config.synthetic_support;
a_true = generate_random_decay_signal(S,config.seed);
sigma_values = config.noise_sigma_values;
num_trials = config.sampling_num_trials;
errors = nan(numel(sigma_values),num_trials);

fprintf('\n=== NOISE SWEEP (N = %d) ===\n',config.noise_sweep_n_samples);
for sigma_idx = 1:numel(sigma_values)
    for trial = 1:num_trials
        options = struct('use_analytical_p3',false, ...
            'n_samples',config.noise_sweep_n_samples, ...
            'noise_sigma',sigma_values(sigma_idx), ...
            'rng_seed',config.seed+1000*sigma_idx+trial, ...
            'enable_backfill',true,'verbose',false); %#ok<NASGU>
        [~,recovered] = evalc('frequency_marching_recovery(a_true,S,options)');
        errors(sigma_idx,trial) = recovered.recovered_region_error;
    end
    fprintf('sigma=%g: mean error %.4e\n',sigma_values(sigma_idx), ...
        mean(errors(sigma_idx,:)));
end

results = struct('support',S,'a_true',a_true,'sigma_values',sigma_values, ...
    'n_samples',config.noise_sweep_n_samples,'errors',errors, ...
    'num_trials',num_trials, ...
    'mean_error',mean(errors,2),'std_error',std(errors,0,2), ...
    'median_error',median(errors,2), ...
    'q25_error',row_quantile(errors,0.25), ...
    'q75_error',row_quantile(errors,0.75));
save(fullfile(config.results_dir,'results_noise_sweep.mat'),'results','config','-v7.3');

figure('Color','w');
lower_error = results.median_error-results.q25_error;
upper_error = results.q75_error-results.median_error;
errorbar(sigma_values,results.median_error,lower_error,upper_error,'-o', ...
    'LineWidth',2,'MarkerSize',8);
set(gca,'XScale','log','YScale','log');
xlabel('Noise standard deviation \sigma');
ylabel('Relative error on marched shells');
ylim([5e-3,config.error_plot_ceiling]);
grid on;
apply_publication_style(gcf);
exportgraphics(gcf,fullfile(config.figures_dir, ...
    'frequency_marching_error_vs_noise.pdf'),'ContentType','vector');
end
