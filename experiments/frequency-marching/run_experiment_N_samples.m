function results = run_experiment_N_samples(config)
%RUN_EXPERIMENT_N_SAMPLES Recovery error versus number of observations.

S = config.synthetic_support;
a_true = generate_random_decay_signal(S,config.seed);
signal_power = projected_signal_power(a_true,S);
observation_snr = config.observation_sweep_snr;
noise_power = signal_power/observation_snr;
noise_sigma = sqrt(noise_power);
N_values = config.observation_values;
num_trials = config.sampling_num_trials;
errors = nan(numel(N_values),num_trials);

fprintf('\n=== OBSERVATION SWEEP (SNR = %g) ===\n',observation_snr);
fprintf('Mean clean projected-coefficient signal power P_signal: %.6g\n',signal_power);
fprintf('Noise power P_noise=sigma^2: %.6g (sigma=%g)\n', ...
    noise_power,noise_sigma);
fprintf('SNR = %.6g (not dB)\n',observation_snr);
max_N = max(N_values);
for trial = 1:num_trials
    cache_key = sprintf('observation_sweep_trial_%d',trial);
    clean_seed = config.seed+trial;
    for N_idx = 1:numel(N_values)
        options = struct('use_analytical_p3',false, ...
            'n_samples',N_values(N_idx), ...
            'noise_sigma',noise_sigma, ...
            'rng_seed',config.seed+1000*N_idx+trial, ...
            'clean_sample_cache_key',cache_key, ...
            'clean_sample_rng_seed',clean_seed, ...
            'clean_sample_cache_n_samples',max_N, ...
            'enable_backfill',true,'verbose',false); %#ok<NASGU>
        [~,recovered] = evalc('frequency_marching_recovery(a_true,S,options)');
        errors(N_idx,trial) = recovered.recovered_region_error;
    end
    fprintf('Trial %d / %d complete\n',trial,num_trials);
end
for N_idx = 1:numel(N_values)
    fprintf('N=%d: mean error %.4e\n',N_values(N_idx),mean(errors(N_idx,:)));
end

results = struct('support',S,'a_true',a_true,'N_values',N_values, ...
    'signal_power',signal_power,'noise_power',noise_power, ...
    'observation_snr',observation_snr, ...
    'noise_sigma',noise_sigma,'errors',errors, ...
    'num_trials',num_trials, ...
    'mean_error',mean(errors,2),'std_error',std(errors,0,2));
save(fullfile(config.results_dir,'results_N_samples_sweep.mat'), ...
    'results','config','-v7.3');

figure('Color','w');
plot(N_values,results.mean_error,'-o','LineWidth',2,'MarkerSize',8);
set(gca,'XScale','log','YScale','log');
xlabel('Number of observations');
ylabel('Relative error');
log_x_limits = log10([min(N_values),max(N_values)]);
log_x_padding = 0.02*diff(log_x_limits);
xlim(10.^[log_x_limits(1)-log_x_padding, ...
    log_x_limits(2)+log_x_padding]);
log_y_limits = log10([min(results.mean_error),max(results.mean_error)]);
log_y_padding = 0.05*diff(log_y_limits);
ylim(10.^[log_y_limits(1)-log_y_padding, ...
    log_y_limits(2)+log_y_padding]);
grid on;
apply_publication_style(gcf);
exportgraphics(gcf,fullfile(config.figures_dir, ...
    'frequency_marching_error_vs_observations.pdf'),'ContentType','vector');
end
