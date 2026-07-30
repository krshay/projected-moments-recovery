function results = run_experiment_noise(config)
%RUN_EXPERIMENT_NOISE Recovery error versus linear observation SNR.

S = config.synthetic_support;
a_true = generate_random_decay_signal(S,config.seed);
snr_values = config.noise_snr_values;
signal_power = projected_signal_power(a_true,S);
sigma_values = sqrt(signal_power./snr_values);
noise_power_values = sigma_values.^2;
num_trials = config.sampling_num_trials;
errors = nan(numel(snr_values),num_trials);
realized_snr = nan(numel(snr_values),num_trials);

fprintf('\n=== NOISE SWEEP (N = %d) ===\n',config.noise_sweep_n_samples);
fprintf('Mean clean projected-coefficient signal power P_signal: %.6g\n',signal_power);
fprintf('Linear observation SNR is P_signal/P_noise = P_signal/sigma^2 (not dB).\n');
for trial = 1:num_trials
    cache_key = sprintf('noise_sweep_trial_%d',trial);
    clean_seed = config.seed+trial;
    for snr_idx = 1:numel(snr_values)
        options = struct('use_analytical_p3',false, ...
            'n_samples',config.noise_sweep_n_samples, ...
            'noise_sigma',sigma_values(snr_idx), ...
            'rng_seed',config.seed+1000*snr_idx+trial, ...
            'clean_sample_cache_key',cache_key, ...
            'clean_sample_rng_seed',clean_seed, ...
            'clean_sample_cache_n_samples',config.noise_sweep_n_samples, ...
            'enable_backfill',true,'verbose',false); %#ok<NASGU>
        [~,recovered] = evalc('frequency_marching_recovery(a_true,S,options)');
        errors(snr_idx,trial) = recovered.recovered_region_error;
        realized_snr(snr_idx,trial) = recovered.realized_observation_snr;
    end
    fprintf('Trial %d / %d complete\n',trial,num_trials);
end
for snr_idx = 1:numel(snr_values)
    fprintf(['SNR=P_signal/P_noise=%g (P_noise=sigma^2=%g, ', ...
        'sigma=%g): mean error %.4e, realized power SNR %.4g\n'], ...
        snr_values(snr_idx),noise_power_values(snr_idx),sigma_values(snr_idx), ...
        mean(errors(snr_idx,:)),mean(realized_snr(snr_idx,:)));
end

results = struct('support',S,'a_true',a_true,'snr_values',snr_values, ...
    'signal_power',signal_power,'noise_power_values',noise_power_values, ...
    'sigma_values',sigma_values, ...
    'realized_snr',realized_snr, ...
    'n_samples',config.noise_sweep_n_samples,'errors',errors, ...
    'num_trials',num_trials, ...
    'mean_error',mean(errors,2),'std_error',std(errors,0,2));
save(fullfile(config.results_dir,'results_noise_sweep.mat'),'results','config','-v7.3');

figure('Color','w');
plot(snr_values,results.mean_error,'-o','LineWidth',2,'MarkerSize',8);
set(gca,'XScale','log','YScale','log');
xlabel('SNR');
ylabel('Relative error');
log_x_limits = log10([min(snr_values),max(snr_values)]);
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
    'frequency_marching_error_vs_snr.pdf'),'ContentType','vector');
end
