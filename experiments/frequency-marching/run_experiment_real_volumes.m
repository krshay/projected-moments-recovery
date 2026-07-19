function summary = run_experiment_real_volumes(config)
%RUN_EXPERIMENT_REAL_VOLUMES Noiseless failure/success on TRPV1 and 80S.

n_volumes = numel(config.real_volumes);
unmodified_error = nan(1,n_volumes);
modified_error = nan(1,n_volumes);
condition_before = false(1,n_volumes);
condition_after = false(1,n_volumes);
details = cell(1,n_volumes);

options = struct('use_analytical_p3',true,'enable_backfill',true, ...
    'verify_analytical',false,'verbose',true,'rng_seed',config.seed);

for idx = 1:n_volumes
    volume = config.real_volumes(idx);
    fprintf('\n============================================================\n');
    fprintf(' Real-volume experiment: %s\n',volume.name);
    fprintf('============================================================\n');
    cache_path = fullfile(config.processed_dir, ...
        sprintf('%s_L%d_grid%d_fourier_bessel.mat',lower(volume.name), ...
        config.volume_options.L_max,config.volume_options.grid_size));
    if exist(cache_path,'file') == 2
        cached = load(cache_path,'a','available_shells','metadata');
        a = cached.a;
        available_shells = cached.available_shells;
        metadata = cached.metadata;
    else
        [a,available_shells,metadata] = ...
            expand_real_volume_fourier_bessel(volume.path,config.volume_options);
        save(cache_path,'a','available_shells','metadata','-v7.3');
    end

    before = check_decay_by_three(available_shells);
    condition_before(idx) = before.satisfies_decay_by_three;
    fprintf('Unmodified supports: [%s]\n',num2str(available_shells));
    fprintf('Decay-by-3 satisfied: %d (minimum parity gap %g)\n', ...
        before.satisfies_decay_by_three,before.minimum_parity_gap);
    [a_decay,S_decay,~,after] = enforce_decay_by_three( ...
        a,available_shells,config.terminal_shells);
    condition_after(idx) = after.satisfies_decay_by_three;
    fprintf('Solver support:      [%s]\n',num2str(S_decay));

    % Use the same theorem-compatible solver support in both runs. The
    % unmodified volume then contains an out-of-model tail, whereas the
    % modified volume differs only by zeroing that tail.
    unmodified = frequency_marching_recovery(a,S_decay,options);
    modified = frequency_marching_recovery(a_decay,S_decay,options);

    unmodified_error(idx) = unmodified.global_relative_error;
    modified_error(idx) = modified.global_relative_error;
    assert(~condition_before(idx), ...
        '%s unexpectedly satisfies decay-by-3 before modification.',volume.name);
    assert(condition_after(idx), ...
        '%s does not satisfy decay-by-3 after modification.',volume.name);
    assert(unmodified_error(idx) > config.noiseless_failure_floor, ...
        '%s unmodified run did not exhibit the expected failure.',volume.name);
    assert(modified_error(idx) < config.noiseless_success_tolerance, ...
        '%s modified run error %.3e exceeds the noiseless tolerance.', ...
        volume.name,modified_error(idx));
    details{idx} = struct('metadata',metadata,'support_before',available_shells, ...
        'support_after',S_decay,'diagnostic_before',before, ...
        'diagnostic_after',after,'unmodified',unmodified,'modified',modified);
end

summary = struct('names',{ {config.real_volumes.name} }, ...
    'unmodified_error',unmodified_error,'modified_error',modified_error, ...
    'condition_before',condition_before,'condition_after',condition_after, ...
    'details',{details});
save(fullfile(config.results_dir,'results_real_volumes_noiseless.mat'), ...
    'summary','config','-v7.3');

figure('Color','w');
bar([unmodified_error(:),modified_error(:)]);
set(gca,'YScale','log','XTick',1:n_volumes, ...
    'XTickLabel',{config.real_volumes.name});
ylabel('Relative coefficient error');
legend({'Unmodified','Decay-by-3 enforced'},'Location','best');
grid on;
apply_publication_style(gcf);
exportgraphics(gcf,fullfile(config.figures_dir, ...
    'frequency_marching_real_volumes.pdf'),'ContentType','vector');
end
