%RUN_DELTA_SHELL_REAL_VOLUMES Molecular frequency marching on radial shells.

clear variables;
close all;
clc;

experiment_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(experiment_dir));
addpath(fullfile(repo_root,'src','frequency-marching'));

settings = struct('L_max',5,'grid_size',31,'rho_max',pi, ...
    'shell_count',12,'n_theta',16,'n_phi',32,'terminal_shells',3, ...
    'rng_seed',1);
names = {'TRPV1','80S'};
paths = {fullfile(repo_root,'data','raw','vol_TRPV1_true.mrc'), ...
    fullfile(repo_root,'data','raw','vol_80S_true.mrc')};

timestamp = datestr(now,'yyyy-mm-dd_HH-MM-SS');
run_dir = fullfile(repo_root,'results','frequency-marching', ...
    ['delta-shell_' timestamp]);
mrc_dir = fullfile(run_dir,'mrc');
mkdir(mrc_dir);

solver_options = struct('use_analytical_p3',true, ...
    'enable_backfill',true,'verify_analytical',false, ...
    'verbose',true,'rng_seed',settings.rng_seed);
n_molecules = numel(names);
unmodified_error = nan(n_molecules,1);
decay_error = nan(n_molecules,1);
unmodified_global_error = nan(n_molecules,1);
decay_global_error = nan(n_molecules,1);
unmodified_seconds = nan(n_molecules,1);
decay_seconds = nan(n_molecules,1);
representation_error = nan(n_molecules,1);
native_reality_defect = nan(n_molecules,1);
details = cell(n_molecules,1);

for molecule = 1:n_molecules
    fprintf('\n============================================================\n');
    fprintf(' Delta-shell molecular experiment: %s\n',names{molecule});
    fprintf(' L=%d, R=%d\n',settings.L_max,settings.shell_count);
    fprintf('============================================================\n');
    assert(exist(paths{molecule},'file') == 2, ...
        'Missing input volume: %s',paths{molecule});

    [a,available_shells,metadata] = ...
        expand_real_volume_delta_shells(paths{molecule},settings);
    native_reality_defect(molecule) = metadata.native_reality_defect;
    shell_volume = reconstruct_delta_shell_volume(a,metadata);
    representation_error(molecule) = norm(shell_volume(:)- ...
        metadata.reference_volume(:))/max(norm(metadata.reference_volume(:)),eps);
    [a_decay,S_decay,before,after] = enforce_decay_by_three( ...
        a,available_shells,settings.terminal_shells);

    fprintf('Available shells: [%s]\n',num2str(available_shells));
    fprintf('Solver support:   [%s]\n',num2str(S_decay));
    fprintf('Native reality convention: %s (defect %.3e)\n', ...
        metadata.native_reality_convention,metadata.native_reality_defect);
    fprintf('Delta-shell analysis/synthesis volume error: %.3e\n', ...
        representation_error(molecule));

    timer = tic;
    unmodified = frequency_marching_recovery(a,S_decay,solver_options);
    unmodified_seconds(molecule) = toc(timer);
    timer = tic;
    decay = frequency_marching_recovery(a_decay,S_decay,solver_options);
    decay_seconds(molecule) = toc(timer);

    unmodified_error(molecule) = unmodified.recovered_region_error;
    decay_error(molecule) = decay.recovered_region_error;
    unmodified_global_error(molecule) = unmodified.global_relative_error;
    decay_global_error(molecule) = decay.global_relative_error;

    stem = lower(names{molecule});
    files = struct( ...
        'reference',fullfile(mrc_dir,[stem '_reference_preprocessed.mrc']), ...
        'shell_true',fullfile(mrc_dir,[stem '_delta_shell_true.mrc']), ...
        'unmodified_reconstruction',fullfile(mrc_dir, ...
            [stem '_unmodified_reconstruction.mrc']), ...
        'decay_true',fullfile(mrc_dir,[stem '_decay_by_three_true.mrc']), ...
        'decay_reconstruction',fullfile(mrc_dir, ...
            [stem '_decay_by_three_reconstruction.mrc']));
    write_mrc_volume(files.reference,metadata.reference_volume);
    write_mrc_volume(files.shell_true,shell_volume);
    write_mrc_volume(files.unmodified_reconstruction, ...
        reconstruct_delta_shell_volume(unmodified.a_est,metadata));
    write_mrc_volume(files.decay_true, ...
        reconstruct_delta_shell_volume(a_decay,metadata));
    write_mrc_volume(files.decay_reconstruction, ...
        reconstruct_delta_shell_volume(decay.a_est,metadata));

    details{molecule} = struct('a',a,'a_decay',a_decay, ...
        'available_shells',available_shells,'S_decay',S_decay, ...
        'metadata',metadata,'diagnostic_before',before, ...
        'diagnostic_after',after,'unmodified',unmodified,'decay',decay, ...
        'mrc_files',files);
end

summary_table = table(names.',unmodified_error,decay_error, ...
    unmodified_global_error,decay_global_error,unmodified_seconds, ...
    decay_seconds,unmodified_seconds+decay_seconds,representation_error, ...
    native_reality_defect,'VariableNames',{'molecule', ...
    'unmodified_recovered_error','decay_recovered_error', ...
    'unmodified_global_error','decay_global_error','unmodified_seconds', ...
    'decay_seconds','total_seconds','representation_error', ...
    'native_reality_defect'});
writetable(summary_table,fullfile(run_dir,'summary.csv'));

condition_molecule = cell(0,1);
condition_coefficients = cell(0,1);
condition_phase = cell(0,1);
condition_minimum = zeros(0,1);
condition_median = zeros(0,1);
condition_maximum = zeros(0,1);
variant_names = {'unmodified','modified'};
phase_names = {'march','backfill'};
for molecule = 1:n_molecules
    variant_results = {details{molecule}.unmodified,details{molecule}.decay};
    for variant = 1:numel(variant_names)
        systems = variant_results{variant}.condition_numbers;
        for phase = 1:numel(phase_names)
            selected = strcmp(systems.phase,phase_names{phase});
            values = systems.reality_constrained_condition(selected);
            condition_molecule{end+1,1} = names{molecule}; %#ok<SAGROW>
            condition_coefficients{end+1,1} = variant_names{variant}; %#ok<SAGROW>
            condition_phase{end+1,1} = phase_names{phase}; %#ok<SAGROW>
            condition_minimum(end+1,1) = min(values); %#ok<SAGROW>
            condition_median(end+1,1) = median(values); %#ok<SAGROW>
            condition_maximum(end+1,1) = max(values); %#ok<SAGROW>
        end
    end
end
condition_summary = table(condition_molecule,condition_coefficients, ...
    condition_phase,condition_minimum,condition_median,condition_maximum, ...
    'VariableNames',{'molecule','coefficients','phase','minimum','median','maximum'});
writetable(condition_summary,fullfile(run_dir,'condition_numbers.csv'));
save(fullfile(run_dir,'results.mat'),'settings','summary_table', ...
    'condition_summary','details','-v7.3');

text_path = fullfile(run_dir,'results.txt');
fid = fopen(text_path,'w');
assert(fid >= 0,'Could not create %s.',text_path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,'Delta-shell molecular frequency marching\n');
fprintf(fid,'L_max=%d, R=%d, grid=%d, theta=%d, phi=%d\n', ...
    settings.L_max,settings.shell_count,settings.grid_size, ...
    settings.n_theta,settings.n_phi);
fprintf(fid,'Shell radii are midpoint samples on (0, pi).\n');
fprintf(fid,'Decay-by-three support: [%s]\n\n',num2str(details{1}.S_decay));
for molecule = 1:n_molecules
    fprintf(fid,'%s\n',names{molecule});
    fprintf(fid,'  unmodified recovered-region error: %.12e\n', ...
        unmodified_error(molecule));
    fprintf(fid,'  decay-by-three recovered-region error: %.12e\n', ...
        decay_error(molecule));
    fprintf(fid,'  unmodified global error: %.12e\n', ...
        unmodified_global_error(molecule));
    fprintf(fid,'  decay-by-three global error: %.12e\n', ...
        decay_global_error(molecule));
    fprintf(fid,'  unmodified runtime: %.6f s\n',unmodified_seconds(molecule));
    fprintf(fid,'  decay-by-three runtime: %.6f s\n',decay_seconds(molecule));
    fprintf(fid,'  analysis/synthesis volume error: %.12e\n', ...
        representation_error(molecule));
    fprintf(fid,'  native reality defect: %.12e\n\n', ...
        native_reality_defect(molecule));
end
fprintf(fid,'Condition numbers of the reality-constrained real systems\n');
for row = 1:height(condition_summary)
    fprintf(fid,'  %s, %s, %s: min %.12e, median %.12e, max %.12e\n', ...
        condition_summary.molecule{row},condition_summary.coefficients{row}, ...
        condition_summary.phase{row},condition_summary.minimum(row), ...
        condition_summary.median(row),condition_summary.maximum(row));
end
clear cleanup;

disp(summary_table);
disp(condition_summary);
fprintf('\nRun complete: %s\n',run_dir);
