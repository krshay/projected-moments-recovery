function summary = run_noise_sweep(experiment,snrs,sample_count)
%RUN_NOISE_SWEEP SNR sweep for the three gradient-descent experiments.
%   RUN_NOISE_SWEEP(EXPERIMENT) runs one of
%     'synthetic'  Gaussian coefficients, identity radial basis
%     'real'       molecular volume, delta coordinates (u = Phi*A)
%     'shell'      synthetic with a within-band shell profile, S = 1000
%   over a grid of signal-to-noise ratios, run clean to noisy. Each observed
%   moment is estimated
%   by Monte Carlo from N uniformly rotated projections with iid complex
%   Gaussian noise added to c_m[k], rather than taken from the exact
%   population formula.
%
%   RUN_NOISE_SWEEP(EXPERIMENT,SNRS,N) overrides the SNR grid and the sample
%   count. SNR is a plain power ratio, not decibels:
%       SNR = mean|c_m[k]|^2 / sigma^2,
%   so SNR = 1 means equal signal and noise power and Inf means noiseless.
%   The grid is sorted descending, so the sweep always runs clean to noisy.
%   The noise level is derived per signal as sigma = rms(c)/sqrt(SNR), which
%   makes a given SNR mean the same thing across the three experiments even
%   though their projections differ in scale. Both are reported.
%
%   WHY THE READOUT DIFFERS FROM THE POPULATION RUNS. At N = 1e5 the empirical
%   moment carries a sampling error of 6.29e-03 relative even at sigma = 0,
%   measured against the exact moment. The loss at the true coefficients is
%   therefore bounded below by roughly (6.29e-03)^2 = 4e-05 rather than 1e-18,
%   and the achievable coefficient error is bounded near 2e-03 (P3 is cubic in
%   the coefficients, so a relative perturbation delta appears as ~3*delta in
%   the moment). The 1e-4 success criterion used for the population runs is
%   unreachable at this sample count and would read 0/20 in every cell, so it
%   is not used here. Instead this driver reports:
%
%     median and best relative l>=2 error   -- threshold free
%     loss at the TRUE coefficients         -- the achievable floor per cell
%     restarts in the lowest loss cluster   -- threshold-free reliability
%
%   The population-moment drivers keep the 1e-4 criterion so their results stay
%   comparable to what is already on disk.
%
%   THE GRID. SNR = [Inf, 30, 10, 3, 1, 0.3, 0.1, 0.03] by default, N = 1e5.
%   Measured moment error against the exact moment, synthetic signal:
%   6.29e-03 at SNR = Inf and 30 (indistinguishable -- sampling limited),
%   9.15e-03 at 3.4, 4.98e-02 at 0.30, 8.69e-01 at 0.034, 3.04e+01 at 0.003.
%   Cells above SNR ~10 measure N rather than SNR; cells below ~0.03 need far
%   more samples than 1e5 to say anything, since the moment error there
%   exceeds the moment itself.
%
%   SAMPLING VERSUS NOISE. The moment error scales as 1/sqrt(N): 3.859 at
%   N = 1e5 against 1.224 at N = 1e6 for the same noise level, a ratio of
%   3.154 versus a predicted sqrt(10) = 3.162. The third moment needs no
%   debiasing under this noise model -- every bias term carries E[n] = 0 or
%   E[n^2] = 0 for circularly symmetric complex noise. Physical noise on a
%   real-valued projection is Hermitian, c_{-m} = conj(c_m), which would make
%   E[n_m n_{-m}] = sigma^2 and bias exactly the stored m1+m2+m3 = 0 blocks.
%   That case is not modelled here.

experiment_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(experiment_dir)));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','frequency-marching'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(experiment)
    experiment = 'synthetic';
end
if nargin < 2 || isempty(snrs)
    % Clean to noisy. Above ~10 the cells are sampling limited at N = 1e5;
    % 0.1 down to 0.01 is the cryo-EM single-particle range.
    snrs = [Inf,30,10,3,1,0.3,0.1,0.03];
end
snrs = sort(snrs(:).','descend');
if nargin < 3 || isempty(sample_count)
    sample_count = 100000;
end

L_max = 5;
shell_count = 12;
restart_count = 20;
seed = 42;
shell_ratio = 1000;          % fixed for the 'shell' experiment
max_iterations = 3000;
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');
volume_file = 'vol_TRPV1_true.mrc';

output_dir = fullfile(repo_root,'results','gradient-descent','noise-sweep');
if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end
diary(fullfile(output_dir,sprintf('noise_sweep_%s.log',experiment)));
diary_cleanup = onCleanup(@() diary('off'));
total_timer = tic;

fprintf('============================================================\n');
fprintf(' Noise sweep: %s\n',experiment);
fprintf(' N = %d samples per moment, %d restarts, %d iterations\n', ...
    sample_count,restart_count,max_iterations);
fprintf(' SNR grid (power ratio, clean to noisy): [%s]\n',num2str(snrs));
fprintf(' Uniform SO(3) rotations; iid complex Gaussian noise on c_m[k]\n');
fprintf('============================================================\n');

[coefficients,support,radial_basis,label] = build_signal(experiment, ...
    repo_root,L_max,shell_count,shell_ratio,volume_file);
fprintf('\n  signal: %s\n',label);
report_structure(coefficients,L_max,support);

% Reference: exact population moment, for the sampling-error diagnostic.
exact_problem = gd_create_problem(coefficients,support,radial_basis, ...
    struct('name',label,'kind','noise-sweep','moment_source','exact', ...
           'success_threshold',1e-4));
exact_moment = exact_problem.observed_moment;
exact_energy = block_energy(exact_moment);
projection_rms = projection_scale(coefficients,support,radial_basis, ...
    exact_problem.model,L_max);
sigmas = sigma_from_snr(snrs,projection_rms);
fprintf('  ||P3_exact||^2 = %.4e   projection RMS = %.4e\n', ...
    exact_energy,projection_rms);
fprintf('  sigma per SNR: [%s]\n',num2str(sigmas,'%.4g '));

records = cell(numel(snrs),1);
for index = 1:numel(snrs)
    records{index} = run_cell(snrs(index),sigmas(index),coefficients, ...
        support,radial_basis,label,sample_count,seed,restart_count, ...
        optimizer,max_iterations,exact_moment,exact_energy,output_dir);
end

report_summary(records,experiment,restart_count,sample_count);
write_summary(records,output_dir,experiment,restart_count,sample_count);
summary = struct('experiment',experiment,'snrs',snrs,'sigmas',sigmas, ...
    'records',{records},'sample_count',sample_count, ...
    'projection_rms',projection_rms,'runtime_seconds',toc(total_timer));
save(fullfile(output_dir,sprintf('noise_sweep_%s.mat',experiment)), ...
    'summary','-v7.3');
fprintf('\nFinished in %.1f s (%.1f min). Written to %s\n', ...
    summary.runtime_seconds,summary.runtime_seconds/60,output_dir);
clear diary_cleanup;
end

%% ==================== signal construction ====================
function [coefficients,support,radial_basis,label] = build_signal( ...
    experiment,repo_root,L_max,shell_count,shell_ratio,volume_file)
support = shell_count*ones(1,L_max+1);
switch lower(experiment)
    case 'synthetic'
        coefficients = gd_generate_josh_synthetic_coefficients( ...
            L_max,shell_count,42);
        coefficients = gd_impose_band_dynamic_range(coefficients,L_max, ...
            support,1);
        radial_basis = repmat({eye(shell_count)},1,L_max+1);
        label = sprintf('synthetic L=%d R=%d, identity basis, D=1', ...
            L_max,shell_count);
    case 'shell'
        coefficients = gd_generate_josh_synthetic_coefficients( ...
            L_max,shell_count,42);
        coefficients = gd_impose_shell_dynamic_range(coefficients,L_max, ...
            support,shell_ratio);
        radial_basis = repmat({eye(shell_count)},1,L_max+1);
        label = sprintf('synthetic, shell profile S=%g, band norms equal', ...
            shell_ratio);
    case 'real'
        % NDFT-exact shell projection. This REPLACED the Fourier-Bessel
        % least-squares expansion (expand_real_volume_fourier_bessel), which
        % is not a well-defined function of the volume: it fits all (l,s,m)
        % jointly through normal equations at cond(B) ~ 3e9 and truncates 25
        % of 432 directions at a 1e-8 cutoff, so it returns one arbitrary
        % member of a solution set. Changing the FB shell count with the
        % volume held fixed moved the resulting array by 1.36 relative; the
        % NDFT array reproduces to 4.1e-17 under radial refinement, because
        % each shell is an independent quadrature projection with no solve.
        %
        % The radial basis is the identity: the shell samples ARE the
        % coefficients, so there is no Phi layer and no radial_count.
        coefficients = nu_expand_volume_shells( ...
            fullfile(repo_root,'data','raw',volume_file), ...
            struct('L_max',L_max,'grid_size',31,'rho_max',pi, ...
                   'shell_count',shell_count,'mode','exact'));
        % NATIVE band norms. gd_impose_band_dynamic_range(...,1) is band-to-
        % band spectral whitening, and it is not moment-accessible: the
        % projection mixes l inside each m block, so the whitened moment
        % cannot be formed from the native moment without already knowing
        % the answer. Only the overall scale is fixed, which the cubic
        % moment leaves free anyway.
        coefficients = coefficients/max(norm(coefficients(:)),eps);
        support = shell_count*ones(1,L_max+1);
        radial_basis = repmat({eye(shell_count)},1,L_max+1);
        label = sprintf('%s, NDFT shells, K=%d, native band norms', ...
            volume_file,shell_count);
    otherwise
        error(['Unknown experiment "%s". Use synthetic, real or shell.'], ...
            experiment);
end
end

function report_structure(a,L_max,support)
band = zeros(1,L_max+1);
shell = ones(1,L_max+1);
for l = 0:L_max
    block = a(1:support(l+1),gd_band_indices(l));
    band(l+1) = norm(block,'fro');
    energies = sqrt(sum(abs(block).^2,2));
    positive = energies(energies > 0);
    if ~isempty(positive)
        shell(l+1) = max(positive)/min(positive);
    end
end
fprintf('  band norms          : [%s]\n',num2str(band,'%.4g '));
fprintf('  shell max/min per l : [%s]\n',num2str(shell,'%.4g '));
end

function value = projection_scale(a,support,radial_basis,model,L_max)
profiles = gd_apply_radial_basis(a,radial_basis,support,L_max);
radial_count = size(profiles,1);
projected = zeros(2*L_max+1,radial_count);
for m = -L_max:L_max
    value_m = zeros(radial_count,1);
    for l = abs(m):L_max
        weight = model.normalization(l+1,m+L_max+1)* ...
            model.legendre_zero(l+1,m+L_max+1);
        value_m = value_m+weight*profiles(:,gd_sh_index(l,m));
    end
    projected(m+L_max+1,:) = value_m.';
end
value = sqrt(mean(abs(projected(:)).^2));
end

%% ==================== one sigma cell ====================
function record = run_cell(snr,sigma,coefficients,support,radial_basis, ...
    label,sample_count,seed,restart_count,optimizer,max_iterations, ...
    exact_moment,exact_energy,output_dir)
fprintf('\n============================================================\n');
fprintf(' SNR = %s   (sigma = %.4g)\n',snr_text(snr),sigma);
fprintf('============================================================\n');

timer = tic;
options = struct('name',sprintf('%s SNR=%g',label,snr), ...
    'kind','noise-sweep','moment_source','empirical', ...
    'success_threshold',1e-4, ...
    'empirical',struct('n_samples',sample_count,'noise_sigma',sigma, ...
        'seed',seed));
problem = gd_create_problem(coefficients,support,radial_basis,options);
fprintf('  moment estimated in %.1f s\n',toc(timer));

% How far the estimated moment sits from the exact one, and the loss the
% true coefficients achieve against it. The latter is the floor no optimizer
% can beat in this cell.
moment_error = sqrt(block_difference(problem.observed_moment,exact_moment)/ ...
    max(exact_energy,eps));
floor_loss = gd_third_moment_objective(problem.x_true,problem);
fprintf('  ||P3_emp - P3_exact||/||P3_exact|| = %.4e\n',moment_error);
fprintf('  loss at TRUE coefficients (floor)  = %.4e\n',floor_loss);

restart_options = struct('restart_count',restart_count,'seed',seed, ...
    'optimizer',optimizer,'initialization','real');
experiment = gd_run_josh_baseline_restarts(problem,restart_options);

restarts = experiment.restarts;
errors = [restarts.optimized_band_error];
losses = [restarts.objective];
capped = sum([restarts.iterations] >= max_iterations);
converged = sum([restarts.exitflag] > 0);
basins = gd_cluster_basins(losses,[restarts.exitflag]);
lowest = basins(1);

fprintf('\n  --- SNR = %s ---\n',snr_text(snr));
fprintf('  median l>=2 error : %.4e\n',median(errors));
fprintf('  best   l>=2 error : %.4e\n',min(errors));
fprintf('  floor loss        : %.4e   (best achieved %.4e)\n', ...
    floor_loss,min(losses));
fprintf('  lowest cluster    : %d of %d restarts at loss %.4e\n', ...
    lowest.count,restart_count,lowest.loss);
fprintf('  distinct basins   : %d%s\n',numel(basins),truncation_note(capped));
fprintf('  converged %d/%d, capped %d/%d, %.1f s\n', ...
    converged,restart_count,capped,restart_count,toc(timer));

record = struct('snr',snr,'sigma',sigma, ...
    'moment_error',moment_error,'floor_loss',floor_loss, ...
    'median_error',median(errors),'best_error',min(errors), ...
    'best_loss',min(losses),'lowest_count',lowest.count, ...
    'lowest_loss',lowest.loss,'basins',numel(basins), ...
    'converged',converged,'capped',capped,'restarts',restarts, ...
    'runtime_seconds',toc(timer));
write_cell_csv(record,output_dir,label);
end

%% ==================== helpers ====================
function sigmas = sigma_from_snr(snrs,projection_rms)
%SIGMA_FROM_SNR sigma = rms(c)/sqrt(SNR); SNR = Inf gives sigma = 0.
sigmas = projection_rms./sqrt(snrs);
sigmas(~isfinite(snrs)) = 0;
end

function text = snr_text(snr)
if ~isfinite(snr)
    text = 'Inf';
elseif snr >= 1
    text = sprintf('%g',snr);
else
    text = sprintf('%.3g',snr);
end
end

function value = block_energy(blocks)
value = 0;
for index = 1:numel(blocks)
    value = value+sum(abs(blocks{index}(:)).^2);
end
end

function value = block_difference(first,second)
value = 0;
for index = 1:numel(first)
    difference = first{index}-second{index};
    value = value+sum(abs(difference(:)).^2);
end
end


function note = truncation_note(capped)
if capped > 0
    note = sprintf('  << UPPER BOUND, %d truncated >>',capped);
else
    note = '  (measured)';
end
end

function report_summary(records,experiment,restart_count,sample_count)
fprintf('\n============================================================\n');
fprintf(' Noise sweep summary: %s (%d restarts, N = %d)\n', ...
    experiment,restart_count,sample_count);
fprintf('============================================================\n');
fprintf('%10s %10s %12s %12s %12s %12s %8s %8s\n', ...
    'SNR','sigma','moment err','floor loss','median e','best e', ...
    'lowest','basins');
for index = 1:numel(records)
    r = records{index};
    fprintf('%10s %10.4g %12.3e %12.3e %12.3e %12.3e %5d/%-2d %8d\n', ...
        snr_text(r.snr),r.sigma,r.moment_error,r.floor_loss, ...
        r.median_error,r.best_error,r.lowest_count,restart_count,r.basins);
end
fprintf(['\nSNR is a power ratio, mean|c|^2/sigma^2, not decibels.\n', ...
    '"lowest" counts restarts in the lowest loss cluster -- the ', ...
    'threshold-free\nreliability measure. "floor loss" is the loss at the ', ...
    'true coefficients,\nwhich no optimizer can beat in that cell.\n']);
end

function write_cell_csv(record,output_dir,label)
name = regexprep(lower(label),'[^a-z0-9]+','_');
path = fullfile(output_dir,sprintf('restarts_%s_snr_%g.csv', ...
    name(1:min(30,numel(name))),record.snr));
fid = fopen(path,'w');
assert(fid >= 0,'Could not open %s.',path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,['restart,seed,loss,relative_error,optimized_band_error,', ...
    'gradient_norm,iterations,exitflag,runtime_seconds\n']);
for k = 1:numel(record.restarts)
    r = record.restarts(k);
    fprintf(fid,'%d,%d,%.12e,%.12e,%.12e,%.12e,%d,%d,%.4f\n', ...
        r.restart,r.seed,r.objective,r.relative_error, ...
        r.optimized_band_error,r.gradient_norm,r.iterations,r.exitflag, ...
        r.runtime_seconds);
end
clear cleanup;
end

function write_summary(records,output_dir,experiment,restart_count,samples)
path = fullfile(output_dir,sprintf('noise_sweep_%s_summary.csv',experiment));
fid = fopen(path,'w');
assert(fid >= 0,'Could not open %s.',path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,['snr,sigma,n_samples,restarts,moment_error,floor_loss,', ...
    'median_error,best_error,best_loss,lowest_cluster_count,', ...
    'lowest_cluster_loss,distinct_basins,converged,capped\n']);
for index = 1:numel(records)
    r = records{index};
    fprintf(fid,'%g,%.12e,%d,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%.12e,%d,%d,%d\n', ...
        r.snr,r.sigma,samples,restart_count,r.moment_error,r.floor_loss, ...
        r.median_error,r.best_error,r.best_loss,r.lowest_count, ...
        r.lowest_loss,r.basins,r.converged,r.capped);
end
clear cleanup;
end
