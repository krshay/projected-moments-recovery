function summary = run_crop_convention(volume,crop,snr,sample_count,max_iterations)
%RUN_CROP_CONVENTION Shell-model recovery with and without the margin crop.
%   RUN_CROP_CONVENTION(VOLUME,CROP) runs one cell at SNR = 30 from a sampled
%   moment. CROP = 1 is the current convention (no crop) and must reproduce the
%   published numbers; CROP = 0.697 is the crop used by
%   krshay/Non-Uniform-Second-Moment (indices 50:300 of a 360^3 map).
%
%   Everything except the crop is identical to RUN_NOISE_SWEEP's 'real' branch:
%   NDFT-exact shell expansion, L = 5, K = 12 shells, identity radial basis,
%   native band norms, bands 0 and 1 pinned, 20 restarts, 3000 iterations,
%   seed 42, N = 1e5. The crop is applied inside NU_LOAD_VOLUME, so no part of
%   the preprocessing is reimplemented here -- an earlier attempt rebuilt the
%   resize (Fourier instead of linear interpolation) and produced an array
%   0.797 away from the pipeline's, which invalidated the comparison.
%
%   REFERENCE VALUES at CROP = 1, for checking the harness before believing
%   any cropped result:
%       80S    moment err 8.981e-04, floor 8.057e-07, median l>=2 5.250e-02,
%              lowest 20/20, capped 0/20
%       TRPV1  moment err 7.458e-04, floor 5.559e-07, median l>=2 5.468e-01,
%              capped 16/20
%   If the CROP = 1 run does not reproduce these, stop: something other than
%   the crop has changed.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(volume),       volume = 'vol_80S_true.mrc'; end
if nargin < 2 || isempty(crop),         crop = 1;      end
if nargin < 3 || isempty(snr),          snr = 30;      end
if nargin < 4 || isempty(sample_count), sample_count = 100000; end
% The uncropped cell converges in at most 2613 iterations and never caps, so
% raising this leaves the baseline bit-identical while letting the cropped
% cells finish. Note it cannot rescue restarts that CONVERGE to a wrong
% solution (||g|| ~ 1e-10) -- only those still descending at the cap.
if nargin < 5 || isempty(max_iterations), max_iterations = 3000; end

L_max = 5; K = 12; grid_size = 31; restart_count = 20; seed = 42;
support = K*ones(1,L_max+1);
basis = repmat({eye(K)},1,L_max+1);
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');
model = gd_precompute_model(L_max);

output_dir = fullfile(repo_root,'results','gradient-descent','crop-convention');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('%s_crop%g_snr%g_it%d',regexprep(volume,'\.[^.]*$',''),crop,snr,max_iterations);
diary(fullfile(output_dir,sprintf('crop_%s.log',tag)));
diary_cleanup = onCleanup(@() diary('off'));
timer_all = tic;

% ---- signal: identical to run_noise_sweep 'real', plus the crop ----
coefficients = nu_expand_volume_shells( ...
    fullfile(repo_root,'data','raw',volume), ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',crop));
coefficients = coefficients/max(norm(coefficients(:)),eps);

band = zeros(1,L_max+1); participation = zeros(1,L_max+1);
kappa = zeros(1,L_max+1); shell = zeros(1,L_max+1);
for l = 0:L_max
    block = coefficients(:,gd_band_indices(l));
    band(l+1) = norm(block,'fro');
    M = zeros(K,2*l+1); M(:,1) = real(block(:,l+1));
    for m = 1:l
        M(:,2*m)   = sqrt(2)*real(block(:,m+l+1));
        M(:,2*m+1) = sqrt(2)*imag(block(:,m+l+1));
    end
    values = svd(M);
    kappa(l+1) = values(1)/max(values(min(end,2*l+1)),eps);
    participation(l+1) = (sum(values.^2)^2/sum(values.^4))/min(K,2*l+1);
    rows = sqrt(sum(M.^2,2)); rows = rows(rows > 0);
    shell(l+1) = max(rows)/min(rows);
end

fprintf('\n============================================================\n');
fprintf(' Crop convention: %s, crop = %g\n',volume,crop);
fprintf(' L=%d, K=%d shells, grid %d, native band norms\n',L_max,K,grid_size);
fprintf(' SNR = %g, N = %d, %d restarts, %d iterations, seed %d\n', ...
    snr,sample_count,restart_count,max_iterations,seed);
fprintf('============================================================\n');
fprintf('  band norms    : [%s]\n',num2str(band,'%.4f '));
fprintf('  participation : %.4f   kappa* %.1f   shell range %.4g\n', ...
    mean(participation(3:end)),exp(mean(log(kappa(3:end)))), ...
    exp(mean(log(shell(3:end)))));

% ---- sampled moment ----
projection_rms = projection_scale(coefficients,support,basis,model,L_max);
sigma = projection_rms/sqrt(snr);
exact = gd_projected_third_moment( ...
    gd_apply_radial_basis(coefficients,basis,support,L_max),model);
energy = 0;
for b = 1:numel(exact), energy = energy+sum(abs(exact{b}(:)).^2); end

problem = gd_create_problem(coefficients,support,basis, ...
    struct('name',sprintf('%s crop=%g SNR=%g',volume,crop,snr), ...
           'kind','crop-convention','moment_source','empirical', ...
           'success_threshold',1e-4, ...
           'empirical',struct('n_samples',sample_count, ...
               'noise_sigma',sigma,'seed',seed)));
difference = 0;
for b = 1:numel(exact)
    difference = difference+sum(abs(problem.observed_moment{b}(:)-exact{b}(:)).^2);
end
moment_error = sqrt(difference/max(energy,eps));
floor_loss = gd_third_moment_objective(problem.x_true,problem);
fprintf('  moment err %.4e | floor loss %.4e\n',moment_error,floor_loss);

% ---- recovery ----
experiment = gd_run_josh_baseline_restarts(problem, ...
    struct('restart_count',restart_count,'seed',seed, ...
           'optimizer',optimizer,'initialization','real'));
restarts = experiment.restarts;
errors = [restarts.optimized_band_error];
capped = sum([restarts.iterations] >= max_iterations);
estimates = cell(1,restart_count);
for k = 1:restart_count, estimates{k} = restarts(k).coefficients_estimate; end
clusters = gd_cluster_solutions(estimates,support,L_max);

fprintf('\n--------------- RESULT (crop = %g) ---------------\n',crop);
fprintf('  median l>=2 error : %.4e\n',median(errors));
fprintf('  best   l>=2 error : %.4e\n',min(errors));
fprintf('  restarts agreeing : %d/%d   (distinct solutions %d)\n', ...
    clusters(1).count,restart_count,numel(clusters));
fprintf('  converged %d/%d, capped %d/%d\n', ...
    sum([restarts.exitflag] > 0),restart_count,capped,restart_count);
if crop == 1
    fprintf('\n  CHECK against the published uncropped values above.\n');
end

summary = struct('volume',volume,'crop',crop,'snr',snr, ...
    'L_max',L_max,'shell_count',K,'grid_size',grid_size, ...
    'sample_count',sample_count,'max_iterations',max_iterations, ...
    'band',band,'participation',mean(participation(3:end)), ...
    'kappa',exp(mean(log(kappa(3:end)))),'moment_error',moment_error, ...
    'floor_loss',floor_loss,'errors',errors,'capped',capped, ...
    'agreeing',clusters(1).count,'distinct',numel(clusters), ...
    'restarts',restarts,'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,sprintf('crop_%s.mat',tag)),'summary','-v7.3');
fprintf('\nFinished in %.1f min. Written to %s\n', ...
    summary.runtime_seconds/60,output_dir);
clear diary_cleanup;
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
