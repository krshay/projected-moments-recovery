function summary = run_single_restart(seed,L_max,volume,crop,snr,sample_count,max_iterations,shell_count)
%RUN_SINGLE_RESTART One restart of shell-model recovery, for a chosen seed.
%   RUN_SINGLE_RESTART(SEED) runs a single random start at L = 13 with the
%   exact moment, so it can be repeated with different seeds until one lands
%   at low error. Each call saves its own file, keyed by the seed, so nothing
%   is overwritten and EXPORT_SOLVED_VOLUME can render any of them.
%
%     SEED            random start                                     (1)
%     L_MAX           angular band limit                              (13)
%     VOLUME          file in data/raw                 ('vol_80S_true.mrc')
%     CROP            margin crop fraction, 1 for none             (0.697)
%     SNR             power ratio; Inf uses the exact moment         (Inf)
%     SAMPLE_COUNT    projections, ignored when SNR = Inf            (1e5)
%     MAX_ITERATIONS  optimizer cap                                 (5000)
%     SHELL_COUNT     radial shells                                    (12)
%
%   COST. At L = 13 the moment has 763 terms and 547 blocks against 69 and 91
%   at L = 5, and there are 2304 free parameters rather than 384 -- so roughly
%   ten times the work per iteration on top of six times the unknowns. Budget
%   30-60 minutes for one restart. A sampled moment costs a further ~27 minutes
%   per call at N = 1e5, which is why SNR defaults to Inf: for a figure, the
%   exact moment removes the sampling floor and the wait.
%
%   THE REPORTED FIGURE. The side-by-side rendering in the paper is
%       run_single_restart(1,13,'vol_80S_true.mrc',1,2,200000,5000,12)
%   which reached loss 1.0418e-06 against a floor of 1.4542e-06, overall error
%   2.366e-02, gradient norm 1.1e-09, in 2804 of 5000 iterations and 342
%   minutes. Two settings differ from RUN_NOISE_SWEEP_UNPINNED and both matter
%   when reading it: degrees 0 and 1 are FIXED TO THEIR TRUE VALUES, through
%   GD_CREATE_PROBLEM's default fixed_bands = [0,1], so this is not end-to-end
%   recovery; and the moment is sampled and fitted in the twisted convention
%   with GD_PRECOMPUTE_MODEL rather than GD_PHYSICAL_MODEL. The pair is
%   matched either way -- the floor loss proves it -- but the two drivers
%   answer slightly different questions.
%
%   ONE RESTART IS NOT A SUCCESS RATE. Re-rolling seeds until one looks good
%   gives a picture of what recovery looks like WHEN IT WORKS, not how often it
%   works. At L = 5 the cropped 80S cell had 11 of 20 restarts agreeing; the
%   equivalent fraction at L = 13 is unmeasured and could be far worse, since
%   the landscape has six times the dimensions. Report the seed and say the
%   figure is a selected restart.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(seed),           seed = 1;                    end
if nargin < 2 || isempty(L_max),          L_max = 13;                  end
if nargin < 3 || isempty(volume),         volume = 'vol_80S_true.mrc'; end
if nargin < 4 || isempty(crop),           crop = 0.697;                end
if nargin < 5 || isempty(snr),            snr = Inf;                   end
if nargin < 6 || isempty(sample_count),   sample_count = 100000;       end
if nargin < 7 || isempty(max_iterations), max_iterations = 5000;       end
if nargin < 8 || isempty(shell_count),    shell_count = 12;            end
K = shell_count; grid_size = 31;

support = K*ones(1,L_max+1);
basis = repmat({eye(K)},1,L_max+1);
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');

output_dir = fullfile(repo_root,'results','gradient-descent','single-restart');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('%s_L%d_K%d_crop%g_snr%g_seed%d', ...
    regexprep(volume,'\.[^.]*$',''),L_max,K,crop,snr,seed);
diary(fullfile(output_dir,sprintf('%s.log',tag)));
diary_cleanup = onCleanup(@() diary('off'));
timer_all = tic;

fprintf('\n============================================================\n');
fprintf(' Single restart: %s, L = %d, K = %d, crop = %g\n',volume,L_max,K,crop);
fprintf(' SNR = %g, seed = %d, %d iterations\n',snr,seed,max_iterations);
fprintf('============================================================\n');

model = gd_precompute_model(L_max);
fprintf('  model: %d terms, %d blocks, %d free dof\n', ...
    numel(model.terms),numel(model.blocks),K*sum(2*(2:L_max)+1));

coefficients = nu_expand_volume_shells( ...
    fullfile(repo_root,'data','raw',volume), ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',crop));
coefficients = coefficients/max(norm(coefficients(:)),eps);

options = struct('name',sprintf('%s L%d seed%d',volume,L_max,seed), ...
    'kind','single-restart','success_threshold',1e-4);
if isfinite(snr)
    projection_rms = projection_scale(coefficients,support,basis,model,L_max);
    sigma = projection_rms/sqrt(snr);
    options.moment_source = 'empirical';
    options.empirical = struct('n_samples',sample_count, ...
        'noise_sigma',sigma,'seed',42);
    fprintf('  sampled moment: sigma %.4e, N = %d\n',sigma,sample_count);
else
    options.moment_source = 'exact';
    fprintf('  exact moment\n');
end
problem = gd_create_problem(coefficients,support,basis,options);
floor_loss = gd_third_moment_objective(problem.x_true,problem);
fprintf('  floor loss at truth: %.4e\n',floor_loss);

fprintf('  optimizing (this is the slow part) ...\n');
experiment = gd_run_josh_baseline_restarts(problem, ...
    struct('restart_count',1,'seed',seed,'optimizer',optimizer, ...
           'initialization','real'));
r = experiment.restarts(1);

fprintf('\n--------------- RESULT (seed %d) ---------------\n',seed);
fprintf('  l>=2 error : %.4e\n',r.optimized_band_error);
% The moment is invariant under rotation about z; the pinned bands normally
% break that, but not for a molecule whose l=1 band is purely m=0. Report the
% aligned figure too, and say whether it is the one to read.
if isfield(r,'axial_gauge_free')
    if r.axial_gauge_free
        fprintf('  l>=2 aligned : %.4e  at alpha = %.1f deg  <- READ THIS\n', ...
            r.optimized_band_error_aligned,r.axial_alpha*180/pi);
        fprintf('     (axial rotation is an unbroken gauge here: the pinned\n');
        fprintf('      bands are m=0 only, so any point on the circle is correct)\n');
    else
        fprintf('  l>=2 aligned : %.4e  at alpha = %.1f deg  (informational:\n', ...
            r.optimized_band_error_aligned,r.axial_alpha*180/pi);
        fprintf('      the pinned bands DO break the rotation, so the raw\n');
        fprintf('      error above is the honest one)\n');
    end
end
fprintf('  loss       : %.4e   (floor %.4e)\n',r.objective,floor_loss);
fprintf('  ||g||      : %.3e | iterations %d/%d | exitflag %d\n', ...
    r.gradient_norm,r.iterations,max_iterations,r.exitflag);
if r.iterations >= max_iterations
    fprintf('  NOT CONVERGED (hit the cap) -- error is not a landscape result\n');
end

summary = struct('volume',volume,'crop',crop,'snr',snr,'seed',seed, ...
    'L_max',L_max,'shell_count',K,'grid_size',grid_size, ...
    'sample_count',sample_count,'max_iterations',max_iterations, ...
    'errors',r.optimized_band_error,'floor_loss',floor_loss, ...
    'capped',double(r.iterations >= max_iterations), ...
    'restarts',r,'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,sprintf('%s.mat',tag)),'summary','-v7.3');
fprintf('\nFinished in %.1f min.\n  %s\n', ...
    summary.runtime_seconds/60,fullfile(output_dir,[tag '.mat']));
fprintf('  visualize: export_solved_volume(''%s'')\n', ...
    fullfile('results','gradient-descent','single-restart',[tag '.mat']));
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
