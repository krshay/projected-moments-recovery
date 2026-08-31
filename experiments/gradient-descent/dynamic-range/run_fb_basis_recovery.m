function summary = run_fb_basis_recovery(volume,snr,sample_count,n_basis,n_samples_radial)
%RUN_FB_BASIS_RECOVERY Recovery in the Fourier-Bessel basis from a sample moment.
%   RUN_FB_BASIS_RECOVERY() runs 80S at SNR = 1 with the defaults below.
%   RUN_FB_BASIS_RECOVERY(VOLUME,SNR,N,S,K) overrides them.
%
%     VOLUME  file in data/raw            (default 'vol_80S_true.mrc')
%     SNR     power ratio mean|c|^2/sigma^2, Inf for the exact moment  (1)
%     N       projections per moment estimate                          (1e5)
%     S       Fourier-Bessel functions per band -- THE UNKNOWNS         (12)
%     K       radial sample points -- DISCRETIZATION ONLY               (16)
%
%   THE MODEL. u_l(rho) is taken to lie in the span of S Fourier-Bessel radial
%   functions, u_l = sum_s A_{l,s} j_l(k_{l,s} rho/rho_max). The A are the
%   unknowns; K only sets how finely that function is sampled to evaluate the
%   moment. Truth is A_true = Phi \ u, the least-squares fit of the model to
%   the volume's NDFT shell expansion, so the recovery target is the best
%   rank-S radial model of the molecule. The residual ||Phi*A_true - u||/||u||
%   is reported: it is the part of the molecule the model cannot represent,
%   and it bounds the accuracy of any FB result from below.
%
%   K MUST EXCEED S. At K = S, Phi is square and interpolating, and the S-th
%   Bessel function -- which has S zeros in the interval -- is nearly
%   indistinguishable from a combination of its predecessors at exactly S
%   points. Measured for 80S: cond(Phi_5) is 6758 at K = S = 12 and 5.40 at
%   K = 16, and the recovery goes from 7/7 restarts cap-limited and never
%   converging, to 14/20 agreeing at ||g|| ~ 8e-10. One extra sample point
%   collapses the condition number by three orders of magnitude; beyond
%   K = S+2 it saturates (K = 96 gives 5.378). This is a property of the
%   discretization, not of the Fourier-Bessel basis.
%
%   Bands 0 and 1 are pinned to truth, as elsewhere. Band norms are native.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(volume),           volume = 'vol_80S_true.mrc'; end
if nargin < 2 || isempty(snr),              snr = 1;       end
if nargin < 3 || isempty(sample_count),     sample_count = 100000; end
if nargin < 4 || isempty(n_basis),          n_basis = 12;  end
if nargin < 5 || isempty(n_samples_radial), n_samples_radial = 16; end
S = n_basis; K = n_samples_radial;
assert(K > S, ['K (%d) must exceed S (%d): at K = S the Fourier-Bessel ', ...
    'design matrix is square and near-singular and the recovery will not ', ...
    'converge. Use K = S+2 or more.'],K,S);
assert(isscalar(snr) && ~isnan(snr) && snr > 0, ...
    'SNR must be positive; use Inf for the exact population moment.');

L_max = 5; restart_count = 20; max_iterations = 3000; seed = 42;
support = S*ones(1,L_max+1);
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');
model = gd_precompute_model(L_max);

output_dir = fullfile(repo_root,'results','gradient-descent','fb-basis');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('%s_S%d_K%d_snr%g',regexprep(volume,'\.[^.]*$',''),S,K,snr);
diary(fullfile(output_dir,sprintf('fb_basis_%s.log',tag)));
diary_cleanup = onCleanup(@() diary('off'));
timer_all = tic;

% ---- the molecule, sampled on K spheres (well-posed NDFT expansion) ----
u = nu_expand_volume_shells(fullfile(repo_root,'data','raw',volume), ...
    struct('L_max',L_max,'grid_size',31,'rho_max',pi, ...
           'shell_count',K,'mode','exact'));
u = u/max(norm(u(:)),eps);                       % native band norms

% ---- Fourier-Bessel basis: S functions on the SAME K radii ----
fb_metadata = struct('L_max',L_max,'rho_max',pi,'available_shells',support, ...
    'bessel_zeros',zeros(L_max+1,S));
for l = 0:L_max
    for s = 1:S
        fb_metadata.bessel_zeros(l+1,s) = nu_spherical_bessel_zero(l,s);
    end
end
Phi = gd_fourier_bessel_radial_basis(fb_metadata,K);
condition = cellfun(@cond,Phi);

% ---- truth in the model: least-squares fit ----
A_true = zeros(S,(L_max+1)^2);
for l = 0:L_max
    columns = gd_band_indices(l);
    A_true(:,columns) = Phi{l+1}\u(:,columns);
end
u_model = gd_apply_radial_basis(A_true,Phi,support,L_max);
model_error = norm(u_model-u,'fro')/norm(u,'fro');

fprintf('\n============================================================\n');
fprintf(' Fourier-Bessel recovery: %s\n',volume);
fprintf(' S = %d basis functions (unknowns), K = %d radial samples\n',S,K);
fprintf(' SNR = %g, N = %d, %d restarts, %d iterations, seed %d\n', ...
    snr,sample_count,restart_count,max_iterations,seed);
fprintf('============================================================\n');
fprintf('  cond(Phi_l)        : [%s]\n',num2str(condition,'%.3g '));
fprintf('  FB modelling error : %.4e  (floor on any FB result)\n',model_error);

% ---- sample moment of the model volume ----
noisy = isfinite(snr);
projection_rms = projection_scale(A_true,support,Phi,model,L_max);
if noisy, sigma = projection_rms/sqrt(snr); else, sigma = 0; end
exact = gd_projected_third_moment(u_model,model);
exact_energy = 0;
for b = 1:numel(exact), exact_energy = exact_energy+sum(abs(exact{b}(:)).^2); end

options = struct('name',sprintf('%s FB S%d K%d SNR=%g',volume,S,K,snr), ...
    'kind','fb-basis','success_threshold',1e-4);
if noisy
    options.moment_source = 'empirical';
    options.empirical = struct('n_samples',sample_count, ...
        'noise_sigma',sigma,'seed',seed);
else
    options.moment_source = 'exact';
end
problem = gd_create_problem(A_true,support,Phi,options);
difference = 0;
for b = 1:numel(exact)
    difference = difference+sum(abs(problem.observed_moment{b}(:)-exact{b}(:)).^2);
end
fprintf('  projection RMS %.4e -> sigma %.4e\n',projection_rms,sigma);
fprintf('  moment err vs exact: %.4e | floor loss %.4e\n', ...
    sqrt(difference/max(exact_energy,eps)), ...
    gd_third_moment_objective(problem.x_true,problem));
fprintf('  free dof (l>=2): %d\n',numel(problem.x_true));

% ---- recovery from random starts ----
experiment = gd_run_josh_baseline_restarts(problem, ...
    struct('restart_count',restart_count,'seed',seed, ...
           'optimizer',optimizer,'initialization','real'));
restarts = experiment.restarts;
errors = [restarts.optimized_band_error];
iterations = [restarts.iterations];
flags = [restarts.exitflag];
capped = sum(iterations >= max_iterations);

profile_error = zeros(1,restart_count);
estimates = cell(1,restart_count);
for k = 1:restart_count
    estimates{k} = restarts(k).coefficients_estimate;
    estimated = gd_apply_radial_basis(estimates{k},Phi,support,L_max);
    profile_error(k) = min(norm(estimated-u_model,'fro'), ...
        norm(estimated+u_model,'fro'))/norm(u_model,'fro');
end
clusters = gd_cluster_solutions(estimates,support,L_max);

fprintf('\n--------------- RESULT ---------------\n');
fprintf('  restarts agreeing on one solution : %d/%d\n', ...
    clusters(1).count,restart_count);
fprintf('  distinct solutions                : %d\n',numel(clusters));
fprintf('  converged %d/%d, capped %d/%d, median ||g|| %.2e\n', ...
    sum(flags > 0),restart_count,capped,restart_count, ...
    median([restarts.gradient_norm]));
fprintf('  FB-coefficient error : best %.4e  median %.4e\n', ...
    min(errors),median(errors));
fprintf('  profile-space error  : best %.4e  median %.4e\n', ...
    min(profile_error),median(profile_error));
fprintf('  (total error against the molecule combines the profile error\n');
fprintf('   with the %.4e modelling error, in quadrature -- they are\n',model_error);
fprintf('   orthogonal, the second being the least-squares residual)\n');

summary = struct('volume',volume,'snr',snr,'S',S,'K',K, ...
    'sample_count',sample_count,'cond_phi',condition, ...
    'model_error',model_error,'errors',errors, ...
    'profile_error',profile_error,'capped',capped, ...
    'agreeing',clusters(1).count,'distinct',numel(clusters), ...
    'restarts',restarts,'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,sprintf('fb_basis_%s.mat',tag)),'summary','-v7.3');
fprintf('\nFinished in %.1f s (%.1f min). Written to %s\n', ...
    summary.runtime_seconds,summary.runtime_seconds/60,output_dir);
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
