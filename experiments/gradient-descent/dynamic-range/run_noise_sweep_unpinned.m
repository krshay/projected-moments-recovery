function summary = run_noise_sweep_unpinned(snr,volume,grid_size,sample_count,restart_count,max_iterations,snr_reference)
%RUN_NOISE_SWEEP_UNPINNED One SNR cell of the noise sweep, unpinned and physical.
%   Three deliberate differences from RUN_NOISE_SWEEP's 'real' branch:
%
%   1. NO BAND IS PINNED. A_0 and A_1 are optimisation variables, so this is
%      end-to-end recovery. The solution is then determined only up to rotation
%      and reflection, and every restart is scored after
%      GD_ALIGN_ROTATION_REFLECTION; the raw reflection-only error is kept
%      alongside it and is NOT the number to read.
%   2. THE COEFFICIENT CONVENTION IS THE PHYSICAL ONE.
%      NU_EXPAND_VOLUME_SHELLS multiplies band l by i^l so that its output
%      satisfies the standard reality condition A_{l,-m} = (-1)^m conj(A_{lm}).
%      GD_PRECOMPUTE_MODEL is the forward model for whatever array it is
%      handed -- the Monte Carlo sampler reproduces it on any input, to
%      sampling error -- and GD_PHYSICAL_MODEL(x) equals GD_PRECOMPUTE_MODEL
%      evaluated at (-i)^l x; that identity holds to 0 in double precision.
%      So the choice of model is exactly the choice of which array counts as
%      the volume's expansion, and the two choices give moments 10.4% apart
%      for 80S and 27.9% apart for TRPV1.
%
%      This driver samples the observed moment from the UNTWISTED array with
%      the default model and fits the twisted array with GD_PHYSICAL_MODEL:
%      a matched pair, floor loss ~1e-05 at N = 1e5. The earlier sweep
%      sampled and fitted the twisted array with the default model, which is
%      also a matched pair, but a different inverse problem. Mixing the halves
%      of either pair raises the floor loss to ~8e-02.
%   3. DEFAULT VOLUME IS EMD-24084, the C1 TRPV1 refinement. The repo's
%      vol_TRPV1_true.mrc is EMD-8117, which is C4-symmetrised: 94.5% of its
%      l >= 2 energy sits in m = 0 mod 4, which is what made it hard.
%
%   Otherwise it matches the original sweep: L = 5, K = 12, native band norms,
%   'real' initialisation, 20 restarts from seed 42, and SNR = Inf meaning
%   sigma = 0 with the moment still estimated from SAMPLE_COUNT projections.
%   MAX_ITERATIONS defaults to 8000 rather than 3000: at 3000 most restarts
%   were still descending, and a capped restart says nothing about the landscape.
%
%   SNR_REFERENCE selects the reference power in the SNR definition and
%   defaults to 'slice', the convention the reported sweeps used. Changing it
%   changes the noise level at a given SNR, so leave it alone to reproduce the
%   published figure; see SIGMA_REFERENCE_POWER.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 2 || isempty(volume),         volume = 'vol_TRPV1_emd24084.mat'; end
if nargin < 3 || isempty(grid_size),      grid_size = 32; end
if nargin < 4 || isempty(sample_count),   sample_count = 100000; end
if nargin < 5 || isempty(restart_count),  restart_count = 20; end
if nargin < 6 || isempty(max_iterations), max_iterations = 8000; end
if nargin < 7 || isempty(snr_reference),  snr_reference = 'slice'; end
L_max = 5; K = 12; seed = 42;
support = K*ones(1,L_max+1);
basis = repmat({eye(K)},1,L_max+1);

% One folder per molecule, named after the token following "vol_" so it sits
% beside the pinned sweeps ("TRPV1_noise sweep", "80S_noise sweep").
token = regexp(volume,'^vol_([^_]+)','tokens','once');
if isempty(token), label = regexprep(volume,'\.[^.]*$',''); else, label = token{1}; end
output_dir = fullfile(repo_root,'results','gradient-descent','noise-sweep', ...
    sprintf('%s_unpinned_noise sweep',label));
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('unpinned_%s_snr_%g',regexprep(volume,'\.[^.]*$',''),snr);
diary(fullfile(output_dir,[tag '.log'])); cleanup = onCleanup(@() diary('off'));
timer_all = tic;

a = nu_expand_volume_shells(fullfile(repo_root,'data','raw',volume), ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',1));
a = a/max(norm(a(:)),eps);
physical = a;                                   % untwisted: what the slice sees
for l = 0:L_max
    physical(:,gd_band_indices(l)) = (-1i)^l*a(:,gd_band_indices(l));
end
default_model = gd_precompute_model(L_max);
model = gd_physical_model(L_max);

% The noise level is sigma^2 = (reference power per coefficient)/SNR. The
% reference power is the mean power of the equatorial slice of the UNROTATED
% volume, which is the convention used for the reported sweeps; see
% SIGMA_REFERENCE_POWER below for that and for the Haar-averaged alternative.
D_proj = (2*L_max+1)*K;
projection_rms = sqrt(sigma_reference_power(physical,basis,support, ...
    default_model,L_max,K,snr_reference));
sigma = projection_rms/sqrt(snr);
if ~isfinite(sigma), sigma = 0; end

observed = gd_empirical_projected_third_moment(physical,support,basis, ...
    default_model,struct('n_samples',sample_count,'noise_sigma',sigma,'seed',seed));
exact = gd_projected_third_moment(physical,default_model);
energy = sum(cellfun(@(t) sum(abs(t(:)).^2),exact));
moment_error = sqrt(sum(cellfun(@(x,y) sum(abs(x(:)-y(:)).^2),observed,exact))/energy);

problem = gd_create_problem(a,support,basis, ...
    struct('name',sprintf('%s unpinned SNR=%g',volume,snr),'kind','noise-sweep-unpinned', ...
           'moment_source','exact','success_threshold',1e-4,'fixed_bands',[]));
problem.model = model;
problem.observed_moment = observed;
problem.moment_scale = sum(cellfun(@(t) sum(abs(t(:)).^2),observed));
floor_loss = gd_third_moment_objective(problem.x_true,problem);

fprintf('\n============================================================\n');
fprintf(' %s, UNPINNED, physical moment\n',volume);
fprintf(' SNR reference power: ''%s'' (D_proj = %d)\n',snr_reference,D_proj);
fprintf(' L=%d K=%d grid=%d | SNR=%g sigma=%.4e N=%d | %d restarts cap %d\n', ...
    L_max,K,grid_size,snr,sigma,sample_count,restart_count,max_iterations);
fprintf(' free dof %d | moment err %.4e | floor loss %.4e\n', ...
    numel(problem.x_true),moment_error,floor_loss);
fprintf('============================================================\n');

optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');
experiment = gd_run_josh_baseline_restarts(problem, ...
    struct('restart_count',restart_count,'seed',seed, ...
           'optimizer',optimizer,'initialization','real'));
R = experiment.restarts;

rows = zeros(restart_count,9);
fprintf('\n seed      loss   raw_l>=2   O3_l>=2    O3_all  refl  iters flag\n');
for k = 1:restart_count
    o = gd_align_rotation_reflection(R(k).coefficients_estimate,a,L_max);
    rows(k,:) = [R(k).seed,R(k).objective,R(k).optimized_band_error, ...
        o.error_free,o.error_all,o.reflected,R(k).iterations,R(k).exitflag, ...
        R(k).iterations >= max_iterations];
    fprintf('%5d %9.2e %10.4e %10.4e %9.4e %5d %6d %4d\n',rows(k,1:8));
end
T = array2table(rows,'VariableNames',{'seed','loss','raw_l2_error', ...
    'aligned_l2_error','aligned_all_band_error','reflected','iterations', ...
    'exitflag','hit_cap'});
writetable(T,fullfile(output_dir,[tag '.csv']));

fprintf('\nSNR %g: median aligned l>=2 %.4e | best %.4e | <1e-3 %d/%d | capped %d/%d\n', ...
    snr,median(T.aligned_l2_error),min(T.aligned_l2_error), ...
    sum(T.aligned_l2_error<1e-3),restart_count,sum(T.hit_cap),restart_count);

summary = struct('volume',volume,'snr',snr,'sigma',sigma, ...
    'snr_reference',snr_reference,'grid_size',grid_size, ...
    'L_max',L_max,'shell_count',K,'sample_count',sample_count, ...
    'max_iterations',max_iterations,'moment_error',moment_error, ...
    'floor_loss',floor_loss,'table',T,'restarts',R, ...
    'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,[tag '.mat']),'summary','-v7.3');
fprintf('Finished in %.1f min.\n',summary.runtime_seconds/60);
clear cleanup;
end

function power = sigma_reference_power(coefficients,basis,support,model,L_max,K,reference)
%SIGMA_REFERENCE_POWER Reference power per projected coefficient for the SNR.
%   The noise level is sigma^2 = POWER/SNR, so this function fixes what "SNR"
%   means. Two choices, which differ because a projection depends on the
%   viewing direction:
%
%   'slice' (DEFAULT, and the convention of the reported sweeps)
%       The mean power of the equatorial slice of the volume in its ORIGINAL
%       orientation, 1/D_proj * sum_{k,s} |c_k[s]|^2 with D_proj = (2L+1)K.
%       A fixed reference orientation, not an average over the sampled views.
%
%   'haar'
%       The same quantity averaged over the viewing distribution, which is the
%       Haar average, and which has a closed form. With
%       a^R_{lk} = sum_{k'} D^l_{kk'} a_{lk'} and the orthogonality relation
%       int D^l_{kk'} conj(D^{l'}_{kk''}) dR = delta_{ll'} delta_{k'k''}/(2l+1),
%       the cross-degree terms drop and
%           E_R sum_{k,s} |c_k[s]|^2 = sum_l W_l ||a_l||_F^2 / (2l+1),
%           W_l = sum_{k=-l}^{l} |N_{lk} P_l^k(0)|^2.
%       The spherical-harmonic addition theorem gives W_l = (2l+1)/(4 pi) at
%       every degree, so the sum collapses to ||a||_F^2/(4 pi): the averaged
%       projection energy depends only on the volume's norm, not its shape.
%       The sum is written out and checked against that value. Verified
%       against 20000-rotation Monte Carlo to 4e-4.
%
%   The two agree to the extent that the volume's energy is isotropic. For the
%   unit-norm expansions used here, 'slice' gives a power 1.6% below 'haar' for
%   80S but 22.3% above it for TRPV1 (EMD-24084). Switching 'slice' -> 'haar'
%   therefore scales sigma by 1.008 for 80S and by 0.904 for TRPV1, i.e. the
%   TRPV1 cells would carry 9.6% less noise amplitude at the same labelled SNR.
%   SWITCHING THIS CHANGES THE RESULTS: use 'slice' to reproduce the figures.

D_proj = (2*L_max+1)*K;
switch lower(reference)
    case 'slice'
        projected = project_equator( ...
            gd_apply_radial_basis(coefficients,basis,support,L_max),model,L_max);
        power = mean(abs(projected(:)).^2);
    case 'haar'
        total = 0;
        for l = 0:L_max
            weight = sum(abs(model.normalization(l+1,(-l:l)+L_max+1).* ...
                             model.legendre_zero(l+1,(-l:l)+L_max+1)).^2);
            band = coefficients(:,gd_band_indices(l));
            total = total+weight*sum(abs(band(:)).^2)/(2*l+1);
        end
        assert(abs(total-sum(abs(coefficients(:)).^2)/(4*pi)) < 1e-12*total, ...
            'Haar-averaged projection energy disagrees with ||a||_F^2/(4 pi).');
        power = total/D_proj;
    otherwise
        error('SNR_REFERENCE must be ''slice'' or ''haar''.');
end
end

function P = project_equator(profiles,model,L_max)
n = size(profiles,1);
P = zeros(2*L_max+1,n);
for m = -L_max:L_max
    v = zeros(n,1);
    for l = abs(m):L_max
        v = v+model.normalization(l+1,m+L_max+1)* ...
            model.legendre_zero(l+1,m+L_max+1)*profiles(:,gd_sh_index(l,m));
    end
    P(m+L_max+1,:) = v.';
end
end
