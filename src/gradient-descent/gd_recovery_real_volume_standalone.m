%% Projected THIRD-moment GD recovery on REAL volumes (uniform SO(3)).
%
% SELF-CONTAINED: single file, base MATLAB only (no toolboxes, no aspire).
% To run: set emd_path below to your volume file, then press Run.
% (expand_volume_to_bessel is inlined at the bottom as a local function.)
%
% Recovers spherical-harmonic x sub-Nyquist Bessel shell coefficients from
% the PROJECTED (tomographic) third moment, by gradient descent on the
% projected M_3 residual.  Bands l = 0, 1 are fixed from the volume's
% truth (the Lemma 3.1 / Kam assumption); GD recovers bands 2..L from a
% random Gaussian initialization in Bessel-coefficient space.
%
% FORWARD MODEL.  The volume is expressed as
%   x(rho, theta, phi) = sum_l sum_s sum_m a_{l,m}[s] phi_{l,s}(rho) Y_l^m
% with phi_{l,s}(rho) = nf * j_l(k_{l,s} rho/rho_max).  At the projection
% equator (theta = pi/2),
%   c_m(rho_k) = sum_l N_{l,m} P_l^m(0) sum_s a_{l,m}[s] phi_{l,s}(rho_k)
% on a 2D-Fourier radial grid {rho_k}.  Define
%   u_l[k, m] = sum_s phi_{l,s}(rho_k) a_{l,m}[s]   (functional radial profile)
% so that c_m[k] = sum_l beta_{l,m} u_l[k, m].  The GD machinery operates on
% u (matching the synthetic script's structure); the unknowns are a_{l,m}[s]
% (per-band Bessel coefficients) and the forward pass applies Phi_l per band.
%
% This script is a real-volume analog of the synthetic GD demo (see
% gd_third_moment_recovery_synthetic.m / equivalent).  Same loss function,
% same trilinear-adjoint gradient, same L-BFGS optimizer, same parity-pruned
% term list.  Only difference: a_true comes from a real volume's Bessel
% expansion (not random Gaussian) and the forward pass evaluates phi_{l,s}
% on a 2D radial grid.
%
% USAGE.  Edit the PARAMETERS block, press Run.  For quick testing, flip
% `quick_test_mode = true` to use L_max = 5 with reduced grid (much faster).
%
% OUTPUTS.  Per-restart: loss, signal_err in functional u-space, signal_err
% in Bessel a-space, and the reflection-twin error.  Recovery is considered
% successful if min(err_u, err_u_refl) is below the expected noise/MC floor.
%
% ----------------------------------------------------------------------------
% FIXED VERSION (convention handling).  The previous version unconditionally
% applied a_S = i^l(-1)^m a, which only converts FRIEDEL -> STANDARD reality
% and corrupted the (already-standard) real-volume coefficients -> ~100%
% reality violation -> GD could not represent the target (identical loss across
% restarts).  enforce_standard_reality() now detects the native convention and
% converts only when needed.  Honest best-restart reporting + optional FD
% gradient check added.  See real_volume_fixes.m for the rationale.
% ----------------------------------------------------------------------------

clear; clc; close all;
fprintf('========================================================\n');
fprintf(' gd_recovery_real_volume.m   (GD third-moment recovery)\n');
fprintf('========================================================\n');

%% ===================== PARAMETERS  (edit these, then press Run) =====================
quick_test_mode  = true;   % true = L=5, K=18 (for development); false = paper2 production scale

if quick_test_mode
    L_max        = 5;
    L_downsample = 31;
    K_radial     = 18;
else
    L_max        = 7;
    L_downsample = 47;
    K_radial     = 30;
end

rho_max          = pi;
basis_mode       = 'kc';                     % 'kc' (sub-Nyquist) or 'manual'
N_vec_manual     = [8 8 8 8 8 8 8 8];        % only used if basis_mode = 'manual'

% Optimization basis: false -> recover Bessel coeffs a_l[s,m] (u = Phi*a);
%                     true  -> recover the radial profile u_l[k,m] DIRECTLY on
%                              the K-grid (the synthetic "radial shell" model),
%                              bypassing the Bessel Phi layer. Same P3 operator.
use_shell_basis  = false;

% Optimization
n_restarts       = 5;
maxiter          = 3000;
seed             = 42;

% P_3 source: analytical (exact) or sampled empirical (with noise)
use_empirical    = true;                    % false = exact analytical P_3 from u_true
n_samples        = 1000;
noise_sigma      = 0;                        % iid CN(0, sigma^2) on c_m[k]

% Volume  <-- EDIT THIS PATH. Point to your .mat (MATLAB var) or MRC/CCP4 map.
%            A 3D real-space cubic volume; .mat is loaded via load(), MRC/CCP4
%            via the inlined raw-header parser (load_volume_mrc).
emd_path         = '/Users/joshuakatz/Documents/MATLAB/TRPV1.mat';

% Threshold for "recovered global" tag (analytical runs only; empirical judged
% against the MC floor in the summary below).
success_thresh   = 1e-5;

% Optional: finite-difference check of the gradient through the Phi layer.
do_fd_grad_check = true;

% Sanity test: initialize AT/NEAR the true volume instead of random.
%   init_near_truth   = true  -> x0 = x_true .* (1 + init_perturbation*randn)
%   init_perturbation = 0     -> exact truth (loss should be ~0, err ~0)
%   multiplicative perturbation respects the huge dynamic range.
init_near_truth   = false;
init_perturbation = 0.0;

%% ====================================================================================
rng(seed);
shift = L_max;
nM    = 2*L_max + 1;

%% =========== PRECOMPUTE TABLES ===========
fprintf('\n=== Precomputing CG / Legendre / Y-norm tables (L_max = %d) ===\n', L_max);
cg       = precompute_cg_table(L_max);
leg0     = precompute_legendre0_table(L_max);
norm_tab = precompute_normalization(L_max);

%% =========== BUILD BESSEL BASIS (per-band) ===========
fprintf('\n=== Building Bessel basis (mode = %s) ===\n', basis_mode);
shells_per_band = cell(1, L_max+1);
switch lower(basis_mode)
    case 'kc'
        k_cap = pi * L_downsample / 2;
        for l = 0:L_max
            s_idx = 1;
            while sph_bessel_zero(l, s_idx) <= k_cap
                s_idx = s_idx + 1;
            end
            shells_per_band{l+1} = 1:(s_idx - 1);
        end
    case 'manual'
        assert(numel(N_vec_manual) >= L_max+1, ...
               'N_vec_manual must have length >= L_max+1');
        for l = 0:L_max
            shells_per_band{l+1} = 1:N_vec_manual(l+1);
        end
    otherwise
        error('unknown basis_mode: %s', basis_mode);
end

N_l = cellfun(@length, shells_per_band);
for l = 0:L_max
    fprintf('  l = %d:  N_l = %d\n', l, N_l(l+1));
end
N_max = max(N_l);
M_tot = sum(N_l);
fprintf('  total Bessel shells: M_tot = %d\n', M_tot);

% Bessel zeros table
bessel_zeros = zeros(L_max+1, max(cellfun(@max, shells_per_band)));
for l = 0:L_max
    for s = shells_per_band{l+1}
        bessel_zeros(l+1, s) = sph_bessel_zero(l, s);
    end
end

% Global (band, within) tables for expand_volume_to_bessel
shell_band   = zeros(M_tot, 1);
shell_within = zeros(M_tot, 1);
kk = 0;
for l = 0:L_max
    for s = shells_per_band{l+1}
        kk = kk + 1;
        shell_band(kk)   = l;
        shell_within(kk) = s;
    end
end

%% =========== RADIAL GRID AND Phi_l MATRICES ===========
rho_grid = linspace(1e-10, rho_max, K_radial).';
Phi_band = cell(1, L_max+1);
for l = 0:L_max
    ss = shells_per_band{l+1};
    Phi_band{l+1} = zeros(K_radial, length(ss));
    for j = 1:length(ss)
        s = ss(j);
        k_ls = bessel_zeros(l+1, s);
        nf   = sqrt(2/rho_max^3) / abs(sph_bessel_j(l+1, k_ls));
        Phi_band{l+1}(:, j) = nf * sph_bessel_j(l, k_ls * rho_grid / rho_max);
    end
end

%% =========== LOAD VOLUME, EXPAND IN BESSEL ===========
fprintf('\n=== Loading volume: %s ===\n', emd_path);
t = tic;
vol = load_volume(emd_path);
fprintf('  raw size: [%s],  load time: %.2fs\n', num2str(size(vol)), toc(t));

fprintf('  Downsampling to %d^3 ...\n', L_downsample);
t = tic;
vol_ds = fourier_downsample_3d(vol, L_downsample);
fprintf('  downsample time: %.2fs\n', toc(t));

fprintf('  Expanding into Bessel basis ...\n');
t = tic;
a_paper2 = expand_volume_to_bessel(vol_ds, shell_band, shell_within, ...
                                   bessel_zeros, L_max, rho_max);
fprintf('  expansion time: %.2fs\n', toc(t));

%% =========== CONVERT a_paper2 -> a_bessel_true (per-band GD format) ===========
% a_bessel_true is (N_max, (L+1)^2): a_bessel_true(s, sh_index(l, m)) is the
% Bessel coefficient at band l, within-band index s, channel m.  Entries
% with s > N_l(l+1) are zero by convention.
a_bessel_true = zeros(N_max, (L_max+1)^2);
for k = 1:M_tot
    l = shell_band(k);
    s = shell_within(k);
    % Within-band index (position in shells_per_band{l+1})
    s_in = find(shells_per_band{l+1} == s, 1);
    for m = -l:l
        a_bessel_true(s_in, sh_index(l, m)) = a_paper2(k, m + shift + 1);
    end
end

% --- CONVENTION: detect native reality, convert to STANDARD only if needed ---
% x_to_a_bessel generates ONLY standard-reality signals, so a_bessel_true must
% be standard reality (a_{l,-m} = (-1)^m conj(a_{l,m})) for GD to represent it.
% enforce_standard_reality auto-detects standard vs Friedel and converts iff
% needed; errors out (pointing at expand_volume_to_bessel) if neither holds.
a_bessel_true = enforce_standard_reality(a_bessel_true, L_max);

% Build u_true = Phi @ a_bessel_true per band (functional radial profile).
% u_true is (K_radial, (L+1)^2).
u_true = build_u_from_a(a_bessel_true, Phi_band, L_max, N_l);

% Per-band energy reporting
fprintf('\n  Per-band coefficient + functional energies:\n');
for l = 0:L_max
    idx_l = arrayfun(@(m) sh_index(l, m), -l:l);
    a_norm = norm(a_bessel_true(1:N_l(l+1), idx_l), 'fro');
    u_norm = norm(u_true(:, idx_l), 'fro');
    fprintf('    l=%d:  ||a_l||_F = %.4e,  ||u_l||_F = %.4e\n', l, a_norm, u_norm);
end

%% =========== BUILD GD MODEL (terms / blocks for the synthetic-script pipeline) ===========
fprintf('\n=== Building parity-pruned (l_1,l_2,l_3) term list ===\n');
[terms, blocks] = build_model(L_max, cg, leg0, norm_tab);
fprintf('  %d terms, %d (m_1, m_2) blocks\n', numel(terms), numel(blocks));

%% =========== COMPUTE P_3 ===========
if use_empirical
    fprintf('\n=== Simulating %d projection samples (sigma = %g) ===\n', n_samples, noise_sigma);
    t = tic;
    cs = simulate_samples_real_volume(a_bessel_true, L_max, N_l, ...
                                      Phi_band, K_radial, n_samples, noise_sigma, ...
                                      leg0, norm_tab);
    fprintf('  sampling time: %.2fs\n', toc(t));
    fprintf('  Estimating P_3 ...\n');
    t = tic;
    P3full = estimate_P3(cs, L_max);
    fprintf('  estimate time: %.2fs\n', toc(t));
else
    fprintf('\n=== Computing analytical P_3 from u_true ===\n');
    t = tic;
    P3full = compute_analytical_P3(u_true, L_max, cg, leg0, norm_tab);
    fprintf('  analytical P_3 time: %.2fs\n', toc(t));
end

% Cache obs into blocks
for b = 1:numel(blocks)
    m1 = blocks(b).m1; m2 = blocks(b).m2; m3 = -m1-m2;
    blocks(b).obs = P3full(:,:,:, m1+shift+1, m2+shift+1, m3+shift+1);
end

% Loss scale (so we report a unit-free number)
scale = norm(P3full(:))^2;
fprintf('  ||P3||^2 = %.4e\n', scale);

%% =========== PIN BANDS 0, 1 FROM TRUTH ===========
% base_Bessel holds bands 0, 1 (truth); bands 2..L start at zero and are
% added to via the optimization variables.
base_Bessel = zeros(size(a_bessel_true));
for l = 0:min(1, L_max)
    for m = -l:l
        base_Bessel(:, sh_index(l, m)) = a_bessel_true(:, sh_index(l, m));
    end
end

%% =========== PARAMETERIZE THE UNKNOWNS (bands 2..L) ===========
% Reflection (O(3)) twin and u-norm (common to both modes).
a_bessel_inv = a_bessel_true;
for l = 0:L_max
    idx_l = arrayfun(@(m) sh_index(l, m), -l:l);
    a_bessel_inv(:, idx_l) = (-1)^l * a_bessel_inv(:, idx_l);
end
u_inv = build_u_from_a(a_bessel_inv, Phi_band, L_max, N_l);
tn_u  = norm(u_true(:));

if use_shell_basis
    % Optimize u_l[k,m] DIRECTLY on the K-grid (synthetic shell model).
    Nshell   = K_radial * ones(1, L_max+1);
    par      = make_param(L_max, Nshell);
    nd       = size(par, 1);
    base_opt = zeros(size(u_true));
    for l = 0:min(1, L_max)
        for m = -l:l, base_opt(:, sh_index(l, m)) = u_true(:, sh_index(l, m)); end
    end
    objfun    = @(x) obj_shell(x, par, base_opt, terms, blocks, scale, L_max);
    recover_u = @(x) x_to_a_bessel(x, par, base_opt, L_max);   % optimized array IS u
    truth_arr = u_true;
    fprintf('\n  SHELL BASIS: %d real DOF (u_l[k,m] direct, bands 2..%d, K=%d grid).\n', ...
            nd, L_max, K_radial);
else
    % Optimize Bessel coefficients a_l[s,m]; u = Phi*a.
    Nshell    = N_l;
    par       = make_param(L_max, N_l);
    nd        = size(par, 1);
    objfun    = @(x) obj_real_volume(x, par, base_Bessel, Phi_band, N_l, ...
                                     terms, blocks, scale, L_max);
    recover_u = @(x) build_u_from_a(x_to_a_bessel(x, par, base_Bessel, L_max), ...
                                    Phi_band, L_max, N_l);
    truth_arr = a_bessel_true;
    fprintf('\n  BESSEL BASIS: %d real DOF (a_l[s,m], bands 2..%d).\n', nd, L_max);
end

%% =========== OPTIMIZE WITH RANDOM RESTARTS ===========
% Per-band init scaling from the truth in the OPTIMIZED space.
band_scale = zeros(1, L_max+1);
for l = 0:L_max
    idx_l = sh_indices_for_band(l);
    if Nshell(l+1) > 0
        col = truth_arr(1:Nshell(l+1), idx_l);
        band_scale(l+1) = norm(col, 'fro') / sqrt(Nshell(l+1) * (2*l+1));
    end
end
x_scale = zeros(nd, 1);
for j = 1:nd, x_scale(j) = band_scale(par(j,2)+1); end
fprintf('  per-band init scales: [%s]\n', num2str(band_scale, '%.2e '));

% Optional finite-difference gradient check.
if do_fd_grad_check
    x0 = randn(nd, 1) .* x_scale;  [~, g0] = objfun(x0);
    nfd = min(12, nd);  gfd = zeros(nfd, 1);  ee = 1e-6;
    for k = 1:nfd
        d = zeros(nd, 1); d(k) = ee;
        gfd(k) = (objfun(x0+d) - objfun(x0-d)) / (2*ee);
    end
    fprintf('  FD grad check (first %d DOF): rel err = %.2e\n', nfd, ...
            norm(gfd - g0(1:nfd)) / max(norm(g0(1:nfd)), eps));
end

% Extract the TRUE optimization vector x_true (so x_to_a_bessel(x_true)=truth)
% for the near-truth sanity test.
x_true = zeros(nd, 1);
for j = 1:nd
    r = par(j,1); l = par(j,2); m = par(j,3); code = par(j,4);
    if code == 0,     x_true(j) = real(truth_arr(r, sh_index(l, 0)));
    elseif code == 1, x_true(j) = real(truth_arr(r, sh_index(l, m)));
    else,             x_true(j) = imag(truth_arr(r, sh_index(l, m))); end
end
if init_near_truth
    fprintf('  INIT NEAR TRUTH: x0 = x_true .* (1 + %.3g*randn);  loss@truth = %.2e\n', ...
            init_perturbation, objfun(x_true));
end

fprintf('\n=== GD with %d restarts (maxiter = %d) ===\n', n_restarts, maxiter);
results = struct('loss', {}, 'err_u', {}, 'err_u_refl', {});
best_err = inf;
for t_restart = 1:n_restarts
    if init_near_truth
        x0 = x_true .* (1 + init_perturbation * randn(nd, 1));
    else
        x0 = randn(nd, 1) .* x_scale;
    end
    [x_opt, fval] = minimize(objfun, x0, maxiter);
    u_est      = recover_u(x_opt);
    err_u      = norm(u_est(:) - u_true(:)) / tn_u;
    err_u_refl = norm(u_est(:) - u_inv(:))  / tn_u;
    results(t_restart) = struct('loss', fval, 'err_u', err_u, ...
                                'err_u_refl', err_u_refl); %#ok<AGROW>
    best_err = min(best_err, min(err_u, err_u_refl));
    if err_u < success_thresh,           tag = '-> recovered TRUTH';
    elseif err_u_refl < success_thresh,  tag = '-> recovered reflected truth';
    else,                                tag = '-> local min / noise floor';
    end
    fprintf('  restart %d: loss=%.3e  err_u=%.3e  (vs refl=%.3e)  %s\n', ...
            t_restart, fval, err_u, err_u_refl, tag);
end

%% =========== POST-PROCESS: SUMMARIZE BEST RESTART (honest reporting) ===========
[~, ibest] = min(arrayfun(@(r) min(r.err_u, r.err_u_refl), results));
bestE = min(results(ibest).err_u, results(ibest).err_u_refl);
fprintf('\n=== best restart #%d:  loss=%.3e  err_u=%.3e  (refl %.3e) ===\n', ...
        ibest, results(ibest).loss, results(ibest).err_u, results(ibest).err_u_refl);
if use_empirical
    n_rec = sum(arrayfun(@(r) min(r.err_u, r.err_u_refl) < success_thresh, results));
    fprintf('  empirical run: judge best err (%.3e) against the P3 MC floor, NOT 1e-5.\n', bestE);
    fprintf('  (%d/%d restarts under the 1e-5 tag; expect 0 unless n_samples is huge.)\n', ...
            n_rec, n_restarts);
else
    if bestE < success_thresh
        fprintf('  analytical run: RECOVERED to %.3e (up to reflection). Pipeline consistent.\n', bestE);
    else
        fprintf('  analytical run STALLED at err %.3e with exact P3.\n', bestE);
        fprintf('  => not a sampling issue. Check convention / expand_volume_to_bessel.\n');
    end
end

fprintf('\nDone.\n');


%% ====================================================================================
%% ============================ HELPER FUNCTIONS ======================================
%% ====================================================================================

function idx = sh_index(l, m)
    idx = l^2 + m + l + 1;
end

function idx = sh_indices_for_band(l)
    idx = arrayfun(@(m) sh_index(l, m), -l:l);
end

% ===================================== convention detection / enforcement
function a_std = enforce_standard_reality(a_bessel, L_max)
% Return coefficients in the STANDARD convention that x_to_a_bessel assumes:
%        a_{l,-m} = (-1)^m conj(a_{l,m}).
% A real volume MUST satisfy one of the reality pairings exactly. The TRPV1
% expansion is closest to FRIEDEL (a_{l,-m}=(-1)^{l+m}conj(a_{l,m})) but with a
% few-percent residual from angular-sampling inaccuracy in expand_volume_to_
% bessel. That residual is physically impossible (no real density can produce
% it), so we ORTHOGONALLY PROJECT onto the closer convention's symmetric
% subspace (averaging each +m with its conjugate partner -- this is the least-
% squares-optimal valid signal and actually reduces error vs. truth), then, if
% the native convention is Friedel, convert Friedel -> standard via i^l(-1)^m.
%
% PARAMETER: project_tol = max fraction of energy we will discard as nonphysical
% residual. If both conventions are violated by more than this, we error (that
% would indicate a genuine sign/indexing bug, not sampling noise).
    project_tol = 0.25;

    % --- relative (energy) violation of each convention over m>0 pairs ---
    [std_v, sig] = conv_defect(a_bessel, L_max, +1);   % +1 -> standard sign
    [fri_v, ~  ] = conv_defect(a_bessel, L_max, -1);   % -1 -> friedel sign (extra (-1)^l)
    if sig == 0, a_std = a_bessel; return; end
    std_r = sqrt(std_v/sig); fri_r = sqrt(fri_v/sig);
    fprintf('  raw expansion reality:  standard=%.1f%%   friedel=%.1f%%\n', ...
            100*std_r, 100*fri_r);

    use_friedel = fri_r < std_r;
    native_r    = min(std_r, fri_r);
    if native_r > project_tol
        error(['Bessel coeffs are %.1f%% from the nearest reality pairing -- too ', ...
               'large for sampling noise. Likely a sign/indexing bug in ', ...
               'expand_volume_to_bessel.'], 100*native_r);
    end
    if use_friedel
        fprintf('  -> native = FRIEDEL; projecting onto it (discarding %.1f%% nonphysical), then converting to standard.\n', 100*fri_r);
    else
        fprintf('  -> native = STANDARD; projecting onto it (discarding %.1f%% nonphysical).\n', 100*std_r);
    end

    % --- orthogonal projection onto the chosen symmetric subspace ---
    a_std = a_bessel;
    for l = 0:L_max
        sgn = 1; if use_friedel, sgn = (-1)^l; end   % friedel = standard sign * (-1)^l
        for m = 1:l
            s  = sgn * (-1)^m;                        % a_{l,-m} = s * conj(a_{l,m})
            ap = a_bessel(:, sh_index(l, m));
            am = a_bessel(:, sh_index(l,-m));
            apr = (ap + s*conj(am)) / 2;              % least-squares projection
            a_std(:, sh_index(l, m)) =  apr;
            a_std(:, sh_index(l,-m)) =  s*conj(apr);
        end
        % m=0 channel: a_{l,0} = s0 conj(a_{l,0}), s0 = sgn (standard: +1)
        a0 = a_bessel(:, sh_index(l, 0));
        if (sgn > 0), a_std(:, sh_index(l,0)) = real(a0);
        else,         a_std(:, sh_index(l,0)) = 1i*imag(a0); end
    end

    % --- if native was Friedel, convert to STANDARD ---
    if use_friedel
        for l = 0:L_max
            for m = -l:l
                a_std(:, sh_index(l,m)) = (1i)^l * (-1)^m * a_std(:, sh_index(l,m));
            end
        end
    end

    % hard post-check: standard reality must now hold to machine precision
    v = 0; scale = max(abs(a_std(:)));
    for l = 0:L_max
        for m = 1:l
            v = max(v, norm(a_std(:, sh_index(l,-m)) - (-1)^m * conj(a_std(:, sh_index(l,m)))));
        end
    end
    fprintf('  standard-reality after projection+convert: %.2e  (must be ~0)\n', v);
    assert(v <= 1e-9*max(scale,eps), 'standard reality not achieved');
end

function [defect, sig] = conv_defect(a_bessel, L_max, base_sign)
% Squared energy of the reality-pairing residual over m>0 (and m=0), plus the
% signal energy. base_sign=+1 -> standard pairing; the friedel pairing differs
% by an extra (-1)^l, passed as base_sign=-1 and applied per band below.
    defect = 0; sig = 0;
    for l = 0:L_max
        sgn = 1; if base_sign < 0, sgn = (-1)^l; end
        for m = 1:l
            s  = sgn * (-1)^m;
            ap = a_bessel(:, sh_index(l, m));
            am = a_bessel(:, sh_index(l,-m));
            defect = defect + norm(am - s*conj(ap))^2;
            sig    = sig + norm(ap)^2 + norm(am)^2;
        end
        a0 = a_bessel(:, sh_index(l, 0));
        if (sgn > 0), d0 = imag(a0); else, d0 = real(a0); end
        defect = defect + norm(d0)^2;
        sig    = sig + norm(a0)^2;
    end
end

% ============================================================ Bessel -> functional
function u = build_u_from_a(a_bessel, Phi_band, L_max, N_l)
% Forward map: a_l[s, m]  ->  u_l[k, m] = sum_s Phi_l[k, s] a_l[s, m].
% a_bessel is (N_max, (L+1)^2); u is (K_radial, (L+1)^2).
    K_radial = size(Phi_band{1}, 1);
    u = zeros(K_radial, (L_max+1)^2);
    for l = 0:L_max
        idx_l = sh_indices_for_band(l);
        a_l = a_bessel(1:N_l(l+1), idx_l);
        u(:, idx_l) = Phi_band{l+1} * a_l;
    end
end

function ga = grad_u_to_a(gu, Phi_band, L_max, N_l, N_max)
% Adjoint of build_u_from_a: dL/da_l[s, m] = sum_k Phi_l[k, s] dL/du_l[k, m].
    ga = zeros(N_max, (L_max+1)^2);
    for l = 0:L_max
        idx_l = sh_indices_for_band(l);
        ga(1:N_l(l+1), idx_l) = Phi_band{l+1}.' * gu(:, idx_l);
    end
end

% =========================================== synthetic-style P_3 model & gradient
function [terms, blocks] = build_model(L_max, cg, leg0, norm_tab)
    shift = L_max;
    terms = struct('l1', {}, 'l2', {}, 'l3', {}, 'i1', {}, 'i2', {}, 'i3', {}, 'c', {});
    for l1 = 0:L_max
        for l2 = 0:L_max
            for l3 = abs(l1-l2):min(l1+l2, L_max)
                if mod(l1+l2+l3, 2), continue; end   % parity prune
                i1=[]; i2=[]; i3=[]; cc=[];
                for k1 = -l1:l1
                    for k2 = -l2:l2
                        k3 = -k1 - k2;
                        if abs(k3) > l3, continue; end
                        c = cg(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
                        if abs(c) < 1e-14, continue; end
                        i1(end+1) = sh_index(l1, k1); %#ok<AGROW>
                        i2(end+1) = sh_index(l2, k2); %#ok<AGROW>
                        i3(end+1) = sh_index(l3, k3); %#ok<AGROW>
                        cc(end+1) = ((-1)^k1) * c / (2*l1+1); %#ok<AGROW>
                    end
                end
                if ~isempty(cc)
                    terms(end+1) = struct('l1', l1, 'l2', l2, 'l3', l3, ...
                                          'i1', i1, 'i2', i2, 'i3', i3, ...
                                          'c', cc); %#ok<AGROW>
                end
            end
        end
    end
    blocks = struct('m1', {}, 'm2', {}, 'ti', {}, 'w', {}, 'obs', {});
    for m1 = -L_max:L_max
        for m2 = -L_max:L_max
            m3 = -m1 - m2;
            if abs(m3) > L_max, continue; end
            ti = []; wv = [];
            for t = 1:numel(terms)
                l1 = terms(t).l1; l2 = terms(t).l2; l3 = terms(t).l3;
                if l1 < abs(m1) || l2 < abs(m2) || l3 < abs(m3), continue; end
                c = cg(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1);
                if abs(c) < 1e-14, continue; end
                w = ((-1)^m1) * ...
                    norm_tab(l1+1, m1+shift+1) * norm_tab(l2+1, m2+shift+1) * ...
                    norm_tab(l3+1, m3+shift+1) * leg0(l1+1, m1+shift+1) * ...
                    leg0(l2+1, m2+shift+1) * leg0(l3+1, m3+shift+1) * c;
                if abs(w) > 1e-16
                    ti(end+1) = t; %#ok<AGROW>
                    wv(end+1) = w; %#ok<AGROW>
                end
            end
            if ~isempty(ti)
                blocks(end+1) = struct('m1', m1, 'm2', m2, ...
                                       'ti', ti, 'w', wv, 'obs', []); %#ok<AGROW>
            end
        end
    end
end

function I3 = forward(u, terms)
% Compute I3_t[k1, k2, k3] for each term t, using the trilinear sum over
% inner (k1, k2, k3) channels.
    R = size(u, 1);
    I3 = cell(numel(terms), 1);
    for t = 1:numel(terms)
        T = terms(t);
        acc = zeros(R, R, R);
        for j = 1:numel(T.c)
            v1 = u(:, T.i1(j));
            v2 = u(:, T.i2(j));
            v3 = u(:, T.i3(j));
            acc = acc + T.c(j) * (reshape(v1, R, 1, 1) .* ...
                                  reshape(v2, 1, R, 1) .* ...
                                  reshape(v3, 1, 1, R));
        end
        I3{t} = acc;
    end
end

function [loss, gu] = loss_grad(u, terms, blocks)
    R = size(u, 1);
    I3 = forward(u, terms);
    Rt = cell(numel(terms), 1);
    for t = 1:numel(terms), Rt{t} = zeros(R, R, R); end
    loss = 0;
    for b = 1:numel(blocks)
        B = blocks(b);
        model = zeros(R, R, R);
        for q = 1:numel(B.ti), model = model + B.w(q) * I3{B.ti(q)}; end
        resid = model - B.obs;
        loss = loss + sum(abs(resid(:)).^2);
        cr = conj(resid);
        for q = 1:numel(B.ti), Rt{B.ti(q)} = Rt{B.ti(q)} + B.w(q) * cr; end
    end
    gu = zeros(size(u));
    for t = 1:numel(terms)
        if ~any(Rt{t}(:)), continue; end
        T = terms(t); Rtt = Rt{t};
        for j = 1:numel(T.c)
            v1 = u(:, T.i1(j)); v2 = u(:, T.i2(j)); v3 = u(:, T.i3(j));
            cj = T.c(j);
            gu(:, T.i1(j)) = gu(:, T.i1(j)) + cj * contract23(Rtt, v2, v3);
            gu(:, T.i2(j)) = gu(:, T.i2(j)) + cj * contract13(Rtt, v1, v3);
            gu(:, T.i3(j)) = gu(:, T.i3(j)) + cj * contract12(Rtt, v1, v2);
        end
    end
end

function r = contract23(T, v2, v3)
    R = size(T, 1);
    A = reshape(reshape(T, R*R, R) * v3, R, R);
    r = A * v2;
end

function s = contract13(T, v1, v3)
    R = size(T, 1);
    A = reshape(reshape(T, R*R, R) * v3, R, R);
    s = A.' * v1;
end

function t = contract12(T, v1, v2)
    R = size(T, 1);
    C = squeeze(sum(T .* reshape(v1, R, 1, 1), 1));
    t = C.' * v2;
end

% ====================================== parameterization (Bessel coeff DOF)
function par = make_param(L_max, N_l)
% par(:, [r, l, m, code]).  code: 0 = m=0 real, 1 = re(m>0), 2 = im(m>0).
    par = zeros(0, 4);
    for l = 2:L_max
        for r = 1:N_l(l+1)
            par(end+1, :) = [r, l, 0, 0]; %#ok<AGROW>
            for m = 1:l
                par(end+1, :) = [r, l, m, 1]; %#ok<AGROW>
                par(end+1, :) = [r, l, m, 2]; %#ok<AGROW>
            end
        end
    end
end

function a_bessel = x_to_a_bessel(x, par, base_Bessel, L_max) %#ok<INUSD>
    a_bessel = base_Bessel;
    for j = 1:size(par, 1)
        r = par(j, 1); l = par(j, 2); m = par(j, 3); code = par(j, 4); v = x(j);
        if code == 0
            a_bessel(r, sh_index(l, 0)) = a_bessel(r, sh_index(l, 0)) + v;
        elseif code == 1
            a_bessel(r, sh_index(l,  m)) = a_bessel(r, sh_index(l,  m)) + v;
            a_bessel(r, sh_index(l, -m)) = a_bessel(r, sh_index(l, -m)) + (-1)^m * v;
        else
            a_bessel(r, sh_index(l,  m)) = a_bessel(r, sh_index(l,  m)) + 1i * v;
            a_bessel(r, sh_index(l, -m)) = a_bessel(r, sh_index(l, -m)) - (-1)^m * 1i * v;
        end
    end
end

function g = grad_a_to_x(ga, par)
% Adjoint of x_to_a_bessel (real-x parameterization).
    g = zeros(size(par, 1), 1);
    for j = 1:size(par, 1)
        r = par(j, 1); l = par(j, 2); m = par(j, 3); code = par(j, 4);
        if code == 0
            g(j) = 2 * real(ga(r, sh_index(l, 0)));
        elseif code == 1
            g(j) = 2 * real(ga(r, sh_index(l,  m)) + (-1)^m * ga(r, sh_index(l, -m)));
        else
            g(j) = 2 * real(1i * ga(r, sh_index(l,  m)) - (-1)^m * 1i * ga(r, sh_index(l, -m)));
        end
    end
end

function [f, gx] = obj_real_volume(x, par, base_Bessel, Phi_band, N_l, ...
                                   terms, blocks, scale, L_max)
% Loss = || P3_model(u(a(x))) - P3_obs ||^2 / scale.
% Gradient via chain rule:  dL/dx  =  dL/du  *  Phi  *  da/dx.
    a_bessel = x_to_a_bessel(x, par, base_Bessel, L_max);
    u        = build_u_from_a(a_bessel, Phi_band, L_max, N_l);
    [loss, gu] = loss_grad(u, terms, blocks);
    ga = grad_u_to_a(gu, Phi_band, L_max, N_l, size(a_bessel, 1));
    gx = grad_a_to_x(ga, par);
    f  = loss / scale;
    gx = gx / scale;
end

function [f, gx] = obj_shell(x, par, base_u, terms, blocks, scale, L_max)
% SHELL-BASIS objective: optimize u_l[k,m] directly (no Phi layer).
% x_to_a_bessel/grad_a_to_x just (de)serialize a standard-reality array, so
% they apply unchanged with u playing the role of the coefficient array.
    u = x_to_a_bessel(x, par, base_u, L_max);    % u_l[k,m] on the K-grid
    [loss, gu] = loss_grad(u, terms, blocks);
    gx = grad_a_to_x(gu, par);
    f  = loss / scale;
    gx = gx / scale;
end

% =================================== analytical P_3 (synthetic-script style)
function P3 = compute_analytical_P3(u, L_max, cg, leg0, norm_tab)
% Closed-form P_3 from u (functional radial profile).  u is (R, (L+1)^2).
    R = size(u, 1); nM = 2*L_max + 1; shift = L_max;
    P3 = zeros(R, R, R, nM, nM, nM);
    I3 = zeros(R, R, R, L_max+1, L_max+1, L_max+1);
    for l1 = 0:L_max
        for l2 = 0:L_max
            for l3 = abs(l1-l2):min(l1+l2, L_max)
                if mod(l1+l2+l3, 2), continue; end
                for k1 = -l1:l1
                    for k2 = -l2:l2
                        k3 = -k1 - k2;
                        if abs(k3) > l3, continue; end
                        c = cg(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
                        if abs(c) < 1e-14, continue; end
                        w = ((-1)^k1) * c / (2*l1+1);
                        v1 = u(:, sh_index(l1, k1));
                        v2 = u(:, sh_index(l2, k2));
                        v3 = u(:, sh_index(l3, k3));
                        outer = reshape(v1, R, 1, 1) .* ...
                                reshape(v2, 1, R, 1) .* ...
                                reshape(v3, 1, 1, R);
                        I3(:, :, :, l1+1, l2+1, l3+1) = ...
                            I3(:, :, :, l1+1, l2+1, l3+1) + w * outer;
                    end
                end
            end
        end
    end
    for m1 = -L_max:L_max
        for m2 = -L_max:L_max
            m3 = -m1 - m2;
            if abs(m3) > L_max, continue; end
            acc = zeros(R, R, R);
            for l1 = abs(m1):L_max
                for l2 = abs(m2):L_max
                    for l3 = max([abs(m3), abs(l1-l2)]):min(l1+l2, L_max)
                        if mod(l1+l2+l3, 2), continue; end
                        c = cg(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1);
                        if abs(c) < 1e-14, continue; end
                        w = ((-1)^m1) * ...
                            norm_tab(l1+1, m1+shift+1) * norm_tab(l2+1, m2+shift+1) * ...
                            norm_tab(l3+1, m3+shift+1) * leg0(l1+1, m1+shift+1) * ...
                            leg0(l2+1, m2+shift+1) * leg0(l3+1, m3+shift+1) * c;
                        if abs(w) < 1e-16, continue; end
                        acc = acc + w * I3(:, :, :, l1+1, l2+1, l3+1);
                    end
                end
            end
            P3(:, :, :, m1+shift+1, m2+shift+1, m3+shift+1) = acc;
        end
    end
end

% =========================================== sampling / empirical P_3
function cs = simulate_samples_real_volume(a_bessel_true, L_max, N_l, ...
                                           Phi_band, K_radial, N, noise_sigma, ...
                                           leg0, norm_tab)
% For each Haar-uniform rotation, rotate the Bessel coefficients in
% coefficient space (via Wigner D per band), evaluate the radial profile
% at the K-grid via Phi_l, and project to 2D Fourier via beta_{l, m}.
    cs = zeros(2*L_max+1, K_radial, N);
    for i = 1:N
        alpha = 2*pi*rand(); gamma = 2*pi*rand(); beta = acos(2*rand() - 1);
        a_rot = apply_wigner_rotation_bessel(a_bessel_true, alpha, beta, gamma, L_max, N_l);
        u_rot = build_u_from_a(a_rot, Phi_band, L_max, N_l);
        c = project_to_2d_fourier(u_rot, L_max, leg0, norm_tab);
        if noise_sigma > 0
            c = c + (noise_sigma/sqrt(2)) * (randn(size(c)) + 1i*randn(size(c)));
        end
        cs(:, :, i) = c;
    end
end

function a_rot = apply_wigner_rotation_bessel(a_bessel, alpha, beta, gamma, L_max, N_l)
% Rotate the Bessel coefficients band-by-band:  a_l <- D^l(R) a_l.
    a_rot = a_bessel;
    for l = 0:L_max
        if N_l(l+1) == 0, continue; end
        idx_l = sh_indices_for_band(l);
        D = wigner_d_matrix(l, alpha, beta, gamma);
        A_l = a_bessel(1:N_l(l+1), idx_l);   % N_l x (2l+1)
        a_rot(1:N_l(l+1), idx_l) = (D * A_l.').';
    end
end

function c = project_to_2d_fourier(u, L_max, leg0, norm_tab)
% c_m[k] = sum_{l>=|m|} beta_{l,m} u_l[k, m]
    K = size(u, 1); shift = L_max;
    c = zeros(2*L_max+1, K);
    for m = -L_max:L_max
        s = zeros(K, 1);
        for l = abs(m):L_max
            w = norm_tab(l+1, m+shift+1) * leg0(l+1, m+shift+1);
            if w == 0, continue; end
            s = s + w * u(:, sh_index(l, m));
        end
        c(m+L_max+1, :) = s.';
    end
end

function P3 = estimate_P3(c_samples, L_max)
    [nM, K, N] = size(c_samples); %#ok<ASGLU>
    R = K;
    P3 = zeros(R, R, R, nM, nM, nM);
    for m1 = -L_max:L_max
        for m2 = -L_max:L_max
            m3 = -m1 - m2;
            if abs(m3) > L_max, continue; end
            x = reshape(c_samples(m1+L_max+1, :, :), R, N);
            y = reshape(c_samples(m2+L_max+1, :, :), R, N);
            z = reshape(c_samples(m3+L_max+1, :, :), R, N);
            T = zeros(R, R, R);
            for r3 = 1:R
                T(:, :, r3) = ((x .* z(r3, :)) * y.') / N;
            end
            P3(:, :, :, m1+L_max+1, m2+L_max+1, m3+L_max+1) = T;
        end
    end
end

% ====================================================== Wigner D / small d
function d = wigner_small_d(l, m, mp, beta)
    if abs(m) > l || abs(mp) > l, d = 0; return; end
    k_min = max([0, m - mp]); k_max = min([l + m, l - mp]);
    s = 0;
    for k = k_min:k_max
        s = s + (-1)^(k+mp-m) / ( factorial(l+m-k)*factorial(l-mp-k)* ...
            factorial(k+mp-m)*factorial(k) ) * ...
            (cos(beta/2))^(2*l+m-mp-2*k) * (sin(beta/2))^(2*k+mp-m);
    end
    d = sqrt(factorial(l+m)*factorial(l-m)*factorial(l+mp)*factorial(l-mp)) * s;
end

function D = wigner_d_matrix(l, alpha, beta, gamma)
    mv = -l:l; n = 2*l+1; D = zeros(n, n);
    for i = 1:n
        for j = 1:n
            D(i, j) = exp(-1i*alpha*mv(i)) * wigner_small_d(l, mv(i), mv(j), beta) * ...
                      exp(-1i*gamma*mv(j));
        end
    end
end

% ====================================================== tables (CG, P_l^m(0), N_lm)
function cg = clebsch_gordan(j1, m1, j2, m2, j3, m3)
    if m1+m2 ~= m3 || j3 < abs(j1-j2) || j3 > j1+j2, cg = 0; return; end
    tmin = max([0, j2-m1-j3, j1+m2-j3]);
    tmax = min([j1+j2-j3, j1-m1, j2+m2]);
    s = 0;
    for t = tmin:tmax
        s = s + ((-1)^t) / ( factorial(t)*factorial(j1+j2-j3-t)* ...
            factorial(j1-m1-t)*factorial(j2+m2-t)* ...
            factorial(j3-j2+m1+t)*factorial(j3-j1-m2+t) );
    end
    pref = sqrt( (2*j3+1)*factorial(j1+j2-j3)*factorial(j1-j2+j3)* ...
        factorial(-j1+j2+j3)/factorial(j1+j2+j3+1)* ...
        factorial(j3+m3)*factorial(j3-m3)* ...
        factorial(j1+m1)*factorial(j1-m1)* ...
        factorial(j2+m2)*factorial(j2-m2) );
    cg = pref * s;
end

function cg_tab = precompute_cg_table(d)
    shift = d;
    cg_tab = zeros(d+1, 2*shift+1, d+1, 2*shift+1, d+1);
    for l1 = 0:d
        for m1 = -l1:l1
            for l2 = 0:d
                for m2 = -l2:l2
                    for l3 = abs(l1-l2):min(l1+l2, d)
                        m3 = -m1 - m2;
                        if abs(m3) > l3, continue; end
                        cg_tab(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1) = ...
                            clebsch_gordan(l2, m2, l3, m3, l1, -m1);
                    end
                end
            end
        end
    end
end

function leg0_tab = precompute_legendre0_table(L_max)
    shift = L_max;
    leg0_tab = zeros(L_max+1, 2*shift+1);
    for l = 0:L_max
        for m = 0:l
            if mod(l-m, 2) ~= 0
                v = 0;
            else
                v = (-1)^((l-m)/2) * factorial(l+m) / ...
                    ( 2^l * factorial((l-m)/2) * factorial((l+m)/2) );
            end
            leg0_tab(l+1, m+shift+1) = v;
            if m > 0
                leg0_tab(l+1, -m+shift+1) = (-1)^m * ...
                    (factorial(l-m)/factorial(l+m)) * v;
            end
        end
    end
end

function norm_tab = precompute_normalization(L_max)
    shift = L_max;
    norm_tab = zeros(L_max+1, 2*shift+1);
    for l = 0:L_max
        for m = -l:l
            norm_tab(l+1, m+shift+1) = sqrt( (2*l+1)*factorial(l-m) / ...
                                             (4*pi*factorial(l+m)) );
        end
    end
end

% ========================================================= optimizer wrapper
function [x, f] = minimize(fun, x0, maxiter)
    if exist('fminunc', 'file') == 2
        opts = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
            'SpecifyObjectiveGradient', true, 'MaxIterations', maxiter, ...
            'MaxFunctionEvaluations', 20*maxiter, ...
            'OptimalityTolerance', 1e-12, 'StepTolerance', 1e-16, ...
            'FunctionTolerance', 1e-18, 'Display', 'off');
        [x, f] = fminunc(fun, x0, opts);
    else
        [x, f] = lbfgs(fun, x0, maxiter);
    end
end

function [x, f] = lbfgs(fun, x0, maxiter)
    m = 20; x = x0; n = numel(x);
    [f, g] = fun(x);
    S = zeros(n, m); Y = zeros(n, m); rho = zeros(1, m);
    nstore = 0; head = 0;
    for it = 1:maxiter
        if norm(g, inf) < 1e-12, break; end
        q = g; alpha = zeros(1, m);
        for i = 1:nstore
            idx = mod(head - i, m) + 1;
            alpha(idx) = rho(idx) * (S(:, idx).' * q);
            q = q - alpha(idx) * Y(:, idx);
        end
        if nstore > 0
            idx = mod(head - 1, m) + 1;
            gamma = (S(:, idx).' * Y(:, idx)) / (Y(:, idx).' * Y(:, idx));
        else
            gamma = 1;
        end
        z = gamma * q;
        for i = nstore:-1:1
            idx = mod(head - i, m) + 1;
            beta = rho(idx) * (Y(:, idx).' * z);
            z = z + (alpha(idx) - beta) * S(:, idx);
        end
        d = -z; gtd = g.' * d;
        if gtd >= 0, d = -g; gtd = g.' * d; end
        step = 1; c1 = 1e-4; ok = false;
        for ls = 1:40
            xn = x + step * d; [fnew, gnew] = fun(xn);
            if fnew <= f + c1 * step * gtd, ok = true; break; end
            step = step * 0.5;
        end
        if ~ok, break; end
        s = step * d; y = gnew - g; sy = s.' * y;
        if sy > 1e-12
            head = mod(head, m) + 1;
            S(:, head) = s; Y(:, head) = y; rho(head) = 1 / sy;
            nstore = min(nstore + 1, m);
        end
        x = xn; f = fnew; g = gnew;
        if abs(f) < 1e-20, break; end
    end
end

% ====================================================================================
% ============ Helpers from paper2_functional_proj.m (volume I/O + Bessel) ===========
% These are local functions in paper2_functional_proj.m; not on the MATLAB path.

function v = sph_bessel_j(l, x)
    v = zeros(size(x));
    nz = abs(x) > 1e-12;
    if l == 0
        v(nz)  = sin(x(nz)) ./ x(nz);
        v(~nz) = 1;
    else
        v(nz)  = sqrt(pi ./ (2*x(nz))) .* besselj(l + 0.5, x(nz));
    end
end

function root = sph_bessel_zero(l, s)
    if l == 0
        root = s * pi;
        return;
    end
    % McMahon asymptotic estimate, then bracket and use fzero.
    a   = pi * (s + l/2);
    mu  = 4 * l * (l + 1);
    est = max(a - mu / (8 * a), 0.5);
    half = pi / 2;
    a_lo = max(est - half, 0.05);
    a_hi = est + half;
    f_lo = sph_bessel_j(l, a_lo);
    f_hi = sph_bessel_j(l, a_hi);
    expand = 1;
    while f_lo * f_hi > 0 && expand < 6
        expand = expand + 1;
        a_lo = max(est - half * expand, 0.05);
        a_hi = est + half * expand;
        f_lo = sph_bessel_j(l, a_lo);
        f_hi = sph_bessel_j(l, a_hi);
    end
    if f_lo * f_hi > 0
        % Fallback: dense linspace + sign change search.
        x_max  = (s + l + 5) * pi;
        x_test = linspace(0.05, x_max, max(4000, 400 * s));
        f_test = sph_bessel_j(l, x_test);
        sc = find(diff(sign(f_test)) ~= 0);
        if length(sc) < s, error('sph_bessel_zero failed (l=%d, s=%d)', l, s); end
        a_lo = x_test(sc(s));
        a_hi = x_test(sc(s)+1);
    end
    root = fzero(@(x) sph_bessel_j(l, x), [a_lo, a_hi]);
end

function vol = load_volume(p)
    fid = fopen(p, 'r');
    if fid < 0, error('cannot open %s', p); end
    head = fread(fid, 6, 'uint8=>uint8');  fclose(fid);
    if numel(head) >= 6 && all(char(head(1:6).') == 'MATLAB')
        S = load(p);  fns = fieldnames(S);  vol = S.(fns{1});  return;
    end
    vol = load_volume_mrc(p);
end

function vol = load_volume_mrc(p)
    fid = fopen(p, 'r', 'l');
    if fid < 0, error('cannot open %s', p); end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    NX = fread(fid, 1, 'int32');  NY = fread(fid, 1, 'int32');
    NZ = fread(fid, 1, 'int32');  MODE = fread(fid, 1, 'int32');
    if NX <= 0 || NY <= 0 || NZ <= 0 || NX > 65536
        fclose(fid);  fid = fopen(p, 'r', 'b');
        cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
        NX = fread(fid, 1, 'int32');  NY = fread(fid, 1, 'int32');
        NZ = fread(fid, 1, 'int32');  MODE = fread(fid, 1, 'int32');
    end
    fseek(fid, 64, 'bof');
    MAPC = fread(fid, 1, 'int32');  MAPR = fread(fid, 1, 'int32');
    MAPS = fread(fid, 1, 'int32');
    fseek(fid, 208, 'bof');
    magic = char(fread(fid, 4, 'uint8=>uint8').');
    if ~strcmp(magic, 'MAP ')
        error('%s is not MAT or MRC', p);
    end
    fseek(fid, 92, 'bof');
    NSYMBT = fread(fid, 1, 'int32');
    fseek(fid, 1024 + NSYMBT, 'bof');
    switch MODE
        case 0, tp = 'int8';
        case 1, tp = 'int16';
        case 2, tp = 'float32';
        case 6, tp = 'uint16';
        otherwise, error('unsupported MODE');
    end
    raw = fread(fid, NX*NY*NZ, ['*' tp]);
    vol = double(reshape(raw, [NX, NY, NZ]));
    if ~(MAPC == 1 && MAPR == 2 && MAPS == 3)
        [~, inv_perm] = sort([MAPC, MAPR, MAPS]);
        vol = permute(vol, inv_perm);
    end
end

function vol_ds = fourier_downsample_3d(vol, szout)
    szin = size(vol);
    if numel(szout) < 3, szout = szout(1) * [1 1 1]; end
    x = fftshift(fftn(vol));
    fx = fft_crop_3d(x, szout);
    vol_ds = ifftn(ifftshift(fx)) * (prod(szout) / prod(szin));
    if isreal(vol), vol_ds = real(vol_ds); end
end

function mc = fft_crop_3d(m, n)
    sz = size(m);
    if numel(n) < 3, n = n(1)*[1 1 1]; end
    ns = floor(sz/2) - floor(n/2);
    if all(ns >= 0)
        mc = m(ns(1)+1:ns(1)+n(1), ns(2)+1:ns(2)+n(2), ns(3)+1:ns(3)+n(3));
    else
        mc = zeros(n);
        mc(1-ns(1):sz(1)-ns(1), 1-ns(2):sz(2)-ns(2), 1-ns(3):sz(3)-ns(3)) = m;
    end
end


% ============================================================================
% INLINED DEPENDENCY (was: aspire/expand_volume_to_bessel.m)
% Folded in so this script is a single self-contained file. Uses only base
% MATLAB (fftn, legendre, besselj, eig). No toolboxes required.
% ============================================================================
function a_true = expand_volume_to_bessel(vol, shell_band, shell_within, ...
    bessel_zeros, L_max, rho_max)
%EXPAND_VOLUME_TO_BESSEL  Project a real-space cubic volume onto the
% spherical-Bessel x Y_l^m basis used by paper-2-style recovery.
%
% Uses CHUNKED LS to avoid materializing the full B matrix.  Process
% in-sphere voxels in batches; accumulate B'*B (small) and B'*y (small)
% incrementally.  Peak memory ~ chunk_size x n_cols x 16 bytes regardless
% of L_downsample.  Works at L=255+ where the full B would be 30+ GB.

    L     = size(vol, 1);
    M_tot = length(shell_band);
    nM    = 2*L_max + 1;
    shift = L_max;

    %% Fourier-side volume (zero-frequency at the center).
    vol_hat = fftshift(fftn(ifftshift(vol)));

    %% 3D Fourier-grid coordinates.
    if mod(L, 2) == 0
        k_range = -L/2 : L/2-1;
    else
        k_range = -(L-1)/2 : (L-1)/2;
    end
    [kx, ky, kz] = ndgrid(k_range, k_range, k_range);
    k_nyq = max(abs(k_range));
    rho_grid = sqrt(kx.^2 + ky.^2 + kz.^2) * (rho_max / k_nyq);

    rho_safe = max(rho_grid, 1e-10);
    cos_theta_grid = max(min(kz ./ rho_safe, 1 - 1e-12), -1 + 1e-12);
    phi_ang = atan2(ky, kx);

    in_sphere = rho_grid <= rho_max & rho_grid > 1e-10;
    in_sphere_idx = find(in_sphere);
    n_in = length(in_sphere_idx);

    %% Column index mapping for the basis (column j <-> (k, m) triple).
    n_cols = 0;
    for k = 1:M_tot
        n_cols = n_cols + (2*shell_band(k) + 1);
    end
    col_to_km = zeros(n_cols, 2);
    col       = 0;
    for k = 1:M_tot
        l_band = shell_band(k);
        for m = -l_band:l_band
            col = col + 1;
            col_to_km(col, :) = [k, m + shift + 1];
        end
    end

    target_all = vol_hat(in_sphere_idx);

    %% Chunked accumulation of B'*B and B'*y.
    % Memory: B_chunk is chunk_size x n_cols complex doubles = 16 bytes each.
    % chunk_size = 200k, n_cols ~ 300 -> ~1 GB peak.  Adjustable below.
    chunk_size = 200000;
    BtB = zeros(n_cols, n_cols);
    Bty = zeros(n_cols, 1);

    for chunk_start = 1:chunk_size:n_in
        chunk_end = min(chunk_start + chunk_size - 1, n_in);
        idx_chunk = in_sphere_idx(chunk_start:chunk_end);
        n_chunk = length(idx_chunk);

        rho_chunk     = rho_grid(idx_chunk);
        cos_chunk     = cos_theta_grid(idx_chunk);
        phi_chunk     = phi_ang(idx_chunk);
        target_chunk  = target_all(chunk_start:chunk_end);

        % Precompute Y_l^m on the chunk (m >= 0; use conj symmetry for m<0).
        Y_pos_chunk = cell(L_max+1, 1);
        for l_band = 0:L_max
            P_full = legendre(l_band, cos_chunk);
            Y_pos_chunk{l_band+1} = cell(l_band+1, 1);
            for m = 0:l_band
                P_lm = P_full(m+1, :).';
                N_lm = sqrt((2*l_band+1) / (4*pi) ...
                            * factorial(l_band-m) / factorial(l_band+m));
                Y_pos_chunk{l_band+1}{m+1} = (-1)^m * N_lm * P_lm .* exp(1i*m*phi_chunk);
            end
        end

        % Build B_chunk one column at a time.
        B_chunk = zeros(n_chunk, n_cols);
        col = 0;
        for k = 1:M_tot
            l_band = shell_band(k);
            s      = shell_within(k);
            k_ls   = bessel_zeros(l_band+1, s);

            x = k_ls * rho_chunk / rho_max;
            j_l = zeros(size(x));
            nz = rho_chunk > 1e-12;
            if l_band == 0
                j_l(nz)  = sin(x(nz)) ./ x(nz);
                j_l(~nz) = 1;
            else
                j_l(nz)  = sqrt(pi ./ (2 * x(nz))) .* besselj(l_band + 0.5, x(nz));
            end
            j_lp1_at_zero = sph_bessel_j_scalar(l_band+1, k_ls);
            norm_factor = sqrt(2/rho_max^3) / abs(j_lp1_at_zero);
            phi_rad_chunk = norm_factor * j_l;

            for m = -l_band:l_band
                col = col + 1;
                if m >= 0
                    Y_lm = Y_pos_chunk{l_band+1}{m+1};
                else
                    Y_lm = (-1)^abs(m) * conj(Y_pos_chunk{l_band+1}{abs(m)+1});
                end
                B_chunk(:, col) = phi_rad_chunk .* Y_lm;
            end
        end

        % Accumulate.
        BtB = BtB + B_chunk' * B_chunk;
        Bty = Bty + B_chunk' * target_chunk;
    end

    %% LS solve via truncated SVD on the small (n_cols x n_cols) BtB.
    [V, D] = eig((BtB + BtB')/2);
    ev     = real(diag(D));
    [ev, perm] = sort(ev, 'descend');
    V      = V(:, perm);
    tol    = max(ev) * 1e-8;
    rk     = sum(ev > tol);
    a_flat = V(:, 1:rk) * ((V(:, 1:rk)' * Bty) ./ ev(1:rk));

    %% Enforce exact Friedel reality (remove roundoff-amplified asymmetry).
    % The normal-equations solve (B'*B) squares the condition number; roundoff
    % in the ill-conditioned near-null-space is NOT inversion-symmetric and gets
    % amplified into a non-physical, non-Friedel component (worst in even-l).
    % A real volume MUST satisfy  a_{l,-m} = (-1)^(l+m) conj(a_{l,m}), so the
    % asymmetric part is provably nonphysical; project it out (orthogonal
    % projection = average each +m with its conjugate partner). This is a no-op
    % when a_flat is already symmetric, so it cannot corrupt a correct result.
    km2col = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for c = 1:n_cols
        km2col(sprintf('%d_%d', col_to_km(c,1), col_to_km(c,2))) = c;
    end
    for k = 1:M_tot
        l_band = shell_band(k);
        for m = 1:l_band
            cp = km2col(sprintf('%d_%d', k,  m + shift + 1));
            cm = km2col(sprintf('%d_%d', k, -m + shift + 1));
            s  = (-1)^(l_band + m);
            ap = (a_flat(cp) + s*conj(a_flat(cm))) / 2;
            a_flat(cp) = ap;
            a_flat(cm) = s*conj(ap);
        end
        c0 = km2col(sprintf('%d_%d', k, 0 + shift + 1));
        if mod(l_band, 2) == 0, a_flat(c0) = real(a_flat(c0));
        else,                   a_flat(c0) = 1i*imag(a_flat(c0)); end
    end

    %% Pack into (M_tot, 2*L_max+1).
    a_true = zeros(M_tot, nM);
    for c = 1:n_cols
        a_true(col_to_km(c, 1), col_to_km(c, 2)) = a_flat(c);
    end
end


function v = sph_bessel_j_scalar(l, x)
    if l == 0
        v = sin(x)/x;
    else
        v = sqrt(pi/(2*x)) * besselj(l + 0.5, x);
    end
end
