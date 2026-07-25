function result = frequency_marching_recovery(a_true, S_assumed, options)
%FREQUENCY_MARCHING_RECOVERY Recover Fourier--Bessel coefficients from P3.
%   RESULT = FREQUENCY_MARCHING_RECOVERY(A_TRUE, S_ASSUMED, OPTIONS) runs
%   the parity-aware frequency march. A_TRUE is R-by-(L+1)^2 in the
%   standard complex spherical-harmonic convention
%       a_{l,-m}(r) = (-1)^m conj(a_{l,m}(r)).
%   S_ASSUMED(l+1) is the radial support used by the solver. Coefficients
%   outside this support may be nonzero; their energy and the resulting
%   model-misspecification error are reported in RESULT.
%
%   OPTIONS fields (all optional):
%     use_analytical_p3  true for the population moment (default true)
%     n_samples          observations for an empirical moment (default 1e3)
%     noise_sigma        circular complex noise standard deviation (default 0)
%     rng_seed           simulation seed (default 42)
%     enable_backfill    recover inner shells after the march (default true)
%     verify_analytical  compare empirical and population P3 (default false)
%     verbose            print progress (default true)
%
%   Bands l=0,1 are supplied exactly, matching the numerical convention in
%   the accompanying theory and in the reference experiment repositories.
%
%  Numerical validation of Theorem 3.3 (Projected Moment Recovery with
%  Radial Decay). Uniform distribution over SO(3). Coefficient-space
%  simulation: for each sample we apply a Wigner-D rotation to the
%  spherical-harmonic coefficients, project to 2D angular-Fourier
%  coefficients c_m[r] = sum_l N_{l,m} P_l^m(0) b_{l,m}[r], add Gaussian
%  noise, estimate the projected third moment P_3, and run frequency
%  marching per Theorem 3.3.
%
%  Ground-truth signal is real (a_{l,-m}[r] = (-1)^m conj(a_{l,m}[r])).
%  S(l) is the decay profile ASSUMED by the algorithm; the true signal may
%  violate it (see GENERAL-INPUT note below).
%
%  Bands l = 0 and l = 1 are treated as known (as in linsolv_rotation.m
%  and as in Sec. 6.2 of the orbit-recovery paper). The march solves l = 2..L
%  by inverting the highest-weight projected P_3 equations.
%
%  PARITY-AWARE VARIANT of projected_moment_recovery.m. Only the marching
%  ranges differ: since P_l^m(0) = 0 when l - m is odd, bands of opposite
%  parity NEVER enter the projected coefficient c_m[r]. Hence the
%  highest-weight collapse (Lemma 3.1) needs only r_i > S(|m_i|+2), and the
%  gap condition relaxes to S(l) - S(l+2) >= 3: the number of shells needs
%  to drop by 3 only every TWO bands (Dan's parity remark).
%
%  The true coefficients and assumed solve profile are deliberately separate,
%  so this routine handles both exact decay and support misspecification.
%  Bands 0,1 are known at all shells. Reported errors separate the assumed
%  support from the unestimated coefficient tail.

if nargin < 3
    options = struct();
end

%% ========== OPTIONS AND INPUT VALIDATION ==========
L_max = length(S_assumed) - 1;
N_samples = get_option(options, 'n_samples', 1000);
noise_sigma = get_option(options, 'noise_sigma', 0);
rng_seed = get_option(options, 'rng_seed', 42);
verify_analytical = get_option(options, 'verify_analytical', false);
use_analytical_P3_for_solve = get_option(options, 'use_analytical_p3', true);
enable_backfill = get_option(options, 'enable_backfill', true);
verbose = get_option(options, 'verbose', true);
S_vec = S_assumed(:).';

% Sanity-check parameter consistency
assert(length(S_assumed) == L_max+1, 'S_assumed must have length L_max+1');
assert(all(diff(S_assumed) <= 0), 'S_assumed must be monotonically non-increasing');
assert(all(S_assumed >= 0 & S_assumed == floor(S_assumed)), ...
    'S_assumed must contain nonnegative integer shell counts');
assert(size(a_true, 2) == (L_max+1)^2, ...
    'A_TRUE must have (L_max+1)^2 columns');
assert(max(S_assumed) <= size(a_true, 1), ...
    'S_assumed cannot exceed the number of rows in A_TRUE');
assert(N_samples >= 1 && N_samples == floor(N_samples), ...
    'n_samples must be a positive integer');
assert(noise_sigma >= 0, 'noise_sigma must be nonnegative');
gaps2 = S_assumed(1:end-2) - S_assumed(3:end);  % S(l) - S(l+2) of the ASSUMED profile
if any(gaps2 < 3) || any(S_assumed(max(1,end-1):end) < 3)
    warning(['Parity gap condition S_assumed(l) - S_assumed(l+2) >= 3 violated. ', ...
        'The parity-aware march may not apply; expect rank-deficient systems.']);
end

R_max = size(a_true, 1);
n_coeffs = (L_max+1)^2;                 % spherical-harmonic coefficients per shell

%% ========== PRECOMPUTE TABLES ==========
fprintf('=== PRECOMPUTING TABLES ===\n');
rng(rng_seed);
cg_tab   = precompute_cg_table(L_max);
leg0_tab = precompute_legendre0_table(L_max);   % leg0_tab(l+1, m+shift+1) = P_l^m(0)
norm_tab = precompute_normalization(L_max);     % norm_tab(l+1, m+shift+1) = N_{l,m}

%% ========== REPORT GROUND-TRUTH SIGNAL ==========
fprintf('\n=== GROUND TRUTH ===\n');
fprintf('  Signal: %d shells, max band %d, n_coeffs = %d\n', ...
        R_max, L_max, n_coeffs);
for l = 0:L_max
    idx = arrayfun(@(m) sh_index(l, m), -l:l);
    tail_n  = norm(a_true(S_assumed(l+1)+1:end, idx), 'fro');   % energy beyond assumed support
    total_n = max(norm(a_true(:, idx), 'fro'), 1e-12);
    fprintf('  band l=%d: assumed %d shells, tail/band = %.3e\n', ...
            l, S_assumed(l+1), tail_n / total_n);
end

%% ========== SIMULATE/ESTIMATE PROJECTED 3rd MOMENT ==========
P3_emp = [];
clean_observation_energy = 0;
noise_observation_energy = 0;
if ~use_analytical_P3_for_solve || verify_analytical
    fprintf('\n=== SIMULATING %d PROJECTION SAMPLES ===\n', N_samples);
    fprintf('  Noise std: %.4f\n', noise_sigma);
    % c_samples(m+L_max+1, r, i) = c_m^{(i)}[r]
    c_samples = zeros(2*L_max+1, R_max, N_samples);
    tic;
    for i = 1:N_samples
        % Haar SO(3): alpha,gamma uniform; cos(beta) uniform on [-1,1].
        alpha = 2*pi*rand();
        gamma = 2*pi*rand();
        beta  = acos(2*rand() - 1);
        b = apply_wigner_rotation(a_true, alpha, beta, gamma, L_max);
        c = project_to_2d_fourier(b, L_max, leg0_tab, norm_tab);
        noise = (noise_sigma/sqrt(2)) * ...
            (randn(size(c)) + 1i*randn(size(c)));
        clean_observation_energy = clean_observation_energy + norm(c,'fro')^2;
        noise_observation_energy = noise_observation_energy + norm(noise,'fro')^2;
        c_samples(:,:,i) = c + noise;
        if verbose && mod(i, max(1, floor(N_samples/10))) == 0
            fprintf('  %d / %d samples\n', i, N_samples);
        end
    end
    fprintf('  Sampling time: %.2f s\n', toc);

    fprintf('\n=== ESTIMATING P_3 ===\n');
    tic;
    P3_emp = estimate_P3(c_samples, L_max);
    fprintf('  Empirical P_3 estimation time: %.2f s\n', toc);
end

P3_ana = [];
if verify_analytical || use_analytical_P3_for_solve
    fprintf('\n=== COMPUTING ANALYTICAL P_3 ===\n');
    tic;
    P3_ana = compute_analytical_P3(a_true, L_max, S_vec, cg_tab, leg0_tab, norm_tab);
    fprintf('  Analytical P_3 time: %.2f s\n', toc);
    if verify_analytical
        rel = norm(P3_emp(:) - P3_ana(:)) / max(norm(P3_ana(:)), 1e-12);
        fprintf('  || P_3_empirical - P_3_analytical || / || P_3_analytical || = %.4e\n', rel);
    end
end

if use_analytical_P3_for_solve
    P3_for_solve = P3_ana;
    fprintf('  [using ANALYTICAL P_3 in the solver -- bypass Monte Carlo]\n');
else
    P3_for_solve = P3_emp;
end

%% ========== FREQUENCY-MARCHING SOLVER ==========
fprintf('\n=== FREQUENCY MARCHING (Theorem 3.3) ===\n');
a_est = zeros(size(a_true));
% Fix bands 0 and 1 as known
for l = 0:min(1, L_max)
    for m = -l:l
        idx = sh_index(l, m);
        a_est(1:R_max, idx) = a_true(1:R_max, idx);   % known at ALL shells
    end
end
fprintf('  Bands 0 and 1 fixed at all shells (assumed recovered from P_1/P_2).\n');

band_rel_err = nan(1, L_max+1);                % error over recovered shells only
band_rel_err_full = nan(1, L_max+1);           % error over ALL active shells (for reference)
condition_records = empty_condition_records();
for l = 0:min(1, L_max)
    band_rel_err(l+1)      = band_error_recovered(a_est, a_true, l, S_vec, L_max);
    band_rel_err_full(l+1) = band_error(a_est, a_true, l, S_vec);
end

for l = 2:L_max
    fprintf('  --- Band l = %d ---\n', l);
    % PARITY-AWARE RANGE: band l+1 has opposite parity and is silenced in
    % c_m by P_l^m(0) = 0, so band l is the highest CONTRIBUTING band at
    % shell r3 already for r3 > S(l+2):
    r3_hi = S_vec(l+1);
    r3_lo = get_S(l+2, S_vec, L_max) + 1;
    if r3_lo > r3_hi
        fprintf('    No new shells at band %d (S(l)=%d, S(l+2)=%d).\n', ...
                l, S_vec(l+1), get_S(l+2, S_vec, L_max));
        continue;
    end

    for r3 = r3_lo:r3_hi
        [A, b] = build_highest_weight_system( ...
            a_est, l, r3, L_max, S_vec, cg_tab, leg0_tab, norm_tab, P3_for_solve);

        if isempty(A)
            fprintf('    r3=%d: no usable equations.\n', r3);
            continue;
        end

        [x, system_condition, was_regularized] = ...
            solve_standard_reality_system(A,b,l);
        condition_records(end+1) = make_condition_record( ...
            'march',l,r3,A,system_condition,was_regularized); %#ok<AGROW>
        regtxt = '';
        if was_regularized, regtxt = ' (regularized)'; end

        % Place solution into a_est at band l, all m, shell r3
        m_vals = -l:l;
        for kk = 1:length(m_vals)
            a_est(r3, sh_index(l, m_vals(kk))) = x(kk);
        end
        fprintf('    r3=%d: %dx%d system, cond=%.2e%s\n', ...
                r3, size(A,1), size(A,2), system_condition, regtxt);
    end

    band_rel_err(l+1)      = band_error_recovered(a_est, a_true, l, S_vec, L_max);
    band_rel_err_full(l+1) = band_error(a_est, a_true, l, S_vec);
    fprintf('  Band %d recovered-shell rel. err.: %.4e  (full-band rel. err.: %.4e)\n', ...
            l, band_rel_err(l+1), band_rel_err_full(l+1));
end

%% ========== BACKFILL INNER SHELLS ==========
if enable_backfill
    fprintf('\n=== BACKFILL PASS (descending bands, inner shells) ===\n');
    [a_est, backfill_conditions] = backfill_inner_shells(a_est, L_max, ...
        S_vec, cg_tab, leg0_tab, norm_tab, P3_for_solve);
    condition_records = [condition_records,backfill_conditions];
    % Recompute band errors now that inner shells are filled
    for l = 0:L_max
        band_rel_err(l+1)      = band_error_recovered(a_est, a_true, l, S_vec, L_max);
        band_rel_err_full(l+1) = band_error(a_est, a_true, l, S_vec);
    end
end

%% ========== RESULTS ==========
fprintf('\n================= RESULTS =================\n');
if enable_backfill
    fprintf('Full recovery pass: highest-weight march + inner-shell backfill.\n\n');
else
    fprintf('Theorem 3.3 validation: recover A_l at "new" shells r in (S(l+2), S(l)].\n');
    fprintf('Shells OUTSIDE those ranges are not touched by this march and remain zero\n');
    fprintf('in a_est -- that is expected, not a bug.\n\n');
end
fprintf('  %-4s  %-22s  %-18s\n', 'band', 'recovered-shell err', 'full-band err');
for l = 0:L_max
    tag = ''; if l <= 1, tag = ' [FIXED]'; end
    fprintf('  l=%-2d  %-22.4e  %-18.4e%s\n', ...
            l, band_rel_err(l+1), band_rel_err_full(l+1), tag);
end
mask = false(size(a_true));
march_mask = false(size(a_true));
for l = 0:L_max
    idx = arrayfun(@(m) sh_index(l, m), -l:l);
    mask(1:S_vec(l+1), idx) = true;
    if l >= 2
        r_lo = get_S(l+2,S_vec,L_max)+1;
        r_hi = get_S(l,S_vec,L_max);
        if r_lo <= r_hi
            march_mask(r_lo:r_hi,idx) = true;
        end
    end
end
err_assumed = norm(a_est(mask) - a_true(mask)) / max(norm(a_true(mask)), 1e-12);
err_marched = norm(a_est(march_mask) - a_true(march_mask)) / ...
    max(norm(a_true(march_mask)),1e-12);
tail_ratio  = norm(a_true(~mask)) / max(norm(a_true(:)), 1e-12);
global_err  = norm(a_est(:) - a_true(:)) / max(norm(a_true(:)), 1e-12);
fprintf('Error over theorem-marched shells:                         %.4e\n',err_marched);
fprintf('\nError over ASSUMED support (what the algorithm estimates): %.4e\n', err_assumed);
fprintf('Signal tail outside assumed support (not estimated):        %.4e\n', tail_ratio);
fprintf('Global rel. err. over all coefficients:                     %.4e\n', global_err);

result = struct();
result.a_est = a_est;
result.a_true = a_true;
result.S_assumed = S_vec;
result.band_relative_error = band_rel_err;
result.band_full_relative_error = band_rel_err_full;
result.assumed_support_error = err_assumed;
result.recovered_region_error = err_marched;
result.tail_energy_ratio = tail_ratio;
result.global_relative_error = global_err;
result.options = options;
result.condition_numbers = struct2table(condition_records);
if noise_observation_energy > 0
    result.realized_observation_snr = ...
        clean_observation_energy/noise_observation_energy;
elseif clean_observation_energy > 0
    result.realized_observation_snr = Inf;
else
    result.realized_observation_snr = NaN;
end
if verify_analytical
    result.empirical_moment_relative_error = rel;
else
    result.empirical_moment_relative_error = NaN;
end
end

%% ========== HELPER FUNCTIONS ==========

function idx = sh_index(l, m)
    idx = l^2 + m + l + 1;
end

function norm_tab = precompute_normalization(L_max)
    % N_{l,m} = sqrt( (2l+1)(l-m)! / (4 pi (l+m)!) )  with m in -L_max..L_max
    shift = L_max;
    norm_tab = zeros(L_max+1, 2*shift+1);
    for l = 0:L_max
        for m = -l:l
            norm_tab(l+1, m+shift+1) = sqrt( (2*l+1) * factorial(l-m) / ...
                                             (4*pi * factorial(l+m)) );
        end
    end
end

function c = project_to_2d_fourier(b, L_max, leg0_tab, norm_tab)
    % b: (R x (L_max+1)^2) spherical-harmonic coefficients per shell
    % c(m+L_max+1, r) = sum_{l>=|m|} N_{l,m} P_l^m(0) b(r, sh_index(l,m))
    R = size(b, 1);
    nM = 2*L_max+1;
    c = zeros(nM, R);
    shift = L_max;
    for m = -L_max:L_max
        s = zeros(R, 1);
        for l = abs(m):L_max
            w = norm_tab(l+1, m+shift+1) * leg0_tab(l+1, m+shift+1);
            if w == 0, continue; end
            s = s + w * b(:, sh_index(l, m));
        end
        c(m+L_max+1, :) = s.';
    end
end

function rotated_signal = apply_wigner_rotation(signal, alpha, beta, gamma, d)
    % Taken from linsolv_rotation.m convention: signal is (num_shells x n_coeffs).
    [num_shells, n_coeffs] = size(signal);
    rotated_signal = zeros(size(signal));
    for l = 0:d
        m_values = -l:l;
        indices = arrayfun(@(m) sh_index(l,m), m_values);
        if max(indices) <= n_coeffs
            D_matrix = wigner_d_matrix(l, alpha, beta, gamma);
            for shell = 1:num_shells
                old_coeffs = signal(shell, indices);
                new_coeffs = D_matrix * old_coeffs.';
                rotated_signal(shell, indices) = new_coeffs.';
            end
        end
    end
end

function D = wigner_d_matrix(l, alpha, beta, gamma)
    m_values = -l:l;
    n = 2*l + 1;
    D = zeros(n, n);
    for i = 1:n
        for j = 1:n
            m  = m_values(i);
            mp = m_values(j);
            d_beta = wigner_small_d(l, m, mp, beta);
            D(i, j) = exp(-1i*alpha*m) * d_beta * exp(-1i*gamma*mp);
        end
    end
end

function d = wigner_small_d(l, m, mp, beta)
    if abs(m) > l || abs(mp) > l, d = 0; return; end
    k_min = max([0, m - mp]);
    k_max = min([l + m, l - mp]);
    sum_term = 0;
    for k = k_min:k_max
        if l+m-k >= 0 && l-mp-k >= 0 && k+mp-m >= 0 && k >= 0
            coeff = (-1)^(k+mp-m) / ( factorial(l+m-k) * factorial(l-mp-k) * ...
                                      factorial(k+mp-m) * factorial(k) );
            sum_term = sum_term + coeff * (cos(beta/2))^(2*l+m-mp-2*k) * ...
                                          (sin(beta/2))^(2*k+mp-m);
        end
    end
    prefactor = sqrt(factorial(l+m) * factorial(l-m) * factorial(l+mp) * factorial(l-mp));
    d = prefactor * sum_term;
end

function s = get_S(l, S_vec, L_max)
    % S(l) with boundary: 0 for l > L_max or l < 0.
    if l < 0 || l > L_max
        s = 0;
    else
        s = S_vec(l+1);
    end
end

function err = band_error(a_est, a_true, l, S_vec)
    % Relative Frobenius error over the NONZERO entries of a_true at band l
    % (shells 1..S(l), all m). Includes both march-recovered shells AND
    % inner shells that the march of Theorem 3.3 does not touch (so for
    % l < L_max this number will be dominated by the un-touched shells and
    % is NOT a validation of the theorem -- see band_error_recovered below).
    idx = arrayfun(@(m) sh_index(l, m), -l:l);
    R_active = S_vec(l+1);
    if R_active == 0
        err = 0; return;
    end
    diff = a_est(1:R_active, idx) - a_true(1:R_active, idx);
    denom = norm(a_true(1:R_active, idx), 'fro');
    err = norm(diff, 'fro') / max(denom, 1e-12);
end

function err = band_error_recovered(a_est, a_true, l, S_vec, L_max)
    % Relative Frobenius error at band l over ONLY the shells the march
    % actually solves for: r in (S(l+2), S(l)] for l >= 2, or all shells
    % 1..S(l) for l <= 1 (treated as fixed).
    idx = arrayfun(@(m) sh_index(l, m), -l:l);
    if l <= 1
        r_lo = 1; r_hi = S_vec(l+1);
    else
        r_lo = get_S(l+2, S_vec, L_max) + 1;   % parity-aware march range
        r_hi = get_S(l, S_vec, L_max);
    end
    if r_hi < r_lo, err = 0; return; end
    diff = a_est(r_lo:r_hi, idx) - a_true(r_lo:r_hi, idx);
    denom = norm(a_true(r_lo:r_hi, idx), 'fro');
    err = norm(diff, 'fro') / max(denom, 1e-12);
end

function [A, b] = build_highest_weight_system(a_est, l_target, r3, ...
        L_max, S_vec, cg_tab, leg0_tab, norm_tab, P3_emp)
    % For the band l_target (= L+1 in the theorem), at a given shell r3 with
    % S(l_target+1) < r3 <= S(l_target), build the linear system whose
    % unknowns are {a_{l_target, k3}[r3] : k3 = -l_target .. l_target} (2 l_target + 1
    % complex unknowns) and whose right-hand sides are highest-weight projected
    % P_3 values.
    %
    % Equations: pick m3 = l_target (highest weight). For every (m1, m2) with
    %   * m1, m2 nonzero
    %   * m1 + m2 = -l_target
    %   * |m1|, |m2| <= l_target - 1  (so the involved A_k for k <= l_target-1 are
    %                                   already recovered)
    %   * triangle |l1 - l2| <= l_target <= l1 + l2 with l_i = |m_i|
    % and for every (r1, r2) with
    %   * S(|m1|+2) < r1 <= S(|m1|)   (parity-aware: band |m_i|+1 has the
    %   * S(|m2|+2) < r2 <= S(|m2|)    wrong parity and cannot contribute)
    % we get one complex equation linear in the unknowns.

    n_unknowns = 2*l_target + 1;
    shift = L_max;
    m3 = l_target;

    A = zeros(0, n_unknowns);
    b = zeros(0, 1);

    for m1 = -(l_target-1):-1
        m2 = -l_target - m1;
        if m2 >= 0 || m2 < -(l_target-1), continue; end
        l1 = abs(m1); l2 = abs(m2); l3 = l_target;
        if l3 < abs(l1-l2) || l3 > l1+l2, continue; end

        cg_outer = cg_tab(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1);
        if abs(cg_outer) < 1e-14, continue; end

        N1 = norm_tab(l1+1, m1+shift+1);
        N2 = norm_tab(l2+1, m2+shift+1);
        N3 = norm_tab(l3+1, m3+shift+1);
        P1 = leg0_tab(l1+1, m1+shift+1);
        P2 = leg0_tab(l2+1, m2+shift+1);
        P3n = leg0_tab(l3+1, m3+shift+1);

        prefactor = ((-1)^m1) * N1*N2*N3 * P1*P2*P3n * cg_outer;
        if abs(prefactor) < 1e-14, continue; end
        prefactor = prefactor / (2*l1 + 1);    % the 1/(2 l1 + 1) from I_3 definition

        r1_min = get_S(l1+2, S_vec, L_max) + 1;   % parity-aware
        r1_max = get_S(l1,   S_vec, L_max);
        r2_min = get_S(l2+2, S_vec, L_max) + 1;   % parity-aware
        r2_max = get_S(l2,   S_vec, L_max);
        if r1_min > r1_max || r2_min > r2_max, continue; end

        for r1 = r1_min:r1_max
            for r2 = r2_min:r2_max
                % Build row: coefficient of each unknown a_{l3, k3}[r3]
                row = zeros(1, n_unknowns);
                for k3 = -l3:l3
                    col = k3 + l3 + 1;
                    coef_k3 = 0;
                    for k1 = -l1:l1
                        k2 = -k1 - k3;
                        if abs(k2) > l2, continue; end
                        cg_in = cg_tab(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
                        if abs(cg_in) < 1e-14, continue; end
                        a1 = a_est(r1, sh_index(l1, k1));
                        a2 = a_est(r2, sh_index(l2, k2));
                        coef_k3 = coef_k3 + ((-1)^k1) * cg_in * a1 * a2;
                    end
                    row(col) = prefactor * coef_k3;
                end
                if norm(row) < 1e-14, continue; end
                rhs = P3_emp(r1, r2, r3, m1+shift+1, m2+shift+1, m3+shift+1);
                A(end+1, :) = row;  %#ok<AGROW>
                b(end+1, 1) = rhs;  %#ok<AGROW>
            end
        end
    end
end

function P3 = estimate_P3(c_samples, L_max)
    % P3(r1, r2, r3, m1_idx, m2_idx, m3_idx) =
    %   (1/N) sum_i c_samples(m1_idx, r1, i) * c_samples(m2_idx, r2, i) * c_samples(m3_idx, r3, i)
    % Only entries with m1 + m2 + m3 == 0 are computed; the rest remain 0.
    [nM, R, N] = size(c_samples);
    assert(nM == 2*L_max+1);
    P3 = zeros(R, R, R, nM, nM, nM);
    for m1 = -L_max:L_max
        for m2 = -L_max:L_max
            m3 = -m1 - m2;
            if abs(m3) > L_max, continue; end
            x = reshape(c_samples(m1+L_max+1, :, :), R, N);   % R x N
            y = reshape(c_samples(m2+L_max+1, :, :), R, N);
            z = reshape(c_samples(m3+L_max+1, :, :), R, N);
            T = zeros(R, R, R);
            for r3 = 1:R
                xz = x .* z(r3, :);                 % R x N
                T(:, :, r3) = (xz * y.') / N;       % R x R
            end
            P3(:, :, :, m1+L_max+1, m2+L_max+1, m3+L_max+1) = T;
        end
    end
end

function P3 = compute_analytical_P3(a, L_max, ~, cg_tab, leg0_tab, norm_tab)
    % Direct evaluation of the notes' equation (for P_3):
    %   P_3(r1,m1,r2,m2,r3,m3) = (-1)^{m1} * sum_{l1,l2,l3: triangle}
    %       N_{l1,m1} N_{l2,m2} N_{l3,m3} P_{l1}^{m1}(0) P_{l2}^{m2}(0) P_{l3}^{m3}(0)
    %       * <l2 m2 l3 m3 | l1 -m1> * I_3(r1,l1,r2,l2,r3,l3)
    % where I_3(r1,l1,r2,l2,r3,l3) =
    %       (1/(2 l1+1)) sum_{k1+k2+k3=0, |ki|<=li} (-1)^{k1} <l2 k2 l3 k3 | l1 -k1>
    %           * a_{l1,k1}[r1] a_{l2,k2}[r2] a_{l3,k3}[r3]
    R = size(a, 1);
    nM = 2*L_max+1;
    shift = L_max;
    P3 = zeros(R, R, R, nM, nM, nM);

    % Precompute all I_3(r1,l1,r2,l2,r3,l3) -- respects radial decay via a(r,:) = 0 for r > S(l).
    I3 = zeros(R, R, R, L_max+1, L_max+1, L_max+1);
    for l1 = 0:L_max
      for l2 = 0:L_max
        for l3 = abs(l1-l2):min(l1+l2, L_max)
            for k1 = -l1:l1
                for k2 = -l2:l2
                    k3 = -k1 - k2;
                    if abs(k3) > l3, continue; end
                    cg = cg_tab(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
                    if abs(cg) < 1e-14, continue; end
                    w = ((-1)^k1) * cg / (2*l1+1);
                    v1 = a(:, sh_index(l1, k1));    % Rx1
                    v2 = a(:, sh_index(l2, k2));
                    v3 = a(:, sh_index(l3, k3));
                    % Accumulate outer product
                    outer = reshape(v1, R,1,1) .* reshape(v2, 1,R,1) .* reshape(v3, 1,1,R);
                    I3(:,:,:,l1+1,l2+1,l3+1) = I3(:,:,:,l1+1,l2+1,l3+1) + w * outer;
                end
            end
        end
      end
    end

    % Now the projected P_3
    for m1 = -L_max:L_max
      for m2 = -L_max:L_max
        m3 = -m1 - m2;
        if abs(m3) > L_max, continue; end
        acc = zeros(R, R, R);
        for l1 = abs(m1):L_max
          for l2 = abs(m2):L_max
            for l3 = max([abs(m3), abs(l1-l2)]):min(l1+l2, L_max)
                cg = cg_tab(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1);
                if abs(cg) < 1e-14, continue; end
                N1 = norm_tab(l1+1, m1+shift+1);
                N2 = norm_tab(l2+1, m2+shift+1);
                N3 = norm_tab(l3+1, m3+shift+1);
                P1 = leg0_tab(l1+1, m1+shift+1);
                P2 = leg0_tab(l2+1, m2+shift+1);
                P3n = leg0_tab(l3+1, m3+shift+1);
                w = ((-1)^m1) * N1*N2*N3 * P1*P2*P3n * cg;
                if abs(w) < 1e-16, continue; end
                acc = acc + w * I3(:,:,:,l1+1,l2+1,l3+1);
            end
          end
        end
        P3(:,:,:, m1+L_max+1, m2+L_max+1, m3+L_max+1) = acc;
      end
    end
end

function leg0_tab = precompute_legendre0_table(L_max)
    % P_l^m(0) for integer l in 0..L_max, m in -l..l.
    % Uses closed form:
    %   P_l^m(0) = 0 if (l-m) is odd;
    %   P_l^m(0) = (-1)^{(l-m)/2} * (l+m)! / ( 2^l * ((l-m)/2)! * ((l+m)/2)! )
    %              for (l-m) even and m >= 0.
    %   P_l^{-m}(0) = (-1)^m * (l-m)!/(l+m)! * P_l^m(0)  (standard relation)
    shift = L_max;
    leg0_tab = zeros(L_max+1, 2*shift+1);
    for l = 0:L_max
        for m = 0:l
            if mod(l - m, 2) ~= 0
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

function cg_tab = precompute_cg_table(d)
    % Port of the CG table from linsolv_rotation.m. Indexed as
    %   cg_tab(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1) = <l2 m2 l3 m3 | l1 -m1>
    % where m3 = -m1 - m2 and shift = d.
    shift = d;
    cg_tab = zeros(d+1, 2*shift+1, d+1, 2*shift+1, d+1);
    for l1 = 0:d
        for m1 = -l1:l1
            for l2 = 0:d
                for m2 = -l2:l2
                    for l3 = abs(l1-l2):min(l1+l2, d)
                        m3 = -m1 - m2;
                        if abs(m3) > l3, continue; end
                        cg = clebsch_gordan(l2, m2, l3, m3, l1, -m1);
                        cg_tab(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1) = cg;
                    end
                end
            end
        end
    end
end

function [a_est, condition_records] = backfill_inner_shells(a_est, L_max, ...
        S_vec, cg_tab, leg0_tab, norm_tab, P3)
    % After the parity-aware highest-weight march recovers A_l at
    % r in (S(l+2), S(l)] for each band l >= 2, this backfill pass fills in
    % A_l at the inner shells r <= S(l+2)
    % for l = L_max-1 down to 2. (Band L_max is already fully recovered by the
    % main march since S(L_max+1) = 0; bands 0, 1 are fixed.)
    %
    % Key idea: at step l_target (descending), use P_3 equations with m_3 = l_target
    % at each unrecovered shell r_3. The outer l_3 sum then restricts l_3 >= l_target,
    % and since A_{l'} at r_3 for l' > l_target is already recovered (from the main
    % march or earlier backfill steps), those terms become known constants that move
    % to the RHS. The remaining unknowns are {a_{l_target, k3}[r_3] : k3 = -l..l}.
    %
    % r_1 and r_2 are drawn from the recovered outer annuli of the partner
    % bands |m_1| and |m_2|. This is weaker than requiring the whole radial
    % shell to be known: the support decay and the parity zero of P_l^m(0)
    % isolate the minimal contributing band at those partner shells.

    R_max = size(a_est, 1);
    condition_records = empty_condition_records();

    % recovered(l+1, r) = true iff A_l[r] is known
    recovered = false(L_max+1, R_max);
    for l = 0:min(1, L_max)
        recovered(l+1, 1:S_vec(l+1)) = true;
    end
    for l = 2:L_max
        r_lo = get_S(l+2, S_vec, L_max) + 1;   % parity-aware march range
        r_hi = get_S(l,   S_vec, L_max);
        if r_lo <= r_hi
            recovered(l+1, r_lo:r_hi) = true;
        end
    end

    for l_target = (L_max-1) : -1 : 2
        fprintf('  --- Backfill band l = %d ---\n', l_target);

        % Find unrecovered shells for this band
        shells_to_fill = [];
        for r = 1:S_vec(l_target+1)
            if ~recovered(l_target+1, r), shells_to_fill(end+1) = r; end %#ok<AGROW>
        end

        if isempty(shells_to_fill)
            fprintf('    No shells to backfill.\n'); continue;
        end

        for r3 = shells_to_fill
            [A, b] = build_backfill_system(a_est, l_target, r3, ...
                L_max, S_vec, cg_tab, leg0_tab, norm_tab, P3);
            if isempty(A)
                fprintf('    r3=%d: no equations.\n', r3); continue;
            end
            [x, system_condition, was_regularized] = ...
                solve_standard_reality_system(A,b,l_target);
            condition_records(end+1) = make_condition_record( ...
                'backfill',l_target,r3,A,system_condition,was_regularized); %#ok<AGROW>
            regtxt = '';
            if was_regularized, regtxt = ' (regularized)'; end
            for k3 = -l_target:l_target
                a_est(r3, sh_index(l_target, k3)) = x(k3 + l_target + 1);
            end
            recovered(l_target+1, r3) = true;
            fprintf('    r3=%d: %dx%d system, cond=%.2e%s\n', ...
                    r3, size(A,1), size(A,2), system_condition, regtxt);
        end
    end
end

function [A, b] = build_backfill_system(a_est, l_target, r3, ...
        L_max, S_vec, cg_tab, leg0_tab, norm_tab, P3)
    % Build the linear system for a_{l_target, k_3}[r_3] (k_3 = -l..l) using
    % P_3(r_1, m_1, r_2, m_2, r_3, m_3 = l_target) equations. For each partner
    % projected mode m_i, set l_i = |m_i| and draw r_i from the outer annulus
    % S(l_i+2) < r_i <= S(l_i). On these shells, higher same-parity bands are
    % outside support and wrong-parity bands vanish after projection, so the
    % partner contribution is exactly the already recovered band l_i.
    %
    % In the outer (l_1, l_2, l_3) sum of P_3, l_3 >= l_target (because |m_3| = l).
    % For l_3 = l_target, a_{l_3, k_3}[r_3] is UNKNOWN -> becomes a coefficient on
    % the LHS. For l_3 > l_target, a_{l_3, k_3}[r_3] is KNOWN (prior march / backfill
    % step), so it contributes to a scalar that is subtracted from the RHS.

    n_unknowns = 2*l_target + 1;
    shift = L_max;
    m3 = l_target;

    A = zeros(0, n_unknowns);
    b = zeros(0, 1);

    for m1 = -L_max:L_max
        m2 = -m3 - m1;
        if abs(m2) > L_max, continue; end

        l1 = abs(m1);
        l2 = abs(m2);
        r1_min = get_S(l1+2, S_vec, L_max) + 1;
        r1_max = get_S(l1,   S_vec, L_max);
        r2_min = get_S(l2+2, S_vec, L_max) + 1;
        r2_max = get_S(l2,   S_vec, L_max);
        if r1_min > r1_max || r2_min > r2_max, continue; end

        for r1 = r1_min:r1_max
            for r2 = r2_min:r2_max
                row = zeros(1, n_unknowns);
                rhs_known = 0;

                l3_min = max([abs(l1-l2), l_target]);
                l3_max = min(l1+l2, L_max);
                for l3 = l3_min:l3_max
                        cg_outer = cg_tab(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1);
                        if abs(cg_outer) < 1e-14, continue; end

                        N1 = norm_tab(l1+1, m1+shift+1);
                        N2 = norm_tab(l2+1, m2+shift+1);
                        N3 = norm_tab(l3+1, m3+shift+1);
                        P1 = leg0_tab(l1+1, m1+shift+1);
                        P2 = leg0_tab(l2+1, m2+shift+1);
                        P3n = leg0_tab(l3+1, m3+shift+1);

                        outer = ((-1)^m1) * N1*N2*N3 * P1*P2*P3n * cg_outer / (2*l1+1);
                        if abs(outer) < 1e-16, continue; end

                        if l3 == l_target
                            % Unknown contribution (linear in a_{l_target, k3}[r3])
                            for k3 = -l3:l3
                                col = k3 + l3 + 1;
                                coef_k3 = 0;
                                for k1 = -l1:l1
                                    k2 = -k1 - k3;
                                    if abs(k2) > l2, continue; end
                                    cg_in = cg_tab(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
                                    if abs(cg_in) < 1e-14, continue; end
                                    a1 = a_est(r1, sh_index(l1, k1));
                                    a2 = a_est(r2, sh_index(l2, k2));
                                    coef_k3 = coef_k3 + ((-1)^k1) * cg_in * a1 * a2;
                                end
                                row(col) = row(col) + outer * coef_k3;
                            end
                        else
                            % Known contribution (l3 > l_target, A_{l3}[r3] already recovered)
                            inner = 0;
                            for k1 = -l1:l1
                                for k2 = -l2:l2
                                    k3 = -k1 - k2;
                                    if abs(k3) > l3, continue; end
                                    cg_in = cg_tab(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
                                    if abs(cg_in) < 1e-14, continue; end
                                    a1 = a_est(r1, sh_index(l1, k1));
                                    a2 = a_est(r2, sh_index(l2, k2));
                                    a3 = a_est(r3, sh_index(l3, k3));
                                    inner = inner + ((-1)^k1) * cg_in * a1 * a2 * a3;
                                end
                            end
                            rhs_known = rhs_known + outer * inner;
                        end
                end

                if norm(row) < 1e-14, continue; end
                rhs = P3(r1, r2, r3, m1+shift+1, m2+shift+1, m3+shift+1) - rhs_known;
                A(end+1, :) = row;  %#ok<AGROW>
                b(end+1, 1) = rhs;  %#ok<AGROW>
            end
        end
    end
end

function cg = clebsch_gordan(j1, m1, j2, m2, j3, m3)
    % Port from linsolv_rotation.m
    if m1+m2 ~= m3 || j3 < abs(j1-j2) || j3 > j1+j2, cg = 0; return; end
    tmin = max([0, j2-m1-j3, j1+m2-j3]);
    tmax = min([j1+j2-j3, j1-m1, j2+m2]);
    sumTerm = 0;
    for t = tmin:tmax
        sumTerm = sumTerm + ((-1)^t) / ( factorial(t) * factorial(j1+j2-j3-t) * ...
            factorial(j1-m1-t) * factorial(j2+m2-t) * ...
            factorial(j3-j2+m1+t) * factorial(j3-j1-m2+t) );
    end
    pref = sqrt( (2*j3+1) * factorial(j1+j2-j3) * factorial(j1-j2+j3) * ...
                 factorial(-j1+j2+j3) / factorial(j1+j2+j3+1) * ...
                 factorial(j3+m3) * factorial(j3-m3) * ...
                 factorial(j1+m1) * factorial(j1-m1) * ...
                 factorial(j2+m2) * factorial(j2-m2) );
    cg = pref * sumTerm;
end

function [x, system_condition, was_regularized] = ...
        solve_standard_reality_system(A,b,l)
    % Parameterize one band by its 2l+1 real degrees of freedom:
    % x_0 real and x_{-m}=(-1)^m conj(x_m). Solving the stacked real and
    % imaginary equations avoids spurious complex null directions.
    n = 2*l+1;
    T = zeros(n,n);
    T(l+1,1) = 1;
    for m = 1:l
        real_col = 2*m;
        imag_col = 2*m+1;
        plus_idx = l+m+1;
        minus_idx = l-m+1;
        T(plus_idx,real_col) = 1;
        T(minus_idx,real_col) = (-1)^m;
        T(plus_idx,imag_col) = 1i;
        T(minus_idx,imag_col) = -1i*(-1)^m;
    end
    A_real = [real(A*T);imag(A*T)];
    b_real = [real(b);imag(b)];
    system_condition = cond(A_real);
    was_regularized = size(A_real,1) < size(A_real,2) || ...
        system_condition > 1e12;
    if was_regularized
        lambda = 1e-10*trace(A_real'*A_real)/max(size(A_real,2),1);
        coordinates = (A_real'*A_real+lambda*eye(n))\(A_real'*b_real);
    else
        coordinates = A_real\b_real;
    end
    x = T*coordinates;
end

function records = empty_condition_records()
records = struct('phase',{},'band',{},'shell',{},'num_equations',{}, ...
    'num_real_unknowns',{},'complex_condition',{}, ...
    'reality_constrained_condition',{},'regularized',{});
end

function record = make_condition_record(phase,band,shell,A, ...
        reality_condition,regularized)
record = struct();
record.phase = phase;
record.band = band;
record.shell = shell;
record.num_equations = size(A,1);
record.num_real_unknowns = size(A,2);
record.complex_condition = cond(A);
record.reality_constrained_condition = reality_condition;
record.regularized = regularized;
end

function value = get_option(options, name, default_value)
    if isfield(options, name)
        value = options.(name);
    else
        value = default_value;
    end
end
