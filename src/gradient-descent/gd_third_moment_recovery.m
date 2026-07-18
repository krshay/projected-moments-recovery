%% Projected THIRD-moment GD recovery (uniform SO(3)) -- SCRIPT
%  Edit the PARAMETERS block below and press Run (F5); no command line needed.
%
%  Recovers spherical-harmonic shell coefficients from the
%   PROJECTED (tomographic) THIRD moment under Haar-uniform SO(3) rotations, by
%   gradient descent from a RANDOM initialization. Self-contained: no toolboxes
%   strictly required (uses fminunc if present, else a built-in L-BFGS).
%
%   MODEL.  A real band-limited signal on R^3 is expanded in spherical harmonics
%   with radially-discretized coefficients a_{l,m}[r], l=0..L, m=-l..l, shells
%   r=1..S(l). Each observation is the signal rotated by a Haar-random g in SO(3)
%   and tomographically projected to a 2-D image; in 2-D polar Fourier coordinates
%   the per-shell angular coefficients are
%        c_m[r] = sum_{l>=|m|} N_{l,m} P_l^m(0) b_{l,m}[r],   b = D^l(g) a,
%   i.e. the equatorial slice. Note this SUMS over bands l: the observable carries
%   only an angular index m and a radial shell r -- the band structure is lost.
%   The recovery target is the projected third moment
%        P3(r1,m1,r2,m2,r3,m3) = E_g[ c_{m1}[r1] c_{m2}[r2] c_{m3}[r3] ].
%
%   METHOD.  Minimize  || P3_model(a) - P3_obs ||^2  over the real degrees of
%   freedom of bands l = 2..L (reality a_{l,-m}=(-1)^m conj(a_{l,m}) built in;
%   bands 0,1 fixed from the first/second moments as usual), starting from a
%   random Gaussian. The map a -> P3 is cubic; the gradient is exact (trilinear
%   adjoint). Multiple random restarts; report per-restart error vs the truth and
%   vs the global-reflection (O(3)) twin.
%
%   ARGUMENTS (all optional after the first two):
%     L_max         max band-limit.
%     S_vec         1 x (L_max+1) shell counts per band, S_vec(l+1)=S(l).
%                   No radial decay: S_vec = [R R ... R].
%     n_restarts    number of random restarts                 (default 5)
%     maxiter       optimizer iterations                      (default 3000)
%     use_empirical false -> exact (population) P3 (default);
%                   true  -> estimate P3 from n_samples sampled rotations + noise.
%     n_samples     samples for the empirical moment          (default 5000)
%     noise_sigma   per-coefficient complex-Gaussian noise std on c (default 0)
%     seed          RNG seed                                  (default 42)
%
%   EXAMPLES:
%     gd_third_moment_recovery(3, [6 6 6 6])            % L=3, 6 shells, no decay
%     gd_third_moment_recovery(5, [9 9 6 6 3 3], 6)     % decaying profile
%     gd_third_moment_recovery(3, [6 6 6 6], 4, 3000, true, 20000, 0.1) % noisy
%
%   Successful restarts drive signal_err to ~1e-8 with the exact moment; with a
%   sampled moment they floor at the Monte-Carlo / noise level. Random-init GD
%   does not always land in the global basin -- run several restarts and keep the
%   lowest-loss one. (Provided as research code for collaborators.)

%% ===================== PARAMETERS  (edit these, then press Run) =====================
clear; clc;
L_max         = 5;            % max band-limit
S_vec         = [6 6 6 6 6 6];    % shells per band (length L_max+1); no radial decay = all equal
n_restarts    = 5;            % number of random restarts
maxiter       = 3000;         % optimizer iterations per restart
use_empirical = true;        % false = exact (population) P3;  true = sampled / noisy P3
n_samples     = 1000000;         % rotations sampled if use_empirical
noise_sigma   = 3;          % per-coefficient complex-Gaussian noise std on c (if empirical)
seed          = 42;           % RNG seed
%% ====================================================================================
assert(numel(S_vec) == L_max+1, 'S_vec must have length L_max+1');

    rng(seed);
    cg       = precompute_cg_table(L_max);
    leg0     = precompute_legendre0_table(L_max);
    norm_tab = precompute_normalization(L_max);
    R = max(S_vec);
    a_true = generate_signal(L_max, S_vec, R);
    [terms, blocks] = build_model(L_max, cg, leg0, norm_tab);

    if use_empirical
        cs = simulate_samples(a_true, L_max, n_samples, noise_sigma, leg0, norm_tab);
        P3full = estimate_P3(cs, L_max);
        fprintf('Projected 3rd-moment GD recovery: L=%d, S=[%s], %d restarts, EMPIRICAL P3 (N=%d, sigma=%g)\n', ...
                L_max, num2str(S_vec), n_restarts, n_samples, noise_sigma);
    else
        P3full = compute_analytical_P3(a_true, L_max, S_vec, cg, leg0, norm_tab);
        fprintf('Projected 3rd-moment GD recovery: L=%d, S=[%s], %d restarts, exact P3\n', ...
                L_max, num2str(S_vec), n_restarts);
    end
    shift = L_max;
    for b = 1:numel(blocks)
        m1 = blocks(b).m1; m2 = blocks(b).m2; m3 = -m1-m2;
        blocks(b).obs = P3full(:,:,:, m1+shift+1, m2+shift+1, m3+shift+1);
    end

    base = zeros(size(a_true));            % bands 0,1 fixed from truth
    for l = 0:min(1, L_max)
        for m = -l:l, base(:, sh_index(l,m)) = a_true(:, sh_index(l,m)); end
    end
    par = make_param(L_max, S_vec);
    nd = size(par, 1);
    scale = norm(P3full(:))^2;

    a_inv = a_true;                        % global-reflection (O(3)) twin
    for l = 0:L_max
        for m = -l:l, a_inv(:, sh_index(l,m)) = (-1)^l * a_inv(:, sh_index(l,m)); end
    end
    tn = norm(a_true(:));
    objfun = @(x) obj(x, par, base, terms, blocks, scale);

    fprintf('  %d unknown real DOF (bands 2..%d); bands 0,1 fixed.\n', nd, L_max);
    results = struct('loss', {}, 'err', {}, 'err_inv', {});
    best = inf;
    for t = 1:n_restarts
        [x, fval] = minimize(objfun, randn(nd,1), maxiter);
        a_est = x_to_a(x, par, base);
        err  = norm(a_est(:) - a_true(:)) / tn;
        erri = norm(a_est(:) - a_inv(:))  / tn;
        results(t) = struct('loss', fval, 'err', err, 'err_inv', erri); %#ok<AGROW>
        best = min(best, min(err, erri));
        if err < 1e-5,        tag = '-> recovered TRUTH';
        elseif erri < 1e-5,   tag = '-> recovered reflected truth';
        else,                 tag = '-> local min / noise floor';
        end
        fprintf('  restart %d: loss=%.3e  signal_err=%.3e  (vs reflection=%.3e)  %s\n', ...
                t, fval, err, erri, tag);
    end
    nrec = sum(arrayfun(@(r) min(r.err, r.err_inv) < 1e-5, results));
    fprintf('  -> %d/%d restarts reached the global solution; best error = %.3e\n', ...
            nrec, n_restarts, best);

% ======================================================================= signal
function idx = sh_index(l, m)
    idx = l^2 + m + l + 1;
end

function a = generate_signal(L_max, S_vec, R_max)
    % Random real signal: a(r,sh_index(l,-m)) = (-1)^m conj(a(r,sh_index(l,m))),
    % nonzero on shells 1..S_vec(l+1) of band l (no decay => S_vec all equal).
    a = zeros(R_max, (L_max+1)^2);
    for l = 0:L_max
        for r = 1:S_vec(l+1)
            a(r, sh_index(l,0)) = randn();
            for m = 1:l
                z = (randn() + 1i*randn())/sqrt(2);
                a(r, sh_index(l, m)) = z;
                a(r, sh_index(l,-m)) = (-1)^m * conj(z);
            end
        end
    end
end

% ================================================================ lookup tables
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
    % cg_tab(l1+1,m1+d+1,l2+1,m2+d+1,l3+1) = <l2 m2 l3 m3 | l1 -m1>, m3=-m1-m2.
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
    % P_l^m(0) (no Condon-Shortley; it cancels in the moments).
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

% ============================================================== analytical P3
function P3 = compute_analytical_P3(a, L_max, S_vec, cg, leg0, norm_tab) %#ok<INUSD>
    R = size(a, 1); nM = 2*L_max+1; shift = L_max;
    P3 = zeros(R, R, R, nM, nM, nM);
    I3 = zeros(R, R, R, L_max+1, L_max+1, L_max+1);
    for l1 = 0:L_max
      for l2 = 0:L_max
        for l3 = abs(l1-l2):min(l1+l2, L_max)
          for k1 = -l1:l1
            for k2 = -l2:l2
              k3 = -k1 - k2;
              if abs(k3) > l3, continue; end
              c = cg(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
              if abs(c) < 1e-14, continue; end
              w = ((-1)^k1) * c / (2*l1+1);
              v1 = a(:, sh_index(l1,k1)); v2 = a(:, sh_index(l2,k2)); v3 = a(:, sh_index(l3,k3));
              outer = reshape(v1,R,1,1).*reshape(v2,1,R,1).*reshape(v3,1,1,R);
              I3(:,:,:,l1+1,l2+1,l3+1) = I3(:,:,:,l1+1,l2+1,l3+1) + w*outer;
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
              c = cg(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1);
              if abs(c) < 1e-14, continue; end
              w = ((-1)^m1) * norm_tab(l1+1,m1+shift+1)*norm_tab(l2+1,m2+shift+1)* ...
                  norm_tab(l3+1,m3+shift+1)*leg0(l1+1,m1+shift+1)* ...
                  leg0(l2+1,m2+shift+1)*leg0(l3+1,m3+shift+1)*c;
              if abs(w) < 1e-16, continue; end
              acc = acc + w * I3(:,:,:,l1+1,l2+1,l3+1);
            end
          end
        end
        P3(:,:,:, m1+shift+1, m2+shift+1, m3+shift+1) = acc;
      end
    end
end

% ================================================ rotation / projection / sampling
function d = wigner_small_d(l, m, mp, beta)
    if abs(m) > l || abs(mp) > l, d = 0; return; end
    k_min = max([0, m - mp]); k_max = min([l + m, l - mp]);
    s = 0;
    for k = k_min:k_max
        s = s + (-1)^(k+mp-m) / ( factorial(l+m-k)*factorial(l-mp-k)* ...
            factorial(k+mp-m)*factorial(k) ) * ...
            (cos(beta/2))^(2*l+m-mp-2*k) * (sin(beta/2))^(2*k+mp-m);
    end
    d = sqrt(factorial(l+m)*factorial(l-m)*factorial(l+mp)*factorial(l-mp))*s;
end

function D = wigner_d_matrix(l, alpha, beta, gamma)
    mv = -l:l; n = 2*l+1; D = zeros(n,n);
    for i = 1:n
        for j = 1:n
            D(i,j) = exp(-1i*alpha*mv(i)) * wigner_small_d(l, mv(i), mv(j), beta) ...
                     * exp(-1i*gamma*mv(j));
        end
    end
end

function rotated = apply_wigner_rotation(signal, alpha, beta, gamma, d)
    [num_shells, n_coeffs] = size(signal);
    rotated = zeros(size(signal));
    for l = 0:d
        idx = arrayfun(@(m) sh_index(l,m), -l:l);
        if max(idx) <= n_coeffs
            D = wigner_d_matrix(l, alpha, beta, gamma);
            for s = 1:num_shells
                rotated(s, idx) = (D * signal(s, idx).').';
            end
        end
    end
end

function c = project_to_2d_fourier(b, L_max, leg0, norm_tab)
    R = size(b,1); shift = L_max;
    c = zeros(2*L_max+1, R);
    for m = -L_max:L_max
        s = zeros(R,1);
        for l = abs(m):L_max
            w = norm_tab(l+1, m+shift+1)*leg0(l+1, m+shift+1);
            if w == 0, continue; end
            s = s + w * b(:, sh_index(l,m));
        end
        c(m+L_max+1, :) = s.';
    end
end

function c_samples = simulate_samples(a_true, L_max, N, noise_sigma, leg0, norm_tab)
    R = size(a_true,1);
    c_samples = zeros(2*L_max+1, R, N);
    for i = 1:N
        alpha = 2*pi*rand(); gamma = 2*pi*rand(); beta = acos(2*rand()-1);
        b = apply_wigner_rotation(a_true, alpha, beta, gamma, L_max);
        c = project_to_2d_fourier(b, L_max, leg0, norm_tab);
        if noise_sigma > 0
            c = c + (noise_sigma/sqrt(2))*(randn(size(c)) + 1i*randn(size(c)));
        end
        c_samples(:,:,i) = c;
    end
end

function P3 = estimate_P3(c_samples, L_max)
    [nM, R, N] = size(c_samples);
    P3 = zeros(R, R, R, nM, nM, nM);
    for m1 = -L_max:L_max
        for m2 = -L_max:L_max
            m3 = -m1 - m2;
            if abs(m3) > L_max, continue; end
            x = reshape(c_samples(m1+L_max+1,:,:), R, N);
            y = reshape(c_samples(m2+L_max+1,:,:), R, N);
            z = reshape(c_samples(m3+L_max+1,:,:), R, N);
            T = zeros(R,R,R);
            for r3 = 1:R
                T(:,:,r3) = ((x .* z(r3,:)) * y.') / N;
            end
            P3(:,:,:, m1+L_max+1, m2+L_max+1, m3+L_max+1) = T;
        end
    end
end

% ================================================= differentiable model + grad
function [terms, blocks] = build_model(L_max, cg, leg0, norm_tab)
    shift = L_max;
    terms = struct('l1', {}, 'l2', {}, 'l3', {}, 'i1', {}, 'i2', {}, 'i3', {}, 'c', {});
    for l1 = 0:L_max
      for l2 = 0:L_max
        for l3 = abs(l1-l2):min(l1+l2, L_max)
          if mod(l1+l2+l3, 2), continue; end          % parity: odd-sum vanishes
          i1=[]; i2=[]; i3=[]; cc=[];
          for k1 = -l1:l1
            for k2 = -l2:l2
              k3 = -k1 - k2;
              if abs(k3) > l3, continue; end
              c = cg(l1+1, k1+shift+1, l2+1, k2+shift+1, l3+1);
              if abs(c) < 1e-14, continue; end
              i1(end+1)=sh_index(l1,k1); i2(end+1)=sh_index(l2,k2); %#ok<AGROW>
              i3(end+1)=sh_index(l3,k3); cc(end+1)=((-1)^k1)*c/(2*l1+1); %#ok<AGROW>
            end
          end
          if ~isempty(cc)
            terms(end+1) = struct('l1',l1,'l2',l2,'l3',l3, ...
                'i1',i1,'i2',i2,'i3',i3,'c',cc); %#ok<AGROW>
          end
        end
      end
    end
    blocks = struct('m1', {}, 'm2', {}, 'ti', {}, 'w', {}, 'obs', {});
    for m1 = -L_max:L_max
      for m2 = -L_max:L_max
        m3 = -m1 - m2;
        if abs(m3) > L_max, continue; end
        ti=[]; wv=[];
        for t = 1:numel(terms)
          l1=terms(t).l1; l2=terms(t).l2; l3=terms(t).l3;
          if l1 < abs(m1) || l2 < abs(m2) || l3 < abs(m3), continue; end
          c = cg(l1+1, m1+shift+1, l2+1, m2+shift+1, l3+1);
          if abs(c) < 1e-14, continue; end
          w = ((-1)^m1)*norm_tab(l1+1,m1+shift+1)*norm_tab(l2+1,m2+shift+1)* ...
              norm_tab(l3+1,m3+shift+1)*leg0(l1+1,m1+shift+1)* ...
              leg0(l2+1,m2+shift+1)*leg0(l3+1,m3+shift+1)*c;
          if abs(w) > 1e-16, ti(end+1)=t; wv(end+1)=w; end %#ok<AGROW>
        end
        if ~isempty(ti)
          blocks(end+1) = struct('m1',m1,'m2',m2,'ti',ti,'w',wv,'obs',[]); %#ok<AGROW>
        end
      end
    end
end

function I3 = forward(a, terms)
    R = size(a,1); I3 = cell(numel(terms),1);
    for t = 1:numel(terms)
        T = terms(t); acc = zeros(R,R,R);
        for j = 1:numel(T.c)
            v1=a(:,T.i1(j)); v2=a(:,T.i2(j)); v3=a(:,T.i3(j));
            acc = acc + T.c(j)*(reshape(v1,R,1,1).*reshape(v2,1,R,1).*reshape(v3,1,1,R));
        end
        I3{t} = acc;
    end
end

function [loss, gc] = loss_grad(a, terms, blocks)
    R = size(a,1);
    I3 = forward(a, terms);
    Rt = cell(numel(terms),1); for t=1:numel(terms), Rt{t}=zeros(R,R,R); end
    loss = 0;
    for b = 1:numel(blocks)
        B = blocks(b); model = zeros(R,R,R);
        for q = 1:numel(B.ti), model = model + B.w(q)*I3{B.ti(q)}; end
        resid = model - B.obs;
        loss = loss + sum(abs(resid(:)).^2);
        cr = conj(resid);
        for q = 1:numel(B.ti), Rt{B.ti(q)} = Rt{B.ti(q)} + B.w(q)*cr; end
    end
    gc = zeros(size(a));
    for t = 1:numel(terms)
        if ~any(Rt{t}(:)), continue; end
        T = terms(t); Rtt = Rt{t};
        for j = 1:numel(T.c)
            v1=a(:,T.i1(j)); v2=a(:,T.i2(j)); v3=a(:,T.i3(j)); cj=T.c(j);
            gc(:,T.i1(j)) = gc(:,T.i1(j)) + cj*contract23(Rtt, v2, v3);
            gc(:,T.i2(j)) = gc(:,T.i2(j)) + cj*contract13(Rtt, v1, v3);
            gc(:,T.i3(j)) = gc(:,T.i3(j)) + cj*contract12(Rtt, v1, v2);
        end
    end
end

function r = contract23(T, v2, v3)   % sum_{s,t} T(r,s,t) v2(s) v3(t)
    R = size(T,1); A = reshape(reshape(T, R*R, R) * v3, R, R); r = A * v2;
end
function s = contract13(T, v1, v3)   % sum_{r,t} T(r,s,t) v1(r) v3(t)
    R = size(T,1); A = reshape(reshape(T, R*R, R) * v3, R, R); s = A.' * v1;
end
function t = contract12(T, v1, v2)   % sum_{r,s} T(r,s,t) v1(r) v2(s)
    R = size(T,1); C = squeeze(sum(T .* reshape(v1, R,1,1), 1)); t = C.' * v2;
end

% =============================================== real (reality-constrained) DOF
function par = make_param(L_max, S_vec)
    par = zeros(0, 4);    % [r, l, m, code]  code: 0=r0, 1=re, 2=im
    for l = 2:L_max
        for r = 1:S_vec(l+1)
            par(end+1,:) = [r, l, 0, 0]; %#ok<AGROW>
            for m = 1:l
                par(end+1,:) = [r, l, m, 1]; %#ok<AGROW>
                par(end+1,:) = [r, l, m, 2]; %#ok<AGROW>
            end
        end
    end
end

function a = x_to_a(x, par, base)
    a = base;
    for j = 1:size(par,1)
        r=par(j,1); l=par(j,2); m=par(j,3); code=par(j,4); v=x(j);
        if code == 0
            a(r, sh_index(l,0)) = a(r, sh_index(l,0)) + v;
        elseif code == 1
            a(r, sh_index(l, m)) = a(r, sh_index(l, m)) + v;
            a(r, sh_index(l,-m)) = a(r, sh_index(l,-m)) + (-1)^m * v;
        else
            a(r, sh_index(l, m)) = a(r, sh_index(l, m)) + 1i*v;
            a(r, sh_index(l,-m)) = a(r, sh_index(l,-m)) - (-1)^m * 1i*v;
        end
    end
end

function g = grad_to_x(gc, par)
    g = zeros(size(par,1), 1);
    for j = 1:size(par,1)
        r=par(j,1); l=par(j,2); m=par(j,3); code=par(j,4);
        if code == 0
            g(j) = 2*real(gc(r, sh_index(l,0)));
        elseif code == 1
            g(j) = 2*real(gc(r, sh_index(l,m)) + (-1)^m*gc(r, sh_index(l,-m)));
        else
            g(j) = 2*real(1i*gc(r, sh_index(l,m)) - (-1)^m*1i*gc(r, sh_index(l,-m)));
        end
    end
end

function [f, g] = obj(x, par, base, terms, blocks, scale)
    a = x_to_a(x, par, base);
    [loss, gc] = loss_grad(a, terms, blocks);
    f = loss / scale;
    g = grad_to_x(gc, par) / scale;
end

% ============================================================ optimizer
function [x, f] = minimize(fun, x0, maxiter)
    % fminunc if available (Optimization Toolbox), else self-contained L-BFGS.
    if exist('fminunc', 'file') == 2
        opts = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
            'SpecifyObjectiveGradient', true, 'MaxIterations', maxiter, ...
            'MaxFunctionEvaluations', 20*maxiter, 'OptimalityTolerance', 1e-12, ...
            'StepTolerance', 1e-16, 'FunctionTolerance', 1e-18, 'Display', 'off');
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
            alpha(idx) = rho(idx) * (S(:,idx).' * q);
            q = q - alpha(idx) * Y(:,idx);
        end
        if nstore > 0
            idx = mod(head - 1, m) + 1;
            gamma = (S(:,idx).' * Y(:,idx)) / (Y(:,idx).' * Y(:,idx));
        else
            gamma = 1;
        end
        z = gamma * q;
        for i = nstore:-1:1
            idx = mod(head - i, m) + 1;
            beta = rho(idx) * (Y(:,idx).' * z);
            z = z + (alpha(idx) - beta) * S(:,idx);
        end
        d = -z; gtd = g.' * d;
        if gtd >= 0, d = -g; gtd = g.'*d; end
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
            S(:,head) = s; Y(:,head) = y; rho(head) = 1 / sy;
            nstore = min(nstore + 1, m);
        end
        x = xn; f = fnew; g = gnew;
        if abs(f) < 1e-20, break; end
    end
end
