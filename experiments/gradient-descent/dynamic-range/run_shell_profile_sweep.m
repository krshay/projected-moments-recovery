function summary = run_shell_profile_sweep(ratios,snr,sample_count)
%RUN_SHELL_PROFILE_SWEEP Radial decay within bands, band norms held equal.
%   Sweeps the within-band radial dynamic range S while every band carries
%   identical total energy. Complements the band sweep D, which does the
%   opposite. Synthetic Gaussian signal, identity radial basis (so shells are
%   the radial index directly), projected third moment, bands 0 and 1 pinned,
%   20 restarts, 3000 iterations.
%
%   Motivation: for 80S in the delta basis the band-to-band spread is only
%   8.1 while the within-band shell spread is 178-2785 (geometric mean 856).
%   The D sweep manipulated the small quantity; this manipulates the large
%   one.
%
%   Row scaling is not spectrally neutral, so the realized within-band
%   sigma_1/sigma_n is measured and reported at every S. If recovery tracks
%   that rather than S itself, this sweep is the kappa ladder in disguise and
%   should be read as such.
%
%   RATIOS may mix positive numbers with the string 'natural', which is the
%   control row: band norms equalized exactly as in every other row, but the
%   radial profile left untouched instead of overwritten. S = 1 is NOT that
%   control -- it forces every shell to the same norm, which is itself a
%   modification of the signal, and one that costs recovery. On the seed-42
%   synthetic the natural profile recovers 20/20 in a single basin while S = 1
%   recovers 17/20 in two, despite S = 1 having the lower kappa*. Read the
%   sweep as a family of imposed profiles measured against 'natural', not as a
%   ladder rising from an unmodified baseline at S = 1.

if nargin < 1 || isempty(ratios)
    ratios = {'natural',1,10,100,1000};
end
ratios = normalize_ratios(ratios);
% SNR = Inf uses the exact population moment (the original behaviour).
% A finite SNR estimates the moment by Monte Carlo from SAMPLE_COUNT
% uniformly rotated projections with iid complex Gaussian noise on c_m[k],
% where SNR is a power ratio, mean|c|^2/sigma^2, not decibels.
if nargin < 2 || isempty(snr)
    snr = Inf;
end
if nargin < 3 || isempty(sample_count)
    sample_count = 100000;
end
assert(isscalar(snr) && ~isnan(snr) && snr > 0, ...
    ['SNR must be a positive power ratio. Use Inf for the noiseless ', ...
     '(exact population moment) case; SNR = 0 would mean infinite noise.']);
noisy = isfinite(snr);
if ~noisy
    sample_count = 0;   % nothing is sampled; recorded as such
end

experiment_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(fileparts(experiment_dir)));
addpath(fullfile(repo_root,'src','gradient-descent'));

L_max = 5;
shell_count = 12;
restart_count = 20;
seed = 42;
threshold = 1e-4;
max_iterations = 3000;
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');

output_dir = fullfile(repo_root,'results','gradient-descent','shell-profile');
if exist(output_dir,'dir') ~= 7
    mkdir(output_dir);
end
diary(fullfile(output_dir,'shell_profile_sweep.log'));
diary_cleanup = onCleanup(@() diary('off'));
timer_all = tic;

fprintf('============================================================\n');
fprintf(' Within-band radial dynamic range S, band norms held equal\n');
fprintf(' Synthetic, identity basis, projected, L=%d, R=%d\n',L_max,shell_count);
fprintf(' %d restarts, %d iterations, S = [%s]\n', ...
    restart_count,max_iterations,strjoin(cellfun(@ratio_label,ratios, ...
    'UniformOutput',false),' '));
if noisy
    fprintf(' SNR = %g (power ratio), N = %d samples per moment\n', ...
        snr,sample_count);
else
    fprintf(' Exact population moment\n');
end
fprintf('============================================================\n');

support = shell_count*ones(1,L_max+1);
radial_basis = repmat({eye(shell_count)},1,L_max+1);
base = gd_generate_josh_synthetic_coefficients(L_max,shell_count,seed);

records = cell(numel(ratios),1);
for index = 1:numel(ratios)
    ratio = ratios{index};
    label = ratio_label(ratio);
    if is_natural(ratio)
        % Control: same band-norm equalization every other row gets, but the
        % radial profile is left as drawn instead of overwritten.
        coefficients = gd_impose_band_dynamic_range(base,L_max,support,1);
    else
        coefficients = gd_impose_shell_dynamic_range(base,L_max,support,ratio);
    end

    fprintf('\n============================================================\n');
    fprintf(' S = %s\n',label);
    fprintf('============================================================\n');
    [band_norms,shell_ratio,kappa] = describe(coefficients,L_max,support);
    fprintf('  band norms ||A_l||_F        : [%s]\n',num2str(band_norms,'%.4f '));
    fprintf('  realized shell max/min       : [%s]\n',num2str(shell_ratio,'%.4g '));
    fprintf('  realized sigma_1/sigma_n     : [%s]\n',num2str(kappa,'%.4g '));

    options = struct('name',sprintf('shell profile S=%s',label), ...
        'kind','synthetic','moment_source','exact', ...
        'success_threshold',threshold);
    if noisy
        sigma = projection_scale(coefficients,support,radial_basis, ...
            gd_precompute_model(L_max),L_max)/sqrt(snr);
        options.moment_source = 'empirical';
        options.empirical = struct('n_samples',sample_count, ...
            'noise_sigma',sigma,'seed',seed);
        fprintf('  SNR %g -> sigma %.4g, N = %d samples\n', ...
            snr,sigma,sample_count);
    end
    problem = gd_create_problem(coefficients,support,radial_basis,options);
    round_trip = gd_third_moment_objective(problem.x_true,problem);
    fprintf('  loss at TRUE coefficients    : %.3e%s\n',round_trip, ...
        floor_note(noisy));
    if ~noisy
        assert(round_trip < 1e-12, ...
            'Truth does not round-trip (%.3e).',round_trip);
    end

    hessian = hessian_report(problem);
    fprintf('  cond(H) at truth             : %.4e   flat(<1e-6): %d of %d\n', ...
        hessian.condition,hessian.flat_count,hessian.dof);

    restart_options = struct('restart_count',restart_count,'seed',seed, ...
        'optimizer',optimizer,'initialization','real');
    experiment = gd_run_josh_baseline_restarts(problem,restart_options);

    restarts = experiment.restarts;
    optimized = [restarts.optimized_band_error];
    losses = [restarts.objective];
    capped = sum([restarts.iterations] >= max_iterations);
    basins = gd_cluster_basins(losses,[restarts.exitflag]);

    fprintf('\n  --- S = %s summary ---\n',label);
    fprintf('  success (l>=2 < %.0e): %d/%d\n',threshold, ...
        sum(optimized < threshold),restart_count);
    fprintf('  best l>=2 error:   %.4e\n',min(optimized));
    fprintf('  median l>=2 error: %.4e\n',median(optimized));
    fprintf('  best loss:         %.4e\n',min(losses));
    fprintf('  converged: %d/%d;  hit cap: %d/%d\n', ...
        sum([restarts.exitflag] > 0),restart_count,capped,restart_count);
    if capped > 0
        fprintf('  basins: %d  << UPPER BOUND, %d truncated >>\n', ...
            numel(basins),capped);
    else
        fprintf('  basins: %d  (measured)\n',numel(basins));
    end
    for k = 1:numel(basins)
        fprintf('    %.6e   %5d restarts, %d converged\n', ...
            basins(k).loss,basins(k).count,basins(k).converged);
    end

    records{index} = struct('ratio',ratio,'label',label, ...
        'band_norms',band_norms, ...
        'shell_ratio',shell_ratio,'kappa',kappa,'hessian',hessian, ...
        'basins',basins,'successes',sum(optimized < threshold), ...
        'floor_loss',round_trip, ...
        'best_error',min(optimized),'median_error',median(optimized), ...
        'best_loss',min(losses),'capped',capped, ...
        'converged',sum([restarts.exitflag] > 0),'restarts',restarts);
    write_csv(records{index},output_dir);
end

report(records,restart_count);
write_summary(records,output_dir,restart_count);
% Both RECORDS and RATIOS are cells, so both need the extra brace: struct()
% treats a bare cell as a template for a struct ARRAY, one element per cell
% entry. RATIOS became a cell when 'natural' was added as a row.
summary = struct('records',{records},'ratios',{ratios}, ...
    'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,'shell_profile_sweep.mat'),'summary','-v7.3');
fprintf('\nFinished in %.1f s. Written to %s\n', ...
    summary.runtime_seconds,output_dir);
clear diary_cleanup;
end

function ratios = normalize_ratios(ratios)
%NORMALIZE_RATIOS Accept a numeric vector or a cell mixing numbers and
%   'natural', and return a cell array of validated entries.
if isnumeric(ratios)
    ratios = num2cell(ratios(:).');
elseif ischar(ratios) || isstring(ratios)
    ratios = {char(ratios)};
end
assert(iscell(ratios) && ~isempty(ratios), ...
    'ratios must be a numeric vector or a cell of numbers and ''natural''.');
for k = 1:numel(ratios)
    entry = ratios{k};
    if ischar(entry) || isstring(entry)
        entry = char(entry);
        assert(strcmpi(entry,'natural'), ...
            'Unknown ratio ''%s''. The only string entry is ''natural''.', ...
            entry);
        ratios{k} = 'natural';
    else
        validateattributes(entry,{'numeric'}, ...
            {'scalar','real','positive','finite'});
        ratios{k} = double(entry);
    end
end
end

function tf = is_natural(ratio)
tf = ischar(ratio) && strcmp(ratio,'natural');
end

function label = ratio_label(ratio)
if is_natural(ratio)
    label = 'natural';
else
    label = sprintf('%g',ratio);
end
end

function note = floor_note(noisy)
if noisy
    note = '   (empirical moment: this is the achievable floor)';
else
    note = '   (exact moment)';
end
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

function [band_norms,shell_ratio,kappa] = describe(a,L_max,support)
band_norms = zeros(1,L_max+1);
shell_ratio = zeros(1,L_max+1);
kappa = zeros(1,L_max+1);
for l = 0:L_max
    rows = 1:support(l+1);
    block = a(rows,gd_band_indices(l));
    band_norms(l+1) = norm(block,'fro');
    row_norms = sqrt(sum(abs(block).^2,2));
    shell_ratio(l+1) = max(row_norms)/max(min(row_norms),eps);
    values = svd(to_real(block,l));
    kappa(l+1) = values(1)/max(values(min(end,2*l+1)),eps);
end
end

function M = to_real(block,l)
M = zeros(size(block,1),2*l+1);
M(:,1) = real(block(:,l+1));
for m = 1:l
    M(:,2*m) = sqrt(2)*real(block(:,m+l+1));
    M(:,2*m+1) = sqrt(2)*imag(block(:,m+l+1));
end
end

function hessian = hessian_report(problem)
x_true = problem.x_true;
n = numel(x_true);
step = 1e-6*max(norm(x_true)/sqrt(n),1e-8);
H = zeros(n,n);
for j = 1:n
    forward = x_true;  forward(j) = forward(j)+step;
    backward = x_true; backward(j) = backward(j)-step;
    [~,gf] = gd_third_moment_objective(forward,problem);
    [~,gb] = gd_third_moment_objective(backward,problem);
    H(:,j) = (gf-gb)/(2*step);
end
H = (H+H')/2;
eigenvalues = sort(real(eig(H)),'descend');
positive = eigenvalues(eigenvalues > 0);
relative = eigenvalues/max(eigenvalues(1),eps);
hessian = struct('dof',n,'lambda_max',eigenvalues(1), ...
    'lambda_min',eigenvalues(end), ...
    'condition',eigenvalues(1)/max(min(positive),eps), ...
    'flat_count',sum(relative < 1e-6));
end


function report(records,restart_count)
fprintf('\n============================================================\n');
fprintf(' Shell-profile sweep summary (%d restarts each)\n',restart_count);
fprintf('============================================================\n');
fprintf('%8s %9s %8s %11s %11s %11s %8s %8s\n', ...
    'S','lowest','basins','best e','median e','floor loss','kappa*','capped');
for index = 1:numel(records)
    r = records{index};
    fprintf('%8s %6d/%-2d %8d %11.3e %11.3e %11.3e %8.1f %5d/%-2d\n', ...
        r.label,r.basins(1).count,restart_count,numel(r.basins), ...
        r.best_error,r.median_error,r.floor_loss, ...
        exp(mean(log(r.kappa(3:end)))),r.capped,restart_count);
end
fprintf(['\n"lowest" counts restarts in the lowest loss cluster. ', ...
    '"floor loss" is the loss\nat the true coefficients -- zero for an ', ...
    'exact moment, the sampling+noise\nfloor for an empirical one, and ', ...
    'the bound no optimizer can beat in that cell.\n']);
fprintf(['\n S = natural is the control: band norms equalized, radial ', ...
    'profile left as drawn.\n S = 1 is not a control -- it forces every ', ...
    'shell to the same norm.\n']);
fprintf(['\n kappa* = geometric mean of realized sigma_1/sigma_n over ', ...
    'bands 2..L.\n Row scaling is not spectrally neutral, so compare ', ...
    'against the kappa ladder.\n']);
end

function write_csv(record,output_dir)
path = fullfile(output_dir,sprintf('restarts_S_%s.csv',record.label));
fid = fopen(path,'w');
assert(fid >= 0);
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

function write_summary(records,output_dir,restart_count)
path = fullfile(output_dir,'shell_profile_summary.csv');
fid = fopen(path,'w');
assert(fid >= 0);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid,['shell_ratio,restarts,successes,lowest_cluster,floor_loss,best_error,median_error,', ...
    'best_loss,hessian_condition,hessian_flat,basins,converged,capped,', ...
    'realized_kappa_geomean\n']);
for index = 1:numel(records)
    r = records{index};
    fprintf(fid,['%s,%d,%d,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%d,%d,%d,%d,', ...
        '%.12e\n'], ...
        r.label,restart_count,r.successes,r.basins(1).count, ...
        r.floor_loss,r.best_error,r.median_error,r.best_loss, ...
        r.hessian.condition,r.hessian.flat_count, ...
        numel(r.basins),r.converged,r.capped, ...
        exp(mean(log(r.kappa(3:end)))));
end
clear cleanup;
end
