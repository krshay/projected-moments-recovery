function summary = run_pinning_comparison(volume,fixed_bands,restart_count,L_max,max_iterations)
%RUN_PINNING_COMPARISON Gradient descent with and without pinning bands 0 and 1.
%   RUN_PINNING_COMPARISON(VOLUME,FIXED_BANDS) runs RESTART_COUNT random restarts
%   against the ANALYTIC projected third moment, with FIXED_BANDS pinned to
%   truth ([0,1] is the historical setting, [] frees every band). Everything
%   else matches RUN_SINGLE_RESTART: L = 5, K = 12, crop = 1, native band norms,
%   'real' initialisation, seeds 1..RESTART_COUNT, 20000-iteration cap.
%
%   WHY. Pinning l = 1 to truth fixes one point of the SO(3) x parity orbit of
%   exact solutions. The claim under test is that the other points of that
%   orbit then become spurious minima -- rotated or mirrored copies whose free
%   bands clash with the pinned l = 1 -- and that this, not the landscape of
%   the free problem, is what traps many restarts.
%
%   SCORING. With nothing pinned the answer is determined only up to rotation
%   and reflection, so every restart is scored after GD_ALIGN_ROTATION_REFLECTION.
%   The same alignment is applied to the pinned arm so the two are scored
%   identically; for a pinned solution it can only lower the error, and is
%   reported next to the raw reflection-only figure so any difference is
%   visible. Both the all-band and the l >= 2 errors are recorded.
%
%   The i^l twist in NU_EXPAND_VOLUME_SHELLS is NOT undone here. Both arms fit
%   the same moment, so the comparison isolates pinning; absolute real-molecule
%   claims still wait on that fix.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(volume),         volume = 'vol_TRPV1_true.mrc'; end
if nargin < 2,                            fixed_bands = [0,1]; end
if nargin < 3 || isempty(restart_count),  restart_count = 10; end
if nargin < 4 || isempty(L_max),          L_max = 5; end
if nargin < 5 || isempty(max_iterations), max_iterations = 20000; end
K = 12; grid_size = 31; crop = 1; seed = 1;

support = K*ones(1,L_max+1);
basis = repmat({eye(K)},1,L_max+1);
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');

if isempty(fixed_bands), arm = 'unpinned'; else, arm = ['pinned' sprintf('%d',fixed_bands)]; end
output_dir = fullfile(repo_root,'results','gradient-descent','pinning-comparison');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('%s_L%d_K%d_%s_cap%d',regexprep(volume,'\.[^.]*$',''),L_max,K,arm,max_iterations);
diary(fullfile(output_dir,[tag '.log']));
diary_cleanup = onCleanup(@() diary('off'));
timer_all = tic;

u = nu_expand_volume_shells(fullfile(repo_root,'data','raw',volume), ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',crop));
u = u/max(norm(u(:)),eps);
problem = gd_create_problem(u,support,basis, ...
    struct('name',sprintf('%s %s',volume,arm),'kind','pinning-comparison', ...
           'moment_source','exact','success_threshold',1e-4, ...
           'fixed_bands',fixed_bands));

fprintf('\n============================================================\n');
fprintf(' Pinning comparison: %s, arm = %s\n',volume,arm);
fprintf(' L=%d, K=%d, crop=%g, analytic moment, %d restarts, cap %d\n', ...
    L_max,K,crop,restart_count,max_iterations);
fprintf(' free dof %d, floor loss %.3e\n',numel(problem.x_true), ...
    gd_third_moment_objective(problem.x_true,problem));
fprintf('============================================================\n');

experiment = gd_run_josh_baseline_restarts(problem, ...
    struct('restart_count',restart_count,'seed',seed, ...
           'optimizer',optimizer,'initialization','real'));
R = experiment.restarts;

rows = zeros(restart_count,10);
fprintf('\n seed      loss    raw_l>=2   O3_l>=2    O3_all  refl  iters flag      ||g||\n');
for k = 1:restart_count
    aligned = gd_align_rotation_reflection(R(k).coefficients_estimate,u,L_max);
    rows(k,:) = [R(k).seed,R(k).objective,R(k).optimized_band_error, ...
        aligned.error_free,aligned.error_all,aligned.reflected, ...
        R(k).iterations,R(k).exitflag,R(k).gradient_norm,R(k).runtime_seconds];
    fprintf('%5d %9.2e %10.4e %10.4e %9.4e %5d %6d %4d %10.2e\n',rows(k,1:9));
end

o3_free = rows(:,4);
fprintf('\n exact recoveries (O3-aligned l>=2 error < 1e-4): %d/%d\n', ...
    sum(o3_free < 1e-4),restart_count);
fprintf(' median O3-aligned l>=2 error: %.4e   capped: %d/%d\n', ...
    median(o3_free),sum(rows(:,7) >= max_iterations),restart_count);

table_out = array2table(rows,'VariableNames',{'seed','loss', ...
    'raw_l2_error','o3_l2_error','o3_all_error','reflected','iterations', ...
    'exitflag','gradient_norm','runtime_seconds'});
writetable(table_out,fullfile(output_dir,[tag '.csv']));
summary = struct('volume',volume,'arm',arm,'fixed_bands',fixed_bands, ...
    'L_max',L_max,'shell_count',K,'crop',crop,'max_iterations',max_iterations, ...
    'table',table_out,'restarts',R,'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,[tag '.mat']),'summary','-v7.3');
fprintf('\nFinished in %.1f min. Written to %s\n',summary.runtime_seconds/60,output_dir);
clear diary_cleanup;
end
