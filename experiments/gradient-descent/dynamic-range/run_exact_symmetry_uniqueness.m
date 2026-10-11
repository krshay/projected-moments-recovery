function summary = run_exact_symmetry_uniqueness(volume,symmetry,L_max,restart_count,max_iterations)
%RUN_EXACT_SYMMETRY_UNIQUENESS Does an exactly Cn-symmetric volume have a second solution?
%   Imposes exact Cn symmetry on VOLUME's shell coefficients (keeping only
%   m = 0 mod N), then runs many random restarts against the ANALYTIC third
%   moment. The question is global uniqueness, which a Hessian cannot answer:
%   is there a point far from the truth that reproduces the moment exactly?
%
%   WHAT IS ALREADY KNOWN, and what this adds. At an exactly C4-symmetric
%   volume the Jacobian of the moment map has 288 of 384 columns identically
%   zero (the m /= 0 mod 4 directions), so the Hessian is singular -- measured
%   as 289 eigenvalues at the finite-difference floor against 1 for the
%   deposited TRPV1. That kills the usual "full-rank Jacobian implies finitely
%   many solutions" argument. It does NOT show non-identifiability: perturbing
%   those directions changes the moment at second order (relative change
%   2.756e-05, 2.756e-03, 2.756e-01 at t = 1e-3, 1e-2, 1e-1 -- clean quadratic
%   scaling), so the truth stays a strict local minimum with a quartic basin.
%
%   READING THE RESULT. The floor is machine zero, so a restart that reaches
%   loss ~1e-20 or below has found an exact solution.
%     - all such restarts at l>=2 error ~0  -> no second solution was found;
%       evidence for uniqueness, not proof, since restarts sample finitely.
%     - some at low loss with large error   -> a genuine second exact solution,
%       i.e. the moment does not determine the volume. That would be a much
%       stronger claim than anything measured so far and should be checked by
%       re-running the forward model on that solution directly.
%     - nothing near the floor at all       -> inconclusive about uniqueness;
%       it only says the quartic basin is hard to descend, which is expected.
%
%   Convergence is slow by construction: a quartic minimum has gradients that
%   vanish faster than the error, so MAX_ITERATIONS should be generous and
%   restarts that stop early must not be read as landscape results.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(volume),         volume = 'vol_TRPV1_true.mrc'; end
if nargin < 2 || isempty(symmetry),       symmetry = 4;  end
if nargin < 3 || isempty(L_max),          L_max = 5;     end
if nargin < 4 || isempty(restart_count),  restart_count = 20;   end
if nargin < 5 || isempty(max_iterations), max_iterations = 20000; end
K = 12; grid_size = 31; seed = 42;

support = K*ones(1,L_max+1);
basis = repmat({eye(K)},1,L_max+1);
optimizer = struct('method','auto','max_iterations',max_iterations, ...
    'optimality_tolerance',1e-12,'step_tolerance',1e-16, ...
    'function_tolerance',1e-18,'display','off');

output_dir = fullfile(repo_root,'results','gradient-descent','symmetry-uniqueness');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('%s_C%d_L%d',regexprep(volume,'\.[^.]*$',''),symmetry,L_max);
diary(fullfile(output_dir,sprintf('%s.log',tag)));
diary_cleanup = onCleanup(@() diary('off'));
timer_all = tic;

u = nu_expand_volume_shells(fullfile(repo_root,'data','raw',volume), ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',1));
u = u/max(norm(u(:)),eps);
before = norm(u,'fro');
for l = 0:L_max
    for m = -l:l
        if mod(m,symmetry) ~= 0
            u(:,gd_sh_index(l,m)) = 0;
        end
    end
end
kept = norm(u,'fro')/before;
u = u/max(norm(u(:)),eps);

fprintf('\n============================================================\n');
fprintf(' Exact C%d symmetry imposed on %s\n',symmetry,volume);
fprintf(' L = %d, K = %d, analytic moment, %d restarts, %d iterations\n', ...
    L_max,K,restart_count,max_iterations);
fprintf('============================================================\n');
fprintf('  energy retained by the symmetrisation : %.4f\n',kept);

problem = gd_create_problem(u,support,basis, ...
    struct('name',sprintf('%s C%d',volume,symmetry),'kind','symmetry-uniqueness', ...
           'moment_source','exact','success_threshold',1e-4));
floor_loss = gd_third_moment_objective(problem.x_true,problem);
fprintf('  floor loss at truth : %.4e\n',floor_loss);
fprintf('  free dof            : %d\n\n',numel(problem.x_true));

experiment = gd_run_josh_baseline_restarts(problem, ...
    struct('restart_count',restart_count,'seed',seed, ...
           'optimizer',optimizer,'initialization','real'));
restarts = experiment.restarts;
errors = [restarts.optimized_band_error];
losses = [restarts.objective];
capped = [restarts.iterations] >= max_iterations;

fprintf('\n--------------- RESULT ---------------\n');
exact_hits = losses < 1e-20;
fprintf('  restarts reaching loss < 1e-20 : %d/%d\n',sum(exact_hits),restart_count);
if any(exact_hits)
    fprintf('  their l>=2 errors : %s\n',num2str(errors(exact_hits),'%.3e '));
    if any(errors(exact_hits) > 1e-3)
        fprintf('  *** a SECOND EXACT SOLUTION appears to exist ***\n');
    else
        fprintf('  all exact hits are the truth: no second solution found\n');
    end
else
    fprintf('  nothing reached the floor -- inconclusive about uniqueness\n');
end
fprintf('  median l>=2 error : %.4e   best %.4e\n',median(errors),min(errors));
fprintf('  median loss       : %.4e   best %.4e\n',median(losses),min(losses));
fprintf('  capped %d/%d\n',sum(capped),restart_count);

estimates = cell(1,restart_count);
for k = 1:restart_count, estimates{k} = restarts(k).coefficients_estimate; end
clusters = gd_cluster_solutions(estimates,support,L_max);
fprintf('  distinct solutions (5%% tolerance) : %d, largest holds %d\n', ...
    numel(clusters),clusters(1).count);

summary = struct('volume',volume,'symmetry',symmetry,'L_max',L_max, ...
    'shell_count',K,'grid_size',grid_size,'energy_retained',kept, ...
    'floor_loss',floor_loss,'errors',errors,'losses',losses, ...
    'capped',sum(capped),'restarts',restarts, ...
    'distinct',numel(clusters),'agreeing',clusters(1).count, ...
    'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,sprintf('%s.mat',tag)),'summary','-v7.3');
fprintf('\nFinished in %.1f min. Written to %s\n', ...
    summary.runtime_seconds/60,output_dir);
clear diary_cleanup;
end
