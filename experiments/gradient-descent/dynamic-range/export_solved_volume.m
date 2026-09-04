function summary = export_solved_volume(result_file,restart_index)
%EXPORT_SOLVED_VOLUME Write a recovered volume and its target as MRC maps.
%   EXPORT_SOLVED_VOLUME(RESULT_FILE) takes a saved RUN_CROP_CONVENTION result,
%   rebuilds the recovered volume and the band-limited volume the optimizer was
%   actually solving for, and writes both as MRC so they can be opened in
%   ChimeraX. RESTART_INDEX selects which restart to export; by default the one
%   with the lowest l>=2 error.
%
%   WHAT TO COMPARE. The recovered map should be judged against the BAND-LIMITED
%   truth, not against the deposited map. Those are very different standards:
%   for 80S at L = 5, K = 12 the recovered volume matches its target at
%   correlation 0.998 while both match the original at only 0.42. Practically
%   all of the visual difference from the real ribosome is the L and K
%   truncation, not the third-moment inversion. Reporting the second number as
%   though it measured the recovery would badly understate it; reporting the
%   first without the truncation would badly overstate what the method delivers.
%
%   Both maps are written normalised to unit maximum, so a single ChimeraX
%   contour level means the same thing in each. Without that the viewer picks a
%   different default level per map and the two look different for no reason.
%
%   The exported restart is one draw. Check the run's clustering before
%   presenting it: the cropped 80S cell has 11 of 20 restarts in the dominant
%   solution and 9 elsewhere, so a figure from the best restart shows what
%   success looks like, not what typically happens.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(result_file)
    result_file = fullfile(repo_root,'results','gradient-descent', ...
        'crop-convention','crop_vol_80S_true_crop0.697_snr30_it5000.mat');
end
% Accept a path relative to the repository root, not just to MATLAB's current
% folder, so the call works from wherever the session happens to be.
if exist(result_file,'file') ~= 2
    candidate = fullfile(repo_root,result_file);
    if exist(candidate,'file') == 2
        result_file = candidate;
    else
        error(['Result file not found:\n  %s\nnor relative to the repo:' ...
            '\n  %s'],result_file,candidate);
    end
end
loaded = load(result_file);
s = loaded.summary;

% Read the geometry from the result rather than assuming it. Older results
% predate these fields; fall back to the values every run so far has used, but
% say so, because reconstructing with the wrong L or K yields a plausible and
% silently wrong map.
if isfield(s,'L_max')
    L_max = s.L_max; K = s.shell_count; grid_size = s.grid_size;
else
    L_max = 5; K = 12; grid_size = 31;
    warning(['Result predates geometry recording; assuming L_max=%d, K=%d, ' ...
        'grid=%d. Verify these match the run that produced it.'], ...
        L_max,K,grid_size);
end
if nargin < 2 || isempty(restart_index)
    [~,restart_index] = min(s.errors);
end

% the band-limited target: the same array the optimizer was given
target = nu_expand_volume_shells( ...
    fullfile(repo_root,'data','raw',s.volume), ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',s.crop));
target = target/max(norm(target(:)),eps);

recovered = s.restarts(restart_index).coefficients_estimate;
% the third moment cannot resolve the mirror; align before comparing
if norm(recovered(:)+target(:)) < norm(recovered(:)-target(:))
    recovered = -recovered;
end

V_target    = nu_shells_to_volume(target,L_max,K,grid_size);
V_recovered = nu_shells_to_volume(recovered,L_max,K,grid_size);
V_original  = nu_load_volume(fullfile(repo_root,'data','raw',s.volume), ...
    grid_size,s.crop);

correlation = @(A,B) (A(:)'*B(:))/(norm(A(:))*norm(B(:)));
fprintf('\n%s, crop %g, SNR %g | L=%d, K=%d, grid %d\n', ...
    s.volume,s.crop,s.snr,L_max,K,grid_size);
fprintf('  restart %d (l>=2 error %.4e)\n',restart_index,s.errors(restart_index));
fprintf('  recovered vs band-limited target : %.4f   <- measures the recovery\n', ...
    correlation(V_recovered,V_target));
fprintf('  band-limited target vs original  : %.4f   <- measures the truncation\n', ...
    correlation(V_target,V_original));
fprintf('  recovered vs original            : %.4f\n', ...
    correlation(V_recovered,V_original));

output_dir = fullfile(repo_root,'results','gradient-descent','fb-basis','chimerax');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
% Name the maps after the run. Hardcoded names would let a second export
% silently overwrite the first, and would mislabel an uncropped run.
unit = @(V) V/max(V(:));
stem = sprintf('%s_L%d_K%d_crop%g_snr%g_r%d', ...
    regexprep(s.volume,'\.[^.]*$',''),L_max,K,s.crop,s.snr,restart_index);
files = {sprintf('%s_solved.mrc',stem),     V_recovered; ...
         sprintf('%s_target.mrc',stem),     V_target; ...
         sprintf('%s_original.mrc',stem),   V_original};
for k = 1:size(files,1)
    nu_write_mrc(fullfile(output_dir,files{k,1}),unit(files{k,2}));
    fprintf('  wrote %s\n',files{k,1});
end

fprintf('\nOpen in ChimeraX with:\n');
fprintf('  open %s/%s\n',output_dir,files{2,1});
fprintf('  open %s/%s\n',output_dir,files{1,1});
fprintf('  volume all style surface ; volume all level 0.32 ; tile ; lighting soft\n');
fprintf('  volume #1 color lightblue ; volume #2 color khaki\n');

summary = struct('result_file',result_file,'restart',restart_index, ...
    'error',s.errors(restart_index), ...
    'corr_recovered_target',correlation(V_recovered,V_target), ...
    'corr_target_original',correlation(V_target,V_original), ...
    'files',{files(:,1)}, ...
    'output_dir',output_dir);
end
