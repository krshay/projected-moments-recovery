function rescore_unpinned_noise_sweep(label,volume,grid_size,snr)
%RESCORE_UNPINNED_NOISE_SWEEP Re-align one saved SNR cell with the dense O(3) search.
%   RESCORE_UNPINNED_NOISE_SWEEP('TRPV1','vol_TRPV1_emd24084.mat',32,30)
%   re-aligns every stored restart with the current GD_ALIGN_ROTATION_REFLECTION
%   and rewrites the cell's CSV. The coarse aligner the sweeps originally used
%   mis-aligned one TRPV1 restart per cell (errors near 0.25 for solutions
%   actually at 0.025-0.11). Nothing is re-optimised: only the scoring changes.
%
%   The original CSV is copied once into "original_alignment/" beside it, and
%   the new CSV keeps the old values as *_original columns, so the correction
%   is visible row by row.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));
L_max = 5; K = 12;
folder = fullfile(repo_root,'results','gradient-descent','noise-sweep', ...
    sprintf('%s_unpinned_noise sweep',label));
stem = sprintf('unpinned_%s_snr_%g',regexprep(volume,'\.[^.]*$',''),snr);
csv_path = fullfile(folder,[stem '.csv']);
backup = fullfile(folder,'original_alignment');
if exist(backup,'dir') ~= 7, mkdir(backup); end
if exist(fullfile(backup,[stem '.csv']),'file') ~= 2
    copyfile(csv_path,fullfile(backup,[stem '.csv']));
end

truth = nu_expand_volume_shells(fullfile(repo_root,'data','raw',volume), ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',1));
truth = truth/max(norm(truth(:)),eps);
S = load(fullfile(folder,[stem '.mat']));
R = S.summary.restarts;
T = readtable(fullfile(backup,[stem '.csv']));
T.aligned_l2_error_original = T.aligned_l2_error;
T.aligned_all_band_error_original = T.aligned_all_band_error;
T.reflected_original = T.reflected;
for k = 1:numel(R)
    o = gd_align_rotation_reflection(R(k).coefficients_estimate,truth,L_max);
    T.aligned_l2_error(k) = o.error_free;
    T.aligned_all_band_error(k) = o.error_all;
    T.reflected(k) = o.reflected;
end
writetable(T,csv_path);
changed = abs(T.aligned_l2_error-T.aligned_l2_error_original) > 1e-6;
fprintf('%s SNR %g: %d of %d restarts changed; median %.4e -> %.4e, mean %.4e -> %.4e\n', ...
    label,snr,sum(changed),height(T),median(T.aligned_l2_error_original), ...
    median(T.aligned_l2_error),mean(T.aligned_l2_error_original),mean(T.aligned_l2_error));
for k = find(changed)'
    fprintf('    seed %d: %.4f -> %.4f\n',T.seed(k),T.aligned_l2_error_original(k),T.aligned_l2_error(k));
end
end
