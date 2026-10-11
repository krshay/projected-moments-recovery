function summary = compare_basis_representation(volume,crop,L_max,grid_size)
%COMPARE_BASIS_REPRESENTATION Delta-shell vs Fourier-Bessel, representation only.
%   COMPARE_BASIS_REPRESENTATION() expands 80S at L = 13 in both radial bases,
%   rebuilds the volume from each, and correlates against the preprocessed map.
%   NO RECOVERY IS RUN. The question here is only how well each basis can
%   represent the molecule at a given number of parameters per band -- which is
%   a ceiling on any recovery in that basis, and is cheap to measure, whereas
%   the recovery itself costs hours.
%
%   THE COMPARISON. Both bases are given the same budget, P numbers per (l,m):
%     delta-shell   u_l(rho) sampled at P radii, splined in between
%     Fourier-Bessel u_l(rho) = sum_{s<=P} A_{l,s} j_l(k_{l,s} rho/rho_max)
%   The Fourier-Bessel coefficients are the least-squares fit of the model to
%   the NDFT shell expansion sampled at K_FIT radii, with K_FIT > P always, so
%   the design matrix is never square (see RUN_FB_BASIS_RECOVERY on why K = S
%   is degenerate).
%
%   ONE ASYMMETRY, STATED RATHER THAN HIDDEN. The shell curve at budget P is
%   reconstructed from P radial samples; the Fourier-Bessel curve at budget P
%   is evaluated on K_FIT radii before the same spline. Fourier-Bessel therefore
%   gets a finer radial grid to interpolate through, which flatters it slightly
%   at small P. The effect is an interpolation artefact, not a property of the
%   basis; SHELL_ON_FIT_GRID in the output isolates it by resampling the shell
%   model onto the same K_FIT radii.
%
%   WHAT THIS DOES NOT MEASURE. A basis that represents the molecule well can
%   still be a bad place to invert the third moment, and vice versa -- the
%   landscape is not a function of the approximation error. Read this as a
%   ceiling, not as a prediction of which basis recovers better.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(volume),    volume = 'vol_80S_true.mrc'; end
if nargin < 2 || isempty(crop),      crop = 0.697; end
if nargin < 3 || isempty(L_max),     L_max = 13;   end
if nargin < 4 || isempty(grid_size), grid_size = 31; end

budgets = [6 8 10 12 14];
K_fit = 16;                      % radial sampling for the FB fit; > max(budgets)
assert(K_fit > max(budgets),'K_FIT must exceed every budget.');

volume_path = fullfile(repo_root,'data','raw',volume);
output_dir = fullfile(repo_root,'results','gradient-descent','basis-comparison');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('%s_L%d_crop%g',regexprep(volume,'\.[^.]*$',''),L_max,crop);
diary(fullfile(output_dir,sprintf('%s.log',tag)));
diary_cleanup = onCleanup(@() diary('off'));
timer_all = tic;

correlation = @(A,B) (A(:)'*B(:))/(norm(A(:))*norm(B(:)));
V_original = nu_load_volume(volume_path,grid_size,crop);

fprintf('\n============================================================\n');
fprintf(' Representation only (no recovery): %s\n',volume);
fprintf(' L = %d, crop = %g, grid %d, FB fitted on %d radii\n', ...
    L_max,crop,grid_size,K_fit);
fprintf('============================================================\n');

% ---- the reference expansion, on the fit grid ----
u_fit = nu_expand_volume_shells(volume_path, ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K_fit,'mode','exact','crop',crop));
u_fit = u_fit/max(norm(u_fit(:)),eps);
V_fit = nu_shells_to_volume(u_fit,L_max,K_fit,grid_size);
fprintf('  shell expansion at K = %d (the reference) : corr %.4f\n', ...
    K_fit,correlation(V_fit,V_original));
fprintf('  -- this is the L = %d ceiling; no radial model beats it --\n\n',L_max);

n = numel(budgets);
shell_corr = zeros(1,n); fb_corr = zeros(1,n);
shell_fit_corr = zeros(1,n); fb_radial_error = zeros(1,n);
shell_radial_error = zeros(1,n);

for i = 1:n
    P = budgets(i);

    % ---- delta-shell at budget P: expand directly at P radii ----
    u_shell = nu_expand_volume_shells(volume_path, ...
        struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
               'shell_count',P,'mode','exact','crop',crop));
    u_shell = u_shell/max(norm(u_shell(:)),eps);
    V_shell = nu_shells_to_volume(u_shell,L_max,P,grid_size);
    shell_corr(i) = correlation(V_shell,V_original);

    % the same shell model resampled onto the FB fit grid, to remove the
    % interpolation asymmetry described above
    rho_P   = ((1:P)-0.5)*(pi/P);
    rho_fit = ((1:K_fit)-0.5)*(pi/K_fit);
    u_shell_on_fit = zeros(K_fit,(L_max+1)^2);
    for c = 1:size(u_shell,2)
        u_shell_on_fit(:,c) = interp1(rho_P,u_shell(:,c),rho_fit,'spline','extrap');
    end
    V_shell_fit = nu_shells_to_volume(u_shell_on_fit,L_max,K_fit,grid_size);
    shell_fit_corr(i) = correlation(V_shell_fit,V_original);
    shell_radial_error(i) = norm(u_shell_on_fit-u_fit,'fro')/norm(u_fit,'fro');

    % ---- Fourier-Bessel at budget P: S = P functions fitted on K_fit radii ----
    support = P*ones(1,L_max+1);
    fb_metadata = struct('L_max',L_max,'rho_max',pi,'available_shells',support, ...
        'bessel_zeros',zeros(L_max+1,P));
    for l = 0:L_max
        for s = 1:P
            fb_metadata.bessel_zeros(l+1,s) = nu_spherical_bessel_zero(l,s);
        end
    end
    Phi = gd_fourier_bessel_radial_basis(fb_metadata,K_fit);
    A = zeros(P,(L_max+1)^2);
    for l = 0:L_max
        columns = gd_band_indices(l);
        A(:,columns) = Phi{l+1}\u_fit(:,columns);
    end
    u_fb = gd_apply_radial_basis(A,Phi,support,L_max);
    fb_radial_error(i) = norm(u_fb-u_fit,'fro')/norm(u_fit,'fro');
    V_fb = nu_shells_to_volume(u_fb,L_max,K_fit,grid_size);
    fb_corr(i) = correlation(V_fb,V_original);

    fprintf('  budget %2d/band : shell %.4f (on fit grid %.4f) | FB %.4f\n', ...
        P,shell_corr(i),shell_fit_corr(i),fb_corr(i));
end

fprintf('\n  radial model error vs the K = %d expansion (lower is better):\n',K_fit);
for i = 1:n
    fprintf('    budget %2d : shell %.4f   FB %.4f\n', ...
        budgets(i),shell_radial_error(i),fb_radial_error(i));
end

% ---- maps at the matched budget used for the L = 13 figure ----
match = find(budgets == 12,1);
if ~isempty(match)
    unit = @(V) V/max(V(:));
    P = 12;
    u_shell = nu_expand_volume_shells(volume_path, ...
        struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
               'shell_count',P,'mode','exact','crop',crop));
    u_shell = u_shell/max(norm(u_shell(:)),eps);
    support = P*ones(1,L_max+1);
    fb_metadata = struct('L_max',L_max,'rho_max',pi,'available_shells',support, ...
        'bessel_zeros',zeros(L_max+1,P));
    for l = 0:L_max
        for s = 1:P
            fb_metadata.bessel_zeros(l+1,s) = nu_spherical_bessel_zero(l,s);
        end
    end
    Phi = gd_fourier_bessel_radial_basis(fb_metadata,K_fit);
    A = zeros(P,(L_max+1)^2);
    for l = 0:L_max
        columns = gd_band_indices(l);
        A(:,columns) = Phi{l+1}\u_fit(:,columns);
    end
    maps = {sprintf('%s_original.mrc',tag),  V_original; ...
            sprintf('%s_shell_P12.mrc',tag), nu_shells_to_volume(u_shell,L_max,P,grid_size); ...
            sprintf('%s_fb_S12.mrc',tag),    nu_shells_to_volume( ...
                gd_apply_radial_basis(A,Phi,support,L_max),L_max,K_fit,grid_size)};
    fprintf('\n  maps at budget 12 (unit-normalised, one contour level compares):\n');
    for k = 1:size(maps,1)
        nu_write_mrc(fullfile(output_dir,maps{k,1}),unit(maps{k,2}));
        fprintf('    %s\n',maps{k,1});
    end
    fprintf('\n  !open -a /Applications/ChimeraX-1.10-rc2025.06.20.app %s %s %s\n', ...
        fullfile(output_dir,maps{1,1}),fullfile(output_dir,maps{2,1}), ...
        fullfile(output_dir,maps{3,1}));
    fprintf('  then: volume all style surface ; volume all level 0.32 ; tile ; lighting soft\n');
end

summary = struct('volume',volume,'crop',crop,'L_max',L_max, ...
    'grid_size',grid_size,'K_fit',K_fit,'budgets',budgets, ...
    'ceiling_corr',correlation(V_fit,V_original), ...
    'shell_corr',shell_corr,'shell_on_fit_grid_corr',shell_fit_corr, ...
    'fb_corr',fb_corr,'shell_radial_error',shell_radial_error, ...
    'fb_radial_error',fb_radial_error,'runtime_seconds',toc(timer_all));
save(fullfile(output_dir,sprintf('%s.mat',tag)),'summary','-v7.3');
fprintf('\nFinished in %.1f min. Written to %s\n', ...
    summary.runtime_seconds/60,output_dir);
clear diary_cleanup;
end
