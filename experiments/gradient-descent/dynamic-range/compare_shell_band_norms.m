function summary = compare_shell_band_norms(L_max,shell_count,grid_size)
%COMPARE_SHELL_BAND_NORMS Band and shell energy of 80S and TRPV1, side by side.
%   COMPARE_SHELL_BAND_NORMS() expands both molecules in the delta-shell basis
%   and reports where their energy sits: per band, per shell, and per band-shell
%   cell. The motivating question is why 80S recovers from the projected third
%   moment and TRPV1 does not.
%
%   CROPPING. Each molecule is expanded at three croppings, because the answer
%   is not cropping-invariant and no single choice is neutral:
%
%     crop = 1        the deposited box. THE RECOVERY CONTRAST BEING EXPLAINED
%                     (80S 20/20, TRPV1 capped 16/20) WAS MEASURED HERE, so
%                     these are the numbers any explanation must account for.
%     per-molecule    the tightest crop retaining >99% of density mass above
%                     5% of maximum: 0.95 for 80S, 0.60 for TRPV1. This is
%                     "the molecule, not the solvent" done honestly -- the two
%                     boxes are padded very differently.
%     crop = 0.697    the value borrowed from krshay/Non-Uniform-Second-Moment.
%                     Reported only for continuity with earlier runs. It keeps
%                     99.9% of TRPV1 but only 76.6% of 80S: on 80S it cuts the
%                     ribosome rather than the margin, so quantities computed
%                     there describe a truncated molecule.
%
%   WHAT THE COLUMNS MEAN. For band l, ENERGY is ||u_l||_F^2 as a fraction of
%   the total; the l = 0 and l = 1 bands are pinned to truth in every recovery,
%   so their share is unavailable to the optimizer and inflates any error
%   measured over all bands. DILUTION is sqrt(free-band energy fraction), the
%   factor by which a pinned-band-inclusive error understates the free-band
%   error. PARTICIPATION is the normalised inverse participation ratio of the
%   band's singular values -- how many directions the band actually uses.
%
%   A WARNING ABOUT THE INFERENCE THIS INVITES. Two predictors built from these
%   same quantities have already failed on these same molecules: kappa* said
%   hemoglobin (8478) should be hard and it recovered 17/20 while 80S (32)
%   recovered 20/20; participation ordered the molecules correctly but moved by
%   more than the 80S-TRPV1 gap under crop changes of 0.02. Treat what follows
%   as a description of the two signals, not as a demonstrated cause. A
%   difference that only appears at one cropping is a property of the
%   preprocessing, not of the molecule.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(L_max),       L_max = 5;  end
if nargin < 2 || isempty(shell_count), shell_count = 12; end
if nargin < 3 || isempty(grid_size),   grid_size = 31;   end
K = shell_count;

molecules = {'vol_80S_true.mrc','vol_TRPV1_true.mrc'};
labels    = {'80S','TRPV1'};
% per-molecule crops retaining >99% of density mass above 0.05*max, measured
% on the native maps; see the header.
native_crop = [0.95 0.60];

output_dir = fullfile(repo_root,'results','gradient-descent','band-norms');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
tag = sprintf('band_norms_L%d_K%d',L_max,K);
diary(fullfile(output_dir,sprintf('%s.log',tag)));
diary_cleanup = onCleanup(@() diary('off'));

settings = {'deposited box (crop = 1)',            [1 1]; ...
            'per-molecule crop (>99% of mass)',    native_crop; ...
            'borrowed crop = 0.697',               [0.697 0.697]};

records = cell(1,size(settings,1));
for s = 1:size(settings,1)
    fprintf('\n============================================================\n');
    fprintf(' %s   |  L = %d, K = %d shells, grid %d\n', ...
        settings{s,1},L_max,K,grid_size);
    fprintf('============================================================\n');
    per_molecule = cell(1,2);
    for i = 1:2
        crop = settings{s,2}(i);
        u = nu_expand_volume_shells( ...
            fullfile(repo_root,'data','raw',molecules{i}), ...
            struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
                   'shell_count',K,'mode','exact','crop',crop));
        u = u/max(norm(u(:)),eps);
        stats = band_shell_stats(u,L_max,K);
        stats.crop = crop; stats.label = labels{i};
        per_molecule{i} = stats;
        report(labels{i},crop,stats,L_max,K);
    end
    a = per_molecule{1}; b = per_molecule{2};
    % the side-by-side the question is really about
    fprintf('\n  band energy fraction, %s vs %s:\n',labels{1},labels{2});
    fprintf('    l        ');
    for l = 0:L_max, fprintf('%9d',l); end
    fprintf('\n    %-9s',labels{1}); fprintf('%9.4f',a.band_energy);
    fprintf('\n    %-9s',labels{2}); fprintf('%9.4f',b.band_energy);
    fprintf('\n    ratio    ');
    fprintf('%9.2f',a.band_energy./max(b.band_energy,eps));
    fprintf('\n');
    fprintf('  free-band (l>=2) energy : %s %.4f, %s %.4f\n', ...
        labels{1},a.free_energy,labels{2},b.free_energy);
    fprintf('  dilution factor         : %s %.4f, %s %.4f\n', ...
        labels{1},a.dilution,labels{2},b.dilution);

    % Cell-by-cell ratio. Above 1 means 80S carries more of its energy in that
    % (shell, band) cell than TRPV1 does in the same cell -- both arrays are
    % unit norm, so the comparison is of shares, not of absolute density.
    fprintf('\n  ratio ||u[k,l]||: %s / %s   (>1 favours %s)\n', ...
        labels{1},labels{2},labels{1});
    fprintf('      %-6s','k');
    for l = 0:L_max, fprintf('%10s',sprintf('l=%d',l)); end
    fprintf('\n');
    for k = 1:K
        fprintf('      %-6d',k);
        fprintf('%10.2f',a.cell_norm(k,:)./max(b.cell_norm(k,:),eps));
        fprintf('\n');
    end
    records{s} = struct('setting',settings{s,1},'crops',settings{s,2}, ...
        'stats',{per_molecule});
end

summary = struct('L_max',L_max,'shell_count',K,'grid_size',grid_size, ...
    'molecules',{molecules},'records',{records});
save(fullfile(output_dir,sprintf('%s.mat',tag)),'summary','-v7.3');
fprintf('\nWritten to %s\n',output_dir);
clear diary_cleanup;
end

% ---------------------------------------------------------------------------
function stats = band_shell_stats(u,L_max,K)
band_norm = zeros(1,L_max+1);
participation = zeros(1,L_max+1);
kappa = zeros(1,L_max+1);
shell_range = zeros(1,L_max+1);
cell_norm = zeros(K,L_max+1);
for l = 0:L_max
    block = u(:,gd_band_indices(l));
    band_norm(l+1) = norm(block,'fro');
    cell_norm(:,l+1) = sqrt(sum(abs(block).^2,2));
    % real form of the band, as used elsewhere in the repository
    M = zeros(K,2*l+1); M(:,1) = real(block(:,l+1));
    for m = 1:l
        M(:,2*m)   = sqrt(2)*real(block(:,m+l+1));
        M(:,2*m+1) = sqrt(2)*imag(block(:,m+l+1));
    end
    values = svd(M);
    kappa(l+1) = values(1)/max(values(min(end,2*l+1)),eps);
    participation(l+1) = (sum(values.^2)^2/sum(values.^4))/min(K,2*l+1);
    rows = sqrt(sum(M.^2,2)); rows = rows(rows > 0);
    shell_range(l+1) = max(rows)/min(rows);
end
energy = band_norm.^2/max(sum(band_norm.^2),eps);
free = sum(energy(3:end));
stats = struct('band_norm',band_norm,'band_energy',energy, ...
    'free_energy',free,'dilution',sqrt(free), ...
    'shell_norm',sqrt(sum(cell_norm.^2,2)).', ...
    'cell_norm',cell_norm,'participation',participation, ...
    'kappa',kappa,'shell_range',shell_range);
end

% ---------------------------------------------------------------------------
function report(label,crop,stats,L_max,K)
fprintf('\n  %s (crop %.3f)\n',label,crop);
fprintf('    band          '); fprintf('%9d',0:L_max); fprintf('\n');
fprintf('    ||u_l||_F     '); fprintf('%9.4f',stats.band_norm); fprintf('\n');
fprintf('    energy frac   '); fprintf('%9.4f',stats.band_energy); fprintf('\n');
fprintf('    participation '); fprintf('%9.4f',stats.participation); fprintf('\n');
fprintf('    kappa         '); fprintf('%9.1f',stats.kappa); fprintf('\n');
fprintf('    shell range   '); fprintf('%9.1f',stats.shell_range); fprintf('\n');
fprintf('    shell norms (k = 1..%d, inner to outer):\n      ',K);
fprintf('%8.4f',stats.shell_norm); fprintf('\n');

% The full (shell, band) matrix. Norms rather than energies, so a cell can be
% read against the band and shell totals above without squaring in your head;
% the energy share of each cell is the square, since the array is unit norm.
fprintf('\n    ||u[k, l]||  (rows: shell k, inner to outer; columns: band l)\n');
fprintf('      %-6s','k');
for l = 0:L_max, fprintf('%10s',sprintf('l=%d',l)); end
fprintf('%12s\n','row total');
for k = 1:K
    fprintf('      %-6d',k);
    fprintf('%10.5f',stats.cell_norm(k,:));
    fprintf('%12.5f\n',norm(stats.cell_norm(k,:)));
end
fprintf('      %-6s','tot');
fprintf('%10.5f',stats.band_norm);
fprintf('%12.5f\n',norm(stats.band_norm));
end
