function summary = export_bandlimited_volume(volume,L_max,shell_count,crop,grid_size)
%EXPORT_BANDLIMITED_VOLUME Write a molecule and its band-limited form as MRC.
%   EXPORT_BANDLIMITED_VOLUME(VOLUME,L_MAX,K,CROP) expands VOLUME in the
%   delta-shell basis, rebuilds it, and writes both the preprocessed original
%   and the band-limited version so they can be compared in ChimeraX. NO
%   RECOVERY IS INVOLVED: this is the ceiling a recovery at these settings
%   could reach, useful for deciding whether a given L, K and crop is worth
%   spending hours of optimizer time on.
%
%   Defaults are 80S at L = 13, K = 12, CROP = 1 (uncropped), grid 31.
%
%   ON CROPPING 80S. The 0.697 crop taken from krshay/Non-Uniform-Second-Moment
%   suits that repository's box, not this one: it retains 99.9% of TRPV1's
%   density but only 75.8% of 80S's, so on 80S it cuts the ribosome rather than
%   the margin. Measured at L = 13, correlation against the preprocessed map is
%   flat from crop 1 to crop 0.697 (0.5663, 0.5911, 0.5817, 0.5959, 0.5894,
%   0.5876 at crop 1, 0.95, 0.90, 0.85, 0.75, 0.697) -- non-monotonic, so the
%   spread is resampling jitter. Cropping 80S costs up to a quarter of the
%   molecule and buys nothing, and it lowered the recovery success rate from
%   20/20 to 11/20 by draining the free (l >= 2) bands. Default CROP = 1.
%
%   Both maps are written normalised to unit maximum so one ChimeraX contour
%   level means the same thing in each.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
addpath(fullfile(repo_root,'src','nonuniform'));

if nargin < 1 || isempty(volume),      volume = 'vol_80S_true.mrc'; end
if nargin < 2 || isempty(L_max),       L_max = 13; end
if nargin < 3 || isempty(shell_count), shell_count = 12; end
if nargin < 4 || isempty(crop),        crop = 1;   end
if nargin < 5 || isempty(grid_size),   grid_size = 31; end
K = shell_count;

volume_path = fullfile(repo_root,'data','raw',volume);
assert(exist(volume_path,'file') == 2,'Volume not found: %s',volume_path);

u = nu_expand_volume_shells(volume_path, ...
    struct('L_max',L_max,'grid_size',grid_size,'rho_max',pi, ...
           'shell_count',K,'mode','exact','crop',crop));
u = u/max(norm(u(:)),eps);

V_bandlimited = nu_shells_to_volume(u,L_max,K,grid_size);
V_original    = nu_load_volume(volume_path,grid_size,crop);

correlation = @(A,B) (A(:)'*B(:))/(norm(A(:))*norm(B(:)));
value = correlation(V_bandlimited,V_original);

output_dir = fullfile(repo_root,'results','gradient-descent','bandlimited');
if exist(output_dir,'dir') ~= 7, mkdir(output_dir); end
base = regexprep(volume,'\.[^.]*$','');
stem = sprintf('%s_L%d_K%d_crop%g',base,L_max,K,crop);
unit = @(V) V/max(V(:));
% The original depends only on the grid and the crop, not on L or K -- naming
% it after them produced three byte-identical files called ..._L5_...,
% ..._L8_... and ..._L13_..._original.mrc, which reads as three different
% volumes in a viewer's model list.
files = {sprintf('%s_grid%d_crop%g_original.mrc',base,grid_size,crop), V_original; ...
         sprintf('%s_bandlimited.mrc',stem),                           V_bandlimited};

fprintf('\n%s | L = %d, K = %d, crop = %g, grid %d\n', ...
    volume,L_max,K,crop,grid_size);
fprintf('  band-limited vs preprocessed original : %.4f\n',value);
for k = 1:size(files,1)
    nu_write_mrc(fullfile(output_dir,files{k,1}),unit(files{k,2}));
    fprintf('  wrote %s\n',files{k,1});
end

fprintf('\nView both (from the MATLAB prompt, note the leading !):\n');
fprintf('  !open -a /Applications/ChimeraX-1.10-rc2025.06.20.app %s %s\n', ...
    fullfile(output_dir,files{1,1}),fullfile(output_dir,files{2,1}));
fprintf('then in ChimeraX:\n');
fprintf('  volume all style surface ; volume all level 0.32 ; tile ; lighting soft\n');
fprintf('  volume #1 color lightblue ; volume #2 color khaki\n');

summary = struct('volume',volume,'L_max',L_max,'shell_count',K, ...
    'crop',crop,'grid_size',grid_size,'correlation',value, ...
    'files',{files(:,1)},'output_dir',output_dir);
end
