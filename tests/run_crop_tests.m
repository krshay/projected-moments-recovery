function run_crop_tests()
%RUN_CROP_TESTS The margin crop in NU_LOAD_VOLUME.
%   The crop was added to NU_LOAD_VOLUME rather than to the experiment scripts
%   for a reason worth stating: an earlier attempt reimplemented the
%   preprocessing alongside it -- Fourier resize instead of INTERPN linear,
%   n/2 instead of (n-1)/2 for the mask radius -- and produced an array 0.797
%   away from the pipeline's. The comparison it was meant to support was
%   meaningless. There is now exactly one implementation of the preprocessing,
%   and these tests pin the two properties that matter:
%
%     1. CROP = 1 and the two-argument call are BIT-IDENTICAL, so every result
%        produced before the crop existed is still reproducible.
%     2. A crop < 1 raises the fraction of the grid the molecule occupies,
%        which is the entire point of applying it.
%
%   Test 3 checks that the crop reaches the coefficients through
%   NU_EXPAND_VOLUME_SHELLS, since a silently ignored option would leave every
%   cropped experiment quietly running the uncropped problem.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src','nonuniform'));
addpath(fullfile(root,'src','gradient-descent'));

volume = fullfile(root,'data','raw','vol_80S_true.mrc');
if exist(volume,'file') ~= 2
    fprintf('SKIP: %s not present.\n',volume);
    return
end
grid_size = 31;

% ---- 1. crop = 1 changes nothing ----
baseline = nu_load_volume(volume,grid_size);
explicit = nu_load_volume(volume,grid_size,1);
difference = max(abs(baseline(:)-explicit(:)));
fprintf('1. crop=1 vs two-argument call : max|diff| %.3e\n',difference);
assert(difference == 0, ...
    ['crop=1 must be bit-identical to the two-argument call, else every ' ...
     'pre-crop result in the repository becomes irreproducible.']);

% ---- 2. cropping concentrates the molecule ----
cropped = nu_load_volume(volume,grid_size,0.697);
occupancy = @(V) nnz(V > 0.1*max(V(:)))/numel(V);
fprintf('2. occupancy: uncropped %.1f%%, crop 0.697 %.1f%%\n', ...
    100*occupancy(baseline),100*occupancy(cropped));
assert(occupancy(cropped) > occupancy(baseline), ...
    'The crop must raise occupancy; that is the reason for applying it.');
assert(isequal(size(cropped),size(baseline)), ...
    'Output grid size must be set by GRID_SIZE, not by the crop.');

% ---- 3. the option actually reaches the coefficients ----
options = struct('L_max',5,'grid_size',grid_size,'rho_max',pi, ...
    'shell_count',12,'mode','exact');
uncropped_coefficients = nu_expand_volume_shells(volume,options);
options.crop = 0.697;
cropped_coefficients = nu_expand_volume_shells(volume,options);
unit = @(A) A/max(norm(A(:)),eps);
shift = norm(unit(cropped_coefficients)-unit(uncropped_coefficients),'fro');
fprintf('3. coefficient shift from the crop : %.4f\n',shift);
assert(shift > 1e-6, ...
    ['NU_EXPAND_VOLUME_SHELLS ignored the crop option; every cropped ' ...
     'experiment would silently run the uncropped problem.']);

% ---- 4. an explicit index range is accepted ----
% The reference repository crops 50:300 of a 360^3 map. The fraction form
% keeps the same WIDTH (251/360 = 0.697) but centres it, so it takes 55:305 --
% the same amount of margin removed, shifted by five voxels. The two are
% therefore close but not identical; use the range form to match exactly.
by_range = nu_load_volume(volume,grid_size,[50 300]);
fprintf('4. [50 300] vs centred fraction 0.697 : max|diff| %.3e\n', ...
    max(abs(by_range(:)-cropped(:))));
assert(isequal(size(by_range),size(cropped)), ...
    'Both crop forms must produce the same output grid.');

fprintf('\nAll crop tests passed.\n');
end
