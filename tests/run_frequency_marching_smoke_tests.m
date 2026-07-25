function run_frequency_marching_smoke_tests()
%RUN_FREQUENCY_MARCHING_SMOKE_TESTS Fast deterministic correctness checks.

test_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(test_dir);
addpath(fullfile(repo_root,'src','frequency-marching'));

S = [6,6,3,3];
diagnostic = check_decay_by_three(S);
assert(diagnostic.satisfies_decay_by_three);

a_true = generate_random_decay_signal(S,11);
result = frequency_marching_recovery(a_true,S,struct( ...
    'use_analytical_p3',true,'enable_backfill',true, ...
    'verify_analytical',false,'verbose',false,'rng_seed',11));
assert(result.global_relative_error < 1e-6, ...
    'Noiseless recovery error %.3e exceeds tolerance.', ...
    result.global_relative_error);
assert(height(result.condition_numbers) == 6);
assert(all(result.condition_numbers.reality_constrained_condition > 0));

% Exercise the empirical-moment/noise path without asserting statistical
% accuracy at this deliberately tiny sample count.
S_empirical = [6,3,3];
a_empirical = generate_random_decay_signal(S_empirical,13); %#ok<NASGU>
[~,empirical_result] = evalc([ ...
    'frequency_marching_recovery(a_empirical,S_empirical,struct(', ...
    '''use_analytical_p3'',false,''n_samples'',20,', ...
    '''noise_sigma'',0.05,''rng_seed'',13,', ...
    '''clean_sample_cache_key'',''smoke_empirical'',', ...
    '''clean_sample_rng_seed'',13,', ...
    '''clean_sample_cache_n_samples'',20,', ...
    '''verbose'',false))']);
assert(isfinite(empirical_result.global_relative_error));

% Exercise MAT loading, Fourier resizing, Fourier--Bessel least squares,
% and conversion to the standard spherical-harmonic reality convention.
grid = linspace(-1,1,15);
[x,y,z] = ndgrid(grid,grid,grid);
synthetic_volume = exp(-8*(x.^2+0.8*y.^2+1.2*z.^2));
temporary_mat = [tempname,'.mat'];
save(temporary_mat,'synthetic_volume');
cleanup = onCleanup(@() delete(temporary_mat));
[a_volume,volume_shells,volume_metadata] = expand_real_volume_fourier_bessel( ...
    temporary_mat,struct('L_max',2,'grid_size',15, ...
    'basis_mode','manual','manual_shells',[2,2,2]));
assert(isequal(volume_shells,[2,2,2]));
assert(all(isfinite(a_volume(:))));
reconstructed_volume = reconstruct_fourier_bessel_volume(a_volume,volume_metadata);
assert(isequal(size(reconstructed_volume),[15,15,15]));
assert(isreal(reconstructed_volume) && all(isfinite(reconstructed_volume(:))));

temporary_mrc = [tempname,'.mrc'];
mrc_cleanup = onCleanup(@() delete(temporary_mrc));
write_mrc_volume(temporary_mrc,reconstructed_volume);
fid = fopen(temporary_mrc,'r','l');
file_cleanup = onCleanup(@() fclose(fid));
dimensions = fread(fid,3,'int32').';
mode = fread(fid,1,'int32');
fseek(fid,208,'bof');
magic = char(fread(fid,4,'uint8=>uint8').');
assert(isequal(dimensions,[15,15,15]) && mode == 2 && strcmp(magic,'MAP '));
for l = 0:2
    for m = 1:l
        plus_idx = l^2+m+l+1;
        minus_idx = l^2-m+l+1;
        reality_error = norm(a_volume(:,minus_idx)- ...
            (-1)^m*conj(a_volume(:,plus_idx)));
        assert(reality_error < 1e-12);
    end
end

available = [8,8,8,8];
[a_decay,S_decay,before,after] = enforce_decay_by_three( ...
    generate_full_support_signal(available,12),available,3);
assert(~before.satisfies_decay_by_three);
assert(after.satisfies_decay_by_three);
for l = 0:numel(S_decay)-1
    idx = (l^2+1):(l+1)^2;
    tail = a_decay(S_decay(l+1)+1:end,idx);
    assert(all(abs(tail(:)) < eps));
end

fprintf('Frequency-marching smoke tests passed.\n');
end

function a = generate_full_support_signal(available,seed)
rng(seed);
L_max = numel(available)-1;
a = zeros(max(available),(L_max+1)^2);
for l = 0:L_max
    for r = 1:available(l+1)
        a(r,l^2+l+1) = randn();
        for m = 1:l
            idx_plus = l^2+m+l+1;
            idx_minus = l^2-m+l+1;
            z = (randn()+1i*randn())/sqrt(2);
            a(r,idx_plus) = z;
            a(r,idx_minus) = (-1)^m*conj(z);
        end
    end
end
end
