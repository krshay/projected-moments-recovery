function [a, available_shells, metadata] = ...
    expand_real_volume_fourier_bessel(volume_path, options)
%EXPAND_REAL_VOLUME_FOURIER_BESSEL Expand a MAT/MRC volume without toolboxes.
%   This follows the complex spherical-harmonic and Fourier--Bessel
%   conventions used by gd_recovery_real_volume_standalone.m. The returned
%   array has standard reality a_{l,-m}=(-1)^m conj(a_{l,m}).
%
%   WARNING -- NOT A WELL-DEFINED FUNCTION OF THE VOLUME.
%   PROJECT_VOLUME below fits all (l,s,m) coefficients jointly by least
%   squares through the normal equations B'*B, then inverts with a truncated
%   pseudo-inverse at a 1e-8 relative eigenvalue cutoff. For TRPV1 at
%   L_max = 5 with 12 shells that is 432 columns at cond(B) ~ 3e9 and
%   cond(B'*B) ~ 9e18, of which only 407 survive the cutoff. The returned
%   array is therefore the minimum-norm member of a 25-dimensional affine
%   family of equally good fits, and which member you get depends on the
%   fitting parameters rather than on the molecule: changing the shell count
%   from 12 to 14 with the volume held FIXED moves the resulting delta-shell
%   array by 1.36 relative (i.e. the two are essentially unrelated).
%
%   Do not use this for new work on real volumes. Use
%   NU_EXPAND_VOLUME_SHELLS (src/nonuniform) instead, which extracts each
%   shell by an independent spherical-quadrature projection with no solve,
%   and reproduces to 4.1e-17 under radial refinement. See
%   tests/run_ndft_expansion_tests.m, which measures both.
%
%   This function is retained because the frequency-marching code genuinely
%   needs coefficients in the Fourier--Bessel basis, which the shell
%   expansion does not produce. Whether the instability above affects those
%   results has not been investigated.

if nargin < 2
    options = struct();
end
L_max = get_option(options, 'L_max', 7);
grid_size = get_option(options, 'grid_size', 47);
rho_max = get_option(options, 'rho_max', pi);
basis_mode = get_option(options, 'basis_mode', 'kc');
manual_shells = get_option(options, 'manual_shells', []);

assert(exist(volume_path, 'file') == 2, 'Volume not found: %s', volume_path);
vol = load_volume(volume_path);
assert(ndims(vol) == 3 && all(size(vol) > 1), 'Volume must be a 3-D array.');
vol = double(real(vol));
vol = crop_centered_cube(vol);
vol = mask_unit_ball(vol);
vol = fourier_resize_3d(vol, grid_size);
vol = vol / max(norm(vol(:)), eps);

shells_per_band = cell(1, L_max+1);
switch lower(basis_mode)
    case 'kc'
        k_cap = pi * grid_size / 2;
        for l = 0:L_max
            s = 1;
            while sph_bessel_zero(l, s) <= k_cap
                s = s + 1;
            end
            shells_per_band{l+1} = 1:(s-1);
        end
    case 'manual'
        assert(numel(manual_shells) == L_max+1, ...
            'manual_shells must have L_max+1 entries.');
        for l = 0:L_max
            shells_per_band{l+1} = 1:manual_shells(l+1);
        end
    otherwise
        error('Unknown basis_mode: %s', basis_mode);
end
available_shells = cellfun(@numel, shells_per_band);
assert(all(available_shells > 0), 'Every angular band needs a radial shell.');

max_shell = max(available_shells);
bessel_zeros = zeros(L_max+1, max_shell);
for l = 0:L_max
    for s = shells_per_band{l+1}
        bessel_zeros(l+1, s) = sph_bessel_zero(l, s);
    end
end

M_tot = sum(available_shells);
shell_band = zeros(M_tot, 1);
shell_within = zeros(M_tot, 1);
k = 0;
for l = 0:L_max
    for s = shells_per_band{l+1}
        k = k + 1;
        shell_band(k) = l;
        shell_within(k) = s;
    end
end

a_flat = project_volume(vol, shell_band, shell_within, ...
    bessel_zeros, L_max, rho_max);
a = zeros(max_shell, (L_max+1)^2);
shift = L_max;
for k = 1:M_tot
    l = shell_band(k);
    s = shell_within(k);
    for m = -l:l
        a(s, sh_index(l,m)) = a_flat(k, m+shift+1);
    end
end
[a, native_convention, reality_defect] = enforce_standard_reality(a, L_max);
a = a / max(norm(a(:)), eps);

metadata = struct();
metadata.volume_path = volume_path;
metadata.grid_size = grid_size;
metadata.L_max = L_max;
metadata.rho_max = rho_max;
metadata.basis_mode = basis_mode;
metadata.available_shells = available_shells;
metadata.bessel_zeros = bessel_zeros;
metadata.native_reality_convention = native_convention;
metadata.native_reality_defect = reality_defect;
metadata.reference_volume = vol;
end

function a_out = project_volume(vol, shell_band, shell_within, ...
    bessel_zeros, L_max, rho_max)
L = size(vol, 1);
M_tot = length(shell_band);
shift = L_max;
vol_hat = fftshift(fftn(ifftshift(vol)));

if mod(L,2) == 0
    k_range = -L/2:L/2-1;
else
    k_range = -(L-1)/2:(L-1)/2;
end
[kx,ky,kz] = ndgrid(k_range,k_range,k_range);
k_nyq = max(abs(k_range));
rho = sqrt(kx.^2+ky.^2+kz.^2) * (rho_max/k_nyq);
rho_safe = max(rho, 1e-10);
cos_theta = max(min(kz./rho_safe, 1-1e-12), -1+1e-12);
phi = atan2(ky,kx);
inside = find(rho <= rho_max & rho > 1e-10);

n_cols = sum(2*shell_band+1);
col_to_km = zeros(n_cols,2);
col = 0;
for k = 1:M_tot
    for m = -shell_band(k):shell_band(k)
        col = col+1;
        col_to_km(col,:) = [k,m+shift+1];
    end
end

target = vol_hat(inside);
chunk_size = min(50000, max(1000, floor(2e7/max(n_cols,1))));
BtB = zeros(n_cols,n_cols);
Bty = zeros(n_cols,1);
for first = 1:chunk_size:numel(inside)
    last = min(first+chunk_size-1,numel(inside));
    idx = inside(first:last);
    rho_c = rho(idx);
    cos_c = cos_theta(idx);
    phi_c = phi(idx);

    Y = cell(L_max+1,1);
    for l = 0:L_max
        P = legendre(l,cos_c);
        Y{l+1} = cell(l+1,1);
        for m = 0:l
            Nlm = sqrt((2*l+1)/(4*pi)*factorial(l-m)/factorial(l+m));
            Y{l+1}{m+1} = (-1)^m*Nlm*P(m+1,:).'.*exp(1i*m*phi_c);
        end
    end

    B = zeros(numel(idx),n_cols);
    col = 0;
    for k = 1:M_tot
        l = shell_band(k);
        s = shell_within(k);
        zero = bessel_zeros(l+1,s);
        radial = sqrt(2/rho_max^3)/abs(sph_bessel_j(l+1,zero)) .* ...
            sph_bessel_j(l,zero*rho_c/rho_max);
        for m = -l:l
            col = col+1;
            if m >= 0
                Ylm = Y{l+1}{m+1};
            else
                Ylm = (-1)^abs(m)*conj(Y{l+1}{abs(m)+1});
            end
            B(:,col) = radial.*Ylm;
        end
    end
    BtB = BtB+B'*B;
    Bty = Bty+B'*target(first:last);
end

[V,D] = eig((BtB+BtB')/2);
[eigenvalues,perm] = sort(real(diag(D)),'descend');
V = V(:,perm);
rank_tol = max(eigenvalues)*1e-8;
rank_est = sum(eigenvalues > rank_tol);
assert(rank_est > 0, 'Fourier--Bessel design matrix is numerically singular.');
coeff = V(:,1:rank_est)*((V(:,1:rank_est)'*Bty)./eigenvalues(1:rank_est));

a_out = zeros(M_tot,2*L_max+1);
for c = 1:n_cols
    a_out(col_to_km(c,1),col_to_km(c,2)) = coeff(c);
end
end

function [a_std, native, defect] = enforce_standard_reality(a, L_max)
[standard_defect,energy] = convention_defect(a,L_max,false);
[friedel_defect,~] = convention_defect(a,L_max,true);
standard_relative = sqrt(standard_defect/max(energy,eps));
friedel_relative = sqrt(friedel_defect/max(energy,eps));
use_friedel = friedel_relative < standard_relative;
if use_friedel
    native = 'friedel';
    defect = friedel_relative;
else
    native = 'standard';
    defect = standard_relative;
end
assert(defect <= 0.25, ...
    'Expansion is %.1f%% from the nearest real-volume convention.',100*defect);

a_std = a;
for l = 0:L_max
    parity = 1;
    if use_friedel
        parity = (-1)^l;
    end
    for m = 1:l
        sign_pair = parity*(-1)^m;
        ap = a(:,sh_index(l,m));
        am = a(:,sh_index(l,-m));
        ap = (ap+sign_pair*conj(am))/2;
        a_std(:,sh_index(l,m)) = ap;
        a_std(:,sh_index(l,-m)) = sign_pair*conj(ap);
    end
    a0 = a(:,sh_index(l,0));
    if parity > 0
        a_std(:,sh_index(l,0)) = real(a0);
    else
        a_std(:,sh_index(l,0)) = 1i*imag(a0);
    end
end
if use_friedel
    for l = 0:L_max
        for m = -l:l
            idx = sh_index(l,m);
            a_std(:,idx) = 1i^l*(-1)^m*a_std(:,idx);
        end
    end
end
end

function [defect,energy] = convention_defect(a,L_max,use_friedel)
defect = 0;
energy = 0;
for l = 0:L_max
    parity = 1;
    if use_friedel
        parity = (-1)^l;
    end
    for m = 1:l
        ap = a(:,sh_index(l,m));
        am = a(:,sh_index(l,-m));
        defect = defect+norm(am-parity*(-1)^m*conj(ap))^2;
        energy = energy+norm(ap)^2+norm(am)^2;
    end
    a0 = a(:,sh_index(l,0));
    if parity > 0
        defect = defect+norm(imag(a0))^2;
    else
        defect = defect+norm(real(a0))^2;
    end
    energy = energy+norm(a0)^2;
end
end

function vol = load_volume(path)
[~,~,extension] = fileparts(path);
if strcmpi(extension,'.mat')
    loaded = load(path);
    fields = fieldnames(loaded);
    best_size = 0;
    vol = [];
    for idx = 1:numel(fields)
        candidate = loaded.(fields{idx});
        if isnumeric(candidate) && ndims(candidate) == 3 && numel(candidate) > best_size
            vol = candidate;
            best_size = numel(candidate);
        end
    end
    assert(~isempty(vol),'No numeric 3-D array found in %s.',path);
else
    vol = load_mrc(path);
end
end

function vol = load_mrc(path)
fid = fopen(path,'r','l');
assert(fid >= 0,'Cannot open %s.',path);
cleanup = onCleanup(@() fclose(fid));
NX = fread(fid,1,'int32'); NY = fread(fid,1,'int32');
NZ = fread(fid,1,'int32'); mode = fread(fid,1,'int32');
assert(all([NX,NY,NZ] > 0),'Invalid MRC dimensions in %s.',path);
fseek(fid,64,'bof');
mapc = fread(fid,1,'int32'); mapr = fread(fid,1,'int32'); maps = fread(fid,1,'int32');
fseek(fid,92,'bof'); nsymbt = fread(fid,1,'int32');
fseek(fid,208,'bof'); magic = char(fread(fid,4,'uint8=>uint8').');
assert(strcmp(magic,'MAP '),'Invalid MRC header in %s.',path);
fseek(fid,1024+nsymbt,'bof');
types = {'int8','int16','float32','','','','uint16'};
assert(ismember(mode,[0,1,2,6]),'Unsupported MRC mode %d.',mode);
raw = fread(fid,NX*NY*NZ,['*' types{mode+1}]);
vol = double(reshape(raw,[NX,NY,NZ]));
if ~isequal([mapc,mapr,maps],[1,2,3])
    [~,inverse_permutation] = sort([mapc,mapr,maps]);
    vol = permute(vol,inverse_permutation);
end
end

function vol = crop_centered_cube(vol)
side = min(size(vol));
first = floor((size(vol)-side)/2)+1;
vol = vol(first(1):first(1)+side-1, ...
    first(2):first(2)+side-1, first(3):first(3)+side-1);
end

function vol = mask_unit_ball(vol)
L = size(vol,1);
coordinates = (0:L-1)-(L-1)/2;
[x,y,z] = ndgrid(coordinates,coordinates,coordinates);
mask = x.^2+y.^2+z.^2 <= (L/2)^2;
vol(~mask) = 0;
end

function resized = fourier_resize_3d(vol,out_size)
if isscalar(out_size)
    out_size = out_size*[1,1,1];
end
spectrum = fftshift(fftn(vol));
in_size = size(vol);
resized_spectrum = zeros(out_size);
copy_size = min(in_size,out_size);
in_first = floor((in_size-copy_size)/2)+1;
out_first = floor((out_size-copy_size)/2)+1;
resized_spectrum(out_first(1):out_first(1)+copy_size(1)-1, ...
    out_first(2):out_first(2)+copy_size(2)-1, ...
    out_first(3):out_first(3)+copy_size(3)-1) = ...
    spectrum(in_first(1):in_first(1)+copy_size(1)-1, ...
    in_first(2):in_first(2)+copy_size(2)-1, ...
    in_first(3):in_first(3)+copy_size(3)-1);
resized = real(ifftn(ifftshift(resized_spectrum))) * ...
    (prod(out_size)/prod(in_size));
end

function value = sph_bessel_j(l,x)
value = zeros(size(x));
nonzero = abs(x) > 1e-12;
if l == 0
    value(nonzero) = sin(x(nonzero))./x(nonzero);
    value(~nonzero) = 1;
else
    value(nonzero) = sqrt(pi./(2*x(nonzero))).*besselj(l+0.5,x(nonzero));
end
end

function root = sph_bessel_zero(l,s)
if l == 0
    root = s*pi;
    return;
end
estimate = max(pi*(s+l/2)-4*l*(l+1)/(8*pi*(s+l/2)),0.5);
left = max(estimate-pi/2,0.05);
right = estimate+pi/2;
for expansion = 1:8
    if sph_bessel_j(l,left)*sph_bessel_j(l,right) <= 0
        root = fzero(@(x) sph_bessel_j(l,x),[left,right]);
        return;
    end
    left = max(estimate-expansion*pi/2,0.05);
    right = estimate+expansion*pi/2;
end
grid = linspace(0.05,(s+l+5)*pi,max(4000,400*s));
values = sph_bessel_j(l,grid);
changes = find(diff(sign(values)) ~= 0);
assert(numel(changes) >= s,'Could not bracket Bessel zero l=%d,s=%d.',l,s);
root = fzero(@(x) sph_bessel_j(l,x),grid(changes(s):changes(s)+1));
end

function idx = sh_index(l,m)
idx = l^2+m+l+1;
end

function value = get_option(options,name,default_value)
if isfield(options,name)
    value = options.(name);
else
    value = default_value;
end
end
