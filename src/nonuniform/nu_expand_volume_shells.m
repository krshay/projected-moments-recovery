function [a,available_shells,metadata] = nu_expand_volume_shells(volume_path,options)
%NU_EXPAND_VOLUME_SHELLS Direct Fourier-shell expansion. NO least squares.
%
%   3-D FFT, sample the spectrum on concentric spheres at the shell radii,
%   spherical-harmonic transform per shell to read off A_{l,m}(s). Every step
%   is an explicit quadrature, so the coefficients are a well-defined function
%   of the volume -- unlike expand_real_volume_fourier_bessel, whose
%   rank-truncated pseudo-inverse returns one arbitrary member of a solution
%   set that shifts with L_max.
%
%   SAMPLING MODE:
%     'exact' (default) evaluates the DFT directly at the sphere points,
%       F(q) = sum_x v(x) exp(-2*pi*i*q.x/N), done separably in three passes.
%       No interpolation error, so the only resolution dependence left is the
%       spherical quadrature itself.
%     'linear' uses interpn on the Cartesian FFT grid (what
%       expand_real_volume_delta_shells does). Cheaper, but interpolation
%       roughens the sphere function and makes the SH transform depend on the
%       angular grid.
%
%   The radial coordinate is the shell index, so the companion radial basis is
%   the identity: profiles ARE the coefficients.

if nargin < 2, options = struct(); end
L_max = get_option(options,'L_max',5);
grid_size = get_option(options,'grid_size',31);
rho_max = get_option(options,'rho_max',pi);
shell_count = get_option(options,'shell_count',12);
n_theta = get_option(options,'n_theta',max(2*L_max+8,24));
n_phi = get_option(options,'n_phi',max(4*L_max+8,48));
mode = get_option(options,'mode','exact');
crop = get_option(options,'crop',1);

assert(exist(volume_path,'file') == 2,'Volume not found: %s',volume_path);
volume = nu_load_volume(volume_path,grid_size,crop);

if mod(grid_size,2) == 0
    k_range = (-grid_size/2:grid_size/2-1)';
else
    k_range = (-(grid_size-1)/2:(grid_size-1)/2)';
end
k_nyquist = max(abs(k_range));
rho_grid = ((1:shell_count)-0.5)*(rho_max/shell_count);
[directions,Y,weights] = nu_sphere_quadrature(L_max,n_theta,n_phi);

if strcmpi(mode,'linear')
    spectrum = fftshift(fftn(ifftshift(volume)));
end

a_native = zeros(shell_count,(L_max+1)^2);
for shell = 1:shell_count
    query = directions*(rho_grid(shell)*k_nyquist/rho_max);
    switch lower(mode)
        case 'exact'
            samples = nu_nonuniform_dft(volume,k_range,query);
        case 'linear'
            samples = interpn(k_range,k_range,k_range,spectrum, ...
                query(:,1),query(:,2),query(:,3),'linear',0);
        otherwise
            error('Unknown sampling mode: %s',mode);
    end
    a_native(shell,:) = (Y'*(weights.*samples)).';
end

% The Fourier transform of a real volume obeys Friedel symmetry,
% A_{l,-m} = (-1)^(l+m) conj(A_{l,m}); multiplying band l by i^l converts that
% to the standard reality A_{l,-m} = (-1)^m conj(A_{l,m}) the charts assume.
a = a_native;
for l = 0:L_max
    a(:,gd_band_indices(l)) = (1i)^l*a_native(:,gd_band_indices(l));
end
defect = gd_reality_defect(a,L_max,shell_count*ones(1,L_max+1));
a = a/max(norm(a(:)),eps);
available_shells = shell_count*ones(1,L_max+1);

metadata = struct('volume_path',volume_path,'grid_size',grid_size, ...
    'L_max',L_max,'rho_max',rho_max,'basis_mode','fourier-shell', ...
    'shell_count',shell_count,'rho_grid',rho_grid, ...
    'n_theta',n_theta,'n_phi',n_phi,'sampling_mode',mode, ...
    'available_shells',available_shells,'standard_reality_defect',defect, ...
    'crop',crop, ...
    'reference_volume',volume);
end

function value = get_option(options,name,default)
if isfield(options,name), value = options.(name); else, value = default; end
end
