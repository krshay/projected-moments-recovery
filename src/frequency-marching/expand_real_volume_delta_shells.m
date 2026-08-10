function [a, available_shells, metadata] = ...
    expand_real_volume_delta_shells(volume_path, options)
%EXPAND_REAL_VOLUME_DELTA_SHELLS Sample a molecular Fourier transform on shells.
%   A(r,(l,m)) contains the spherical-harmonic coefficients of the 3-D
%   Fourier transform restricted to radius RHO_GRID(r). Unlike the
%   Fourier--Bessel expansion, the radial coordinate is represented by
%   point samples (delta shells), exactly matching the radial model used by
%   frequency_marching_recovery.

if nargin < 2
    options = struct();
end
L_max = get_option(options,'L_max',5);
grid_size = get_option(options,'grid_size',31);
rho_max = get_option(options,'rho_max',pi);
shell_count = get_option(options,'shell_count',12);
n_theta = get_option(options,'n_theta',max(16,2*L_max+4));
n_phi = get_option(options,'n_phi',max(32,4*L_max+4));

assert(exist(volume_path,'file') == 2,'Volume not found: %s',volume_path);
validateattributes(shell_count,{'numeric'},{'scalar','integer','positive'});
validateattributes(n_theta,{'numeric'},{'scalar','integer','>=',L_max+1});
validateattributes(n_phi,{'numeric'},{'scalar','integer','>=',2*L_max+1});

volume = load_and_preprocess_volume(volume_path,grid_size);
spectrum = fftshift(fftn(ifftshift(volume)));

% Midpoint shells avoid both the directionally degenerate origin and the
% interpolation boundary at the Cartesian Nyquist radius.
rho_grid = ((1:shell_count)-0.5)*(rho_max/shell_count);
[directions,Y,quadrature_weights] = spherical_quadrature(L_max,n_theta,n_phi);

if mod(grid_size,2) == 0
    k_range = -grid_size/2:grid_size/2-1;
else
    k_range = -(grid_size-1)/2:(grid_size-1)/2;
end
k_nyquist = max(abs(k_range));
a_native = zeros(shell_count,(L_max+1)^2);
for shell = 1:shell_count
    query = directions*(rho_grid(shell)*k_nyquist/rho_max);
    samples = interpn(k_range,k_range,k_range,spectrum, ...
        query(:,1),query(:,2),query(:,3),'linear',0);
    coefficients = Y'*(quadrature_weights.*samples);
    a_native(shell,:) = coefficients.';
end

[a,native_convention,reality_defect] = ...
    enforce_standard_reality(a_native,L_max);
a = a/max(norm(a(:)),eps);
available_shells = shell_count*ones(1,L_max+1);

metadata = struct();
metadata.volume_path = volume_path;
metadata.grid_size = grid_size;
metadata.L_max = L_max;
metadata.rho_max = rho_max;
metadata.basis_mode = 'delta-shell';
metadata.shell_count = shell_count;
metadata.rho_grid = rho_grid;
metadata.available_shells = available_shells;
metadata.n_theta = n_theta;
metadata.n_phi = n_phi;
metadata.native_reality_convention = native_convention;
metadata.native_reality_defect = reality_defect;
metadata.reference_volume = volume;
end

function [directions,Y,weights] = spherical_quadrature(L_max,n_theta,n_phi)
[mu,w_mu] = gauss_legendre(n_theta);
phi = (0:n_phi-1)*(2*pi/n_phi);
[mu_grid,phi_grid] = ndgrid(mu,phi);
sin_theta = sqrt(max(0,1-mu_grid.^2));
directions = [sin_theta(:).*cos(phi_grid(:)), ...
    sin_theta(:).*sin(phi_grid(:)),mu_grid(:)];
weights = repmat(w_mu(:),1,n_phi)*(2*pi/n_phi);
weights = weights(:);

Y = zeros(numel(mu_grid),(L_max+1)^2);
for l = 0:L_max
    P = legendre(l,mu.');
    for m = 0:l
        normalization = sqrt((2*l+1)/(4*pi)* ...
            factorial(l-m)/factorial(l+m));
        angular = (-1)^m*normalization* ...
            repmat(P(m+1,:).',1,n_phi).*exp(1i*m*phi_grid);
        Y(:,sh_index(l,m)) = angular(:);
        if m > 0
            Y(:,sh_index(l,-m)) = (-1)^m*conj(angular(:));
        end
    end
end
end

function [nodes,weights] = gauss_legendre(order)
off_diagonal = 0.5./sqrt(1-(2*(1:order-1)).^(-2));
jacobi = diag(off_diagonal,1)+diag(off_diagonal,-1);
[vectors,values] = eig(jacobi,'vector');
[nodes,permutation] = sort(values);
vectors = vectors(:,permutation);
weights = 2*(vectors(1,:).^2).';
end

function [a_standard,native,defect] = enforce_standard_reality(a,L_max)
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
    'Shell expansion is %.1f%% from the nearest real-volume convention.',100*defect);

a_standard = a;
for l = 0:L_max
    parity = 1;
    if use_friedel
        parity = (-1)^l;
    end
    for m = 1:l
        pair_sign = parity*(-1)^m;
        positive = a(:,sh_index(l,m));
        negative = a(:,sh_index(l,-m));
        positive = (positive+pair_sign*conj(negative))/2;
        a_standard(:,sh_index(l,m)) = positive;
        a_standard(:,sh_index(l,-m)) = pair_sign*conj(positive);
    end
    zero_order = a(:,sh_index(l,0));
    if parity > 0
        a_standard(:,sh_index(l,0)) = real(zero_order);
    else
        a_standard(:,sh_index(l,0)) = 1i*imag(zero_order);
    end
end
if use_friedel
    for l = 0:L_max
        for m = -l:l
            idx = sh_index(l,m);
            a_standard(:,idx) = 1i^l*(-1)^m*a_standard(:,idx);
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
        positive = a(:,sh_index(l,m));
        negative = a(:,sh_index(l,-m));
        defect = defect+norm(negative-parity*(-1)^m*conj(positive))^2;
        energy = energy+norm(positive)^2+norm(negative)^2;
    end
    zero_order = a(:,sh_index(l,0));
    if parity > 0
        defect = defect+norm(imag(zero_order))^2;
    else
        defect = defect+norm(real(zero_order))^2;
    end
    energy = energy+norm(zero_order)^2;
end
end

function volume = load_and_preprocess_volume(path,grid_size)
[~,~,extension] = fileparts(path);
if strcmpi(extension,'.mat')
    loaded = load(path);
    fields = fieldnames(loaded);
    volume = [];
    best_size = 0;
    for idx = 1:numel(fields)
        candidate = loaded.(fields{idx});
        if isnumeric(candidate) && ndims(candidate) == 3 && ...
                numel(candidate) > best_size
            volume = candidate;
            best_size = numel(candidate);
        end
    end
    assert(~isempty(volume),'No numeric 3-D array found in %s.',path);
else
    volume = load_mrc(path);
end
volume = double(real(volume));
side = min(size(volume));
first = floor((size(volume)-side)/2)+1;
volume = volume(first(1):first(1)+side-1, ...
    first(2):first(2)+side-1,first(3):first(3)+side-1);

coordinates = (0:side-1)-(side-1)/2;
[x,y,z] = ndgrid(coordinates,coordinates,coordinates);
volume(x.^2+y.^2+z.^2 > (side/2)^2) = 0;
volume = fourier_resize_3d(volume,grid_size);
volume = volume/max(norm(volume(:)),eps);
end

function volume = load_mrc(path)
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
volume = double(reshape(raw,[NX,NY,NZ]));
if ~isequal([mapc,mapr,maps],[1,2,3])
    [~,inverse_permutation] = sort([mapc,mapr,maps]);
    volume = permute(volume,inverse_permutation);
end
end

function resized = fourier_resize_3d(volume,out_size)
if isscalar(out_size)
    out_size = out_size*[1,1,1];
end
spectrum = fftshift(fftn(volume));
in_size = size(volume);
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
resized = real(ifftn(ifftshift(resized_spectrum)))* ...
    (prod(out_size)/prod(in_size));
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
