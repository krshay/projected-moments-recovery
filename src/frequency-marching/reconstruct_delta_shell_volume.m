function volume = reconstruct_delta_shell_volume(a_standard,metadata)
%RECONSTRUCT_DELTA_SHELL_VOLUME Visualize shell coefficients on a 3-D grid.
%   Radial point samples are linearly interpolated only for visualization;
%   this interpolation is not used by the projected-moment calculation or
%   by frequency marching.

L_max = metadata.L_max;
grid_size = metadata.grid_size;
rho_max = metadata.rho_max;
rho_grid = metadata.rho_grid;
assert(size(a_standard,2) == (L_max+1)^2, ...
    'Coefficient array and metadata L_max disagree.');

a_native = a_standard;
if strcmp(metadata.native_reality_convention,'friedel')
    for l = 0:L_max
        for m = -l:l
            idx = sh_index(l,m);
            a_native(:,idx) = (-1)^m*(1i)^(-l)*a_standard(:,idx);
        end
    end
end

if mod(grid_size,2) == 0
    k_range = -grid_size/2:grid_size/2-1;
else
    k_range = -(grid_size-1)/2:(grid_size-1)/2;
end
[kx,ky,kz] = ndgrid(k_range,k_range,k_range);
k_nyquist = max(abs(k_range));
rho = sqrt(kx.^2+ky.^2+kz.^2)*(rho_max/k_nyquist);
rho_safe = max(rho,1e-10);
cos_theta = max(min(kz./rho_safe,1),-1);
phi = atan2(ky,kx);
inside = find(rho <= rho_max);
spectrum = zeros(grid_size,grid_size,grid_size);

chunk_size = 50000;
for first = 1:chunk_size:numel(inside)
    last = min(first+chunk_size-1,numel(inside));
    voxel_idx = inside(first:last);
    radial_coordinates = rho(voxel_idx);
    cos_chunk = cos_theta(voxel_idx);
    phi_chunk = phi(voxel_idx);
    values = zeros(numel(voxel_idx),1);
    for l = 0:L_max
        P = legendre(l,cos_chunk);
        for m = 0:l
            normalization = sqrt((2*l+1)/(4*pi)* ...
                factorial(l-m)/factorial(l+m));
            Y_positive = (-1)^m*normalization*P(m+1,:).' .* ...
                exp(1i*m*phi_chunk);
            positive = interp1(rho_grid,a_native(:,sh_index(l,m)), ...
                radial_coordinates,'linear','extrap');
            values = values+positive.*Y_positive;
            if m > 0
                Y_negative = (-1)^m*conj(Y_positive);
                negative = interp1(rho_grid,a_native(:,sh_index(l,-m)), ...
                    radial_coordinates,'linear','extrap');
                values = values+negative.*Y_negative;
            end
        end
    end
    spectrum(voxel_idx) = values;
end

volume = real(fftshift(ifftn(ifftshift(spectrum))));
volume = volume/max(norm(volume(:)),eps);
end

function idx = sh_index(l,m)
idx = l^2+m+l+1;
end
