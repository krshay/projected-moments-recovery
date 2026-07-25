function volume = reconstruct_fourier_bessel_volume(a_standard,metadata)
%RECONSTRUCT_FOURIER_BESSEL_VOLUME Synthesize a volume from coefficients.
%   A_STANDARD uses a_{l,-m}=(-1)^m conj(a_{l,m}). The expansion basis may
%   natively use the Friedel convention; metadata records that convention
%   and this function applies the inverse conversion before synthesis.

required = {'L_max','grid_size','rho_max','available_shells', ...
    'bessel_zeros','native_reality_convention'};
for idx = 1:numel(required)
    assert(isfield(metadata,required{idx}), ...
        'Expansion metadata is missing field %s.',required{idx});
end

L_max = metadata.L_max;
L = metadata.grid_size;
rho_max = metadata.rho_max;
available_shells = metadata.available_shells;
assert(size(a_standard,2) == (L_max+1)^2, ...
    'Coefficient array and metadata L_max disagree.');

a_native = a_standard;
if strcmp(metadata.native_reality_convention,'friedel')
    for l = 0:L_max
        for m = -l:l
            coefficient_idx = sh_index(l,m);
            a_native(:,coefficient_idx) = (-1)^m*(1i)^(-l) * ...
                a_standard(:,coefficient_idx);
        end
    end
end

if mod(L,2) == 0
    k_range = -L/2:L/2-1;
else
    k_range = -(L-1)/2:(L-1)/2;
end
[kx,ky,kz] = ndgrid(k_range,k_range,k_range);
k_nyq = max(abs(k_range));
rho = sqrt(kx.^2+ky.^2+kz.^2)*(rho_max/k_nyq);
rho_safe = max(rho,1e-10);
cos_theta = max(min(kz./rho_safe,1-1e-12),-1+1e-12);
phi = atan2(ky,kx);
inside = find(rho <= rho_max);

spectrum = zeros(L,L,L);
chunk_size = 50000;
for first = 1:chunk_size:numel(inside)
    last = min(first+chunk_size-1,numel(inside));
    voxel_idx = inside(first:last);
    rho_chunk = rho(voxel_idx);
    cos_chunk = cos_theta(voxel_idx);
    phi_chunk = phi(voxel_idx);
    values = zeros(numel(voxel_idx),1);

    for l = 0:L_max
        P = legendre(l,cos_chunk);
        Y_positive = cell(l+1,1);
        for m = 0:l
            normalization = sqrt((2*l+1)/(4*pi) * ...
                factorial(l-m)/factorial(l+m));
            Y_positive{m+1} = (-1)^m*normalization*P(m+1,:).' .* ...
                exp(1i*m*phi_chunk);
        end

        for shell = 1:available_shells(l+1)
            zero = metadata.bessel_zeros(l+1,shell);
            radial = sqrt(2/rho_max^3)/abs(sph_bessel_j(l+1,zero)) .* ...
                sph_bessel_j(l,zero*rho_chunk/rho_max);
            for m = -l:l
                if m >= 0
                    Ylm = Y_positive{m+1};
                else
                    Ylm = (-1)^abs(m)*conj(Y_positive{abs(m)+1});
                end
                values = values + a_native(shell,sh_index(l,m))*radial.*Ylm;
            end
        end
    end
    spectrum(voxel_idx) = values;
end

volume = real(fftshift(ifftn(ifftshift(spectrum))));
volume = volume/max(norm(volume(:)),eps);
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

function idx = sh_index(l,m)
idx = l^2+m+l+1;
end
