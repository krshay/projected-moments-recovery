function volume = nu_shells_to_volume(coefficients,L_max,shell_count,grid_size,rho_max)
%NU_SHELLS_TO_VOLUME Rebuild a 3-D volume from shell SH coefficients.
%   VOLUME = NU_SHELLS_TO_VOLUME(A,L_MAX,K,GRID_SIZE) evaluates the spherical-
%   harmonic expansion on the Cartesian frequency grid and inverse transforms,
%   inverting NU_EXPAND_VOLUME_SHELLS. Use it to look at a recovered array as a
%   map rather than as numbers.
%
%   THREE THINGS THAT MUST MATCH THE FORWARD TRANSFORM, each of which silently
%   produces a plausible but wrong volume if got wrong:
%     - the i^l twist NU_EXPAND_VOLUME_SHELLS applies to reach standard reality
%       is undone here by (-i)^l;
%     - the harmonics use NU_SPHERE_QUADRATURE's convention, N_lm P_l^m e^{imphi}
%       with MATLAB's LEGENDRE already carrying Condon-Shortley, so there is no
%       additional (-1)^m;
%     - the forward is fftshift(fftn(ifftshift(v))), so the inverse needs the
%       outer FFTSHIFT. Omitting it circularly shifts the volume by half the
%       grid, which shows up as a near-zero correlation with the original while
%       still looking like a plausible map.
%
%   Radially the coefficients are known only at the K shell radii, so values in
%   between are splined. Everything outside the outermost shell is zero: the
%   shells reach |k| = k_nyquist while the grid corners reach sqrt(3) times
%   further, so roughly half the grid's frequency voxels are unrepresented.
%   That is a property of a shell expansion, not of the recovery, and it limits
%   how close any reconstruction can look to the original map.

if nargin < 5 || isempty(rho_max), rho_max = pi; end

if mod(grid_size,2) == 0
    k_range = (-grid_size/2:grid_size/2-1);
else
    k_range = (-(grid_size-1)/2:(grid_size-1)/2);
end
k_nyquist = max(abs(k_range));
[KX,KY,KZ] = ndgrid(k_range,k_range,k_range);
radius = sqrt(KX.^2+KY.^2+KZ.^2);
rho = radius*(rho_max/k_nyquist);
rho_grid = ((1:shell_count)-0.5)*(rho_max/shell_count);

safe = max(radius,1e-12);
cos_theta = max(min(KZ./safe,1),-1);
phi = atan2(KY,KX);
inside = find(rho <= max(rho_grid));

spectrum = zeros(size(radius));
ct = cos_theta(inside); ph = phi(inside); rr = rho(inside);
for l = 0:L_max
    P = legendre(l,ct);
    for m = -l:l
        am = abs(m);
        radial = interp1(rho_grid,coefficients(:,gd_sh_index(l,m)),rr, ...
            'spline','extrap');
        normalization = sqrt((2*l+1)/(4*pi)* ...
            factorial(l-am)/factorial(l+am));
        harmonic = normalization*P(am+1,:).'.*exp(1i*am*ph);
        if m < 0, harmonic = (-1)^am*conj(harmonic); end
        spectrum(inside) = spectrum(inside)+((-1i)^l*radial).*harmonic;
    end
end
volume = real(fftshift(ifftn(ifftshift(spectrum))));
end
