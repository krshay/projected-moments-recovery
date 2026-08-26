function [directions,Y,weights] = nu_sphere_quadrature(L_max,n_theta,n_phi)
%NU_SPHERE_QUADRATURE Gauss-Legendre x trapezoid grid with the SH matrix.
%   Y(i, gd_sh_index(l,m)) = Y_l^m at point i, in the SAME convention as
%   NU_BETA_TABLE (Condon-Shortley phase included, orthonormal on the sphere),
%   so coefficients produced here feed the moment model directly.
%   WEIGHTS integrate over the sphere: sum(WEIGHTS) = 4*pi.

[nodes,gauss_weights] = nu_gauss_legendre(n_theta);
theta = acos(nodes);
phi = 2*pi*(0:n_phi-1)'/n_phi;
n_points = n_theta*n_phi;
directions = zeros(n_points,3);
weights = zeros(n_points,1);
Y = zeros(n_points,(L_max+1)^2);
row = 0;
for j = 1:n_theta
    P = legendre(L_max,nodes(j));       % MATLAB includes Condon-Shortley
    all_P = cell(L_max+1,1);
    for l = 0:L_max
        all_P{l+1} = legendre(l,nodes(j));
    end
    for k = 1:n_phi
        row = row+1;
        directions(row,:) = [sin(theta(j))*cos(phi(k)), ...
                             sin(theta(j))*sin(phi(k)),cos(theta(j))];
        weights(row) = gauss_weights(j)*(2*pi/n_phi);
        for l = 0:L_max
            Pl = all_P{l+1};
            for m = 0:l
                norm_lm = sqrt((2*l+1)/(4*pi)*factorial(l-m)/factorial(l+m));
                value = norm_lm*Pl(m+1)*exp(1i*m*phi(k));
                Y(row,gd_sh_index(l,m)) = value;
                if m > 0
                    Y(row,gd_sh_index(l,-m)) = (-1)^m*conj(value);
                end
            end
        end
    end
end
end
