function defect = gd_reality_defect(coefficients,L_max,support)
%GD_REALITY_DEFECT Relative standard spherical-harmonic reality violation.

if nargin < 3
    support = size(coefficients,1)*ones(1,L_max+1);
end
residual_energy = 0;
signal_energy = 0;
for l = 0:L_max
    rows = 1:support(l+1);
    a0 = coefficients(rows,gd_sh_index(l,0));
    residual_energy = residual_energy+norm(imag(a0))^2;
    signal_energy = signal_energy+norm(a0)^2;
    for m = 1:l
        plus = coefficients(rows,gd_sh_index(l,m));
        minus = coefficients(rows,gd_sh_index(l,-m));
        residual_energy = residual_energy+ ...
            norm(minus-(-1)^m*conj(plus))^2;
        signal_energy = signal_energy+norm(plus)^2+norm(minus)^2;
    end
end
defect = sqrt(residual_energy/max(signal_energy,eps));
end
