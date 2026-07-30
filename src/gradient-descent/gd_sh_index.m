function idx = gd_sh_index(l,m)
%GD_SH_INDEX Column index for the spherical-harmonic coefficient (l,m).

idx = l^2+m+l+1;
end
