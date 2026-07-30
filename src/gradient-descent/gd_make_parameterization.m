function parameterization = gd_make_parameterization(L_max,support)
%GD_MAKE_PARAMETERIZATION Real degrees of freedom in bands 2,...,L_MAX.
%   Rows are [radial_index, l, m, component], where component is 0 for the
%   real m=0 coefficient, 1 for real(a_lm), and 2 for imag(a_lm), m>0.

assert(numel(support) == L_max+1, ...
    'support must contain one radial support size per angular band.');
rows = zeros(sum((2*(2:L_max)+1).*support(3:end)),4);
next = 0;
for l = 2:L_max
    for radial_index = 1:support(l+1)
        next = next+1;
        rows(next,:) = [radial_index,l,0,0];
        for m = 1:l
            rows(next+1,:) = [radial_index,l,m,1];
            rows(next+2,:) = [radial_index,l,m,2];
            next = next+2;
        end
    end
end
parameterization = rows(1:next,:);
end
