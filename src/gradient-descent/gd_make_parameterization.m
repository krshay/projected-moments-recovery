function parameterization = gd_make_parameterization(L_max,support,first_band)
%GD_MAKE_PARAMETERIZATION Real degrees of freedom in bands FIRST_BAND,...,L_MAX.
%   Rows are [radial_index, l, m, component], where component is 0 for the
%   real m=0 coefficient, 1 for real(a_lm), and 2 for imag(a_lm), m>0.
%
%   FIRST_BAND defaults to 2, the historical convention in which bands 0 and 1
%   are pinned to truth. Pass 0 to optimise every band.

assert(numel(support) == L_max+1, ...
    'support must contain one radial support size per angular band.');
if nargin < 3 || isempty(first_band), first_band = 2; end
bands = first_band:L_max;
rows = zeros(sum((2*bands+1).*support(bands+1)),4);
next = 0;
for l = bands
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
