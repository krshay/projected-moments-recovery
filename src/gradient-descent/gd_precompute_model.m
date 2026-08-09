function model = gd_precompute_model(L_max)
%GD_PRECOMPUTE_MODEL Build Josh's parity-pruned projected-P3 operator.
%   The Clebsch--Gordan, equatorial Legendre, and spherical-harmonic
%   normalization conventions are unchanged from gd_third_moment_recovery.m.

cg = precompute_cg_table(L_max);
legendre_zero = precompute_legendre_zero(L_max);
normalization = precompute_normalization(L_max);
shift = L_max;

terms = struct('l1',{},'l2',{},'l3',{},'i1',{},'i2',{},'i3',{},'c',{});
for l1 = 0:L_max
    for l2 = 0:L_max
        for l3 = abs(l1-l2):min(l1+l2,L_max)
            if mod(l1+l2+l3,2)
                continue;
            end
            i1 = [];
            i2 = [];
            i3 = [];
            coefficients = [];
            for k1 = -l1:l1
                for k2 = -l2:l2
                    k3 = -k1-k2;
                    if abs(k3) > l3
                        continue;
                    end
                    value = cg(l1+1,k1+shift+1,l2+1,k2+shift+1,l3+1);
                    if abs(value) < 1e-14
                        continue;
                    end
                    i1(end+1) = gd_sh_index(l1,k1); %#ok<AGROW>
                    i2(end+1) = gd_sh_index(l2,k2); %#ok<AGROW>
                    i3(end+1) = gd_sh_index(l3,k3); %#ok<AGROW>
                    coefficients(end+1) = (-1)^k1*value/(2*l1+1); %#ok<AGROW>
                end
            end
            if ~isempty(coefficients)
                terms(end+1) = struct('l1',l1,'l2',l2,'l3',l3, ... %#ok<AGROW>
                    'i1',i1,'i2',i2,'i3',i3,'c',coefficients);
            end
        end
    end
end

blocks = struct('m1',{},'m2',{},'m3',{},'term_indices',{},'weights',{});
for m1 = -L_max:L_max
    for m2 = -L_max:L_max
        m3 = -m1-m2;
        if abs(m3) > L_max
            continue;
        end
        term_indices = [];
        weights = [];
        for term_index = 1:numel(terms)
            term = terms(term_index);
            if term.l1 < abs(m1) || term.l2 < abs(m2) || term.l3 < abs(m3)
                continue;
            end
            value = cg(term.l1+1,m1+shift+1, ...
                term.l2+1,m2+shift+1,term.l3+1);
            if abs(value) < 1e-14
                continue;
            end
            weight = (-1)^m1* ...
                normalization(term.l1+1,m1+shift+1)* ...
                normalization(term.l2+1,m2+shift+1)* ...
                normalization(term.l3+1,m3+shift+1)* ...
                legendre_zero(term.l1+1,m1+shift+1)* ...
                legendre_zero(term.l2+1,m2+shift+1)* ...
                legendre_zero(term.l3+1,m3+shift+1)*value;
            if abs(weight) > 1e-16
                term_indices(end+1) = term_index; %#ok<AGROW>
                weights(end+1) = weight; %#ok<AGROW>
            end
        end
        if ~isempty(term_indices)
            blocks(end+1) = struct('m1',m1,'m2',m2,'m3',m3, ... %#ok<AGROW>
                'term_indices',term_indices,'weights',weights);
        end
    end
end

model = struct('L_max',L_max,'terms',terms,'blocks',blocks, ...
    'clebsch_gordan',cg,'legendre_zero',legendre_zero, ...
    'normalization',normalization, ...
    'convention','Josh projected-third-moment convention; even parity only');
end

function value = clebsch_gordan(j1,m1,j2,m2,j3,m3)
if m1+m2 ~= m3 || j3 < abs(j1-j2) || j3 > j1+j2
    value = 0;
    return;
end
t_min = max([0,j2-m1-j3,j1+m2-j3]);
t_max = min([j1+j2-j3,j1-m1,j2+m2]);
series = 0;
for t = t_min:t_max
    series = series+(-1)^t/(factorial(t)*factorial(j1+j2-j3-t)* ...
        factorial(j1-m1-t)*factorial(j2+m2-t)* ...
        factorial(j3-j2+m1+t)*factorial(j3-j1-m2+t));
end
prefactor = sqrt((2*j3+1)*factorial(j1+j2-j3)* ...
    factorial(j1-j2+j3)*factorial(-j1+j2+j3)/ ...
    factorial(j1+j2+j3+1)*factorial(j3+m3)*factorial(j3-m3)* ...
    factorial(j1+m1)*factorial(j1-m1)* ...
    factorial(j2+m2)*factorial(j2-m2));
value = prefactor*series;
end

function table = precompute_cg_table(L_max)
shift = L_max;
table = zeros(L_max+1,2*shift+1,L_max+1,2*shift+1,L_max+1);
for l1 = 0:L_max
    for m1 = -l1:l1
        for l2 = 0:L_max
            for m2 = -l2:l2
                for l3 = abs(l1-l2):min(l1+l2,L_max)
                    m3 = -m1-m2;
                    if abs(m3) <= l3
                        table(l1+1,m1+shift+1,l2+1,m2+shift+1,l3+1) = ...
                            clebsch_gordan(l2,m2,l3,m3,l1,-m1);
                    end
                end
            end
        end
    end
end
end

function table = precompute_legendre_zero(L_max)
shift = L_max;
table = zeros(L_max+1,2*shift+1);
for l = 0:L_max
    for m = 0:l
        if mod(l-m,2) ~= 0
            value = 0;
        else
            value = (-1)^((l-m)/2)*factorial(l+m)/ ...
                (2^l*factorial((l-m)/2)*factorial((l+m)/2));
        end
        table(l+1,m+shift+1) = value;
        if m > 0
            table(l+1,-m+shift+1) = (-1)^m* ...
                factorial(l-m)/factorial(l+m)*value;
        end
    end
end
end

function table = precompute_normalization(L_max)
shift = L_max;
table = zeros(L_max+1,2*shift+1);
for l = 0:L_max
    for m = -l:l
        table(l+1,m+shift+1) = sqrt((2*l+1)*factorial(l-m)/ ...
            (4*pi*factorial(l+m)));
    end
end
end
