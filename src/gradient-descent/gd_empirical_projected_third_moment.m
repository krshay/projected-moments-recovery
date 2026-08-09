function moment = gd_empirical_projected_third_moment( ...
    coefficients,support,radial_basis,model,options)
%GD_EMPIRICAL_PROJECTED_THIRD_MOMENT Monte-Carlo projected P3 with noise.
%   Rotations, equatorial projection, and complex-noise conventions match
%   Josh's original scripts. The result uses MODEL.blocks cell storage.

sample_count = get_option(options,'n_samples',5000);
noise_sigma = get_option(options,'noise_sigma',0);
seed = get_option(options,'seed',42);
rng(seed);
L_max = model.L_max;
radial_count = size(radial_basis{1},1);
moment = cell(numel(model.blocks),1);
for block_index = 1:numel(moment)
    moment{block_index} = zeros(radial_count,radial_count,radial_count);
end

for sample_index = 1:sample_count
    alpha = 2*pi*rand();
    gamma = 2*pi*rand();
    beta = acos(2*rand()-1);
    rotated = rotate_coefficients(coefficients,support,L_max,alpha,beta,gamma);
    profiles = gd_apply_radial_basis(rotated,radial_basis,support,L_max);
    projected = project_equator(profiles,model);
    if noise_sigma > 0
        projected = projected+noise_sigma/sqrt(2)*( ...
            randn(size(projected))+1i*randn(size(projected)));
    end
    for block_index = 1:numel(model.blocks)
        block = model.blocks(block_index);
        first = projected(block.m1+L_max+1,:).';
        second = projected(block.m2+L_max+1,:).';
        third = projected(block.m3+L_max+1,:).';
        moment{block_index} = moment{block_index}+ ...
            reshape(first,radial_count,1,1).* ...
            reshape(second,1,radial_count,1).* ...
            reshape(third,1,1,radial_count);
    end
end
for block_index = 1:numel(moment)
    moment{block_index} = moment{block_index}/sample_count;
end
end

function rotated = rotate_coefficients(coefficients,support,L_max,alpha,beta,gamma)
rotated = coefficients;
for l = 0:L_max
    columns = gd_band_indices(l);
    rotation = wigner_d_matrix(l,alpha,beta,gamma);
    band = coefficients(1:support(l+1),columns);
    rotated(1:support(l+1),columns) = (rotation*band.').';
end
end

function projected = project_equator(profiles,model)
L_max = model.L_max;
radial_count = size(profiles,1);
projected = zeros(2*L_max+1,radial_count);
for m = -L_max:L_max
    value = zeros(radial_count,1);
    for l = abs(m):L_max
        weight = model.normalization(l+1,m+L_max+1)* ...
            model.legendre_zero(l+1,m+L_max+1);
        value = value+weight*profiles(:,gd_sh_index(l,m));
    end
    projected(m+L_max+1,:) = value.';
end
end

function matrix = wigner_d_matrix(l,alpha,beta,gamma)
magnetic_indices = -l:l;
matrix = zeros(2*l+1);
for row = 1:2*l+1
    for column = 1:2*l+1
        m = magnetic_indices(row);
        mp = magnetic_indices(column);
        matrix(row,column) = exp(-1i*alpha*m)* ...
            wigner_small_d(l,m,mp,beta)*exp(-1i*gamma*mp);
    end
end
end

function value = wigner_small_d(l,m,mp,beta)
k_min = max([0,m-mp]);
k_max = min([l+m,l-mp]);
series = 0;
for k = k_min:k_max
    series = series+(-1)^(k+mp-m)/ ...
        (factorial(l+m-k)*factorial(l-mp-k)* ...
        factorial(k+mp-m)*factorial(k))* ...
        cos(beta/2)^(2*l+m-mp-2*k)* ...
        sin(beta/2)^(2*k+mp-m);
end
value = sqrt(factorial(l+m)*factorial(l-m)* ...
    factorial(l+mp)*factorial(l-mp))*series;
end

function value = get_option(options,name,default)
if isfield(options,name)
    value = options.(name);
else
    value = default;
end
end
