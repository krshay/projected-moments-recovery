function zero = nu_spherical_bessel_zero(l,s)
%NU_SPHERICAL_BESSEL_ZERO S-th positive zero of the spherical Bessel j_l.
%   Bracket by scanning for a sign change, then bisect. Matches the zeros
%   used by expand_real_volume_fourier_bessel.m (its private sph_bessel_zero).

persistent cache
if isempty(cache), cache = containers.Map('KeyType','char','ValueType','double'); end
key = sprintf('%d_%d',l,s);
if isKey(cache,key), zero = cache(key); return; end

step = 0.05;
x = max(step,l*1.0);          % first zero of j_l is > l
found = 0; previous = x; previous_value = jl(l,x);
while found < s
    x = x+step;
    value = jl(l,x);
    if previous_value == 0
        found = found+1; zero = previous;
    elseif sign(value) ~= sign(previous_value)
        lo = previous; hi = x;
        for iteration = 1:200
            mid = 0.5*(lo+hi);
            if sign(jl(l,mid)) == sign(jl(l,lo)), lo = mid; else, hi = mid; end
        end
        found = found+1; zero = 0.5*(lo+hi);
    end
    previous = x; previous_value = value;
    assert(x < 5000,'No %d-th zero of j_%d found below x = 5000.',s,l);
end
cache(key) = zero;
end

function value = jl(l,x)
if x == 0
    value = double(l == 0); return;
end
value = sqrt(pi/(2*x))*besselj(l+0.5,x);
end
