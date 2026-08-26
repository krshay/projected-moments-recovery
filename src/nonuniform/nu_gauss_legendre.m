function [nodes,weights] = nu_gauss_legendre(n)
%NU_GAUSS_LEGENDRE Nodes/weights on [-1,1] (weights sum to 2).

nodes = zeros(n,1); weights = zeros(n,1);
for i = 1:n
    x = cos(pi*(i-0.25)/(n+0.5));
    for iter = 1:100
        p0 = 1; p1 = 0;
        for k = 1:n
            p2 = p1; p1 = p0;
            p0 = ((2*k-1)*x*p1-(k-1)*p2)/k;
        end
        dp = n*(x*p0-p1)/(x^2-1);
        dx = -p0/dp; x = x+dx;
        if abs(dx) < 1e-15, break; end
    end
    p0 = 1; p1 = 0;
    for k = 1:n
        p2 = p1; p1 = p0;
        p0 = ((2*k-1)*x*p1-(k-1)*p2)/k;
    end
    dp = n*(x*p0-p1)/(x^2-1);
    nodes(i) = x; weights(i) = 2/((1-x^2)*dp^2);
end
end
