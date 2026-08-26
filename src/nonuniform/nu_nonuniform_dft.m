function values = nu_nonuniform_dft(volume,k_range,query)
%NU_NONUNIFORM_DFT Exact centred DFT evaluated at arbitrary query points.
%   values(p) = sum_x volume(x) * exp(-2i*pi*(query(p,:) . x)/N), with x running
%   over the centred grid K_RANGE. Matches fftshift(fftn(ifftshift(volume))) at
%   integer query points, and interpolates it exactly (as the underlying
%   trigonometric polynomial) in between.
%
%   Done separably in three passes so the cost is O(P*N^2 + P*N) rather than
%   O(P*N^3) with a dense P x N^3 matrix.

N = numel(k_range);
n_points = size(query,1);
scale = -2i*pi/N;
E1 = exp(scale*(query(:,1)*k_range.'));      % P x N
E2 = exp(scale*(query(:,2)*k_range.'));
E3 = exp(scale*(query(:,3)*k_range.'));

chunk = max(1,floor(4e6/max(N*N,1)));
values = zeros(n_points,1);
for first = 1:chunk:n_points
    last = min(first+chunk-1,n_points);
    rows = first:last;
    partial = E1(rows,:)*reshape(volume,N,N*N);        % |rows| x (N*N)
    partial = reshape(partial,numel(rows),N,N);
    partial = squeeze(sum(partial.*E2(rows,:),2));     % |rows| x N
    if numel(rows) == 1, partial = reshape(partial,1,N); end
    values(rows) = sum(partial.*E3(rows,:),2);
end
end
