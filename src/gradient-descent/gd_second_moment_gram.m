function second_moment = gd_second_moment_gram(coefficients,support,L_max)
%GD_SECOND_MOMENT_GRAM Exact population Gram data exposed by the second moment.
%   G_l=A_l*A_l^* and tau_l=trace(G_l)=||A_l||_F^2. Experiments pass this
%   structure to the optimizer, mirroring the interface used for observed
%   second-moment Gram matrices instead of passing hidden "true norms."

gram_matrices = cell(1,L_max+1);
tau = zeros(1,L_max+1);
for l = 0:L_max
    band = coefficients(1:support(l+1),gd_band_indices(l));
    gram_matrices{l+1} = band*band';
    tau(l+1) = real(trace(gram_matrices{l+1}));
end
second_moment = struct( ...
    'gram_matrices',{gram_matrices}, ...
    'tau',tau, ...
    'source','exact population second-moment Gram interface', ...
    'normalization','G_l=A_l*A_l^*, tau_l=trace(G_l)=||A_l||_F^2');
end
