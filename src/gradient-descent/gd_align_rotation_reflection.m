function result = gd_align_rotation_reflection(estimate,reference,L_max,free_bands,n_candidates)
%GD_ALIGN_ROTATION_REFLECTION Align REFERENCE to ESTIMATE over O(3) and score it.
%   RESULT = GD_ALIGN_ROTATION_REFLECTION(ESTIMATE,REFERENCE,L_MAX) finds the
%   rotation R and reflection flag minimising ||ESTIMATE - g.REFERENCE||_F over
%   all bands, where g ranges over rotations and the parity a_lm -> (-1)^l a_lm.
%   The projected third moment is invariant under that whole group, so when no
%   band is pinned the recovered coefficients are determined only up to it.
%
%   RESULT.error_all   relative error over all bands at the best g
%   RESULT.error_free  relative error over FREE_BANDS (default 2:L) at that g
%   RESULT.euler, RESULT.reflected, RESULT.aligned_reference
%
%   SEARCH. N_CANDIDATES (default 20000) Haar-random rotations per parity, each
%   scored on ALL bands, then the best 25 refined by Nelder-Mead.
%
%   WHY SO DENSE. An earlier version scored a 1296-point Euler grid on bands
%   1..3 and refined 8 candidates. On TRPV1 (EMD-24084) -- a tetramer that
%   stays roughly four-fold symmetric even when refined without symmetry --
%   it put one restart in a near-symmetric wrong pose in every SNR cell,
%   reporting aligned errors near 0.25 for solutions whose true error was
%   0.025-0.11 and whose loss matched the best restart to five digits. Scoring
%   on bands 1..3 is what let the wrong pose win: those bands barely
%   distinguish the near-equivalent poses of a near-symmetric molecule.
%
%   Rotations use D^l = exp(-i alpha Jz) exp(-i beta Jy) exp(-i gamma Jz),
%   built from the eigendecomposition of Jy, acting on a band as (D*a.').'.
%   This is the group under which the forward model's moment is invariant
%   (checked to 2e-17); any parametrisation of that group gives the same
%   minimum, so the Euler convention itself does not affect the error.

if nargin < 4 || isempty(free_bands), free_bands = 2:L_max; end
if nargin < 5 || isempty(n_candidates), n_candidates = 20000; end
free_columns = cell2mat(arrayfun(@(l) gd_band_indices(l),free_bands,'uni',0));
[V,E] = jy_eigen(L_max);

stream = RandStream('mt19937ar','Seed',11);
candidates = [2*pi*rand(stream,n_candidates,1), ...
              acos(2*rand(stream,n_candidates,1)-1), ...
              2*pi*rand(stream,n_candidates,1)];
reference_norm = max(norm(reference(:)),eps);
references = {reference,gd_reflect_coefficients(reference,L_max)};
search = optimset('TolX',1e-12,'TolFun',1e-15,'MaxFunEvals',6000, ...
    'MaxIter',6000,'Display','off');

best = struct('cost',inf,'parity',1,'angles',[0 0 0]);
for parity = 1:2
    base = references{parity};
    cost = @(g) norm(reshape(estimate-rotate(base,g,L_max,V,E),[],1))/reference_norm;
    values = zeros(n_candidates,1);
    for k = 1:n_candidates, values(k) = cost(candidates(k,:)); end
    [~,order] = sort(values);
    for k = order(1:min(25,n_candidates))'
        [angles,value] = fminsearch(cost,candidates(k,:),search);
        if value < best.cost
            best = struct('cost',value,'parity',parity,'angles',angles);
        end
    end
end

aligned = rotate(references{best.parity},best.angles,L_max,V,E);
result = struct( ...
    'error_all',norm(estimate(:)-aligned(:))/reference_norm, ...
    'error_free',norm(estimate(:,free_columns)-aligned(:,free_columns),'fro')/ ...
        max(norm(reference(:,free_columns),'fro'),eps), ...
    'euler',best.angles,'reflected',best.parity == 2,'aligned_reference',aligned);
end

function [V,E] = jy_eigen(L_max)
V = cell(L_max+1,1); E = cell(L_max+1,1);
for l = 0:L_max
    m = (-l:l)'; raising = zeros(2*l+1);
    for q = 1:2*l, raising(q+1,q) = sqrt(l*(l+1)-m(q)*(m(q)+1)); end
    Jy = (raising-raising')/(2i); Jy = (Jy+Jy')/2;
    [V{l+1},D] = eig(Jy); E{l+1} = diag(D);
end
end

function b = rotate(a,g,L_max,V,E)
b = a;
for l = 0:L_max
    m = (-l:l)'; c = gd_band_indices(l);
    D = diag(exp(-1i*g(1)*m))*(V{l+1}*diag(exp(-1i*g(2)*E{l+1}))*V{l+1}')* ...
        diag(exp(-1i*g(3)*m));
    b(:,c) = (D*a(:,c).').';
end
end
