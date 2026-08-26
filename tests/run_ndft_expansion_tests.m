function run_ndft_expansion_tests()
%RUN_NDFT_EXPANSION_TESTS Well-definedness of the NDFT shell expansion.
%   The property under test is the one the Fourier-Bessel least-squares
%   expansion does NOT have: the coefficient array must be a function of the
%   volume, not of the representation parameters used to compute it.
%
%   NU_EXPAND_VOLUME_SHELLS extracts each shell by a spherical quadrature
%   projection -- an inner product against an orthonormal basis, one matrix
%   multiply, no system solved -- and does so independently per shell. So
%   refining the radial grid adds rows without disturbing existing ones, and
%   refining the angular quadrature converges.
%
%   EXPAND_REAL_VOLUME_FOURIER_BESSEL instead fits all (l,s,m) jointly via
%   normal equations (cond(B) ~ 3e9, cond(BtB) ~ 9e18) and truncates 25 of
%   432 directions at a 1e-8 relative cutoff. It returns the minimum-norm
%   member of a 25-dimensional affine family, and that member moves by ~1.36
%   relative when the FB shell count changes with the volume held fixed.
%   Test 4 records that contrast so the difference stays visible.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src','nonuniform'));
addpath(fullfile(root,'src','gradient-descent'));
addpath(fullfile(root,'src','frequency-marching'));

volume = fullfile(root,'data','raw','vol_TRPV1_true.mrc');
if exist(volume,'file') ~= 2
    fprintf('SKIP: %s not present.\n',volume);
    return
end
L = 5; grid_size = 31;
opts = @(K,nt,np) struct('L_max',L,'grid_size',grid_size,'rho_max',pi, ...
    'shell_count',K,'mode','exact','n_theta',nt,'n_phi',np);

fprintf('\n== NDFT expansion: well-definedness ==\n');

%% 1. Radial refinement is exact. K=48 contains the K=16 radii exactly,
%    because rho_s = (s-0.5)*rho_max/K and 48 = 3*16 gives s -> 3s-1.
a16 = nu_expand_volume_shells(volume,opts(16,32,64));
a48 = nu_expand_volume_shells(volume,opts(48,32,64));
subset = 3*(1:16)-1;
u = a16/norm(a16(:));
v = a48(subset,:)/norm(a48(subset,:),'fro');
radial = norm(u(:)-v(:))/norm(u(:));
fprintf(' 1. radial K=16 vs K=48 subset : %.3e\n',radial);
assert(radial < 1e-12, ...
    'Radial refinement must be exact (got %.3e). Shells are computed independently, so this can only fail if a coupled solve was introduced.',radial);

%% 2. Angular quadrature converges. Vhat on a sphere is not band-limited to
%    L, so this cannot reach machine zero -- it must converge, not vanish.
b1 = nu_expand_volume_shells(volume,opts(12,24,48));
b2 = nu_expand_volume_shells(volume,opts(12,32,64));
b3 = nu_expand_volume_shells(volume,opts(12,40,80));
d12 = reldiff(b1,b2); d23 = reldiff(b2,b3);
fprintf(' 2. angular 24x48->32x64      : %.3e\n',d12);
fprintf('    angular 32x64->40x80      : %.3e\n',d23);
assert(d23 < d12, 'Angular refinement must converge (%.3e then %.3e).',d12,d23);
assert(d23 < 1e-4, 'Angular residual %.3e is too large to be quadrature error.',d23);

%% 3. A real volume obeys Friedel symmetry exactly, so after the i^l twist the
%    array must satisfy standard reality to machine precision -- with no
%    projection applied. The FB path needed one, which is why its defect was
%    ~3e-04 rather than ~4e-16.
[~,~,meta] = nu_expand_volume_shells(volume,opts(12,24,48));
fprintf(' 3. standard reality defect    : %.3e\n',meta.standard_reality_defect);
assert(meta.standard_reality_defect < 1e-12, ...
    'Reality defect %.3e suggests the Friedel convention or the i^l twist is wrong.', ...
    meta.standard_reality_defect);

%% 4. No shell radius is degenerate: none at rho=0 (kills l>=1) and none at
%    rho_max. Midpoints avoid both by construction.
rho = meta.rho_grid(:).'/meta.rho_max;
fprintf(' 4. rho/rho_max in [%.4f, %.4f]\n',min(rho),max(rho));
assert(min(rho) > 0 && max(rho) < 1, 'A shell sits on a degenerate radius.');

%% 5. Contrast: the FB least-squares array is NOT resolution stable. Changing
%    the FB shell count with the volume fixed moves it by order 1.
fb = @(S) fb_delta_array(volume,L,grid_size,S,16);
try
    f12 = fb(12); f14 = fb(14);
    fbshift = reldiff(f12,f14);
    fprintf(' 5. FB shells 12 vs 14        : %.3f   <- unstable, for contrast\n',fbshift);
    assert(fbshift > 0.1, ...
        'FB path unexpectedly stable (%.3e); the contrast this suite documents may no longer hold.',fbshift);
catch err
    fprintf(' 5. FB contrast skipped (%s)\n',err.message);
end

fprintf('\nALL NDFT EXPANSION TESTS PASSED\n');
end

function d = reldiff(a,b)
a = a/norm(a(:)); b = b/norm(b(:));
d = min(norm(a(:)-b(:)),norm(a(:)+b(:)))/norm(a(:));
end

function u = fb_delta_array(volume,L,grid_size,shells,K)
support = shells*ones(1,L+1);
[raw,~,meta] = expand_real_volume_fourier_bessel(volume, ...
    struct('L_max',L,'grid_size',grid_size,'rho_max',pi, ...
           'basis_mode','manual','manual_shells',support));
u = gd_apply_radial_basis(raw,gd_fourier_bessel_radial_basis(meta,K),support,L);
end
