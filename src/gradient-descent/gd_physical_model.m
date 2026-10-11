function model = gd_physical_model(L_max)
%GD_PHYSICAL_MODEL Moment model for the PHYSICAL projected third moment.
%   GD_PRECOMPUTE_MODEL assumes coefficients in the physical (Friedel)
%   convention, but NU_EXPAND_VOLUME_SHELLS returns them twisted by i^l so that
%   they satisfy the standard reality A_{l,-m} = (-1)^m conj(A_{l,m}) that the
%   real parameterisation and the Clebsch-Gordan charts require. Feeding twisted
%   coefficients to the default model therefore fits the moment of a twisted
%   object, not the projected third moment of the volume: measured 10-12% apart
%   on the TRPV1 arrays.
%
%   This undoes the twist inside the model, so that
%       P3(a, gd_physical_model(L)) == P3((-i)^l .* a, gd_precompute_model(L)),
%   with a in the twisted convention the rest of the pipeline uses. Each term is
%   a product of three coefficients, contributing (-i)^(l1+l2+l3); that sum is
%   always even, because the equatorial slice kills l+m odd and the blocks have
%   m1+m2+m3 = 0, so the factor reduces to (-1)^((l1+l2+l3)/2).
%
%   Only TERMS change. BLOCKS, NORMALIZATION and LEGENDRE_ZERO are untouched, so
%   GD_EMPIRICAL_PROJECTED_THIRD_MOMENT is unaffected by this function -- it
%   projects coefficients directly rather than through TERMS. To sample a
%   physical moment, pass it untwisted coefficients ((-i)^l .* a) with the
%   default model, and fit those data with this model. See
%   RUN_NOISE_SWEEP_UNPINNED.
%
%   The correction was identified by A. Balanov's group in their reproduction
%   package (code/fixed_model.m); this is the same transformation.

model = gd_precompute_model(L_max);
for k = 1:numel(model.terms)
    t = model.terms(k);
    assert(mod(t.l1+t.l2+t.l3,2) == 0, ...
        'Term %d has odd l1+l2+l3; the twist correction assumes even.',k);
    model.terms(k).c = t.c*(-1)^((t.l1+t.l2+t.l3)/2);
end
model.convention = 'physical (i^l twist undone)';
end
