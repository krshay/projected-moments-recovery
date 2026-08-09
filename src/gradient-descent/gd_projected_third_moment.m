function moment = gd_projected_third_moment(radial_profiles,model)
%GD_PROJECTED_THIRD_MOMENT Exact population projected third moment.
%   MOMENT is stored only for observable triples m1+m2+m3=0, in the same
%   order as MODEL.blocks. This avoids allocating a mostly-zero 6-D array.

term_tensors = forward_terms(radial_profiles,model.terms);
radial_count = size(radial_profiles,1);
moment = cell(numel(model.blocks),1);
for block_index = 1:numel(model.blocks)
    block = model.blocks(block_index);
    tensor = zeros(radial_count,radial_count,radial_count);
    for q = 1:numel(block.term_indices)
        tensor = tensor+block.weights(q)* ...
            term_tensors{block.term_indices(q)};
    end
    moment{block_index} = tensor;
end
end

function tensors = forward_terms(coefficients,terms)
radial_count = size(coefficients,1);
tensors = cell(numel(terms),1);
for term_index = 1:numel(terms)
    term = terms(term_index);
    tensor = zeros(radial_count,radial_count,radial_count);
    for j = 1:numel(term.c)
        first = coefficients(:,term.i1(j));
        second = coefficients(:,term.i2(j));
        third = coefficients(:,term.i3(j));
        tensor = tensor+term.c(j)*( ...
            reshape(first,radial_count,1,1).* ...
            reshape(second,1,radial_count,1).* ...
            reshape(third,1,1,radial_count));
    end
    tensors{term_index} = tensor;
end
end
