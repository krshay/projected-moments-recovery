function [loss,profile_gradient,model_moment] = ...
    gd_third_moment_loss_gradient(radial_profiles,model,observed_moment)
%GD_THIRD_MOMENT_LOSS_GRADIENT Squared P3 residual and exact analytic gradient.
%   PROFILE_GRADIENT follows dF=2*real(sum(PROFILE_GRADIENT.*dU)).

radial_count = size(radial_profiles,1);
terms = model.terms;
term_tensors = forward_terms(radial_profiles,terms);
adjoint_tensors = cell(numel(terms),1);
for term_index = 1:numel(terms)
    adjoint_tensors{term_index} = ...
        zeros(radial_count,radial_count,radial_count);
end

loss = 0;
model_moment = cell(numel(model.blocks),1);
for block_index = 1:numel(model.blocks)
    block = model.blocks(block_index);
    tensor = zeros(radial_count,radial_count,radial_count);
    for q = 1:numel(block.term_indices)
        tensor = tensor+block.weights(q)* ...
            term_tensors{block.term_indices(q)};
    end
    model_moment{block_index} = tensor;
    residual = tensor-observed_moment{block_index};
    loss = loss+sum(abs(residual(:)).^2);
    conjugate_residual = conj(residual);
    for q = 1:numel(block.term_indices)
        term_index = block.term_indices(q);
        adjoint_tensors{term_index} = adjoint_tensors{term_index}+ ...
            block.weights(q)*conjugate_residual;
    end
end

profile_gradient = zeros(size(radial_profiles));
for term_index = 1:numel(terms)
    adjoint_tensor = adjoint_tensors{term_index};
    if ~any(adjoint_tensor(:))
        continue;
    end
    term = terms(term_index);
    for j = 1:numel(term.c)
        first = radial_profiles(:,term.i1(j));
        second = radial_profiles(:,term.i2(j));
        third = radial_profiles(:,term.i3(j));
        coefficient = term.c(j);
        profile_gradient(:,term.i1(j)) = ...
            profile_gradient(:,term.i1(j))+coefficient* ...
            contract23(adjoint_tensor,second,third);
        profile_gradient(:,term.i2(j)) = ...
            profile_gradient(:,term.i2(j))+coefficient* ...
            contract13(adjoint_tensor,first,third);
        profile_gradient(:,term.i3(j)) = ...
            profile_gradient(:,term.i3(j))+coefficient* ...
            contract12(adjoint_tensor,first,second);
    end
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

function result = contract23(tensor,second,third)
radial_count = size(tensor,1);
matrix = reshape(reshape(tensor,radial_count^2,radial_count)*third, ...
    radial_count,radial_count);
result = matrix*second;
end

function result = contract13(tensor,first,third)
radial_count = size(tensor,1);
matrix = reshape(reshape(tensor,radial_count^2,radial_count)*third, ...
    radial_count,radial_count);
result = matrix.'*first;
end

function result = contract12(tensor,first,second)
radial_count = size(tensor,1);
matrix = squeeze(sum(tensor.*reshape(first,radial_count,1,1),1));
result = matrix.'*second;
end
