function [x,fval,info] = gd_minimize(objective,x0,options,metric_function)
%GD_MINIMIZE fminunc with recorded histories, falling back to internal L-BFGS.

if nargin < 3
    options = struct();
end
if nargin < 4
    metric_function = [];
end
method = get_option(options,'method','auto');
max_iterations = get_option(options,'max_iterations',3000);
history = empty_history();
timer = tic;
last_x = [];
last_gradient = [];
fallback_message = '';

use_fminunc = strcmpi(method,'fminunc') || ...
    (strcmpi(method,'auto') && exist('fminunc','file') == 2);
if use_fminunc
    try
        optimizer_options = optimoptions('fminunc', ...
            'Algorithm','quasi-newton', ...
            'SpecifyObjectiveGradient',true, ...
            'MaxIterations',max_iterations, ...
            'MaxFunctionEvaluations',20*max_iterations, ...
            'OptimalityTolerance',get_option(options,'optimality_tolerance',1e-12), ...
            'StepTolerance',get_option(options,'step_tolerance',1e-16), ...
            'FunctionTolerance',get_option(options,'function_tolerance',1e-18), ...
            'Display',get_option(options,'display','off'), ...
            'OutputFcn',@output_function);
        [x,fval,exitflag,output,gradient] = ...
            fminunc(@tracked_objective,x0,optimizer_options);
        append_history(output.iterations,x,fval,norm(gradient));
        info = struct('optimizer','fminunc','exitflag',exitflag, ...
            'iterations',output.iterations,'function_count',output.funcCount, ...
            'message',output.message,'gradient',gradient, ...
            'gradient_norm',norm(gradient), ...
            'runtime_seconds',toc(timer),'history',history, ...
            'fallback_message',fallback_message);
        return;
    catch exception
        if strcmpi(method,'fminunc')
            rethrow(exception);
        end
        fallback_message = sprintf('fminunc unavailable at runtime: %s', ...
            exception.message);
        history = empty_history();
        timer = tic;
    end
end

[x,fval,lbfgs_info] = run_lbfgs(objective,x0,max_iterations,options);
for index = 1:numel(lbfgs_info.accepted_x)
    append_history(lbfgs_info.iteration(index), ...
        lbfgs_info.accepted_x{index},lbfgs_info.objective(index), ...
        lbfgs_info.accepted_gradient_norm(index));
end
info = rmfield(lbfgs_info,{'accepted_x','accepted_gradient_norm', ...
    'iteration','objective'});
info.optimizer = 'internal L-BFGS';
info.runtime_seconds = toc(timer);
info.history = history;
info.fallback_message = fallback_message;

    function [value,gradient] = tracked_objective(current_x)
        [value,gradient] = objective(current_x);
        last_x = current_x;
        last_gradient = gradient;
    end

    function stop = output_function(current_x,optim_values,state)
        stop = false;
        if strcmp(state,'init') || strcmp(state,'iter') || strcmp(state,'done')
            if isequal(current_x,last_x)
                gradient_norm = norm(last_gradient);
            elseif isfield(optim_values,'firstorderopt')
                gradient_norm = optim_values.firstorderopt;
            else
                [~,current_gradient] = objective(current_x);
                gradient_norm = norm(current_gradient);
            end
            append_history(optim_values.iteration,current_x, ...
                optim_values.fval,gradient_norm);
        end
    end

    function append_history(iteration,current_x,value,gradient_norm)
        metric = struct();
        if ~isempty(metric_function)
            metric = metric_function(current_x);
        end
        if ~isempty(history.iteration) && history.iteration(end) == iteration
            position = numel(history.iteration);
        else
            position = numel(history.iteration)+1;
        end
        history.iteration(position,1) = iteration;
        history.objective(position,1) = value;
        history.gradient_norm(position,1) = gradient_norm;
        history.runtime_seconds(position,1) = toc(timer);
        history.relative_error(position,1) = ...
            get_metric(metric,'relative_error',nan);
        history.optimized_band_error(position,1) = ...
            get_metric(metric,'optimized_band_error',nan);
    end
end

function [x,fval,info] = run_lbfgs(objective,x0,max_iterations,options)
memory = get_option(options,'lbfgs_memory',20);
x = x0;
[fval,gradient] = objective(x);
dimension = numel(x);
S = zeros(dimension,memory);
Y = zeros(dimension,memory);
rho = zeros(1,memory);
stored = 0;
head = 0;
accepted_x = {x};
objective_history = fval;
gradient_history = norm(gradient);
iteration_history = 0;
exitflag = 0;
message = 'Maximum iterations reached.';
function_count = 1;

for iteration = 1:max_iterations
    if norm(gradient,inf) < get_option(options,'optimality_tolerance',1e-12)
        exitflag = 1;
        message = 'First-order optimality tolerance reached.';
        break;
    end
    q = gradient;
    alpha = zeros(1,memory);
    for index = 1:stored
        slot = mod(head-index,memory)+1;
        alpha(slot) = rho(slot)*(S(:,slot).'*q);
        q = q-alpha(slot)*Y(:,slot);
    end
    if stored > 0
        slot = mod(head-1,memory)+1;
        gamma = (S(:,slot).'*Y(:,slot))/(Y(:,slot).'*Y(:,slot));
    else
        gamma = 1;
    end
    z = gamma*q;
    for index = stored:-1:1
        slot = mod(head-index,memory)+1;
        beta = rho(slot)*(Y(:,slot).'*z);
        z = z+(alpha(slot)-beta)*S(:,slot);
    end
    direction = -z;
    directional_derivative = gradient.'*direction;
    if directional_derivative >= 0
        direction = -gradient;
        directional_derivative = -gradient.'*gradient;
    end
    step = 1;
    accepted = false;
    for line_search = 1:get_option(options,'max_line_search_steps',40)
        candidate = x+step*direction;
        [candidate_value,candidate_gradient] = objective(candidate);
        function_count = function_count+1;
        if candidate_value <= fval+1e-4*step*directional_derivative
            accepted = true;
            break;
        end
        step = step/2;
    end
    if ~accepted
        exitflag = -2;
        message = 'Line search failed.';
        break;
    end
    displacement = step*direction;
    gradient_change = candidate_gradient-gradient;
    curvature = displacement.'*gradient_change;
    if curvature > 1e-12
        head = mod(head,memory)+1;
        S(:,head) = displacement;
        Y(:,head) = gradient_change;
        rho(head) = 1/curvature;
        stored = min(stored+1,memory);
    end
    x = candidate;
    fval = candidate_value;
    gradient = candidate_gradient;
    accepted_x{end+1,1} = x; %#ok<AGROW>
    objective_history(end+1,1) = fval; %#ok<AGROW>
    gradient_history(end+1,1) = norm(gradient); %#ok<AGROW>
    iteration_history(end+1,1) = iteration; %#ok<AGROW>
    if abs(fval) < 1e-20
        exitflag = 2;
        message = 'Objective is numerically zero.';
        break;
    end
end
info = struct('exitflag',exitflag,'iterations',iteration_history(end), ...
    'function_count',function_count,'message',message,'gradient',gradient, ...
    'gradient_norm',norm(gradient),'accepted_x',{accepted_x}, ...
    'iteration',iteration_history,'objective',objective_history, ...
    'accepted_gradient_norm',gradient_history);
end

function history = empty_history()
history = struct('iteration',zeros(0,1),'objective',zeros(0,1), ...
    'gradient_norm',zeros(0,1),'runtime_seconds',zeros(0,1), ...
    'relative_error',zeros(0,1),'optimized_band_error',zeros(0,1));
end

function value = get_option(options,name,default)
if isfield(options,name)
    value = options.(name);
else
    value = default;
end
end

function value = get_metric(metric,name,default)
if isstruct(metric) && isfield(metric,name)
    value = metric.(name);
else
    value = default;
end
end
