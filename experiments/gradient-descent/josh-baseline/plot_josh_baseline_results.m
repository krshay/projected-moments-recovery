function files = plot_josh_baseline_results(experiment,figure_dir,stem)
%PLOT_JOSH_BASELINE_RESULTS Simple convergence and error figures.

figure_handle = figure('Visible','off');
hold on;
for restart = 1:numel(experiment.restarts)
    history = experiment.restarts(restart).history;
    plot(history.iteration,max(history.objective,realmin), ...
        'DisplayName',sprintf('Restart %d',restart));
end
set(gca,'YScale','log');
xlabel('Iteration');
ylabel('Normalized P_3 loss');
title({'Josh baseline','A_0,A_1 fixed to truth'});
legend('Location','best');
grid on;
apply_gradient_descent_style(figure_handle);
files.objective = export_pair(figure_handle,figure_dir, ...
    [stem,'_objective_history']);
close(figure_handle);

figure_handle = figure('Visible','off');
hold on;
for restart = 1:numel(experiment.restarts)
    history = experiment.restarts(restart).history;
    plot(history.iteration,max(history.relative_error,realmin), ...
        'DisplayName',sprintf('Restart %d',restart));
end
set(gca,'YScale','log');
xlabel('Iteration');
ylabel('Relative coefficient error');
title({'Josh baseline','A_0,A_1 fixed to truth'});
legend('Location','best');
grid on;
apply_gradient_descent_style(figure_handle);
files.error = export_pair(figure_handle,figure_dir,[stem,'_error_history']);
close(figure_handle);

figure_handle = figure('Visible','off');
errors = [experiment.restarts.relative_error];
bar(1:numel(errors),errors);
set(gca,'YScale','log','XTick',1:numel(errors));
xlabel('Restart');
ylabel('Final relative coefficient error');
title({'Josh baseline final errors','A_0,A_1 fixed to truth'});
grid on;
apply_gradient_descent_style(figure_handle);
files.final_error = export_pair(figure_handle,figure_dir, ...
    [stem,'_final_errors']);
close(figure_handle);
end

function files = export_pair(figure_handle,figure_dir,stem)
files = struct('pdf',fullfile(figure_dir,[stem,'.pdf']), ...
    'png',fullfile(figure_dir,[stem,'.png']));
exportgraphics(figure_handle,files.pdf,'ContentType','vector');
exportgraphics(figure_handle,files.png,'Resolution',300);
end
