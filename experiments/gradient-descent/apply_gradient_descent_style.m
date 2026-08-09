function apply_gradient_descent_style(figure_handle)
%APPLY_GRADIENT_DESCENT_STYLE Publication formatting used by GD figures.

set(figure_handle,'Color','w','Units','inches','Position',[1,1,6.6,4.2]);
axes_handles = findall(figure_handle,'Type','axes');
set(axes_handles,'FontName','Helvetica','FontSize',11, ...
    'LineWidth',0.8,'Box','off','TickDir','out');
line_handles = findall(figure_handle,'Type','line');
set(line_handles,'LineWidth',1.4);
end
