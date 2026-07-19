function apply_publication_style(figure_handle)
%APPLY_PUBLICATION_STYLE Match the reference repositories' figure style.

font_size = 14;
axes_handles = findall(figure_handle,'Type','axes');
set(axes_handles,'FontSize',font_size,'FontName','Arial', ...
    'LineWidth',1.5,'FontWeight','bold','Box','on','TickDir','out');
line_handles = findall(figure_handle,'Type','line');
set(line_handles,'LineWidth',2,'MarkerSize',8);
legend_handles = findall(figure_handle,'Type','legend');
set(legend_handles,'FontSize',font_size-2,'FontName','Arial');
set(figure_handle,'Renderer','painters');
end
