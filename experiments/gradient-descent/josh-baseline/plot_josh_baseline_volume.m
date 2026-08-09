function files = plot_josh_baseline_volume( ...
    truth,estimate,error_value,figure_dir,stem)
%PLOT_JOSH_BASELINE_VOLUME Central slices of true and reconstructed volumes.

middle = ceil(size(truth,3)/2);
limits = [min(truth(:,:,middle),[],'all'),max(truth(:,:,middle),[],'all')];
figure_handle = figure('Visible','off');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile;
imagesc(truth(:,:,middle));
axis image off;
clim(limits);
title('True coefficients');
nexttile;
imagesc(estimate(:,:,middle));
axis image off;
clim(limits);
title(sprintf('Reconstruction, error %.2g',error_value));
colormap(gray(256));
sgtitle({'Josh baseline','A_0,A_1 fixed to truth'});
apply_gradient_descent_style(figure_handle);
files = struct('pdf',fullfile(figure_dir,[stem,'_volume.pdf']), ...
    'png',fullfile(figure_dir,[stem,'_volume.png']));
exportgraphics(figure_handle,files.pdf,'ContentType','vector');
exportgraphics(figure_handle,files.png,'Resolution',300);
close(figure_handle);
end
