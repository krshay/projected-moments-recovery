function plot_noise_sweep_cell(molecule,mat_file,output_name)
%PLOT_NOISE_SWEEP_CELL Noise-sweep curve in the two-series house style.
%   Plots the relative coefficient error over the optimized bands (l >= 2)
%   against SNR on a categorical axis (so SNR = Inf has a place), linear in
%   the error. Two series:
%     'Mean of converged restarts'  -- restarts with exitflag > 0 only
%     'Median across all restarts'  -- every restart, converged or capped
%
%   CAUTION ON THE CONVERGED MEAN. The two series answer different questions
%   and the first is only as meaningful as the number of restarts behind it.
%   Where most restarts hit the iteration cap the mean is taken over a small
%   and self-selected subset -- the restarts that happened to finish -- which
%   is not a random sample of the 20. The printed table below reports the
%   converged count per cell so the reader can see how many points each mean
%   rests on.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
sweep_dir = fullfile(repo_root,'results','gradient-descent','noise-sweep');
loaded = load(fullfile(sweep_dir,mat_file));
records = loaded.summary.records;

n = numel(records);
snr = zeros(1,n); mean_conv = nan(1,n); med_all = zeros(1,n);
n_conv = zeros(1,n); labels = cell(1,n);
for i = 1:n
    r = records{i};
    errors = [r.restarts.optimized_band_error];
    flags  = [r.restarts.exitflag];
    snr(i) = r.snr;
    med_all(i) = median(errors);
    ok = flags > 0;
    n_conv(i) = sum(ok);
    if any(ok), mean_conv(i) = mean(errors(ok)); end
    if isfinite(r.snr), labels{i} = sprintf('%g',r.snr); else, labels{i} = 'Inf'; end
end
[~,order] = sort(snr,'descend');           % Inf first, clean -> noisy
snr = snr(order); mean_conv = mean_conv(order); med_all = med_all(order);
n_conv = n_conv(order); labels = labels(order);
x = 1:n;

figure('Position',[100 100 780 560],'Color','w');
hold on; box on; grid on;
set(gca,'FontSize',13,'LineWidth',1.0,'GridAlpha',0.15, ...
    'XTick',x,'XTickLabel',labels,'XLim',[0.7 n+0.3]);
grid minor;

plot(x,mean_conv,'-o','Color',[0 0.447 0.741],'LineWidth',2.2, ...
    'MarkerSize',9,'MarkerFaceColor','none');
plot(x,med_all,'--s','Color',[0.851 0.325 0.098],'LineWidth',2.0, ...
    'MarkerSize',9,'MarkerFaceColor','none');

xlabel('SNR (power ratio, clean/noisy)','FontWeight','bold','FontSize',14,'Interpreter','none');
ylabel('Relative coefficient error (l >= 2)','FontWeight','bold','FontSize',14,'Interpreter','none');
% Interpreter 'none': molecule and file names contain underscores, which the
% default TeX interpreter renders as subscripts.
title(sprintf('NDFT-shell noise sweep (%s)',molecule), ...
    'FontWeight','bold','FontSize',15,'Interpreter','none');
legend({'Mean of converged restarts','Median across all restarts'}, ...
    'Location','northwest','FontSize',12,'Box','on');
ylim([0 max(1.5,1.05*max([mean_conv,med_all]))]);

fprintf('\n%-6s %14s %16s %12s\n','SNR','mean(converged)','median(all)','converged n');
for i = 1:n
    fprintf('%-6s %14.4f %16.4f %9d/20\n',labels{i},mean_conv(i),med_all(i),n_conv(i));
end

out = fullfile(sweep_dir,output_name);
exportgraphics(gcf,[out '.png'],'Resolution',200);
exportgraphics(gcf,[out '.pdf'],'ContentType','vector');
fprintf('\nwritten to %s.png and .pdf\n',out);
end
