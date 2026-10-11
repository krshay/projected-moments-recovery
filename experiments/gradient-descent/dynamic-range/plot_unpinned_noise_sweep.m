function summary = plot_unpinned_noise_sweep(label,volume_stem,title_text)
%PLOT_UNPINNED_NOISE_SWEEP Aggregate and plot one RUN_NOISE_SWEEP_UNPINNED sweep.
%   PLOT_UNPINNED_NOISE_SWEEP('80S','vol_80S_true','80S, EMD-2660') reads the
%   eight per-SNR CSVs in "<LABEL>_unpinned_noise sweep", writes summary.csv and
%   all_restarts.csv beside them, and plots mean and median O(3)-aligned l >= 2
%   error against SNR in the same style as PLOT_NOISE_SWEEP_CELL.
%
%   Every restart is included: nothing is capped in these sweeps (check the
%   hit_cap column), so there is no "converged only" subset to single out.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
folder = fullfile(repo_root,'results','gradient-descent','noise-sweep', ...
    sprintf('%s_unpinned_noise sweep',label));
snrs = [Inf 30 10 3 1 0.3 0.1 0.03];
n = numel(snrs);
[med,mn,best,sd,merr,capped,med_all,mn_all,sd_all] = deal(zeros(1,n));
labels = cell(1,n);
all_rows = [];
for i = 1:n
    s = snrs(i);
    if isinf(s), labels{i} = 'Inf'; else, labels{i} = sprintf('%g',s); end
    stem = fullfile(folder,sprintf('unpinned_%s_snr_%g',volume_stem,s));
    T = readtable([stem '.csv']);
    e = T.aligned_l2_error;
    [med(i),mn(i),best(i),sd(i),capped(i)] = deal(median(e),mean(e),min(e),std(e),sum(T.hit_cap));
    % error over ALL degrees: the quantity the O(3) alignment itself minimises
    ea = T.aligned_all_band_error;
    [med_all(i),mn_all(i),sd_all(i)] = deal(median(ea),mean(ea),std(ea));
    M = load([stem '.mat']); merr(i) = M.summary.moment_error;
    all_rows = [all_rows; [repmat(s,height(T),1) T.seed T.aligned_l2_error ...
        T.aligned_all_band_error T.raw_l2_error T.loss T.iterations T.hit_cap]]; %#ok<AGROW>
end
writetable(array2table(all_rows,'VariableNames',{'snr','seed','aligned_l2_error', ...
    'aligned_all_band_error','raw_l2_error','loss','iterations','hit_cap'}), ...
    fullfile(folder,'all_restarts.csv'));
summary = table(snrs',merr',med',mn',best',sd',med_all',mn_all',sd_all',capped', ...
    'VariableNames',{'snr','moment_error','median_aligned_l2','mean_aligned_l2', ...
     'best_aligned_l2','std_aligned_l2','median_aligned_all','mean_aligned_all', ...
     'std_aligned_all','capped'});
writetable(summary,fullfile(folder,'summary.csv'));
disp(summary);

x = 1:n;
figure('Position',[100 100 780 560],'Color','w'); hold on; box on; grid on; grid minor;
set(gca,'FontSize',13,'LineWidth',1.0,'GridAlpha',0.15,'XTick',x, ...
    'XTickLabel',labels,'XLim',[0.7 n+0.3]);
plot(x,mn,'-o','Color',[0 0.447 0.741],'LineWidth',2.2,'MarkerSize',9);
plot(x,med,'--s','Color',[0.851 0.325 0.098],'LineWidth',2.0,'MarkerSize',9);
xlabel('SNR (power ratio, clean/noisy)','FontWeight','bold','FontSize',14,'Interpreter','none');
ylabel('Relative coefficient error (l >= 2), O(3)-aligned','FontWeight','bold', ...
    'FontSize',14,'Interpreter','none');
title(sprintf('Unpinned recovery, physical moment (%s)',title_text), ...
    'FontWeight','bold','FontSize',15,'Interpreter','none');
legend({'Mean over 20 restarts','Median over 20 restarts'},'Location','northwest', ...
    'FontSize',12,'Box','on');
ylim([0 1.05*max([mn med])]);
out = fullfile(folder,sprintf('%s_unpinned_noise_sweep',lower(label)));
exportgraphics(gcf,[out '.png'],'Resolution',200);
exportgraphics(gcf,[out '.pdf'],'ContentType','vector');
fprintf('wrote summary.csv, all_restarts.csv, %s.png/.pdf\n',out);
end
