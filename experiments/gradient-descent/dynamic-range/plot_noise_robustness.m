function plot_noise_robustness(output_name)
%PLOT_NOISE_ROBUSTNESS Relative recovery error against SNR, for the paper.
%   Reads the noise-sweep .mat files and plots the median relative error over
%   the optimized bands (l >= 2) as a function of SNR, log-log, with the
%   restart-to-restart range shaded.
%
%   WHICH ERROR IS PLOTTED. The l >= 2 error, not the total. Bands 0 and 1 are
%   pinned to truth, and for TRPV1 they carry 93.5 percent of the energy, so a
%   total-array error understates the error in the bands actually being
%   recovered by a factor of about four.
%
%   SNR = Inf is omitted (it has no place on a log axis). Note it is not the
%   population moment either: it is N samples with sigma = 0, so it measures
%   the sampling floor rather than noiselessness.

repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root,'src','gradient-descent'));
sweep_dir = fullfile(repo_root,'results','gradient-descent','noise-sweep');
if nargin < 1 || isempty(output_name)
    output_name = 'noise_robustness';
end

series = { ...
    '80S',   'noise_sweep_real_vol_80S_true.mat',   [0.00 0.45 0.74]; ...
    'TRPV1', 'noise_sweep_real_vol_TRPV1_true.mat', [0.85 0.33 0.10]};

figure('Position',[100 100 620 460],'Color','w');
hold on; box on; grid on;
set(gca,'XScale','log','YScale','log','GridAlpha',0.15, ...
    'MinorGridAlpha',0.07,'FontSize',11,'LineWidth',0.8);

handles = []; labels = {};
for s = 1:size(series,1)
    [name,file,colour] = series{s,:};
    path = fullfile(sweep_dir,file);
    if exist(path,'file') ~= 2
        fprintf('skip %s (%s not found)\n',name,file);
        continue
    end
    loaded = load(path);
    records = loaded.summary.records;

    snr = []; med = []; lo = []; hi = [];
    for i = 1:numel(records)
        r = records{i};
        if ~isfinite(r.snr), continue; end          % Inf has no log-axis home
        errors = [r.restarts.optimized_band_error];
        snr(end+1) = r.snr;          %#ok<AGROW>
        med(end+1) = median(errors); %#ok<AGROW>
        lo(end+1)  = min(errors);    %#ok<AGROW>
        hi(end+1)  = max(errors);    %#ok<AGROW>
    end
    [snr,order] = sort(snr); med = med(order); lo = lo(order); hi = hi(order);

    fill([snr,fliplr(snr)],[lo,fliplr(hi)],colour, ...
        'FaceAlpha',0.13,'EdgeColor','none','HandleVisibility','off');
    h = plot(snr,med,'-o','Color',colour,'MarkerFaceColor',colour, ...
        'MarkerSize',5,'LineWidth',1.6);
    handles(end+1) = h;                     %#ok<AGROW>
    labels{end+1} = sprintf('%s (%d SNR points)',name,numel(snr)); %#ok<AGROW>
    fprintf('%-6s: %d points, error %.3e (SNR %g) to %.3e (SNR %g)\n', ...
        name,numel(snr),med(end),snr(end),med(1),snr(1));
end

xlabel('SNR','FontWeight','bold');
ylabel('Relative error  (bands \it{l} \geq 2)','FontWeight','bold');
if ~isempty(handles)
    legend(handles,labels,'Location','southwest','Box','off');
end

output = fullfile(sweep_dir,output_name);
exportgraphics(gcf,[output '.pdf'],'ContentType','vector');
exportgraphics(gcf,[output '.png'],'Resolution',200);
fprintf('\nwritten to %s.pdf and .png\n',output);
end
