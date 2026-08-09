function path = write_baseline_csv(experiment,data_dir,stem)
%WRITE_BASELINE_CSV Compact per-restart baseline results.

restarts = experiment.restarts;
restart = [restarts.restart].';
objective = [restarts.objective].';
relative_error = [restarts.relative_error].';
optimized_band_error = [restarts.optimized_band_error].';
iterations = [restarts.iterations].';
gradient_norm = [restarts.gradient_norm].';
runtime_seconds = [restarts.runtime_seconds].';
reached_global = [restarts.reached_global_solution].';
reality_defect = [restarts.reality_defect].';
result_table = table(restart,objective,relative_error, ...
    optimized_band_error,iterations,gradient_norm,runtime_seconds, ...
    reached_global,reality_defect);
path = fullfile(data_dir,[stem,'_restart_summary.csv']);
writetable(result_table,path);
end
