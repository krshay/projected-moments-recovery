function config = dynamic_range_config(repo_root)
%DYNAMIC_RANGE_CONFIG Band-dynamic-range sweep on the Josh baseline objective.
%   Section 4.7 conjectures that stable recovery from the projected third
%   moment needs all frequency bands to be approximately equally normalized.
%   This sweep tests that directly: band norms are forced onto the profile
%   ||A_0|| = ||A_1|| = 1, ||A_l|| = D^(-(l-1)/(L-1)), and D is varied.
%   D = 1 is the equal-normalization control.

config = struct();
config.repo_root = repo_root;
config.seed = 42;
config.output_root = fullfile(repo_root,'results','gradient-descent', ...
    'dynamic-range');
timestamp = datestr(now,'yyyy-mm-dd_HH-MM-SS');
config.run_dir = unique_run_dir(fullfile(config.output_root,timestamp));
config.config_dir = fullfile(config.run_dir,'config');
config.data_dir = fullfile(config.run_dir,'data');
config.logs_dir = fullfile(config.run_dir,'logs');

% Coefficient source: 'synthetic' (Gaussian, recovers at D = 1) or 'real'
% (molecular volume expanded in Fourier-Bessel).
config.source = 'synthetic';
config.synthetic = struct('L_max',5,'shell_count',12,'seed',42);

% Radial parameterization(s) to optimize in. The driver loops over these.
%   'fourier-bessel' recovers Bessel coefficients a_l[s,m]; the forward pass
%                    applies Phi_l to reach the radial profile.
%   'delta'          recovers the radial profile u_l[k,m] directly on the rho
%                    grid (identity radial basis), bypassing Phi_l entirely.
%                    This is the parameterization the synthetic problem uses.
% Band targets are applied to profile norms, so both modes are fitted against
% the identical observed P3 with the identical number of real DOF. The only
% difference is the coordinate system the optimizer works in.
config.radial_basis_modes = {'delta'};

% Sweep points. The driver loops over this vector.
config.ratios = [100,300,1000];

config.restart_count = 20;
% Radial sample count for the Fourier-Bessel forward map Phi_l.
% At 12 samples against 12 Bessel shells the basis is rank deficient
% (rank(Phi_5) = 10/12, cond ~ 1.3e17). 16 samples restore full rank at
% cond(Phi_5) = 5.35. Note the projected third moment is a radial_count^3
% tensor, so cost grows as the cube: 16 costs 2.4x of 12, while 31 costs
% 17x for cond 5.38 instead of 5.35 -- no conditioning benefit.
config.radial_count = 16;

% NDFT shell count for real volumes. The shell samples are the coefficients
% (identity radial basis), so this is the only radial knob on the real path;
% config.radial_count applies to the synthetic Fourier-Bessel path only.
config.shell_count = 12;
config.success_threshold = 1e-4;

% Truth-scaled per-band Gaussian starts. Required for a dynamic-range sweep:
% the unscaled N(0,1) start used by the Section 4.7 synthetic run would put
% band L at O(1) while its truth norm is 1/D, so large D would fail from an
% initialization-scale mismatch rather than from conditioning.
config.initialization = 'real';

% Raised from the josh-baseline real-volume value of 200. At 200 every real
% restart terminated on MaxIterations with ||g|| ~ 1e-3, which conflates
% truncation with genuine failure. Synthetic converges near 250 iterations,
% so this does not change the D = 1 control.
config.optimizer = struct('method','auto','max_iterations',3000, ...
    'optimality_tolerance',1e-10,'step_tolerance',1e-14, ...
    'function_tolerance',1e-16,'display','off');

% Matches josh_baseline_config real-volume settings: L=5, R=12 per band.
config.volume_options = struct('L_max',5,'grid_size',31,'rho_max',pi, ...
    'basis_mode','manual','manual_shells',[12,12,12,12,12,12]);
config.volume = struct('name','hemoglobin', ...
    'path',fullfile(repo_root,'data','raw','hemoglobin.map'));
end

function directory = unique_run_dir(candidate)
directory = candidate;
suffix = 2;
while exist(directory,'dir') == 7
    directory = sprintf('%s_%02d',candidate,suffix);
    suffix = suffix+1;
end
end
