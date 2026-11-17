function run_matlab_all(force)
% LRTV and LRTDTV (official MATLAB codes in ../external) on both scenes, with the parameters chosen on
% PaviaU (see README). Results: results/<scene>/<method>.mat with the restored image (H x W x B, single,
% clipped to [0,1]), the run time and the machine. Finished runs are skipped unless force = true.
%
%   run_matlab_all            % in MATLAB, with this folder as the current folder
%   matlab -batch "cd('<repository>/matlab'); run_matlab_all"     % from Terminal (run_all.sh does this)
%
% Protocol as in our earlier runs: rng(0) before each run (LRTV starts from a random matrix), output
% clipped to [0,1]. CPU threads: half of the cores, the same number as the Python CPU methods.
if nargin < 1; force = false; end
here = fileparts(mfilename('fullpath'));
repo = fileparts(here);
nthreads = max(1, floor(feature('numcores') / 2));
maxNumCompThreads(nthreads);
scenes = {'PaviaU', 'PaviaC'};
methods = {'LRTV', 'LRTDTV'};
for i = 1:numel(scenes)
    for j = 1:numel(methods)
        run_one(repo, scenes{i}, methods{j}, nthreads, force);
    end
end
end

function run_one(repo, scene, method, nthreads, force)
outdir = fullfile(repo, 'results', scene);
outfile = fullfile(outdir, [method '.mat']);
if exist(outfile, 'file') && ~force
    fprintf('skip %s %s: result exists\n', method, scene);
    return;
end
if ~exist(outdir, 'dir'); mkdir(outdir); end
d = load(fullfile(repo, 'data', [scene '_case5_seed0.mat']), 'input', 'gt');
Noi = double(d.input);
gt = double(d.gt);
[M, N, ~] = size(Noi);
oldpath = path;
restore_path = onCleanup(@() path(oldpath));   % the two codes have functions with the same names
switch method
    case 'LRTV'     % He, Zhang, Zhang, Shen, IEEE TGRS 2016; lambda = LS / sqrt(M N) as in the authors' demo
        base = fullfile(repo, 'external', 'LRTV', 'LLRGTV');
        addpath(base, fullfile(base, 'prox_operators'), fullfile(base, 'PROPACK'));
        tau = 0.0025; ls = 10; r = 3; lambda = ls / sqrt(M * N);
        params = struct('tau', tau, 'lambda_scale', ls, 'lambda', lambda, 'rank', r);
        rng(0);
        t0 = tic;
        X = LRTV_accelerate(Noi, tau, lambda, r);
        time_s = toc(t0);
    case 'LRTDTV'   % Wang et al., IEEE JSTARS 2018; rank [0.8 M, 0.8 N, r3]
        base = fullfile(repo, 'external', 'LRTDTV');
        addpath(base, fullfile(base, 'mylib'), fullfile(base, 'prox_operators'));
        tau = 1; lambda = 10; rk = [round(0.8 * M), round(0.8 * N), 4];
        params = struct('tau', tau, 'lambda', lambda, 'rank', rk);
        rng(0);
        t0 = tic;
        X = LRTDTV(Noi, tau, lambda, rk);
        time_s = toc(t0);
end
clear restore_path
X = min(max(X, 0), 1);
mse = squeeze(mean(mean((X - gt).^2, 1), 2));
mpsnr = mean(10 * log10(1 ./ max(mse, 1e-12)));
restored = single(X); %#ok<NASGU>
[~, cpu] = system('sysctl -n machdep.cpu.brand_string');
machine = struct('matlab', version, 'computer', computer, 'cpu', strtrim(cpu), ...
                 'cpu_cores', feature('numcores'), 'threads', nthreads); %#ok<NASGU>
tmp = fullfile(outdir, [method '.partial.mat']);
save(tmp, 'restored', 'time_s', 'params', 'machine', 'mpsnr', '-v7');
movefile(tmp, outfile, 'f');
fprintf('%s %s: MPSNR %.2f (%.1f s, %d threads)\n', method, scene, mpsnr, time_s, nthreads);
end
