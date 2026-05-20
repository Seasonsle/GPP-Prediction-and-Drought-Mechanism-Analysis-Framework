function [gBestScore, gBest, cg_curve] = PSO(N, Max_iteration, lb, ub, dim, fobj, label)
% PSO Particle swarm optimization for hyperparameter search.
%
% Inputs
%   N              Number of particles.
%   Max_iteration  Number of optimization iterations.
%   lb, ub         Lower and upper bounds. Each can be scalar or 1-by-dim.
%   dim            Number of optimized parameters.
%   fobj           Objective function handle.
%   label          Optional initialization label passed to ys.m.
%
% Outputs
%   gBestScore     Best objective value.
%   gBest          Best parameter vector.
%   cg_curve       Best objective value at each iteration.

if max(size(ub)) == 1
    ub = ub * ones(1, dim);
    lb = lb * ones(1, dim);
end

vMin = -6;
vMax = 6;
nParticles = N;
wMax = 0.9;
wMin = 0.6;
c1 = 2;
c2 = 2;

vel = zeros(nParticles, dim);
pBestScore = inf(nParticles, 1);
pBest = zeros(nParticles, dim);
gBest = zeros(1, dim);
gBestScore = inf;
cg_curve = zeros(1, Max_iteration);

pos0 = ys(N, dim, label);
pos = zeros(nParticles, dim);
for i = 1:nParticles
    pos(i, :) = (ub - lb) .* pos0(i, :) + lb;
    vel(i, :) = vMin + rand(1, dim) * (vMax - vMin);
end

for i = 1:nParticles
    pBestScore(i) = fobj(pos(i, :));
    pBest(i, :) = pos(i, :);
    if gBestScore > pBestScore(i)
        gBestScore = pBestScore(i);
        gBest = pBest(i, :);
    end
end

for iter = 1:Max_iteration
    for i = 1:nParticles
        pos(i, :) = min(max(pos(i, :), lb), ub);
    end

    for i = 1:nParticles
        fitness = fobj(pos(i, :));
        if pBestScore(i) > fitness
            pBestScore(i) = fitness;
            pBest(i, :) = pos(i, :);
        end
        if gBestScore > fitness
            gBestScore = fitness;
            gBest = pos(i, :);
        end
    end

    w = wMax - iter * ((wMax - wMin) / Max_iteration);
    for i = 1:nParticles
        for j = 1:dim
            vel(i, j) = w * vel(i, j) ...
                + c1 * rand() * (pBest(i, j) - pos(i, j)) ...
                + c2 * rand() * (gBest(j) - pos(i, j));
            vel(i, j) = min(max(vel(i, j), vMin), vMax);
            pos(i, j) = pos(i, j) + vel(i, j);
        end
    end

    fprintf('PSO iteration %d best RMSE: %.6f\n', iter, gBestScore);
    cg_curve(iter) = gBestScore;
end
end
