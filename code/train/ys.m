function pos = ys(N, dim, ~)
%YS Initialize particle positions in [0, 1] for PSO.
% The third input is retained for compatibility with PSO.m.
pos = rand(N, dim);
end
