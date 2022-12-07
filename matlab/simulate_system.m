% Assumes the following variables are in the workspace:
%   sys      : system to be simulated
%   f_prefix : Prefix of the files where the data is saved
%   trajectories: Cell array containing one array for each trajectory.
%   stoptime : length (in seconds) of the simulation (default: 5)
% 
% Each trajectory must specify a set of gravity vectors, one for each step,
% as a [n_steps x 2] matrix, such that row i is the gravity at step i.
if ~ exist("stoptime", "var"); stoptime = 5; end


% Initialize constant variables for the simulations
global Minv A D k l mask tau ks;
k = interleave(sys.k);  
d = interleave(sys.d);
l = interleave(sys.l);
A = edge_index_to_matrix(sys.adj, sys.n_conn, sys.n_masses);
mask = interleave(sys.mask);
M = 1/sys.n_masses * eye(2 * sys.n_masses);
Minv = inv(M);
D = A' * diag(d) * A;
x_rest = sys.rest(:);
ks = 0.2;
dstar = ks * (A' * A) * x_rest;

run = 1;
for i = 1:length(trajectories)
    gravities = trajectories{i};
    
    % (Re)set initial conditions
    x0 = x_rest;
    dx0 = zeros(size(x0));

    for g = gravities'
        % Prepare external force vector
        g_ = repmat(g, sys.n_masses, 1);
        tau = g_ .* diag(M) + dstar;

        % Actual simulation
        if run ~= 1; disp("- - - - - - - -"); end
        disp("Sim: " + run);
        tic;
        out = sim('spring_network.slx', ...
            'StartTime', '0', 'StopTime', string(stoptime), ...
            'FixedStep', '5e-3');
        time = toc;
        disp("Simulations took: " + time + " seconds");

        % Save simulation to file
        dir = f_prefix + "sim" + run;
        mkdir(dir)
        writematrix(out.x, dir + "/x.csv");
        writematrix(out.dx, dir + "/dx.csv");
        
        % Update variables for the next iteration
        x0 = out.x(end, :)';
        dx0 = out.dx(end, :)';
        run = run + 1;
    end
end