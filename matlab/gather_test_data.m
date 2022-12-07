addpath("shapes\", "utils\")

data_path = "../data/"; % Path where systems data is saved
n_sys = 5;              % Number of systems
seed = 42;              % RNG seed


global Minv A D k l mask tau ks;

warning('off', 'MATLAB:MKDIR:DirectoryExists');
for sys_idx = 1:n_sys
    disp("===== SYSTEM " + sys_idx + " =====");

    sys_prefix = "../data/sys" + sys_idx + "/raw/";
    sys = MassSpringSystem.load(sys_prefix);
    x_rest = sys.rest(:);

    save_prefix = "../results/sys" + sys_idx + "/";
    mkdir(save_prefix)

    % Initialize constant variables for the simulations
    k = interleave(sys.k);
    d = interleave(sys.d);
    l = interleave(sys.l);
    A = edge_index_to_matrix(sys.adj, sys.n_conn, sys.n_masses);
    M = 1/sys.n_masses * eye(2 * sys.n_masses);
    Minv = inv(M);
    D = A' * (d .* A);
    mask = interleave(sys.mask);
    ks = 0.2;
    f_rest = ks * (A' * A) * x_rest;

    % Reset rng
    rng(seed);

    % Set initial conditions
    x0 = x_rest;
    dx0 = zeros(size(x_rest));

    gs = zeros(1,30);
    thetas = zeros(1,30);
    for i=8:37
        disp("Sim: " + i);
        % Select gravity intensity and orientation
        theta = -pi/2 + pi/4*rand() * (-1)^i;
        v = 3 + 14 * rand();
        g = get_gravity(v, theta, size(x0));
        tau = g * (1 / sys.n_masses) + f_rest;

        % Simulation
        out = sim('spring_network.slx', ...
            'StartTime', '0', 'StopTime', "5", ...
            'FixedStep', '5e-3');
        
        % Save data
        folder = save_prefix + "sim" + i + "/";
        mkdir(folder);
        writematrix(out.x(1:2:end, :), folder + "x.csv");
        writematrix(out.dx(1:2:end, :), folder + "dx.csv");
        writematrix(out.tout(1:2:end, :), folder + "t.csv");

        % Select initial configuration for the next simulation
        if rand() > 0.8; x0 = x_rest;
        else; x0 = out.x(end, :)'; end

        gs(i-7) = g;
        thetas(i-7) = g;
    end

    writematrix(gs, "../results/intensity.csv")
    writematrix(thetas, "../results/theta.csv")

    disp("=====================");
end
