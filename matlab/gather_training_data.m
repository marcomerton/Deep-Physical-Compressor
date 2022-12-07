addpath("shapes\", "utils\")

gen = SystemGenerator(6,6,4, 50,0.3, 0.1, 0.4,0.8); % System generator
n_sys = 5;          % Number of systems
path = "../data/";  % Folder where the data is saved
seed = 1234;        % RNG seed
rng(seed);

% Gravity intensity and orientation angles
trajectories = {
    [get_gravity(9.81, -3/4*pi, [1,2]);
     get_gravity(9.81,   -pi/3, [1,2])];

    [get_gravity(9.81,   -pi/4, [1,2]);
     get_gravity(9.81, -2/3*pi, [1,2])];

    [get_gravity(9.81,   -pi/2, [1,2])];
   
    [get_gravity(6,  -1/3*pi, [1,2]);
     get_gravity(14, -2/3*pi, [1,2])];
};

% Small log
disp("Number of systems: " + n_sys);
disp("Saving in: " + path);
disp("Seed: " + seed);

warning('off', 'MATLAB:MKDIR:DirectoryExists');
for i = 1:n_sys
    disp("===== SYSTEM " + i + " =====");
    sys = gen.generate();
    f_prefix = path + "sys" + i + "/raw/";
    mkdir(f_prefix);
    sys.save(f_prefix);
    simulate_system % This is the actual simulation
    disp("=====================");
end