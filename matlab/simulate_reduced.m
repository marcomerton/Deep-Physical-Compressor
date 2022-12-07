%% Parameters
% Assumes the following variables are in the workspace:
%   sys_idx: System number (1 to 5)
%   sim_idx: Simulation number (8 to 37)
%   model_name: Name of the model to use (AE or GAE)

addpath("matlab/python", "matlab/shapes", "matlab/utils")

disp("System number: " + sys_idx);
disp("Simulation number: " + sim_idx);
disp("Model name: " + model_name);


% Load system
sys = MassSpringSystem.load(sys_prefix);
x_rest = sys.rest(:);

x0 = readmatrix("results/sys" + sys_idx + "/sim" + sim_idx + "/x.csv");
x0 = x0(1, :)';

% Load gravity
gs = readmatrix("results/intensity.csv");
thetas = readmatrix("results/theta.csv");

theta = thetas(sim_idx-7);
intensity = gs(sim_idx-7);
gravity = [cos(theta); sin(theta)] * intensity;


% Simulation constants
global Minv A D k l mask tau ks M22 D22 xb;
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

% External force
g = repmat(gravity, sys.n_masses, 1);
external = g * (1 / sys.n_masses) + f_rest;

% These are for the reduced simulation
m = 2 * sys.n_fixed + 1; % Beginning of mobile part
M22 = M(m:end, m:end);
D22 = D(m:end, m:end);
xb = x_rest(1:(m-1));


%% Setup Python variables
% This removes the connection (edges) of the fixed masses
% Assumes the fixed masses have the lowest indeces.
conn_mask = sum((sys.adj - sys.n_fixed) <= 0, 2) == 0;
adj_ = sys.adj(conn_mask, :) - sys.n_fixed;
attr_ = [sys.k(conn_mask); sys.d(conn_mask); sys.l(conn_mask)]';

groups_ = sys.groups(sys.mask == 1);
rest_ = sys.rest(:, sys.mask == 1);

% Reset python environment
terminate(pyenv)
pyenv("ExecutionMode", "OutOfProcess");

py_setup();
py_load_data(rest_, adj_, attr_, groups_);
py_load_model(int32(sys), name);


%% Reduced Siumlation
xm_mat = reshape(x0(m:end), 2, sys.n_masses-(m-1)/2)';
q0 =  py_encode(xm_mat);
dq0 = zeros(size(q0));
tau = external(m:end);

disp("Starting reduced simulation...");
out = sim('spring_network_AE.slx', ...
    'StartTime', '0', 'StopTime', "5", ...
    'FixedStep', '1e-2');

writematrix(out.q, save_prefix + "q.csv");
writematrix(out.dq, save_prefix + "dq.csv");
writematrix(out.ddq, save_prefix + "ddq.csv");
