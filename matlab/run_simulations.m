%% Define paths 
datapath = "data/205masses/sys" + sys_idx + "/raw/";
configfile = "results/sys" + sys_idx + "/models/ae3.json";
modelfile = "results/sys" + sys_idx + "/models/ae3";


%% Load system and select starting and target config.
sys = MassSpringSystem.load(datapath);
xs = readmatrix(datapath + "sim" + start_sim_idx + "/x.csv");
x0 = xs(start_step_idx, :);

writematrix(x0, res_dir + "/start.csv");
xs = readmatrix(datapath + "sim" + stop_sim_idx + "/x.csv");
x_target = xs(stop_step_idx, :);
writematrix(x_target, res_dir + "/target.csv");
writematrix(actuated_mass_idx, res_dir + "/mass_idx.csv")


%% Initialize constants
global Minv A D k l mask tau_ref tau_base ks xi_ref m a S p_ref;
k = interleave(sys.k);  
d = interleave(sys.d);
l = interleave(sys.l);
A = edge_index_to_matrix(sys.adj, sys.n_conn, sys.n_masses);
mask = interleave(sys.mask);
M = 1/sys.n_masses * eye(2 * sys.n_masses);
Minv = inv(M);
D = A' * diag(d) * A;
x_rest = sys.rest(:);
ks = 0.1;
dstar = ks * (A' * A) * x_rest;
tau_base = dstar;
a = actuated_mass_idx;
S = zeros(2, 2 * sys.n_masses);
S(:, (2*a-1):(2*a)) = eye(2);
p_ref = x_target((2*a-1):(2*a))';


%% Setup python env & variables
conn_mask = sum((sys.adj - sys.n_fixed) <= 0, 2) == 0;
adj_ = sys.adj(conn_mask, :) - sys.n_fixed;
attr_ = [sys.k(conn_mask); sys.d(conn_mask); sys.l(conn_mask)]';
groups_ = sys.groups(sys.mask == 1);
rest_ = sys.rest(:, sys.mask == 1);
terminate(pyenv)
pyenv("ExecutionMode", "OutOfProcess");
py_setup();
py_load_data(rest_, adj_, attr_, groups_);
py_load_model(configfile, modelfile);


%% Compute target latent state and reconstructed target config.
m = 2*sys.n_fixed + 1;
xm_bar = x_target(m:end);
xb = x_rest(1:(m-1));
xi_ref = py_encode(xm_bar)';
[rec, J, ~] = py_decoder_dyn(xi_ref, zeros(size(xi_ref)));
writematrix(rec, res_dir + "/target_rec.csv");


%% Compute reference force
S = S(:, m:end);
SJplus = pinv((S*J)');
K = get_stiffness_matrix([xb; rec'], A, k, l, ks);
Kmm = K(m:end, m:end);
Kmb = K(m:end, 1:(m-1));
tau_ref = SJplus * (J' * (Kmm * rec') + J' * (Kmb * xb));


%% Simulation
if exist('stoptime',  'var') == 0;  stoptime = 5;  end
dx0 = zeros(size(x0));

out_ae = sim('sim_AE_dyn_ctrl.slx', ...
        'StartTime', '0', 'StopTime', string(stoptime), ...
        'FixedStep', '5e-3');
writematrix(out_ae.x, res_dir + "/ae-xs.csv");
writematrix(out_ae.dx, res_dir + "/ae-dxs.csv");
