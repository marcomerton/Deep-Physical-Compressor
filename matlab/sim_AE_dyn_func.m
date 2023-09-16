function ddq = sim_AE_dyn_func(x, dx)

global Minv A D k l mask tau_ref tau_base ks xi_ref m a S;

alpha = 1;
beta = 2;

% Autoencoder dynamics
[xi, xi_dot] = py_encoder_dyn(x(m:end), dx(m:end));
J = py_jacobian(xi);

% Compute force
perror = xi_ref - xi;
derror = xi_dot';
tau = tau_base;
tau((2*a-1):(2*a)) = tau_ref + S * J * (alpha * perror - beta * derror);

% Compute acceleration
K = get_stiffness_matrix(x, A, k, l, ks);
ddq = - Minv * (D*dx + K*x - tau);
ddq = ddq .* mask;

end