function ddq = dynamics(x, dx)

global Minv A D k l mask tau ks;

K = get_stiffness_matrix(x, A, k, l, ks);

ddq = - Minv * (D*dx + K*x - tau);
ddq = ddq .* mask;

end