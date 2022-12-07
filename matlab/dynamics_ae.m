function ddq = dynamics_ae(q, dq)

global M22 D22 A k l xb tau ks;

[rec, J, Jdot] = py_decoder_dyn(q, dq);
rec = rec';

m = length(xb) + 1;
K = get_stiffness_matrix([xb; rec], A, k, l, ks);
Kmm = K(m:end, m:end);
Kmb = K(m:end, 1:(m-1));

B = J' * M22 *  J;
C = J' * M22 * Jdot;
G = J' * D22 * J;
Fb = J' * Kmb;
Fm = J' * Kmm;
ext = J' * tau;

ddq = - B \ (C*dq + G*dq + Fb*xb + Fm*rec - ext);

end