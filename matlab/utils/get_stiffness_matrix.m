function K = get_stiffness_matrix(x, A, k, l, ks)
%GET_STIFFNESS_MATRIX ...
%
    n_rows = size(A, 1);
    eps = 1e-12; % Added to distances for stability
    
    % Computes the distances between connected masses
    dist = zeros(n_rows, 1);
    temp = (A * x) .^ 2;
    for i=1:2:n_rows
        d = sqrt(temp(i) + temp(i+1));
        dist([i, i+1]) = d;
    end
    z = k .* (1 - l ./ (dist+eps)) + ks;
    K = (A' .* z') * A;

end