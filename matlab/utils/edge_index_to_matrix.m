function A = edge_index_to_matrix(adj, n_springs, n_masses)
%EDGE_INDEX_TO_MATRIX Build the (signed) adjacency matrix corresponding
%to the given set of edges.
%
%   Each edge is specified as a 2-elements row of the matrix 'adj'.
%   Edges are meant to represent 2-dimensional connections. Therefore, two
%   rows are generated for each edge, one for each dimension.

A = zeros(2*n_springs, 2*n_masses);
for i=1:n_springs
    m1 = adj(i,1); m2 = adj(i,2);

    x1 = 2*m1-1; y1 = x1+1;
    x2 = 2*m2-1; y2 = x2+1;

    idx2 = 2 * i; idx1 = idx2 - 1;

    if x1 > 0; A(idx1, x1) = 1; end
    if x2 > 0; A(idx1, x2) = -1; end
    if y1 > 0; A(idx2, y1) = 1; end
    if y2 > 0; A(idx2, y2) = -1; end

end