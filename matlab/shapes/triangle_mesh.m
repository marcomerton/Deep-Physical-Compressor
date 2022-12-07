function system = triangle_mesh(width, k_, l_, d_)
%TRIANGLE ...
%

% Useful constant
r = l_ * sqrt(3) / 2;

% System constants
system = struct();
system.n_masses = (width+2) * (width+1) / 2;
system.n_conn = 3 * width * (width+1) / 2;
system.n_fixed = width + 1;
system.type = MeshType.TRIANGLE;
system.groups = zeros(1, system.n_masses);

% Initialize connections data
adj = zeros(system.n_conn, 2);
k = k_ * ones(1, system.n_conn);
d = d_ * ones(1, system.n_conn);
l = l_ * ones(1, system.n_conn);

% Build mobile mask
mask = ones(1, system.n_masses);
mask(1:(width+1)) = 0;
system.mask = mask;

% Initialize starting position
q0 = zeros(2, system.n_masses);

% Top connections and masses
q0(:, 1) = [0; 0];
for i=1:width
    adj(i, :) = [i+1, i];
    q0(:, i+1) = q0(:, i) + [l_; 0];
end


% Building loop
m1 = 1;
m2 = 2 + width;
idx = width;

for i=width:-1:1
    for j=1:i
        % Initial position
        q0(:, m2) = q0(:, m1) + [l_/2; -r];
        
        % Connection to masses in previous row
        adj(idx+1, :) = [m2, m1];
        adj(idx+2, :) = [m2, m1+1];
        
        % If possible, connect to mass on the left
        if(j>1)
            adj(idx+3, :) = [m2, m2-1];
            idx = idx + 1;
        end

         % Move forward
        idx = idx + 2;
        m1 = m1 + 1;
        m2 = m2 + 1;
    end

    % Adjust index
    m1 = m1 + 1;
end

% Add everything to struct
system.adj = adj;
system.k = k;
system.l = l;
system.d = d;
system.q0 = q0;


% Attach points
system.top_attach_points = 1:(width+1);

system.left_attach_points = 1:(width+1);
for i=2:(width+1)
    system.left_attach_points(i:end) = ...
        system.left_attach_points(i:end) + width + 2 - i;
end

system.right_attach_points = system.left_attach_points + (width:-1:0);
system.right_attach_points = flip(system.right_attach_points);

end