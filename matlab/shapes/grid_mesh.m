function system = grid_mesh(height, width, k_, l_, d_)
%RECTANGLE ...
%

% Useful constant
r = l_ * sqrt(2);

% System constants
system = struct();
system.n_masses = (width + 1) * (height + 1);
system.n_conn = width + height + 4 * width * height;
system.n_fixed = width + 1;
system.type = MeshType.GRID;
system.groups = zeros(1, system.n_masses);

% Initialize connections data
adj = zeros(system.n_conn, 2);
k = k_ * ones(1, system.n_conn);
d = d_ * ones(1, system.n_conn);
l = l_ * ones(1, system.n_conn);

% Build force mask
mask = ones(1, system.n_masses);
mask(1:(width+1)) = 0;
system.mask = mask;

% Initialize starting position
q0 = zeros(2, system.n_masses);

% Top connections and masses
for i=1:width
    adj(i, :) = [i, i+1];
    q0(:, i) = [(i-1) * l_, 0];
end
q0(:, width+1) = [width*l_, 0];


% Building loop
idx = width;
m_up = 1;
m_low = width + 2;

for h=1:height
    % Left-most connection and mass
    adj(idx+1, :) = [m_low, m_up];
    q0(:, m_low) = q0(:, m_up) - [0; l_];

    % Move forward
    m_up = m_up + 1;
    m_low = m_low + 1;
    idx = idx + 1;

    for w=1:width
        % Initial position
        q0(:, m_low) = q0(:, m_up) - [0; l_];

        % Connections
        adj(idx+1, :) = [m_low, m_up];
        adj(idx+2, :) = [m_low, m_low-1];
        adj(idx+3, :) = [m_low, m_up-1]; l(idx+3) = r; k(idx+3) = 2*k_;
        adj(idx+4, :) = [m_up, m_low-1]; l(idx+4) = r; k(idx+4) = 2*k_;

        % Move forward
        m_up = m_up + 1;
        m_low = m_low + 1;
        idx = idx + 4;
    end
end

% Add everything to struct
system.adj = adj;
system.k = k;
system.l = l;
system.d = d;
system.q0 = q0;


% Attach points
system.top_attach_points = 1:(width+1);
system.left_attach_points = 1:(width+1):system.n_masses;
system.right_attach_points = flip(system.left_attach_points + width);
system.bottom_attach_points = system.n_masses + (-width:0);

end