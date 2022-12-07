classdef MassSpringSystem
%
properties
    n_masses    (1,1)   % Number of masses
    n_conn      (1,1)   % Number of connections
    n_fixed     (1,1)   % Number of fixed masses
    adj         (:,2)   % Connections index
    k           (1,:)   % Stiffness coefficients vector
    l           (1,:)   % Rest length vector
    d           (1,:)   % Damping coefficients vector
    rest        (2,:)   % Masses rest positions
    groups      (1,:)   % Groups index
    mask        (1,:)   % Mobile masses mask

    top_att     (1,:)   % Top attach points indexes
    left_att    (1,:)   % Left attach points indexes
    right_att   (1,:)   % Right attach points indexes
    bottom_att  (1,:)   % Bottom attach points indexes

    rot         (1,1)   % Sides attach angle
end


methods (Access=public)

    function obj = connect(obj, sys, dir)
    %CONNECT ...

        % This also removes the attach points
        switch(dir)
            case "left"
                at1 = obj.left_att;
                obj.left_att = [];
                theta = obj.rot;
            case "right"
                at1 = obj.right_att;
                obj.right_att = [];
                theta = - obj.rot;
            case "bottom"
                at1 = obj.bottom_att;
                obj.bottom_att = [];
                theta = 0;
        end
        at2 = sys.top_att;
        len = length(at2);

        % Transform (rotale + translate) points in the second system
        translation = obj.rest(:, at1(1));
        rotmat = [cos(theta), sin(theta); -sin(theta), cos(theta)];
        sys.rest = rotmat * sys.rest + translation;

        % Remove attach points from second system
        sys.rest(:, at2) = [];
        sys.groups(at2) = [];
        sys.mask(at2) = [];

        % Renumerates nodes and groups in the second system
        sys.adj = sys.adj + obj.n_masses;
        at2 = at2 + obj.n_masses;
        masks = zeros([len, size(sys.adj)], "logical");
        for i=1:len
            masks(i, :, :) = (sys.adj == at2(i));
        end
        for i=1:len
            sys.adj(masks(i, :, :)) = at1(i);
        end
        for el = sort(at2, 'descend')
            grt_i = (sys.adj > el);
            sys.adj(grt_i) = sys.adj(grt_i) - 1;
        end
        sys.groups = sys.groups + max(obj.groups) + 1;

        % Update masses
        obj.rest = [obj.rest, sys.rest];
        obj.n_masses = size(obj.rest, 2);
        obj.groups = [obj.groups, sys.groups];
        obj.mask = [obj.mask, sys.mask];

        % Update connections (this also removes duplicate edges)
        obj.adj = [obj.adj; sys.adj];
        [obj.adj, ia, ~] = unique(sort(obj.adj, 2), 'rows', 'stable');
        obj.n_conn = length(ia);
        obj.k = [obj.k, sys.k]; obj.k = obj.k(ia);
        obj.l = [obj.l, sys.l]; obj.l = obj.l(ia);
        obj.d = [obj.d, sys.d]; obj.d = obj.d(ia);
    end


    function save(obj, f_prefix)
    %SAVE ...
        writematrix(obj.adj, f_prefix + "adj.csv");
        writematrix([obj.k; obj.d; obj.l]', f_prefix + "attr.csv");
        writematrix(obj.groups, f_prefix + "groups.csv");
        writematrix(obj.mask, f_prefix + "mask.csv");
        writematrix(obj.rest', f_prefix + "rest.csv");
    end

end % Methods (Access=public)

    
methods (Static)

    function obj = GridMesh(height, width, k, l, d)
        sys = grid_mesh(height, width, k, l, d);
        obj = MassSpringSystem.fromStruct(sys);
    end

    function obj = TriangleMesh(width, k, l, d)
        sys = triangle_mesh(width, k, l, d);
        obj = MassSpringSystem.fromStruct(sys);
    end

    function obj = fromStruct(sys)
    % FROMSTRUCT ...
    %
        obj = MassSpringSystem();
        obj.n_masses = sys.n_masses;  obj.n_conn = sys.n_conn;
        obj.adj = sys.adj;  obj.k = sys.k;  obj.l = sys.l;  obj.d = sys.d;
        obj.rest = sys.q0;  obj.groups = sys.groups;
        obj.mask = sys.mask;
        obj.n_fixed = obj.n_masses - sum(obj.mask);
    
        obj.top_att = sys.top_attach_points;
        obj.left_att = sys.left_attach_points;
        obj.right_att = sys.right_attach_points;
        

        if sys.type == MeshType.GRID
            obj.bottom_att = sys.bottom_attach_points;
            obj.rot = pi/2;
        elseif sys.type == MeshType.TRIANGLE
            obj.bottom_att = [];
            obj.rot = pi/3;
        end
    end


    function obj = load(f_prefix)
        obj = MassSpringSystem();
        obj.adj = readmatrix(f_prefix + "adj.csv");
        attr = readmatrix(f_prefix + "attr.csv")';
        obj.k = attr(1, :);
        obj.d = attr(2, :);
        obj.l = attr(3, :);
        obj.groups = readmatrix(f_prefix + "groups.csv");
        obj.mask = readmatrix(f_prefix + "mask.csv");
        obj.rest = readmatrix(f_prefix + "rest.csv")';
        obj.n_conn = size(obj.adj, 1);
        obj.n_masses = size(obj.rest, 2);
        obj.n_fixed = obj.n_masses - sum(obj.mask);
    end

end % Methods (Static)

end