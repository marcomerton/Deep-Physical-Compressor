classdef SystemGenerator
    properties
        n_grids         (1,1)   % Number of grid pieces
        n_triangles     (1,1)   % Number of traingle pieces
        width           (1,1)   % Width of the pieces
        k_              (1,1)   % Springs stiffness
        l_              (1,1)   % Springs length
        d_              (1,1)   % Damping coefficients
        beta            (1,1)   % Probability of lateral structure
        gamma           (1,1)   % Beta decay factor
    end


methods (Access=public)

    % Constructor
    function obj = SystemGenerator(ng, nt, w, k, l, d, b, g)
        obj.n_grids = ng;
        obj.n_triangles = nt;
        obj.width = w;
        obj.k_ = k;
        obj.l_ = l;
        obj.d_ = d;
        obj.beta = b;
        obj.gamma = g;
    end

    function sys = generate(obj)
    %GENERATE Returns a randomly generated system as a struct.
    %
        systems = [repmat([MeshType.GRID], 1, obj.n_grids), ...
                   repmat([MeshType.TRIANGLE], 1, obj.n_triangles)];
        systems = systems(randperm(length(systems)));


        dirs = ["left", "right"];
        dirs = dirs(randperm(2));

        [sys, ~, ~] = obj.gen(systems(1), dirs);
        idx = 2;
        while(idx <= length(systems))
            [new_base, dir1, dir2] = obj.gen(systems(idx), dirs);
            if systems(idx) == MeshType.TRIANGLE
                dirs = dirs(end:-1:1);
            end
            idx = idx + 1;

            p = obj.beta;
            if rand() < p && idx <= length(systems)
                dirs_old = dirs;
                dirs = dirs(randperm(2));

                [sys1, ~, ~] = obj.gen(systems(idx), dirs);
                idx = idx + 1;
                p = p * obj.gamma;
                while(rand() < p && idx <= length(systems))
                    [sys2, dir3, ~] = obj.gen(systems(idx), dirs);
                    if systems(idx) == MeshType.TRIANGLE
                        dirs = dirs(end:-1:1);
                    end
                    idx = idx + 1;
                    p = p * obj.gamma;
                    sys1 = sys2.connect(sys1, dir3);
                end
                new_base = new_base.connect(sys1, dir2);
                dirs = dirs_old;
            end
            sys = new_base.connect(sys, dir1);
        end

        [base, dir, ~] = obj.gen(MeshType.GRID);
        sys = base.connect(sys, dir);
    end

end % methods (Access=public)


% Private methods
methods (Access=private)

    function [s, d1, d2] = gen(obj, type, dirs)
    %GEN Returns a single mesh of the specified type together with two
    %random attach points
        if type == MeshType.GRID
            s = MassSpringSystem.GridMesh( ...
                    obj.width, obj.width, obj.k_, obj.l_, obj.d_);
            d1 = "bottom";
            d2 = randsample(["left", "right"], 1);

        elseif type == MeshType.TRIANGLE
            s = MassSpringSystem.TriangleMesh( ...
                    obj.width, obj.k_, obj.l_, obj.d_);
            d1 = dirs(1);
            d2 = dirs(2);
        end
    end

end % methods (Access=private)

end