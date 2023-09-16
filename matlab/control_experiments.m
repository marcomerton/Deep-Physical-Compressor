sys_idx = 2;
actuated_masses = [193, 163, 123];
nruns = 50;
start_from = 1;
rng(42);

datapath = "data/205masses/sys" + sys_idx + "/raw/";

warning('off', 'MATLAB:MKDIR:DirectoryExists');
for run_idx = 1:nruns
    start_sim_idx = 1; % randi([1,7], 1);
    stop_sim_idx = randi([1,7], 1);

    start_step_idx = 1; %randi([0,400], 1);
    stop_step_idx = randi([50,400], 1);

    actuated_mass_idx = randsample(actuated_masses, 1);

    if run_idx < start_from; continue; end
    xs = readmatrix(datapath + "sim" + start_sim_idx + "/x.csv");
    x0 = xs(start_step_idx, :);
    xs = readmatrix(datapath + "sim" + stop_sim_idx + "/x.csv");
    x_target = xs(stop_step_idx, :);

    res_dir = "results/control/sys" + sys_idx + "/run" + run_idx;
    mkdir(res_dir)

    disp("Run " + run_idx + ":")
    disp("Start conf.: sim" + start_sim_idx + "(" + start_step_idx + ")")
    disp("Stop conf.:  sim" + stop_sim_idx + "(" + stop_step_idx + ")")
    disp("Actuated mass: " + actuated_mass_idx);
    run_simulations
end