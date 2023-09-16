import sys
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from utils.plots import draw_graph, update_graph

num_masses = 205
sys_idx = 2

colors = ['#ADA617', '#FAA546', '#1F39AD']
y_range = [-12, 0]
x_range = [-7, 7]
fps = 24

def make_control_gif(run, ctrl):
    datapath = f"data/205masses/sys{sys_idx}/raw/"
    respath = f"results/control/sys{sys_idx}/run{run}"
    edge_index = torch.LongTensor(np.genfromtxt(f"{datapath}adj.csv", delimiter=',')).T - 1

    xs = np.genfromtxt(f"{respath}/{ctrl}-xs.csv", delimiter=",").reshape(-1, 205, 2)
    target = np.genfromtxt(f"{respath}/target.csv", delimiter=',').reshape(205, 2)
    
    fig, ax = plt.subplots(1,1, figsize=(7,6))

    _, lt = draw_graph(target, edge_index, color=colors[1])

    empty_data = np.empty((num_masses, 2))
    sc, ls = draw_graph(empty_data, edge_index, ax=ax, color=colors[2])

    @torch.no_grad()
    def animate(x):
        update_graph(sc, ls, x, edge_index)

    ax.set(ylim=y_range, xlim=x_range)
    ax.legend(
        [lt[0], ls[0]],
        ["Target (Original)", "Simulated system"],
        loc = "lower center",
        ncol = 2
    )

    idxs = range(0, len(xs), len(xs)//(fps*5))
    xs = xs[idxs]
  
    ani = FuncAnimation(fig, animate,
            frames=xs,
            interval=1000/fps,
            save_count=len(xs)
    )

    wr = FFMpegWriter(fps=fps)
    ani.save(f"{respath}/{ctrl}.gif", writer = wr)
    plt.close()



if __name__ == '__main__':
    make_control_gif(sys.argv[1], sys.argv[2])
