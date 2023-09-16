import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


sys_idx = 2
num_masses = 205

normalsize = 14
titlesize = 17


def error_through_time(xs, target):
    return torch.nn.functional.mse_loss(
        xs, target.repeat(xs.shape[0], 1), reduction='none'
    ).mean(dim=1)


def make_plot_quantile(nruns, ctrl, mask, ax, scale='linear', add_mean=False, **plot_kw):
    errors = []
    for flag, run in zip(mask, range(nruns)):
        if not flag: continue

        respath = f"results/control/sys{sys_idx}/run{run+1}"
        target = torch.tensor(np.genfromtxt(f"{respath}/target.csv", delimiter=','))
        xs = torch.tensor(np.genfromtxt(f"{respath}/{ctrl}-xs.csv", delimiter=","))
    
        error = error_through_time(xs, target)
        errors.append(error)
    errors = torch.stack(errors).numpy()

    ax.set_xlabel("$t$ ($s$)", fontsize=titlesize)
    ax.set_ylabel("$MSE$", fontsize=titlesize)
    ax.set_yscale(scale)
    if scale != "log":
        ax.set(ylim=[0,1])
    _x = np.linspace(0, errors.shape[1] // 200, errors.shape[1])
    ax.fill_between(
        _x, np.quantile(errors, .25, axis=0), np.quantile(errors, .75, axis=0),
        alpha=0.4, **plot_kw,
    )
    res = []
    res.append( ax.plot(_x, np.median(errors, axis=0), linestyle='--', **plot_kw)[0] )
    if add_mean:
        res.append( ax.plot(_x, np.mean(errors, axis=0), **plot_kw)[0] )
    return res


def get_outliers_mask(threshold, nruns, ctrl):
    mask = torch.ones(nruns)
    for run in range(nruns):
        respath = f"results/control/sys{sys_idx}/run{run+1}"
        target = np.genfromtxt(f"{respath}/target.csv", delimiter=',')
        xs = np.genfromtxt(f"{respath}/{ctrl}-xs.csv", delimiter=",")

        error = torch.nn.functional.mse_loss(
            torch.Tensor(xs[-1]),
            torch.Tensor(target),
        )
        mask[run] = error < threshold
    return mask



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create error plots.")
    parser.add_argument("nruns", type=int)
    parser.add_argument("-t", "--threshold", type=float, default=1.)
    parser.add_argument("-y", "--yscale", default="linear")
    parser.add_argument("-a", "--all", action="store_true")
    parser.add_argument("-m", "--mean", action="store_true")

    args = parser.parse_args()
    ctrl = "ae"

    filename = "results/control/sys{}/{}-error-{}{}{}.pdf".format(
        sys_idx,
        ctrl,
        args.yscale,
        "-masked" if not args.all else "",
        "-mean" if args.mean else "",
    )


    mask = (
        get_outliers_mask(args.threshold, args.nruns, ctrl) 
        if not args.all else torch.ones(args.nruns)
    )
    print(f"Masked sims: {mask.sum()}")

    labels = ["Median", "Mean"] if args.mean else ["Median"]

    mpl.rcParams["lines.linewidth"] = 2
    mpl.rcParams["font.size"] = normalsize
    mpl.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(1,1, figsize=(10,4))
    lines = make_plot_quantile(
        args.nruns, ctrl, mask, ax,
        scale=args.yscale, add_mean=args.mean, color="tab:blue",
    )
    ax.set_xlim(xmin=0, xmax=5)
    ax.grid()
    fig.legend(lines, labels)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
