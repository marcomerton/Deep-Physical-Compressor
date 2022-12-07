import os
import numpy as np


def get_number_of_simulations(name, path="data/"):
    """Returns the number of simulations for a given dataset name."""
    return len([0 for f in os.listdir(f"{path}{name}/raw/") if "sim" in f])


def get_train_test_simulations_index(name, path="data/"):
    """Returns the train and test simulation indexes for a given dataset name."""
    n_sims = get_number_of_simulations(name, path=path)
    idxs = np.arange(n_sims)
    return idxs[:-2], idxs[-2:]


def remove_masked(mask, edge_index, edge_attr=None):
    """"""
    num_fixed = sum(mask == 0)
    mask_ = ((edge_index - num_fixed) < 0).sum(0) == 0
    ei = edge_index[:, mask_] - num_fixed
    if edge_attr is not None:
        ea = edge_attr[mask_, :]
        return ei, ea
    else:
        return ei
