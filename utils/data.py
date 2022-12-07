import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected

from dataset.utils import remove_masked
from .misc import from_dictionary


def load_system(sys, load_attr=False, dataset="205masses"):
    edge_index = torch.LongTensor(
                np.genfromtxt(f"data/{dataset}/sys{sys}/raw/adj.csv", delimiter=',')).T - 1
    mask = torch.LongTensor(np.genfromtxt(f"data/{dataset}/sys{sys}/raw/mask.csv", delimiter=','))
    rest = torch.Tensor(np.genfromtxt(f"data/{dataset}/sys{sys}/raw/rest.csv", delimiter=','))
    
    num_masses = len(mask)
    num_fixed = sum(mask == 0)

    if load_attr:
        edge_attr = torch.Tensor(np.genfromtxt(f"data/{dataset}/sys{sys}/raw/attr.csv", delimiter=','))
        ei, ea = remove_masked(mask, edge_index, edge_attr)
    else:
        ei = remove_masked(mask, edge_index)
        ea = None
    
    d = Data(pos=rest.view(num_masses,2)[num_fixed:,:], edge_index=ei, edge_attr=ea)
    d = ToUndirected()(d)
    return d, num_masses, num_fixed


def load_model(sys, name, device='cpu'):
    if isinstance(sys, int):
        path = f"results/sys{sys}/models/{name}"
    else:
        path = f"results/{sys}/models/{name}"

    model = from_dictionary(f"{path}.json")
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model


def load_simulation(sys, sim, name=None):
    real = torch.Tensor(np.genfromtxt(f"results/sys{sys}/sim{sim}/x.csv", delimiter=','))
    time = np.genfromtxt(f"results/sys{sys}/sim{sim}/t.csv", delimiter=',')
    
    if name is not None:
        reduced = torch.Tensor(np.genfromtxt(f"results/sys{sys}/sim{sim}/{name}/q.csv", delimiter=','))
        return real, time, reduced
    
    return real, time
