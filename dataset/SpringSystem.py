import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.collate import collate



class SpringSystemDataset(InMemoryDataset):
    """Dataset of configurations coming from the simulation of a mass-spring system.
    Each graph corresponds to one configuration encountred during a simulation of the system.

    Raw data should be organized as follows:
        - 1 file (adj.csv) containing the edge index corresponding to the network's connections.
        This should be a 2-columns matrix containing, in each row, the indexes of a pair of
        connected nodes' (indexes should start from 1 as in Matlab)
        - 1 file (attr.csv) containing the connections attributes. These should be contained in a
        ExA matrix where E is the number of edges and A is the number of attributes
        - 1 file (groups.csv) containing the group index of each node
        - 1 file (mask.csv) containing the fixed nodes' mask. Nodes whose corresponding element is
        zero maintain the same position throughout the simulation (therefore could be treated in a
        special way)
        - 1 file (rest.csv) containing the system's rest positions. This should be Nx2 matrix where
        each row contain the rest position of one node
        - 1 folder per simulation (sim<sim_idx>) containing, a 'x.csv' file storing, one per
        row, the configuration (positions) of the system's masses in one simulation step. The
        positions should be arranged as [x1 y1 x2 y2 ... xn yn]

    The dataset creation is controlled by the following parameters:
        - root: folder containing the raw data folder (eg. 'train' -> 'train/raw/')
        - sim_indexes: indexes of the simulations to include in the dataset
        - device: device where data should be loaded
        - balance: whether to remove similar configuration from the dataset. If set to True,
        configurations within a distance of 5e-2 will be removed. If set to any scalar, that
        value will be used as threshold for removing the configurations. If set to False or
        0 all the configurations are kept
        - remove_fixed: whether to remove fixed nodes from the graph
        - use_edge_attr: whether to add connection attributes (stiffness, rest length,
        damping) to the data
        - **kwargs: any other parameter to PyG.data.InMemoryDataset
    """

    def __init__(self, root, sim_indexes=None, device='cpu',
            balance=True, remove_fixed=True, use_edge_attr=False,
            **kwargs
    ):
        self.balance = balance
        self.remove_fixed = remove_fixed

        super().__init__(root, **kwargs)

        if sim_indexes is None: sim_indexes = torch.arange(0, len(self.processed_paths))
        self.data, self.slices = self._load(sim_indexes, device)

        # A bit tricky but works
        self.masses = self.slices['pos'][1].item()

        self.edge_attributes = 3
        if not use_edge_attr:
            self.data.edge_attr = None
            self.edge_attributes = 0
        
        self.sim_idxs = sim_indexes


    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self):
        suffix = ""
        if self.balance: suffix = suffix + "_balanced"
        if self.remove_fixed: suffix = suffix + "_masked"
        n_sims = len([1 for f in self.raw_file_names if "sim" in f])
        return [f"data_{idx+1}{suffix}.pt" for idx in range(n_sims)]


    @property
    def num_edge_attributes(self):
        return self.edge_attributes

    @property
    def num_masses(self):
        return self.masses
    
    @property
    def num_simulations(self):
        return len(self.sim_idxs)


    def process(self):
        """
        """

        # Load system information
        prefix = self.raw_dir
        edge_index = torch.LongTensor(np.genfromtxt(f"{prefix}/adj.csv", delimiter=',')).T - 1
        edge_attr = torch.Tensor(np.genfromtxt(f"{prefix}/attr.csv", delimiter=','))
        groups = torch.LongTensor(np.genfromtxt(f"{prefix}/groups.csv", delimiter=','))
        mask = torch.IntTensor(np.genfromtxt(f"{prefix}/mask.csv", delimiter=','))
        rest = torch.Tensor(np.genfromtxt(f"{prefix}/rest.csv", delimiter=','))

        # Load simulation data
        for name in self.raw_file_names:
            if "sim" not in name: continue

            data = np.genfromtxt(f"{prefix}/{name}/x.csv", delimiter=',')
            data_list = self._get_data_list(data, edge_index, edge_attr, groups, mask, rest)

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            idx = int(name.split(os.path.sep)[-1].split("m")[1]) - 1 
            torch.save(self.collate(data_list), self.processed_paths[idx])



    def _get_data_list(self, data, edge_index, edge_attr, groups, mask, rest):
        """Prepares the list of 'Data' objects to be saved.
        This also takes care of balancing the dataset and removing the fixed masses,
        if the corresponding parameters are true.
        """
        if self.remove_fixed:
            data = data[:, mask.repeat_interleave(2) != 0]
            rest = rest[mask != 0, :]
            groups = groups[mask != 0]

            # This works assuming masked masses always have the lower indexes
            mask_ = ((edge_index - sum(mask == 0)) < 0).sum(0) == 0
            edge_index = edge_index[:, mask_] - sum(mask == 0)
            edge_attr = edge_attr[mask_, :]

        if self.balance: data = self._balance(data, self.balance)
        data = data.reshape(len(data), edge_index.max().item()+1, 2)

        data_list = []
        for g in data:
            d = Data(
                pos = torch.Tensor(g),
                edge_index = edge_index,
                edge_attr = edge_attr,
                groups = groups,
                mask = mask,
                rest = rest
            )
            data_list.append( ToUndirected()(d) )
        return data_list


    def _balance(self, data, eps):
        """Removes similar configurations leaving only one in a eps-neighbourhood."""
        if eps is True: eps=1e-1
        to_remove = set()
        for i in range(len(data)):
            if i in to_remove:
                continue

            norms = np.linalg.norm(data[i+1:] - data[i], axis=1)
            to_remove.update(
                [i+1+j for j,n in enumerate(norms) if n < eps]
            )

        return np.delete(data, list(to_remove), axis=0)


    def _load(self, indexes, device):
        """"""
        data_list = []
        slices_list = []
        for idx in indexes:
            file = self.processed_paths[idx]
            data, slices = torch.load(file, map_location=torch.device(device))
            data_list.append(data)
            slices_list.append(slices)
        
        data, slices, _ = collate(
            data_list[0].__class__,
            data_list = data_list,
            increment = False,
            add_batch = False,
        )

        sl = slices_list[0]
        for idx, sl1 in enumerate(slices_list[1:]):
            for k in sl1.keys():
                sl[k] = torch.cat([sl[k], sl1[k][1:] + slices[k][idx+1]])
        
        return data, sl


def MultipleSystemsDataset(root, sys_indexes = None, **kwargs):
    """"""
    datasets = []

    for dir in os.listdir(root):
        idx = int(dir[3:])

        if sys_indexes is None or idx in sys_indexes:
            d = SpringSystemDataset(f"{root}/{dir}", **kwargs)
            datasets.append(d)
    
    return ConcatDataset(datasets)
