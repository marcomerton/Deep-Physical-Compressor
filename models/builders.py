import torch_geometric as PyG
from torch.nn import Sequential, Linear, ELU

from .layers import SAGEConv_layer


def build_sequential(input_size, hidden_sizes, actv=ELU):
	"""Builds a fully connected (sequential) neural network.

	ARGS:
		input_size: number of input units.
		hidden_sizes: list of hidden sizes.
		actv: activation function to use between layers (Default: ELU).
			Note the activation is not inserted after the last layer.

	RETURNS:
		torch.nn.Sequential
	"""

	layers = []
	layers.append(Linear(input_size, hidden_sizes[0]))
	for i in range(1, len(hidden_sizes)):
		layers.append(actv())
		layers.append(Linear(hidden_sizes[i-1], hidden_sizes[i]))

	return Sequential(*layers)


def build_pyg_sequential(input_channels, hidden_channels, edge_channels,
						layer_func=SAGEConv_layer, actv=ELU):
	"""Builds a (sequential) graph neural network.

	ARGS:
		input_channels: number of input features per node
		hidden_channels: number of node features in each layer
		edge_channels: number of edge features
		layer_func: function returning a gnn layer. Should take in input
			'input_channels', 'hidden_channels' and 'edge_channels'
			(Default: GCNConv)
		actv: activation function to use between layers (Default: ELU)

	RETURNS:
		torch_geometric.nn.Sequential
	"""

	layers = []

	layers.append(
		(layer_func(input_channels, hidden_channels[0], edge_channels),
		'x, edge_index, edge_attr -> x')
	)

	prev_size = hidden_channels[0]
	for size in hidden_channels[1:]:
		if actv is not None:
			layers.append((actv(), 'x -> x'))

		layers.append(
			(layer_func(prev_size, size, edge_channels),
			'x, edge_index, edge_attr -> x')
		)
		prev_size = size
		
	return PyG.nn.Sequential('x, edge_index, edge_attr', layers)
