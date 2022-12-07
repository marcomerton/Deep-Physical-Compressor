from torch_geometric.nn import SAGEConv, GATv2Conv, CGConv, GENConv, GCNConv


def SAGEConv_layer(in_channels, out_channels, edge_channels):
	return SAGEConv(in_channels, out_channels)


def GATConv_layer(in_channels, out_channels, edge_channels):
	return GATv2Conv(
		in_channels = in_channels,
		out_channels = out_channels,
		edge_dim = edge_channels,
		heads = 1, concat = False
	)


def CGConv_layer(in_channels, out_channels, edge_channels):
	return CGConv(
		channels = (in_channels, out_channels),
		dim = edge_channels,
		aggr = 'add'
	)


def GeneralizedConv(in_channels, out_channels, edge_channels):
	return GENConv(
		in_channels, out_channels,
		aggr='softmax', t=1.0, learn_t=True,
		num_layers=2, norm='layer'
	)


def GCNConv_layer(in_channels, out_channels, edge_channels):
	return GCNConv(in_channels, out_channels, improved=True)
