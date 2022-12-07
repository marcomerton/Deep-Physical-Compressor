from .GraphAutoencoder import GraphAutoencoder
from .builders import build_sequential, build_pyg_sequential

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

pool_dict = {
    "add": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool
}



class GNNEncoder(torch.nn.Module):
	"""
	GNNEncoder
	Graphs are first processed with a GNN after which a node-wise MLP is applied
	to the node states. The resulting node embeddings are concatenated and fed to
	an inner MLP computing the graph embedding.

	[pos] -> GNN -> [xt] -> MLP -> [nodes_emb] -> Flatten -> MLP -> [z]
	"""

	def __init__(self, in_channels, edge_channels, max_size,
				 gnn_sizes, nmlp_sizes, gmlp_sizes, **kwargs):
		super(GNNEncoder, self).__init__()

		self.gnn = build_pyg_sequential(in_channels, gnn_sizes, edge_channels, **kwargs)
		self.node_mlp = build_sequential(gnn_sizes[-1], nmlp_sizes)
		self.inner_enc = build_sequential(max_size * nmlp_sizes[-1], gmlp_sizes)

		self.n = max_size


	def get_node_embeddings(self, data):
		"""Returns the node embeddings.
		These are the nodes' states in the last gnn layer.
		"""
		return self.gnn(data.pos, data.edge_index, data.edge_attr)


	def forward(self, data):
		"""Encode the input."""
		xt = self.gnn(data.pos, data.edge_index, data.edge_attr)
		out = self.node_mlp(xt)
		return self.inner_enc(out.view(-1, self.n * out.shape[-1]))



class GNNDecoder(torch.nn.Module):
    """
    GNNDecoder2
    Improved GNN decoder that features a further node-wise mlp after the gnn processing.
    This takes the final node states and reconstruct, for each of them, the initial node features.
    """

    def __init__(self, emb_size, max_size, edge_channels, gmlp_sizes, gnn_sizes, nmlp_sizes, **kwargs):
        super(GNNDecoder, self).__init__()

        self.inner_dec = build_sequential(emb_size, gmlp_sizes)
        self.gnn = build_pyg_sequential(gmlp_sizes[-1]//max_size, gnn_sizes, edge_channels, **kwargs)
        self.node_mlp = build_sequential(gnn_sizes[-1], nmlp_sizes)
        self.n = max_size


    def forward(self, emb, data):
        """Decode the input embedding."""
        x0 = self.inner_dec(emb)
        x0 = x0.view(-1, x0.shape[-1] // self.n)
        xt = self.gnn(x0, data.edge_index, data.edge_attr)
        return self.node_mlp(xt)



def GAE(in_channels, edge_channels, max_size,
         enc_gnn_sizes, enc_nmlp_sizes, enc_inner_sizes,
         dec_inner_sizes, dec_gnn_sizes, dec_nmlp_sizes,
        **kwargs):

    encoder = GNNEncoder(in_channels, edge_channels, max_size,
                enc_gnn_sizes, enc_nmlp_sizes, enc_inner_sizes, **kwargs)
    decoder = GNNDecoder(enc_inner_sizes[-1], max_size, edge_channels,
                dec_inner_sizes, dec_gnn_sizes, dec_nmlp_sizes, **kwargs)

    return GraphAutoencoder(encoder, decoder)
