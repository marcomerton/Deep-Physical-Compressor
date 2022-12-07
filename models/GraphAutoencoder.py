import torch

class GraphAutoencoder(torch.nn.Module):
    """Base graph autoencoder class.
    This is meant as a template class for graph autoencoder models that
    aim at reconstructing input node positions.

    The behavior of the model is fully specified by the specific encoder and
    decoder models.
    """
    def __init__(self, encoder, decoder):
        super(GraphAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder


    def get_node_embeddings(self, data):
        """Computes the node embeddings"""
        return self.encoder.get_node_embeddings(data)


    def encode(self, data):
        """Encodes the input graph(s)"""
        if data.batch is None:
            data.batch = torch.zeros(len(data.pos), dtype=torch.int64, device=data.pos.device)

        return self.encoder(data)


    def decode(self, emb, data):
        """ Decodes the input graph(s)"""
        if data.batch is None:
            data.batch = torch.zeros(data.edge_index.max()+1, dtype=torch.int64, device=emb.device)

        return self.decoder(emb, data)


    def forward(self, data):
        """Applies the model to the input"""
        if data.batch is None:
            data.batch = torch.zeros(len(data.pos), dtype=torch.int64, device=data.pos.device)

        emb = self.encoder(data)
        return self.decoder(emb, data)
