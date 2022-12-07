import torch
from .builders import build_sequential


class AE(torch.nn.Module):
	"""DeepAutoencoder
	The encoder and the decoder are simple deep neural networks.
	"""

	def __init__(self, input_size, enc_sizes, dec_sizes):
		super(AE, self).__init__()

		self.encoder = build_sequential(input_size, enc_sizes)
		self.decoder = build_sequential(enc_sizes[-1], dec_sizes)

		self.input_size = input_size


	def encode(self, data):
		"""Runs the encoder."""
		return self.encoder(data.pos.reshape(-1, self.input_size))


	def decode(self, emb, data):
		"""Runs the decoder."""
		return self.decoder(emb)


	def forward(self, data):
		"""Applies the whole model to the input."""
		return self.decoder(
				self.encoder(
					data.pos.reshape(-1, self.input_size)
				)
			).reshape(data.pos.shape)
