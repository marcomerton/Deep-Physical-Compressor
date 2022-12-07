import numpy as np
import torch
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def draw_graph(points, adj, ax=None, color='blue', **kwargs):
	"""
	"""
	if ax is None:
		ax = plt.gca()

	if points.ndim == 1:
		x = points
		y = [0] * len(points)

	else:
		x = points[:, 0]
		y = points[:, 1]


	if isinstance(color, str) or isinstance(color, tuple):
		# Normal usecase
		lines = []
		for e in adj.T:
			if all(e < len(x)):
				lines.append(ax.plot(x[e], y[e], color=color, **kwargs)[0])

		sc = ax.scatter(x, y, c=color, **kwargs)

	else:
		# When used to show error
		lines = []
		for e in adj.T:
			if all(e < len(x)):
				lines.append(ax.plot(x[e], y[e], color='black', alpha=0.2)[0])

		sc = ax.scatter(x, y, c=color, **kwargs)

	return sc, lines


def update_graph(sc, lines, new_pos, adj):
	"""
	"""
	sc.set_offsets(np.c_[new_pos[:,0], new_pos[:,1]])
	
	for i, e in enumerate(adj.T):
		lines[i].set_data(new_pos[e, 0], new_pos[e, 1])
		

def animate(data, model=None, fig=None, ax=None, interval=200):
	"""
	"""
	if fig is None or ax is None:
		fig, ax = plt.subplots(1,1, figsize=(10,5))
	
	empty_data = np.empty((data.num_masses, 2))
	
	sc, lines = draw_graph(empty_data, data[0].edge_index,
							ax=ax, color='green', alpha=0.5)
	
	if model is not None:
		sc_rec, lines_rec = draw_graph(empty_data, data[0].edge_index,
										ax=ax, color='blue')
	
	def animate(g):
		update_graph(sc, lines, g.pos, g.edge_index)
		
		if model is not None:
			rec = model(g)
			update_graph(sc_rec, lines_rec, rec, g.edge_index)
	
	ani = FuncAnimation(fig, animate, frames=data, interval=interval)
	return ani


def animate_(data, adj, fig=None, ax=None, interval=200):
	"""
	"""
	if fig is None or ax is None:
		fig, ax = plt.subplots(1,1, figsize=(10,5))
	
	empty_data = np.empty((data.shape[1], 2))
	
	sc, lines = draw_graph(empty_data, adj, ax=ax, color='green', alpha=0.5)
	
	def animate(nodes):
		update_graph(sc, lines, nodes, adj)
	
	ani = FuncAnimation(fig, animate, frames=data, interval=interval)
	return ani



def draw_learning_curve(tr_losses, vl_losses, label=None, ax=None, **kwargs):
	"""Display the training and validation loss/metric"""
	if ax is None:
		ax = plt.gca()
		
	step = len(tr_losses) // len(vl_losses)
		
	p = ax.plot(tr_losses, alpha=0.3, **kwargs)
	kwargs['color'] = p[0].get_color()

	ax.plot(range(step-1, len(tr_losses), step), vl_losses,
		label=label, **kwargs)
