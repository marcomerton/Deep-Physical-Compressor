from argparse import ArgumentParser
import json
import torch

from dataset import SpringSystemDataset
from dataset.utils import get_train_test_simulations_index
from torch_geometric.loader import DataLoader

from sklearn.model_selection import ParameterGrid
from utils.misc import from_dictionary

from utils.training import train



if __name__ == "__main__":
	parser = ArgumentParser(description="Perform a grid-search using an held-out validation set.")
	parser.add_argument("config_file", help="Model configuration file")
	parser.add_argument("dataset", help="Name of the dataset to use")
	parser.add_argument("save_prefix", help="Prefix for the files containing the results")
	parser.add_argument("-s", "--seed", help="Seed for the RNG", type=int, default=1234, required=False)

	args = parser.parse_args()


	# Setup
	config = None
	with open(args.config_file, 'r') as f: config = json.load(f)
	torch.manual_seed(args.seed)
	
	# Assumed to be unique
	batch_size = config['batch_size']
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# Create parameter grid with the specified parameters
	param_grid = {}
	for param_name in config['grid_search']:
		param_grid[param_name] = config[param_name]
	param_grid = ParameterGrid(param_grid)


	# Prapre datasets
	tr_idxs, vl_idxs = get_train_test_simulations_index(args.dataset)
	train_data = SpringSystemDataset(f"data/{args.dataset}", sim_indexes=tr_idxs, device=device)
	valid_data = SpringSystemDataset(f"data/{args.dataset}", sim_indexes=vl_idxs, device=device)
	train_data = train_data.shuffle()
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_data, batch_size=batch_size)


	res = []
	for i, params in enumerate(param_grid):
		print(params)
		config.update(params)

		model = from_dictionary(config)
		model.to(device)

		loss = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['lmb'])
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
						step_size=config['lr_step'], gamma=config['lr_decay'])

		tr, vl = train(
			model, optimizer, scheduler, loss, config['epochs'],
			train_loader, valid_loader, device=device,
			save_model=False, save_best=False, save_scores=True,
			filename=f"{args.save_prefix}{i}", verbose=False
		)
		res.append((params, tr, vl))

		print(f"Training score: {tr:0.6e}")
		print(f"Validation score: {vl:0.6e}")


# Save results summary
with open(f"{args.save_prefix}summary.txt", "w") as f:
	for e in res:
		f.write(str(e) + "\n")
