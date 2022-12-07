import argparse
import json
import torch

from dataset import SpringSystemDataset
from dataset.utils import get_train_test_simulations_index
from torch_geometric.loader import DataLoader

from utils.misc import from_dictionary

from utils.training import train



if __name__ == "__main__":

	# Setup argument parser
	parser = argparse.ArgumentParser(description="Train and saves a GAE model.")
	parser.add_argument("config_file", help="Model configuration file")
	parser.add_argument("dataset", help="Name of the dataset to use")
	parser.add_argument("filename", help="Prefix for the files containing the results")
	parser.add_argument("-s", "--seed", help="Seed for the RNG", type=int, default=1234, required=False)

	args = parser.parse_args()


	# Load configuration
	config = None
	with open(args.config_file, 'r') as f: config = json.load(f)

	batch_size = config['batch_size']
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	torch.manual_seed(args.seed)

	# Prapre datasets
	tr_idxs, vl_idxs = get_train_test_simulations_index(args.dataset)
	train_data = SpringSystemDataset(f"data/{args.dataset}", sim_indexes=tr_idxs, device=device)
	valid_data = SpringSystemDataset(f"data/{args.dataset}", sim_indexes=vl_idxs, device=device)
	train_data = train_data.shuffle()
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
	valid_loader = DataLoader(valid_data, batch_size=batch_size)
	print(f"Training data: {train_data}")
	print(f"Validation data: {valid_data}")


	# Build model
	model = from_dictionary(config)
	model.to(device)
	print(model)

	# Optimizer & Loss
	loss = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['lmb'])
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
					step_size=config['lr_step'], gamma=config['lr_decay'])

	# This also saves model's parameters and train/val losses
	tr, vl = train(
		model, optimizer, scheduler, loss, config['epochs'],
		train_loader, valid_loader, device=device,
		filename=args.filename)

	print(f"Final training score: {tr}")
	print(f"Final validation score: {vl}")
