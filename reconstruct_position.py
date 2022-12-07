from argparse import ArgumentParser
import numpy as np
import torch
from utils.data import load_system, load_model, load_simulation


if __name__ == '__main__':
    parser = ArgumentParser(description="Compute the pointwise/compressed reconstructioed positions for a system")
    parser.add_argument("sys_idx", help="System index")
    parser.add_argument("model", help="Name of model")

    args = parser.parse_args()


    d, num_masses, num_fixed = load_system(args.sys_idx)
    model = load_model(args.sys_idx, args.model)

    for sim in range(8,38):
        print(f"Sim {sim}")

        xs, time, qs = load_simulation(args.sys_idx, sim, args.model)
        

        xs_point = np.ndarray([len(time), 2*num_masses])
        xs_red = np.ndarray([len(time), 2*num_masses])

        for i, (x, q) in enumerate(zip(xs, qs)):
            x_ = x.view(num_masses, 2)
            
            with torch.no_grad():
                d.pos = x_[num_fixed:, :]
                x_point = model(d)
                x_point = torch.cat([x_[:num_fixed, :], x_point], dim=0).view(-1)
                
                x_red = model.decode(q, d).view(-1)
                x_red = torch.cat([x[:2*num_fixed], x_red], dim=0)
            
            xs_point[i,:] = x_point
            xs_red[i,:] = x_red
        
        path = f"results/sys{args.sys_idx}/sim{sim}/{args.model}/"
        np.savetxt(f"{path}xs_point.csv", xs_point, delimiter=',')
        np.savetxt(f"{path}xs_red.csv", xs_red, delimiter=',')
