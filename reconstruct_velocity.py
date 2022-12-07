from argparse import ArgumentParser
import numpy as np
import torch
from torch.autograd.functional import jacobian

from utils.data import load_system, load_model, load_simulation


if __name__ == '__main__':
    parser = ArgumentParser(description="Compute the pointwise/compressed reconstructioed velocities for a system")
    parser.add_argument("sys_idx", help="System index")
    parser.add_argument("model", help="Name of model")

    args = parser.parse_args()

    d, num_masses, num_fixed = load_system(args.sys_idx)

    model = load_model(args.sys_idx, args.model)
    def enc_f(x):
        d.pos = x.view(num_masses-num_fixed, 2)
        return model.encode(d).flatten()
    def dec_f(e):
        return model.decode(e, d).view(-1)


    for sim in range(8,38):
        print(f"Sim {sim}")

        res_path = f"results/sys{args.sys_idx}/sim{sim}/"
        xs, time, qs = load_simulation(args.sys_idx, sim, args.model)
        dxs = torch.Tensor(np.genfromtxt(f"{res_path}dx.csv", delimiter=','))
        dqs = torch.Tensor(np.genfromtxt(f"{res_path}{args.model}/dq.csv", delimiter=','))


        dxs_point = np.ndarray([len(time), 2*(num_masses-num_fixed)])
        dxs_red = np.ndarray([len(time), 2*(num_masses-num_fixed)])

        for i, (x, dx, q, dq) in enumerate(zip(xs, dxs, qs, dqs)):
            d.pos = x.view(num_masses, 2)[num_fixed:, :]
            with torch.no_grad(): q_point = model.encode(d).flatten()
            
            Je = jacobian(enc_f, x[2*num_fixed:])
            Jd = jacobian(dec_f, q_point)
            dxs_point[i,:] = Jd @ (Je @ dx[2*num_fixed:])

            J = jacobian(dec_f, q)
            dxs_red[i,:] = J @ dq
    
        path = f"results/sys{args.sys_idx}/sim{sim}/{args.model}/"
        np.savetxt(f"{path}dxs_point.csv", dxs_point, delimiter=',')
        np.savetxt(f"{path}dxs_red.csv", dxs_red, delimiter=',')
