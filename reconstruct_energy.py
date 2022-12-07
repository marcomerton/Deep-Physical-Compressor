from argparse import ArgumentParser
import numpy as np
import torch
from utils.data import load_system, load_model
from utils.misc import getElasticEnergy, getGravitationalEnergy, getKineticEnergy


if __name__ == '__main__':
    parser = ArgumentParser(description="Compute the pointwise/compressed reconstructed energy for a system")
    parser.add_argument("sys_idx", help="System index")
    parser.add_argument("model", help="Name of model")

    args = parser.parse_args()


    d, num_masses, num_fixed = load_system(args.sys_idx)
    m = np.ones(num_masses - num_fixed) / num_masses

    data_path = f"data/205masses/sys{args.sys_idx}/raw/"
    ei = torch.LongTensor(np.genfromtxt(f"{data_path}adj.csv", delimiter=',')).T - 1
    ea = torch.Tensor(np.genfromtxt(f"{data_path}attr.csv", delimiter=','))
    rest = torch.Tensor(np.genfromtxt(f"{data_path}rest.csv", delimiter=',')).view(num_masses, 2)

    model = load_model(args.sys_idx, args.model)


    thetas = np.genfromtxt("results/thetas.csv", delimiter=',')
    intens = np.genfromtxt("results/intens.csv", delimiter=',')


    kin_point = np.ndarray([501])
    elas_point = np.ndarray([501])
    grav_point = np.ndarray([501])

    kin_red = np.ndarray([501])
    elas_red = np.ndarray([501])
    grav_red = np.ndarray([501])

    for sim in range(8,38):
        print(f"Sim {sim}")
        
        res_path = f"results/sys{args.sys_idx}/sim{sim}/{args.model}/"
        xs_point = torch.Tensor(np.genfromtxt(f"{res_path}xs_point.csv", delimiter=','))
        dxs_point = torch.Tensor(np.genfromtxt(f"{res_path}dxs_point.csv", delimiter=','))
        xs_red = torch.Tensor(np.genfromtxt(f"{res_path}xs_red.csv", delimiter=','))
        dxs_red = torch.Tensor(np.genfromtxt(f"{res_path}dxs_red.csv", delimiter=','))


        for i, (xp, dxp, xr, dxr) in enumerate(zip(xs_point, dxs_point, xs_red, dxs_red)):
            dxp_ = dxp.view(num_masses-num_fixed, 2)
            xp_ = xp.view(num_masses, 2)
            dxr_ = dxr.view(num_masses-num_fixed, 2)
            xr_ = xr.view(num_masses, 2)
            
            kin_point[i] = getKineticEnergy(dxp_.numpy(), m)
            elas_point[i] = getElasticEnergy(xp_, rest, ei.T, ea[:,0], ea[:,2])
            grav_point[i] = getGravitationalEnergy(xp_[num_fixed:, :].numpy(),
                                m, thetas[sim-8], intens[sim-8])
            
            kin_red[i] = getKineticEnergy(dxr_.numpy(), m)
            elas_red[i] = getElasticEnergy(xr_, rest, ei.T, ea[:,0], ea[:,2])
            grav_red[i] = getGravitationalEnergy(xr_[num_fixed:, :].numpy(),
                                m, thetas[sim-8], intens[sim-8])

        np.savetxt(f"{res_path}kin-point.csv", kin_point, delimiter=',')
        np.savetxt(f"{res_path}elas-point.csv", elas_point, delimiter=',')
        np.savetxt(f"{res_path}grav-point.csv", grav_point, delimiter=',')
        
        np.savetxt(f"{res_path}kin-red.csv", kin_red, delimiter=',')
        np.savetxt(f"{res_path}elas-red.csv", elas_red, delimiter=',')
        np.savetxt(f"{res_path}grav-red.csv", grav_red, delimiter=',')
