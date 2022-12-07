import json
import models

import numpy as np



def from_dictionary(configuration, **kwargs):
    """ Load a model from a configuration dictionary.
    The dictionary should specify the model name (as exposed by the module)
    and all the necessary parameters the model takes.
    Keyword arguments are not supported (for now)
    Alternatively, the name of a JSON file containing the dictionary can be given.
    """

    if isinstance(configuration, str):
        # If a file name is given
        with open(configuration, 'r') as f: conf = json.load(f)

    elif isinstance(configuration, dict):
        # If a dictionary is given
        conf = configuration

    model_f = getattr(models, conf['model'])
    if isinstance(model_f, type):
        # When the attribute is a class
        param_names_list = model_f.__init__.__code__.co_varnames[1:model_f.__init__.__code__.co_argcount]
    else:    
        # When the attribute is a function
        param_names_list = model_f.__code__.co_varnames[:model_f.__code__.co_argcount]

    args = [conf[param_name] for param_name in param_names_list]
    return model_f(*args, **kwargs)



def getKineticEnergy(dx, m):
    """"""
    return 0.5 * sum(m @ (dx ** 2))


def getElasticEnergy(x, rest, ei, k, l):
    """"""
    ks = 0.2
    
    energy = 0
    for i, (v, u) in enumerate(ei):
        deltax = x[v] - x[u]
        deltar = rest[v] - rest[u]
        
        energy += 0.5 * k[i] * (deltax.square().sum().sqrt() - l[i]).square()
        energy += 0.5 * ks * (deltax.square().sum() - deltar.square().sum())
        
    return energy


def getGravitationalEnergy(x, m, g_theta, g_int):
    """"""
    costheta = np.cos(-np.pi/2 - g_theta)
    sintheta = np.sin(-np.pi/2 - g_theta)
    rot = np.array([[costheta, -sintheta], [sintheta, costheta]])
    
    x_ = (rot @ x.T)
    
    return g_int * (m * (x_[1,:])).sum()
