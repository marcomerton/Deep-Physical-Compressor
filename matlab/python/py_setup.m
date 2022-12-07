function py_setup()
%PY_SETUP General imports to make python code work. Numpy is required as
%interface between Matlab and PyTorch.

pyrun([ ...
    "import numpy as np" ...
    "import torch" ...
    "from torch.autograd.functional import jacobian, hvp"]);

end