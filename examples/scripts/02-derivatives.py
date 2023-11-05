"""
Validation of derivatives
"""
from functools import partial

import numpy as np
import torch

from torch.func import jacrev, vmap

import epgtorchx as epgx
from epgtorchx import optim

#%% local utils
def crlb_cost(flip, ESP, T1, T2):
    
    # phases
    phases = torch.zeros_like(flip)
    
    # get partial function
    _cost = partial(_crlb_cost, ESP, T1, T2, phases)
    
    # get derivative
    _dcost = jacrev(_cost)
    
    return _cost(flip), _dcost(flip)
    
def _crlb_cost(ESP, T1, T2, phases, flip):
    
    # calculate signal and derivative
    _, grad = epgx.fse(flip, phases, ESP, T1, T2, diff=["T2"], asnumpy=False)
    
    # calculate cost
    return optim.calculate_crlb(grad)
    
# %% params
t1 = 1000.0
t2 = 100.0

angles = torch.ones(50, dtype=torch.float32, requires_grad=True) * 180.0
esp = 5.0

# run
sig, grad = epgx.fse(angles, 0 * angles, esp, t1, t2, diff=["T2"], asnumpy=False)

# numerical derivative
dsig = epgx.fse(angles.clone(), 0 * angles, esp, t1, t2+1)
dsig_dt2_numeric = dsig - sig.detach().numpy()

# cost and derivative
cost, dcost = crlb_cost(angles, esp, t1, t2)

