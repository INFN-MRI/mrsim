"""
Validation of derivatives
"""
from functools import partial

import numpy as np
import torch

from torch.func import jacrev, vmap

import epgtorchx as epgx
from epgtorchx import optim

import matplotlib.pyplot as plt
import time

#%% local utils
fse_grad = epgx.fse

def fse_finitediff_grad(flip, phases, ESP, T1, T2, asnumpy=True):
    
    # run
    sig = epgx.fse(flip, phases, ESP, T1, T2, asnumpy=asnumpy)

    # numerical derivative
    dt = 1.0
    dsig = epgx.fse(flip, phases, ESP, T1, T2+dt, asnumpy=asnumpy)
    
    return sig, (dsig - sig) / dt

def crlb_cost(flip, ESP, T1, T2):
    
    # to tensor
    flip = torch.as_tensor(flip, dtype=torch.float32)
    flip.requires_grad = True
    
    # phases
    phases = torch.zeros_like(flip)
    
    # get partial function
    _cost = partial(_crlb_cost, ESP, T1, T2, phases)
    _dcost = jacrev(_cost)
    
    return _cost(flip).detach().cpu().numpy(), _dcost(flip).detach().cpu().numpy()
 
def _crlb_cost(ESP, T1, T2, phases, flip):
    
    # calculate signal and derivative
    _, grad = epgx.fse(flip, phases, ESP, T1, T2, diff=["T2"], asnumpy=False)
    
    # calculate cost
    return optim.calculate_crlb(grad)

def _crlb_finitediff_cost(ESP, T1, T2, phases, flip):
    
    # calculate signal and derivative
    _, grad = fse_finitediff_grad(flip, phases, ESP, T1, T2, asnumpy=False)
    
    # calculate cost
    return optim.calculate_crlb(grad).cpu().detach().numpy()

def crlb_finitediff_cost(flip, ESP, T1, T2):
    
    # initial cost
    cost0 = _crlb_finitediff_cost(ESP, T1, T2, 0 * flip, flip)
    dcost = []
    
    for n in range(len(flip)):
        # get angles
        angles = flip.copy()
        angles[n] += 1.0
        dcost.append(_crlb_finitediff_cost(ESP, T1, T2, 0 * angles, angles))
                
    return cost0, np.asarray(dcost) - cost0

# %% params
t1 = 1000.0
t2 = 100.0

angles = np.concatenate((np.linspace(0, 180.0, 36), np.ones(60, dtype=np.float32) * 180.0))
esp = 5.0

# run
t0 = time.time()
sig0, grad0 = fse_finitediff_grad(angles, 0 * angles, esp, t1, t2)
t1 = time.time()
tgrad0 = t1 - t0

t0 = time.time()
sig, grad = fse_grad(angles, 0 * angles, esp, t1, t2, diff=["T2"])
t1 = time.time()
tgrad = t1 - t0


# cost and derivative
t0 = time.time()
cost0, dcost0 = crlb_finitediff_cost(angles, esp, t1, t2)
t1 = time.time()
tcost0 = t1 - t0

t0 = time.time()
cost, dcost = crlb_cost(angles, esp, t1, t2)
t1 = time.time()
tcost = t1 - t0

# plot derivative
#%%plots 
fsz = 20
plt.figure()
plt.subplot(2,2,1)
plt.rcParams.update({'font.size': 0.5 * fsz})
plt.plot(angles, '.')
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, 97])
plt.ylabel("Flip Angle [deg]", fontsize=fsz)

plt.subplot(2,2,2)
plt.rcParams.update({'font.size': 0.5 * fsz})
plt.plot(abs(grad), '-k'), plt.plot(abs(grad0), '*r')
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, 97])
plt.ylabel(r"$\frac{\partial signal}{\partial T2}$ [a.u.]", fontsize=fsz)
plt.legend(["Finite Diff", "Auto Diff"])


plt.subplot(2,2,3)
plt.rcParams.update({'font.size': 0.5 * fsz})
plt.plot(abs(dcost), '-k'), plt.plot(abs(dcost0), '*r')
plt.xlabel("Echo #", fontsize=fsz)
plt.xlim([-1, 97])
plt.ylabel(r"$\frac{\partial CRLB}{\partial FA}$ [a.u.]", fontsize=fsz)
plt.legend(["Finite Diff", "Auto Diff"])

plt.subplot(2,2,4)

# define labels
# plot results
labels = ['derivative of signal', 'CRLB objective gradient']
time_finite = [round(tgrad0, 2), round(tcost0, 2)]
time_auto = [round(tgrad, 2), round(tcost, 2)]


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
rects1 = plt.bar(x + width/2, time_finite, width, label='Finite Diff')
rects2 = plt.bar(x - width/2, time_auto, width, label='Auto Diff')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Execution Time [s]', fontsize=fsz)
plt.xticks(x, labels, fontsize=fsz)
# plt.ylim([0, 25])
plt.legend()

plt.bar_label(rects1, padding=3, fontsize=fsz)
plt.bar_label(rects2, padding=3, fontsize=fsz)

