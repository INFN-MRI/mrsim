#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 11:26:38 2023

@author: mcencini
"""
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

import epgtorchx as epgx
from epgtorchx import regression

# %% actual routine
def create_phantom(shape):
    seg, props, _ = epgx.create_shepp_logan(shape[0], shape[-1], True, model="bm")
    
    # maps
    M0 = np.zeros(shape, dtype=np.float32).T
    T1s = np.zeros(shape, dtype=np.float32).T
    T2s = np.zeros(shape, dtype=np.float32).T
    T1f = np.zeros(shape, dtype=np.float32).T
    T2f = np.zeros(shape, dtype=np.float32).T
    k = np.zeros(shape, dtype=np.float32).T
    ff = np.zeros(shape, dtype=np.float32).T
    
    # fill phantoms
    tissues = np.unique(seg)
    idx = [0, -1, 4, 3, 6, 1]
    idx2 = [0, -1, 6, 2, 3, 1]

    for n in tissues:
        mask = (seg == n)
        M0[mask] = props["M0"][idx[n]]
        T1s[mask] = props["T1"][idx[n]] * 1.2
        T2s[mask] = props["T2"][idx[n]]  * 1.2
        T1f[mask] = props["bm"]["T1"][idx2[n]] * 0.8
        T2f[mask] = props["bm"]["T2"][idx2[n]] * 0.8
        k[mask] = props["bm"]["k"][idx2[n]]
        ff[mask] = props["bm"]["weight"][idx2[n]]
        
    return np.stack((M0, T1s, T2s, T1f, T2f, k, ff), axis=0)


def simulate(maps, flip, ESP, phases=None, device="cpu"):
    # default
    if phases is None:
        phases = -np.ones_like(flip) * 90.0
    
    # get ishape
    ishape = maps.shape[1:]    
    output = epgx.fse(flip, phases, ESP, 
                      maps[1].flatten(), 
                      maps[2].flatten(), 
                      T1bm=maps[3].flatten(), 
                      T2bm=maps[4].flatten(), 
                      weight_bm=maps[-1].flatten(),
                      kbm=maps[5].flatten(), 
                      device=device)
    
    # reshape
    return maps[0] * output.T.reshape(-1, *ishape)
    
    
def fitting(input, flip, ESP, phases=None, device="cpu", H=1000, tsize=10000, lamda=2**-1.5, rho=2**-20, sigma=0.01, c=2**0.6):
    
    # default
    if phases is None:
        phases = -np.ones_like(flip) * 90.0
    
    # parameter grid
    ffDistTrain = np.random.uniform(0.03, 0.31, tsize) # Myelin Water fraction
    T2fDistTrain = np.random.uniform(16, 24, tsize)
    T2sDistTrain = np.random.uniform(64, 96, tsize)
    kDistTrain = np.random.uniform(5, 150, tsize) # Ballpark value
    
    # convert to pytorch
    ffDistTrain = torch.as_tensor(ffDistTrain, dtype=torch.float32, device=device)
    T2fDistTrain = torch.as_tensor(T2fDistTrain, dtype=torch.float32, device=device)
    T2sDistTrain = torch.as_tensor(T2sDistTrain, dtype=torch.float32, device=device)
    kDistTrain = torch.as_tensor(kDistTrain, dtype=torch.float32, device=device)
    
    # generate test data
    signal = epgx.fse(flip, 
                      phases, 
                      ESP, 
                      1000.0, 
                      T2sDistTrain, 
                      T1bm=500.0,
                      T2bm=T2fDistTrain,
                      kbm=kDistTrain,
                      weight_bm=ffDistTrain,
                      device=device,
                      asnumpy=False)
        
    # generate noise
    noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) * sigma
    noise = torch.as_tensor(noise, dtype=signal.dtype, device=signal.device)
    
    # add noise
    signal = abs(signal + noise)

    # svd compression
    _, _, v = torch.linalg.svd(signal, full_matrices=False)
    v = v[:, :4] # hardcoded for now
    
    # prepare signals and labels for training 
    train_x = torch.stack((ffDistTrain, T2fDistTrain, T2sDistTrain, kDistTrain), axis=-1)
    train_y = signal @ v
    
    # train 
    kernel = regression.perk_train(train_x, train_y, H=H, lamda=lamda)
    
    # inference
    input = torch.as_tensor(input.copy())
    
    # compress
    print(input.shape)
    ishape = input.shape[1:]
    print(ggg)
    # input = input.
    input = input.reshape(-1, input.shape[-1]) # (nvoxels, nechoes)
    input = input @ v
    input = input.reshape(*ishape, -1).permute(3, 2, 1, 0) # (ncoeff, nvoxels)
    input = input.to(signal.device) 
        
    output = regression.perk_eval(input, kernel, reg=rho)
    
    # output as numpy
    output = output.detach().cpu().numpy()
    
    return output
    
# %% generate map
flip = 180.0 * np.ones(50, dtype=np.float32)
ESP = 5.0
device="cpu"

# prepare phantom
gt = create_phantom([200, 200, 8])

# simulate acquisition
echo_series = simulate(gt, flip, ESP, device=device)

omaps = fitting(echo_series, flip, ESP, device=device)






    




