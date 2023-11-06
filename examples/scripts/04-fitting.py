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
def create_phantom():
    seg, props, _ = epgx.create_shepp_logan(200, 200, True, model="bm")
    
    # maps
    M0 = np.zeros((200, 200, 200), dtype=np.float32)
    T1s = np.zeros((200, 200, 200), dtype=np.float32)
    T2s = np.zeros((200, 200, 200), dtype=np.float32)
    T1f = np.zeros((200, 200, 200), dtype=np.float32)
    T2f = np.zeros((200, 200, 200), dtype=np.float32)
    k = np.zeros((200, 200, 200), dtype=np.float32)
    ff = np.zeros((200, 200, 200), dtype=np.float32)
    
    # fill phantoms
    for n in range(len(props)):
        idx = (seg == n)
        M0[idx] = props["M0"][n]
        T1s[idx] = props["T1"][n] * 1.2
        T2s[idx] = props["T2"][n]  * 1.2
        T1f[idx] = props["bm"]["T1"][n] * 0.8
        T2f[idx] = props["bm"]["T2"][n] * 0.8
        k[idx] = props["bm"]["k"][n]
        ff[idx] = props["bm"]["weight"][n]
        
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
    return output.reshape(-1, *ishape)
    
    
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
    input = torch.as_tensor(input, dtype=signal.dtype, device=signal.device)
        
    output = regression.perk_eval(input, kernel, v, rho)
    
    # output as numpy
    output = output.detach().cpu().numpy()
    
    return output
    
# %% generate map
flip = 180.0 * np.ones(50, dtype=np.float32)
ESP = 5.0
device="cpu"

# prepare phantom
gt = create_phantom()

# simulate acquisition
echo_series = simulate(gt, flip, ESP, device=device)






    




