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
    idx = [0, -1, 4, 2, 3, 1]

    for n in tissues:
        mask = (seg == n)
        M0[mask] = props["M0"][idx[n]]
        T1s[mask] = props["T1"][idx[n]]
        T2s[mask] = props["T2"][idx[n]]
        T1f[mask] = props["bm"]["T1"][idx[n]]
        T2f[mask] = props["bm"]["T2"][idx[n]]
        k[mask] = props["bm"]["k"][idx[n]]
        ff[mask] = props["bm"]["weight"][idx[n]]
        
    return np.stack((M0, T1s, T2s, T1f, T2f, k, ff), axis=0)


def simulate(maps, flip, ESP, phases=None, device="cpu"):
    # default
    if phases is None:
        phases = -np.ones_like(flip) * 90.0
    
    # get ishape
    ishape = maps.shape[1:]    
    # output = epgx.fse(flip, phases, ESP, 
    #                   1000.0, 
    #                   100.0, 
    #                   T1bm=maps[3].flatten(), 
    #                   T2bm=maps[4].flatten(), 
    #                   weight_bm=maps[-1].flatten(),
    #                   kbm=0.0, 
    #                   device=device)
    output = epgx.fse(flip, phases, ESP, 1000.0,  maps[2].flatten(), device=device)
    
    
    # reshape
    # return maps[0] * output.T.reshape(-1, *ishape)
    return abs(output.T.reshape(-1, *ishape))

def fitting(input, flip, ESP, device="cpu"):
    
    # prepare batch
    ishape = input.shape[1:]
    input = input.reshape(input.shape[0], -1)
    
    # prepare for fit
    input = input.T # (nvoxels, nechoes)
    # input = input / (np.linalg.norm(input, axis=-1)[:, None] + 0.00000001)
    input = torch.as_tensor(input, device=device)
    input = abs(input)
    
    # prepare model
    def model(p):
        sig, dsig = epgx.fse(flip, 0 * flip + 90.0, ESP, 1000.0, p[:, 0], diff=["T2"], device=device, asnumpy=False)
        # sig, dsig = epgx.fse(flip, 0 * flip + 90.0, ESP, 1000.0, 
        #                      p[:, 1], T1bm=500.0, T2bm=p[:, 2], kbm=0.0, 
        #                      weight_bm=p[:, 0], 
        #                      diff=["weight", "T2"], 
        #                      device=device, assnumpy=False)
        
        return -sig.imag, -dsig.imag[:, None, :]
        
    # initial parameters
    # f0 = 0.1 * torch.ones(input.shape[0], dtype=input.dtype, device=input.device)    
    T20 = 85.0 * torch.ones(input.shape[0], dtype=input.dtype, device=input.device)    
    # T2bm0 = 10.0 * torch.ones(input.shape[0], dtype=input.dtype, device=input.device)    
    # kbm0 = 10.0 * torch.ones(input.shape[0], dtype=input.dtype, device=input.device)
    # p0 = torch.stack((f0, T20, T2bm0), axis=-1)
    p0 = T20[:, None]

    # actual inference
    prediction = regression.lmdif(model, p0)
    
    # prepare for output
    prediction = prediction.cpu().detach().numpy()
    prediction = prediction.T # (nparams, nvoxels)
    
    return prediction.reshape(-1, *ishape)
    
# %% generate map
flip = 180.0 * np.ones(32, dtype=np.float32)
ESP = 10.0
device="cpu"

# prepare phantom
gt = create_phantom([32, 32, 2])

# simulate acquisition
echo_series = simulate(gt, flip, ESP, device=device)

# fit
# omaps = fitting(echo_series, flip, ESP, device=device)






    




