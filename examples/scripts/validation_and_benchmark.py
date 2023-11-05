# %% plotting utils
import numpy as np
import matplotlib.pyplot as plt

# plotting
def display_signal(input, legend=None, symbol='-', color=None):
    if color is not None:
        plt.gca().set_prop_cycle(plt.cycler("color", color))
    plt.plot(abs(input), symbol)
    plt.xlim([0, len(input)])
    plt.xlabel("TR index")
    plt.ylabel("signal magnitude [a.u.]")
    if legend is not None:
        plt.legend(legend)
    plt.tight_layout()

# %% mri-sim-py implementation
import epg as simpy

def mrisimpy_fse(flip, phases, ESP, T1, T2, T1b=None, T2b=None, k=None, verbose=False):
    pass

# %% Sycomore implementation
import multiprocess as mp
import time

import sycomore
from sycomore.units import *

def _sycomore_fse(flip, ESP, T1, T2, T1b=None, T2b=None, k=None, fb=None):
    # parse 
    npulses = flip.shape[0]
    
    # initialize spin system
    if T1b is None:
        species = sycomore.Species(T1, T2)

        # initialize model
        model = sycomore.epg.Regular(species)
    else:
        species_a = sycomore.Species(T1, T2)
        species_b = sycomore.Species(T1b, T2b)
        M0a = np.asarray([0.0, 0.0, 1-fb], dtype=np.float64)
        M0b = np.asarray([0.0, 0.0, fb], dtype=np.float64)
        k_a = k * fb

        # initialize model
        model = sycomore.epg.Regular(species_a, species_b, M0a, M0b, k_a)
    
    # initialize output
    signal = np.zeros(npulses, dtype=np.complex64)

    # excitation
    model.apply_pulse(90.0 * deg)
    
    # loop over flip angles
    for n in range(npulses):
        
        # apply relaxation
        model.relaxation(0.5 * ESP)

        # shift states
        model.shift()
        
        # apply rf
        model.apply_pulse(flip[n])

        # shift states
        model.shift()

        # apply relaxation
        model.relaxation(0.5 * ESP)

        # record signal
        signal[n] = model.echo

    return signal
    
def sycomore_fse(parallel, flip, phases, ESP, T1, T2, T1b=None, T2b=None, k=None, fb=None, verbose=False):

    # sequence parameters
    flip = np.asarray(flip) * deg
    ESP = ESP * ms

    # convert to array  
    T1 = np.atleast_1d(np.asarray(T1)) 
    T2 = np.atleast_1d(np.asarray(T2))

    if T1b is not None:
        T1b = np.atleast_1d(np.asarray(T1b)) 
        T2b = np.atleast_1d(np.asarray(T2b))
        k = np.atleast_1d(np.asarray(k)) * 1e-3
        fb = np.atleast_1d(np.asarray(fb))
        
    # broadcast
    if T1b is None:
        T1, T2 = np.broadcast_arrays(T1, T2)
    else:
        T1, T2, T1b, T2b, k, fb = np.broadcast_arrays(T1, T2, T1b, T2b, k, fb)

    # units
    T1 = T1 * ms
    T2 = T2 * ms

    if T1b is not None:
        T1b = T1b * ms
        T2b = T2b * ms
        k = k * kHz

    # get natoms
    natoms = T1.shape[0]
    
    # run
    if parallel is False:
        engine = _sycomore_fse
        t0 = time.time()
        if T1b is None:
            signal = [engine(flip, ESP, T1[n], T2[n]) for n in range(natoms)]
        else:
            signal = [engine(flip, ESP, T1[n], T2[n], T1b[n], T2b[n], k[n], fb[n]) for n in range(natoms)]
        t1 = time.time()
        if verbose:
            return np.stack(signal, axis=0).squeeze(), t1-t0
        else:
            return np.stack(signal, axis=0).squeeze()
    else:
        if T1b is None:
            engine = lambda t1, t2 : _sycomore_fse(flip, TR, t1, t2)
        else:
            engine = lambda t1, t2, t1b, t2b, kk, ff : _sycomore_fse(flip, TR, t1, t2, t1b, t2b, kk, ff)
        t0 = time.time()
        if T1b is None:
            with mp.Pool(mp.cpu_count()) as p:
                signal = p.starmap(engine, zip(T1, T2))
        else:
            with mp.Pool(mp.cpu_count()) as p:
                signal = p.starmap(engine, zip(T1, T2, T1b, T2b, k, fb))
        t1 = time.time()
        if verbose:
            return np.stack(signal, axis=0), t1-t0
        else:
            return np.stack(signal, axis=0)
        
# %% execution
import epgtorchx as epgx

# parameters
flip = 50 * [180.0]
phases = 50 * [90.0]
ESP = 5.0

T1 = 1000.0 # [500.0, 833.0, 2569.0]
T2 = 100.0  # 

T1b = 500.0 # ms 
T2b = 20.0 # ms 

f = 0.2
k = 10.0 # Hz  

# computation
sig0, rt0 = sycomore_fse(False, flip, phases, ESP, T1, T2, T1b, T2b, k, f, verbose=True)
sig1, rt1 = epgx.fse(flip, phases, ESP, T1, T2, T1bm=T1b, T2bm=T2b, kbm=k, weight_bm=f, verbose=True)

sig0noex, rt0noex = sycomore_fse(False, flip, phases, ESP, T1, T2, verbose=True)
sig1noex, rt1noex = epgx.fse(flip, phases, ESP, T1, T2, verbose=True)

# plotting
