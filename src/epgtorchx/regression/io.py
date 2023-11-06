# -*- coding: utf-8 -*-
"""
Utils to load MATLAB generated (Bloch) dictionaries.

Created on Fri Mar 10 15:17:32 2023

Ref: https://stackoverflow.com/questions/11955000/how-to-preserve-matlab-struct-when-accessing-in-python

@author: Matteo Cencini
"""
import scipy.io as sio


from pymatch.typing import BlochDictionary


__all__ = ['load']


def load(filename, key='dict'):
    """
    Load a Bloch dictionary generated in matlab

    Args:
        filename (TYPE): DESCRIPTION.
        key (TYPE, optional): DESCRIPTION. Defaults to 'dict'.

    Returns:
        None.

    """
    # load dict from matlab
    matdict = _loadmat(filename)[key]
    
    # return a dictionary
    return BlochDictionary(matdict['D'], matdict['normD'], matdict['lut'], matdict['labels'])
    
    

#%% Utils
def _loadmat(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(input):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in input:
        if isinstance(input[key], sio.matlab.mio5_params.mat_struct):
            input[key] = _todict(input[key])
    return input


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

