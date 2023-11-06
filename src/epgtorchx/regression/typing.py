# -*- coding: utf-8 -*-
"""
Helper data structure to handle Bloch dictionary.


Created on Fri Mar 10 15:32:27 2023

@author: Matteo Cencini
"""
from dataclasses import dataclass


import numpy as np


@dataclass
class BlochDictionary:
    atoms: np.ndarray
    norm: np.ndarray
    lookup_table: np.ndarray
    labels: dict
    
    def __post_init__(self):
        self.atoms = np.ascontiguousarray(self.atoms.transpose())
        self.lookup_table = np.ascontiguousarray(self.lookup_table.transpose())
        self.labels = self.labels.tolist()

