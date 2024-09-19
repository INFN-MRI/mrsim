r"""
SPGR simulation
================
This script shows how to use the package to compute the transverse complex magnetization
using the SPGR signal equation.

Spoiled Gradient-Recalled Echo (SPGR) is a gradient-echo MRI sequence where residual transverse
magnetization is "spoiled" or eliminated between consecutive RF excitations. This results in predominantly
T1-weighted images with reduced steady-state signal, as transverse coherence is destroyed after each TR.


A simplified diagram for a SPGR sequence is this:

.. code-block::
    
    
                      |--------------------TR------------------|
                      |------TE------|
        .                                                                 .
        .     (alpha, phi(n))                                             .
        .                                                                 .
        .            /\                                        /\         .
        . Rf    _/\ /  \ /\_______________________________/\ /  \ /\_...  .               
        .         \/   \/                                   \/   \/       .     
        .                                           _____                 . 
        .                         ________         /     \                .
        . Gread ___________      /        \      _/       \______...      .
        .                  \____/          \____/                         .
        .        |Rf pulse|     | Readout |       | Spoil |               .
        .                                                                 .
        .                                                                 .

Here, ``alpha`` is the flip angle, ``phi(n)`` is the (often quadratically incremented) 
phase for the n-th pulse `n`` is the TR index.

"""

import numpy as np
import matplotlib.pyplot as plt

from mrsim import spgr

# %%
# Data Generation
# ===============
# For realistic 3D images we will use the mrtwin package.
# This can be installed as ``pip install mrtwin``

from mrtwin import shepplogan_phantom

phantom = shepplogan_phantom(ndim=2, shape=200, segtype=False)

# Set up the parameters for the bSSFP function
TR = 15.0  # Repetition time in ms
TE = 2.0  # Echo time in ms
alpha = 60.0  # Flip angle in degrees

# Generate the bSSFP signal for these parameters
Mxy = spgr(
    T1=phantom.T1,
    T2star=phantom.T2s,
    TR=TR,
    TE=TE,
    alpha=alpha,
)

plt.imshow(abs(Mxy), cmap="gray"), plt.axis("off"), plt.title(
    "SPGR signal magnitude [a.u.]"
)
plt.show()

# %%
# the function also supports other vectorial inputs, namely:
#
# 1. ``field_map``: The static field inhomogeneities map in [Hz]
# 2. ``delta_cs``: The point-wise value of tissue chemical shift in [Hz].
# 3. ``M0``: Equilibrium magnetization.
#
# We also support vectorized computation of multple flip angles (``alpha``),
# echo times (``TE``) and repetition times (``TR``).

TE = [2.0, 4.0, 6.0, 8.0]

Mxy = spgr(
    T1=phantom.T1,
    T2star=phantom.T2s,
    TR=TR,
    TE=TE,
    alpha=alpha,
)

display = np.concatenate([vol for vol in Mxy], axis=-1)
plt.imshow(abs(display), cmap="gray"), plt.axis("off"), plt.title("multi-echo SPGR")
plt.show()

# %%
# Automatic Differentiation
# =========================
# ``mrsim`` supports computation of signal derivatives wrt tissue parameter
# via ``torch`` forward mode differentiation.
# For bSSFP, we support derivatives wrt ``T1`` and / or ``T2``:

Mxy, Jacobian = spgr(
    T1=1000.0, T2star=100.0, TR=TR, TE=TE, alpha=alpha, diff=("T1", "T2star")
)

fig, ax = plt.subplots(1, 2)

ax[0].plot(abs(Mxy), ".")
ax[0].set_title("forward simulation pass")
ax[0].set_xlabel("contrast index")
ax[0].set_ylabel("magnetization [a.u.]")

ax[1].plot(abs(Jacobian.T), ".")
ax[1].set_title("jacobians")
ax[1].set_xlabel("contrast index")
ax[1].set_ylabel("magnetization derivative [a.u.]")
ax[1].legend(["$dM_{xy}/dT1$", "$dM_{xy}/dT2*$"])

plt.tight_layout()
plt.show()
