r"""
bSSFP simulation
================
This script shows how to use the package to compute the transverse complex magnetization
using the bSSFP signal equation.

Balanced Steady-State Free Precession (bSSFP) is a fast imaging sequence characterized by
repeated RF excitations, pre-phasing gradients, and alternating phase cycling. It produces
strong T2/T1 contrast and is widely used in clinical MRI due to its short acquisition times.

A simplified diagram for a bSSFP sequence is this:

.. code-block::
    
    
                      |--------------TR----------------|
                      |----TE=TR/2---|
        .                                                        .
        .     (alpha, n * dphi)                                  .
        .                                                        .
        .            /\                               /\         .
        . Rf    _/\ /  \ /\_______________________/\ /  \ /\_... .               
        .         \/   \/                          \/   \/       .     
        .                                                        . 
        .                         ________                       .
        . Gread ___________      /        \      ____________... .
        .                  \____/          \____/                .
        .        |Rf pulse|     | Readout |                      .
        .                                                        .
        .                                                        .

Here, ``alpha`` is the flip angle, ``dphi`` is the constant phase increment 
chosen for the phase cycling (e.g., 180°) and ``n`` is the TR index.

"""

import numpy as np
import matplotlib.pyplot as plt

from mrsim import bssfp

# %%
# Data Generation
# ===============
# For realistic 3D images we will use the mrtwin package.
# This can be installed as ``pip install mrtwin``

from mrtwin import shepplogan_phantom

phantom = shepplogan_phantom(ndim=2, shape=200, segtype=False)

# Set up the parameters for the bSSFP function
TR = 5.0  # Repetition time in ms
alpha = 60.0  # Flip angle in degrees
phase_cyc = 180.0  # Phase-cycling increment in degrees

# Generate the bSSFP signal for these parameters
Mxy = bssfp(
    T1=phantom.T1,
    T2=phantom.T2,
    TR=TR,
    alpha=alpha,
    phase_cyc=phase_cyc,
)

plt.imshow(abs(Mxy), cmap="gray"), plt.axis("off"), plt.title(
    "bSSFP signal magnitude [a.u.]"
)
plt.show()

# %%
# the function also supports other vectorial inputs, namely:
#
# 1. ``field_map``: The static field inhomogeneities map in [Hz]
# 2. ``delta_cs``: The point-wise value of tissue chemical shift in [Hz].
# 3. ``phi_rf``: RF phase offset in radians.
# 4. ``phi_edd``: Phase errors due to eddy current effects in radians.
# 5. ``phi_drift``: Phase errors due to B0 drift in radians.
# 6. ``M0``: Equilibrium magnetization.
#
# We also support vectorized computation of multple flip angles (``alpha``),
# phase increments (``phase_cyc``) and repetition times (``TR``).

alpha = [10.0, 30.0, 60.0, 10.0, 30.0, 60.0]
phase_cyc = [0.0, 0.0, 0.0, 180.0, 180.0, 180.0]

Mxy = bssfp(
    T1=phantom.T1,
    T2=phantom.T2,
    TR=TR,
    alpha=alpha,
    phase_cyc=phase_cyc,
)

display = np.concatenate([vol for vol in Mxy], axis=-1)
plt.imshow(abs(display), cmap="gray"), plt.axis("off"), plt.title(
    "multi-phase VFA bSSFP"
)
plt.show()

# %%
# Automatic Differentiation
# =========================
# ``mrsim`` supports computation of signal derivatives wrt tissue parameter
# via ``torch`` forward mode differentiation.
# For bSSFP, we support derivatives wrt ``T1`` and / or ``T2``:

Mxy, Jacobian = bssfp(
    T1=1000.0, T2=100.0, TR=TR, alpha=alpha, phase_cyc=phase_cyc, diff=("T1", "T2")
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
ax[1].legend(["$dM_{xy}/dT1$", "$dM_{xy}/dT2$"])

plt.tight_layout()
plt.show()
