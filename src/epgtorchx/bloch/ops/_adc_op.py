"""
EPG signal recording operator.

Can be used to record signal during simulation.
"""
__all__ = ["observe", "susceptibility"]

import torch


def observe(states, phi=None):
    """
    Store observable magnetization.

    Args:
        states (dict): input states matrix for free pools.
        phi (torch.Tensor): effective phase for signal demodulation.

    Returns:
        (torch.Tensor): net observable magnetization at current timepoint.

    """
    # parse
    F = states["F"]  # (nstates, nlocs, npools, 3)

    # get transverse magnetization
    mxy = F[0, ..., 0]  # (nlocs, npools)

    # demodulate
    if phi is not None:
        mxy = mxy * torch.exp(-1j * phi)

    # sum across pools
    mxy = mxy.sum(axis=-1).mean(axis=-1)

    return mxy


def susceptibility(signal, time, z):
    """
    Apply static susceptibility effects (bulk decay and dephasing).

    Args:
        (torch.Tensor): net observable magnetization.
        time (torch.Tensor): effective phase for signal demodulation.

    Returns:
        (torch.Tensor): net observable magnetization at current timepoint.

    """
    if time.shape[-1] != 1:  # multiecho
        if signal.shape[-1] != time.shape[-1]:  # assume echo must be broadcasted
            signal = [..., None]

    #  apply effect
    if time.shape[-1] == 1 and time != 0:
        signal = signal * torch.exp(-time * (z[..., 0] + 1j * z[..., 1]))

    return signal
