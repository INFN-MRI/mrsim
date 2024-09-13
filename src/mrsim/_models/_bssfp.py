"""Balanced SSFP simulation sub-routines."""

__all__ = ["bssfp"]


import math


import numpy.typing as npt


import torch


from ._decorators import torchify, simulator


ARGMAP = {"T1": 0, "T2": 1}


@torchify
def bssfp(
    T1: float | npt.ArrayLike,
    T2: float | npt.ArrayLike,
    TR: float | npt.ArrayLike,
    alpha: float | npt.ArrayLike,
    field_map: float | npt.ArrayLike = 0.0,
    phase_cyc: float | npt.ArrayLike = 0.0,
    M0: float | npt.ArrayLike = 1.0,
    delta_cs: float | npt.ArrayLike = 0.0,
    phi_rf: float | npt.ArrayLike = 0.0,
    phi_edd: float = 0.0,
    phi_drift: float = 0.0,
    diff: str | tuple[str] | None = None,
):  # noqa
    r"""bSSFP transverse signal at time TE after excitation.

    Parameters
    ----------
    T1 : float | npt.ArrayLike
        longitudinal exponential decay time constant (in ms).
    T2 : float | npt.ArrayLike
        transverse exponential decay time constant (in ms).
    TR : float | npt.ArrayLike
        repetition time (in ms).
    alpha : float | npt.ArrayLike
        flip angle (in deg).
    field_map : float | npt.ArrayLike, optional
        B0 field map (in Hz).
    phase_cyc : float | npt.ArrayLike, optional
        Linear phase-cycle increment (in deg).
    M0 : float | npt.ArrayLike
        proton density.
    delta_cs : float | npt.ArrayLike, optional
        chemical shift of species w.r.t. the water peak (in Hz).
    phi_rf : float | npt.ArrayLike, optional
        RF phase offset, related to the combin. of Tx/Rx phases (in
        rad).
    phi_edd : float | npt.ArrayLike, optional
        phase errors due to eddy current effects (in rad).
    phi_drift : float | npt.ArrayLike, optional
        phase errors due to B0 drift (in rad).
    diff : str | tuple[str], optional
       String or tuple of strings, saying which arguments
       to get the signal derivative with respect to.
       Defaults to ``None`` (no differentation).

    Returns
    -------
    Mxy : numpy.ndarray
        Transverse complex magnetization.

    Notes
    -----
    `T1`, `T2`, `TR`, `alpha`, `field_map`, `phase_cyc`, `M0`, and `phi_rf` can all be
    either scalars or arrays.

    Output shape is determined by the shapes of input arrays.  All input
    arrays with equal shape will be assumed to have overlapping axes.  All
    input arrays with unique shapes will be assumed to have distinct axes
    and will be broadcast appropriately.

    Implementation of equations [1--2] in [1]_.  These equations are
    based on the Ernst-Anderson derivation [4]_ where off-resonance
    is assumed to be subtracted as opposed to added (as in the
    Freeman-Hill derivation [5]_).  Hoff actually gets Mx and My
    flipped in the paper, so we fix that here.  We also assume that
    the field map will be provided given the Freeman-Hill convention.

    We will additionally assume that linear phase increments
    (phase_cyc) will be given in the form:

    .. math::

        \theta = 2 \pi (\delta_{cs} + \Delta f_0)\text{TR} + \Delta \theta.

    Notice that this is opposite of the convention used in PLANET,
    where phase_cyc is subtracted (see equation [12] in [2]_).

    Also see equations [2.7] and [2.10a--b] from [4]_ and equations
    [3] and [6--12] from [5]_.

    References
    ----------
    .. [1] Xiang, Qing‐San, and Michael N. Hoff. "Banding artifact
           removal for bSSFP imaging with an elliptical signal
           model." Magnetic resonance in medicine 71.3 (2014):
           927-933.

    .. [4] Ernst, Richard R., and Weston A. Anderson. "Application of
           Fourier transform spectroscopy to magnetic resonance."
           Review of Scientific Instruments 37.1 (1966): 93-102.

    .. [5] Freeman R, Hill H. Phase and intensity anomalies in
           fourier transform NMR. J Magn Reson 1971;4:366–383.
    """
    return _bssfp(
        T1,
        T2,
        M0,
        field_map,
        delta_cs,
        phi_rf,
        phi_edd,
        phi_drift,
        TR,
        alpha,
        phase_cyc,
        diff,
    )


# %% subroutines
def _get_theta(TR, field_map, phase_cyc, delta_cs):
    """Get theta, spin phase per repetition time, given off-resonance."""
    # Enable broadcasting
    delta_cs = delta_cs.unsqueeze(-1)
    field_map = field_map.unsqueeze(-1)
    TR = TR.unsqueeze(0)
    phase_cyc = phase_cyc.unsqueeze(0)

    return 2 * math.pi * (delta_cs + field_map) * TR + phase_cyc


def _get_bssfp_phase(T2, TR, field_map, delta_cs, phi_rf, phi_edd, phi_drift):
    """Additional bSSFP phase factors."""
    # Enable broadcasting
    T2 = T2.unsqueeze(-1)
    delta_cs = delta_cs.unsqueeze(-1)
    field_map = field_map.unsqueeze(-1)
    phi_rf = phi_rf.unsqueeze(-1)
    phi_edd = phi_edd.unsqueeze(-1)
    phi_drift = phi_drift.unsqueeze(-1)
    TR = TR.unsqueeze(0)

    # Compute TE
    TE = TR / 2  # assume bSSFP

    # Compute total phase accrual
    phi = 2 * math.pi * (delta_cs + field_map) * TE + phi_rf + phi_edd + phi_drift

    # Compute signal dampening
    exp_term = torch.exp(-TE / T2)
    exp_term = torch.nan_to_num(exp_term, nan=0.0, posinf=0.0, neginf=0.0)

    return torch.exp(1j * phi) * exp_term


@simulator(ARGMAP, n_diff_args=2, n_batched_args=8)
def _bssfp(
    T1,
    T2,
    M0,
    field_map,
    delta_cs,
    phi_rf,
    phi_edd,
    phi_drift,
    TR,
    alpha,
    phase_cyc,
):
    # Unit conversion
    T1 = T1 * 1e-3  # ms -> s
    T2 = T2 * 1e-3  # ms -> s
    TR = TR * 1e-3  # ms -> s
    alpha = torch.deg2rad(alpha)
    phase_cyc = torch.deg2rad(phase_cyc)

    # Broadcast tensors
    TR, alpha, phase_cyc = torch.atleast_1d(TR, alpha, phase_cyc)
    TR, alpha, phase_cyc = torch.broadcast_tensors(TR.unsqueeze(-1), alpha, phase_cyc)
    TR, alpha, phase_cyc = TR.ravel(), alpha.ravel(), phase_cyc.ravel()
    TR, alpha, phase_cyc = TR.squeeze(), alpha.squeeze(), phase_cyc.squeeze()

    # We are assuming Freeman-Hill convention for off-resonance map,
    # so we need to negate to make use with this Ernst-Anderson-based implementation from Hoff
    field_map = -1 * field_map
    phase_cyc = -1 * phase_cyc

    # divide-by-zero risk with PyTorch's nan_to_num
    E1 = torch.exp(
        -1
        * torch.nan_to_num(
            TR.unsqueeze(0) / T1.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0
        )
    )
    E2 = torch.exp(
        -1
        * torch.nan_to_num(
            TR.unsqueeze(0) / T2.unsqueeze(-1), nan=0.0, posinf=0.0, neginf=0.0
        )
    )

    # Precompute theta and some cos, sin
    theta = _get_theta(
        TR=TR, field_map=field_map, phase_cyc=phase_cyc, delta_cs=delta_cs
    )
    ca = torch.cos(alpha).unsqueeze(0)
    sa = torch.sin(alpha).unsqueeze(0)
    ct = torch.cos(theta)
    st = torch.sin(theta)

    # Main calculation
    den = (1 - E1 * ca) * (1 - E2 * ct) - (E2 * (E1 - ca)) * (E2 - ct)
    Mx = -1 * M0 * ((1 - E1) * E2 * sa * st) / den
    My = M0 * ((1 - E1) * sa) * (1 - E2 * ct) / den
    Mxy = Mx + 1j * My
    Mxy = torch.nan_to_num(Mxy, nan=0.0, posinf=0.0, neginf=0.0)

    # Add additional phase factor for readout at TE = TR/2.
    signal = Mxy * _get_bssfp_phase(
        T2, TR, field_map, delta_cs, -1 * phi_rf, -1 * phi_edd, -1 * phi_drift
    )

    # Move multi-contrast in front
    signal = signal.unsqueeze(0)
    signal = signal.swapaxes(0, -1)
    
    return signal.squeeze().to(torch.complex64)
