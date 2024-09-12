"""Balanced SSFP simulation sub-routines."""

__all__ = ["bssfp"]

import math

import numpy as np
import numpy.typing as npt

import torch

from mrinufft.operators.base import with_torch

from ._decorators import force_scalar_tensors

ARGMAP = {"T1": 0, "T2": 1}


@with_torch
@force_scalar_tensors
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
):
    signal = _bssfp(
        T1,
        T2,
        M0,
        field_map,
        delta_cs,
        phi_rf,
        TR,
        alpha,
        phase_cyc,
        phi_edd,
        phi_drift,
    )
    if diff is None:
        return signal
    if isinstance(diff, str):
        assert diff in [
            "T1",
            "T2",
        ], f"Differentiation is only supported wrt T1, T2; got {diff}"
        argnums = ARGMAP[diff]
    else:
        argnums = []
        for arg in diff:
            assert arg in [
                "T1",
                "T2",
            ], f"Differentiation is only supported wrt T1, T2; got {arg}"
            argnums.append(ARGMAP[arg])
        argnums = tuple(argnums)

    # get jacobian
    shape = T1.shape
    T1 = T1.ravel()
    T2 = T2.ravel()
    M0 = M0.ravel()
    field_map = field_map.ravel()
    delta_cs = delta_cs.ravel()
    phi_rf = phi_rf.ravel()

    T1, T2, M0, field_map, delta_cs, phi_rf = torch.broadcast_tensors(
        T1, T2, M0, field_map, delta_cs, phi_rf
    )
    jacfun = _jacobian(argnums, TR, alpha, phase_cyc, phi_edd, phi_drift)
    jacobian = jacfun(T1, T2, M0, field_map, delta_cs, phi_rf)
    
    if isinstance(jacobian, tuple):
        jacobian = [grad[..., 0] + 1j * grad[..., 1] for grad in jacobian]
        jacobian = [grad.T for grad in jacobian]
        jacobian = [grad.reshape(-1, *shape) for grad in jacobian]
        jacobian = torch.stack(jacobian, dim=0)
    else:
        jacobian = jacobian[..., 0] + 1j * jacobian[..., 1]
        jacobian = jacobian.T
        jacobian = jacobian.reshape(-1, *shape)

    return signal, jacobian


# %% subroutines
def _get_theta(TR, field_map, phase_cyc, delta_cs):
    """Get theta, spin phase per repetition time, given off-resonance."""
    delta_cs = delta_cs.unsqueeze(-1)
    field_map = field_map.unsqueeze(-1)
    TR = TR.unsqueeze(0)
    phase_cyc = phase_cyc.unsqueeze(0)

    return 2 * math.pi * (delta_cs + field_map) * TR + phase_cyc


def _get_bssfp_phase(T2, TR, field_map, delta_cs, phi_rf, phi_edd, phi_drift):
    """Additional bSSFP phase factors."""
    T2 = T2.unsqueeze(-1)
    delta_cs = delta_cs.unsqueeze(-1)
    field_map = field_map.unsqueeze(-1)
    phi_rf = phi_rf.unsqueeze(-1)
    TR = TR.unsqueeze(0)

    TE = TR / 2  # assume bSSFP
    phi = 2 * math.pi * (delta_cs + field_map) * TE + phi_rf + phi_edd + phi_drift

    # divide-by-zero risk, similar to np.nan_to_num in numpy
    exp_term = torch.exp(
        -1 * torch.nan_to_num(TE / T2, nan=0.0, posinf=0.0, neginf=0.0)
    )
    return torch.exp(1j * phi) * exp_term


def _bssfp(
    T1,
    T2,
    M0,
    field_map,
    delta_cs,
    phi_rf,
    TR,
    alpha,
    phase_cyc,
    phi_edd,
    phi_drift,
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

    signal = signal.unsqueeze(0)
    signal = signal.swapaxes(0, -1)
    return signal.squeeze()


def _bssfp_jac(TR, alpha, phase_cyc, phi_edd, phi_drift):
    def func(T1, T2, M0, field_map, delta_cs, phi_rf):
        output = _bssfp(
            T1,
            T2,
            M0,
            field_map,
            delta_cs,
            phi_rf,
            TR,
            alpha,
            phase_cyc,
            phi_edd,
            phi_drift,
        )
        return torch.stack((output.real, output.imag), dim=-1)
    return func


def _jacobian(argnums, TR, alpha, phase_cyc, phi_edd, phi_drift):
    _forward = _bssfp_jac(TR, alpha, phase_cyc, phi_edd, phi_drift)
    return torch.vmap(torch.func.jacfwd(_forward, argnums=argnums))
