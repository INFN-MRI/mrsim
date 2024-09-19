"""Test bSSFP sequence"""

import numpy as np
import pytest

import mrsim
import ssfp

ARGLUT = {"T1": 0, "T2": 1}


# Define finite difference approximation for derivatives
def finite_difference(func, args, arg_idx, epsilon=1e-3):
    gradient = []

    args1 = list(args)
    args2 = list(args)

    args1[arg_idx] = args1[arg_idx] + epsilon
    args2[arg_idx] = args2[arg_idx] - epsilon

    f1 = func(*args1)
    f2 = func(*args2)

    gradient = 1e-3 * (f1 - f2) / (2 * epsilon)  # s -> ms

    return gradient


# Test parameters
@pytest.mark.parametrize(
    "T1, T2, TR, alpha, field_map, phase_cyc, M0, delta_cs, phi_rf, phi_edd, phi_drift",
    [
        (1000.0, 100.0, 4.0, 30.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),  # all scalar
        (
            1000.0 * np.ones((32, 32)),
            100.0 * np.ones((32, 32)),
            4.0,
            30.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),  # broadcast maps
        (
            1000.0,
            100.0,
            4.0,
            [15.0, 30.0, 15.0, 30.0],
            0.0,
            [0.0, 0.0, 180.0, 180.0],
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),  # vectorize params
    ],
)
def test_bssfp(
    T1, T2, TR, alpha, field_map, phase_cyc, M0, delta_cs, phi_rf, phi_edd, phi_drift
):
    args = [
        T1,
        T2,
        TR,
        alpha,
        field_map,
        phase_cyc,
        M0,
        delta_cs,
        phi_rf,
        phi_edd,
        phi_drift,
    ]
    args = [np.asarray(arg) for arg in args]
    (
        T1,
        T2,
        TR,
        alpha,
        field_map,
        phase_cyc,
        M0,
        delta_cs,
        phi_rf,
        phi_edd,
        phi_drift,
    ) = args

    result = mrsim.bssfp(
        T1,
        T2,
        TR,
        alpha,
        field_map,
        phase_cyc,
        M0,
        delta_cs,
        phi_rf,
        phi_edd,
        phi_drift,
    )
    expected_output = ssfp.bssfp(
        T1 * 1e-3,
        T2 * 1e-3,
        TR * 1e-3,
        np.deg2rad(alpha),
        field_map,
        np.deg2rad(phase_cyc),
        M0,
        delta_cs,
        phi_rf,
        phi_edd,
        phi_drift,
    )

    np.testing.assert_allclose(result, expected_output)


# Test derivatives
@pytest.mark.parametrize(
    "T1, T2, TR, alpha, field_map, phase_cyc, M0, delta_cs, phi_rf, phi_edd, phi_drift, diff",
    [
        (1000.0, 100.0, 4.0, 30.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, "T1"),
        (1000.0, 100.0, 4.0, 30.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, "T2"),
        # Add more parameters as needed
    ],
)
def test_bssfp_derivatives(
    T1,
    T2,
    TR,
    alpha,
    field_map,
    phase_cyc,
    M0,
    delta_cs,
    phi_rf,
    phi_edd,
    phi_drift,
    diff,
):
    args_ref = [
        T1 * 1e-3,
        T2 * 1e-3,
        TR * 1e-3,
        np.deg2rad(alpha),
        field_map,
        np.deg2rad(phase_cyc),
        M0,
        delta_cs,
        phi_rf,
        phi_edd,
        phi_drift,
    ]
    args_ref = [np.asarray(arg) for arg in args_ref]

    # Call bssfp with diff to get automatic gradient
    _, result = mrsim.bssfp(
        T1,
        T2,
        TR,
        alpha,
        field_map,
        phase_cyc,
        M0,
        delta_cs,
        phi_rf,
        phi_edd,
        phi_drift,
        diff=diff,
    )

    # Calculate finite differences for the derivative
    expected_output = finite_difference(ssfp.bssfp, args_ref, ARGLUT[diff])

    # Compare numerical and analytical gradients
    assert np.abs(result - expected_output) < 1e-5
