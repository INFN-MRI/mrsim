"""Test bSSFP sequence"""

import numpy as np
import pytest

import mrsim
import ssfp

ARGLUT = {"T1": 0, "T2star": 1}


# Define finite difference approximation for derivatives
def finite_difference(func, args, arg_idx, epsilon=1e-3):
    gradient = []

    args1 = list(args)
    args2 = list(args)

    args1[arg_idx] = args1[arg_idx] + epsilon
    args2[arg_idx] = args2[arg_idx] - epsilon

    f1 = func(*args1)
    f2 = func(*args2)

    gradient = 1e-3 * (f1 - f2) / (2 * epsilon)

    return gradient


# Test parameters
@pytest.mark.parametrize(
    "T1, T2star, TR, TE, alpha, M0",
    [
        (1000.0, 100.0, 10.0, 2.0, 30.0, 1.0),  # all scalar
        (
            1000.0 * np.ones((32, 32)),
            100.0 * np.ones((32, 32)),
            10.0,
            2.0,
            30.0,
            1.0,
        ),  # broadcast maps
        (
            1000.0,
            100.0,
            10.0,
            [1.0, 1.0, 6.0, 6.0],
            [15.0, 30.0, 15.0, 30.0],
            1.0,
        ),  # vectorize params
    ],
)
def test_spgr(T1, T2star, TR, TE, alpha, M0):
    args = [
        T1,
        T2star,
        TR,
        TE,
        alpha,
        M0,
    ]
    args = [np.atleast_1d(arg) for arg in args]
    (
        T1,
        T2star,
        TR,
        TE,
        alpha,
        M0,
    ) = args

    result = mrsim.spgr(
        T1,
        T2star,
        TR,
        TE,
        alpha,
        M0,
    )
    if len(TE) == 1:
        expected_output = ssfp.spoiled_gre(
            T1 * 1e-3,
            T2star * 1e-3,
            TR * 1e-3,
            TE * 1e-3,
            np.deg2rad(alpha),
            M0,
        )
    else:
        expected_output = [
            ssfp.spoiled_gre(
                T1 * 1e-3,
                T2star * 1e-3,
                TR * 1e-3,
                TE[n] * 1e-3,
                np.deg2rad(alpha[n]),
                M0,
            )
            for n in range(len(TE))
        ]
        expected_output = np.asarray(expected_output).squeeze()

    np.testing.assert_allclose(abs(result), expected_output)


# Test derivatives
@pytest.mark.parametrize(
    "T1, T2star, TR, TE, alpha, M0, diff",
    [
        (1000.0, 100.0, 10.0, 2.0, 30.0, 1.0, "T1"),
        (1000.0, 100.0, 10.0, 2.0, 30.0, 1.0, "T2star"),
    ],
)
def test_spgr_derivatives(T1, T2star, TR, TE, alpha, M0, diff):
    args_ref = [
        T1 * 1e-3,
        T2star * 1e-3,
        TR * 1e-3,
        TE * 1e-3,
        np.deg2rad(alpha),
        M0,
    ]
    args_ref = [np.atleast_1d(arg) for arg in args_ref]

    # Call spgr with diff to get automatic gradient
    _, result = mrsim.spgr(
        T1,
        T2star,
        TR,
        TE,
        alpha,
        M0,
        diff=diff,
    )

    # Calculate finite differences for the derivative
    expected_output = finite_difference(ssfp.spoiled_gre, args_ref, ARGLUT[diff])

    # Compare numerical and analytical gradients
    assert np.abs(abs(result) - abs(expected_output)) < 1e-5
