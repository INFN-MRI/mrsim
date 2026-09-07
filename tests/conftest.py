"""Rooted here so ``tests/`` is importable, which is what ``utils`` needs."""

import pytest


@pytest.fixture
def always_worth_detecting(monkeypatch):
    """Reach for the subspace verdict however small the problem looks.

    A test whose arms must take the same kernel has to say so: left to the
    threshold, two runs of different sizes -- a batch of trains against the
    trains one by one -- can fall on opposite sides of it and be compared
    across kernels.
    """
    from torchsim.sequence import _accelerators

    monkeypatch.setattr(_accelerators, "detection", lambda kind, device: 0.0)
