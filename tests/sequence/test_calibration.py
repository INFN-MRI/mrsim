"""The two thresholds the dispatcher decides by.

Nothing here asserts where a threshold falls in seconds: the values came off
one laptop GPU and a test that pinned them would be testing that laptop. What
is testable is that every pass the dispatcher asks about has an answer, that
the answers order the way the passes do, and that work either side of one ends
up on the side it claims, which ``tests/sequence/test_execution.py`` covers
for the crossover along with the order of the passes.
"""

import pytest
import torch

from torchsim.sequence._calibration import crossover, detection

KINDS = ["forward", "jvp", "adjoint"]


@pytest.mark.parametrize("kind", KINDS)
def test_every_pass_the_dispatcher_asks_about_has_a_crossover(kind):
    assert crossover(kind) > 0.0


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("where", ["cpu", "cuda"])
def test_every_pass_has_a_detection_threshold_on_either_side(kind, where):
    assert detection(kind, torch.device(where)) > 0.0


def test_a_pass_nobody_runs_is_a_mistake_rather_than_a_default():
    """A missing entry must not quietly read as "launch at any size"."""
    with pytest.raises(KeyError):
        crossover("curvature")
    with pytest.raises(KeyError):
        detection("curvature", torch.device("cpu"))


def test_a_card_earns_the_subspace_test_later_than_the_host():
    """It clears the work behind the verdict fast enough that the test, whose
    cost is fixed, takes a larger problem to repay.
    """
    assert detection("forward", torch.device("cuda")) > detection(
        "forward", torch.device("cpu")
    )
