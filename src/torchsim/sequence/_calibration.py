"""Where the dispatcher's two thresholds sit.

Two decisions in the dispatcher are questions about size: whether a problem is
big enough to repay a device launch, and whether testing a sequence for a real
subspace repays the kernel it would speed up. Both are answered here, from a
table.

The first threshold falls at a couple of dozen voxels on any card measured, so
every problem anyone simulates is orders of magnitude past it and the answer
never turns on which machine is asking. The second falls higher, and the values
below were measured on one laptop GPU: they are a place to start rather than a
description of any particular machine, and they are conservative in the
direction that costs least -- a test skipped runs the complex kernel, which is
correct everywhere.
"""

from __future__ import annotations

__all__ = ["crossover", "detection"]

import torch

# Work below which a launch does not repay itself, per pass. Work counts
# (voxel, train, event) triples, so these are tens of voxels of an echo train.
_CROSSOVER = {"forward": 1500.0, "jvp": 400.0, "adjoint": 500.0}

# Work above which one subspace test repays the kernel it selects. The test
# costs a fixed handful of reductions and one round trip; the saving is the gap
# between the complex and the real kernels, which grows with the problem, and
# is widest for the adjoint.
_DETECTION = {
    ("forward", "cuda"): 100_000.0,
    ("jvp", "cuda"): 100_000.0,
    ("adjoint", "cuda"): 2_000.0,
    ("forward", "cpu"): 5_000.0,
    ("jvp", "cpu"): 5_000.0,
    ("adjoint", "cpu"): 5_000.0,
}


def crossover(kind: str) -> float:
    """Work below which the host beats a device for a ``kind`` of pass."""
    return _CROSSOVER[kind]


def detection(kind: str, device: torch.device) -> float:
    """Work above which testing for a real subspace repays what it costs."""
    return _DETECTION[kind, device.type]
