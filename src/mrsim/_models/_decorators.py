"""Decorator utils."""

__all__ = ["torchify", "simulator"]

import inspect

from functools import wraps
from typing import Callable

import numpy as np

import torch

from mrinufft._array_compat import with_torch
from mrinufft.operators.interfaces.utils import is_cuda_array


def torchify(func: Callable) -> Callable:
    """
    Force all the numeric input argument (including scalars) to be torch.Tensors.

    The first ArrayLike argument is chosen to determine the leading
    array interface (numpy, cupy or torch) and device (cpu or cuda:n).

    Output is automatically converted to the same array interface as
    the leading array.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = _get_args(func, args, kwargs)
        args = [np.asarray(arg) if _could_be_array(arg) else arg for arg in args]

        # run function
        return with_torch(func)(*args)

    return wrapper


def simulator(
    ARGMAP: dict = {}, n_diff_args: int = 0, n_batched_args: int = 0
) -> Callable:
    """
    Transform a simulator into differentiable simulator.

    Parameters
    ----------
    ARGMAP : dict, optional
        DESCRIPTION. The default is {}.
    n_diff_args : int, optional
        Number of differentiable (batched) arguments.
        The default is 0.
    n_batched_args : int, optional
        Number of batched (indluding differentiable) arguments.
        The default is 0.

    Returns
    -------
    Callable
        The differentiable version of decorated simulator.
        The decorated function gains an additional "diff"
        argument (last argument) which takes the name of the
        arguments included in the jacobian calculation.

    Notes
    -----
    Decorated function must receive real-valued input parameters.
    Only positional arguments are supported.
    We assume that args[:n_diff_args] are both batched and differentiable,
    and args[n_diff_args:n_batched_args] are batched and non-differentiable.
    It is recommended to wrap the decorated function to enable keyworder arguments
    We assume diff is the last argument

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            # Get diff
            diff = args[-1]
            args = args[:-1]

            # Forward pass
            output = func(*args)

            # Forward pass only case
            if n_diff_args == 0 or diff is None:
                return output

            # Get batched args
            batched_args = args[:n_batched_args]

            # Get shape of the matrix
            shape = _find_first_nonscalar_shape(batched_args)

            # Ravel batched arguments
            batched_args = [arg.ravel() for arg in batched_args]

            # Broadcast batched arguments
            batched_args = torch.broadcast_tensors(*batched_args)

            # Get differentiable and non-differentiable args
            other_args = args[n_batched_args:]

            # Get the Jacobian if differentiation is requested
            argnums = _get_argnums(diff, ARGMAP)
            jacobian_func = _get_jacobian(func, argnums, other_args)

            # Parallelized forward differentiation
            jacobian = jacobian_func(*batched_args)

            # Reshape to match original dimensions and return jacobian
            jacobian = _reshape_jacobian(jacobian, shape)

            return output, jacobian

        return wrapper

    return decorator


# %% subroutines
def _get_args(func, args, kwargs):
    """Convert input args/kwargs mix to a list of positional arguments.

    This automatically fills missing kwargs with default values.
    """
    signature = inspect.signature(func)

    # Get number of arguments
    n_args = len(args)

    # Create a dictionary of keyword arguments and their default values
    _kwargs = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            _kwargs[k] = v.default
        else:
            _kwargs[k] = None

    # Merge the default keyword arguments with the provided kwargs
    for k in kwargs.keys():
        _kwargs[k] = kwargs[k]

    # Replace args
    _args = list(_kwargs.values())

    return list(args) + _args[n_args:]


_numeric_types = (int, float, complex)


def _could_be_array(arg):
    if isinstance(arg, _numeric_types):
        return True
    elif isinstance(arg, (list, tuple)) and isinstance(arg[0], _numeric_types):
        leading_type = type(arg[0])
        return all(isinstance(el, leading_type) for el in arg)
    return False


def _get_device(args):
    for arg in args:
        if is_cuda_array(arg):
            return arg.device
    return torch.device("cpu")


def _find_first_nonscalar_shape(batched_args):  # noqa
    """Returns shape of first non-scalar tensor."""
    shape = [1]
    for arg in batched_args:
        if arg.ndim != 0:
            return arg.shape
    return shape


def _get_argnums(diff, ARGMAP):  # noqa
    """Helper function to get argument indices for differentiation."""

    if isinstance(diff, str):
        return ARGMAP[diff]
    elif isinstance(diff, (tuple, list)):
        return tuple([ARGMAP[d] for d in diff])
    else:
        raise ValueError(f"Unsupported diff type: {diff}")


def _get_jacobian(func, argnums, non_batched_args):  # noqa
    """
    Helper function to get the Jacobian with respect to the differentiable arguments.
    This version captures the batched_args and other_args to create a modified
    function that only takes diff_batched_args as input for differentiation.
    """

    def _forward(*batched_args):
        output = func(*batched_args, *non_batched_args)

        # Call the original function with all arguments
        return torch.stack((output.real, output.imag), dim=-1)

    # Return the Jacobian computation on the modified _forward function
    return torch.vmap(torch.func.jacfwd(_forward, argnums=argnums))


def _reshape_jacobian(jacobian, shape):
    """Reshape the Jacobian to match the original input shape."""
    if isinstance(jacobian, tuple):
        jacobian = [grad[..., 0] + 1j * grad[..., 1] for grad in jacobian]
        jacobian = [grad.T if grad.ndim == 2 else grad for grad in jacobian]
        jacobian = [grad.reshape(-1, *shape).squeeze() for grad in jacobian]
        jacobian = torch.stack(jacobian, dim=0)
    else:
        jacobian = jacobian[..., 0] + 1j * jacobian[..., 1]
        if jacobian.ndim == 2:
            jacobian = jacobian.T
        jacobian = jacobian.reshape(-1, *shape).squeeze()

    return torch.nan_to_num(jacobian.to(torch.complex64), posinf=0.0, neginf=0.0)
