"""Decorator utils."""

import inspect

from functools import wraps

import numpy as np

import torch

from mrinufft.operators.interfaces.utils import is_cuda_array


def force_scalar_tensors(fun):
    """Ensure all scalar arguments are tensors."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        kwargs = _get_defaults(fun, kwargs)

        # get device of first torch tensor
        device = _get_device(args, kwargs)

        # convert all to torch
        args = [
            torch.as_tensor(arg, device=device) if _could_be_tensor(arg) else arg
            for arg in args
        ]
        kwargs = {
            k: (torch.as_tensor(v, device=device) if _could_be_tensor(v) else v)
            for k, v in kwargs.items()
        }

        # run function
        return fun(*args, **kwargs)

    return wrapper


def _could_be_tensor(arg):
    if np.isscalar(arg) or isinstance(arg, (list, tuple)):
        if isinstance(arg, (list, tuple)) and isinstance(arg[0], str):
            return False
        elif isinstance(arg, str):
            return False
        else:
            return True
    else:
        return False


def _get_device(args, kwargs):
    for arg in args:
        if is_cuda_array(arg):
            return arg.device
    for arg in kwargs.values():
        if is_cuda_array(arg):
            return arg.device
    return torch.device("cpu")


def _get_defaults(func, kwargs):
    # Get the function signature
    sig = inspect.signature(func)

    # Create a dictionary of keyword arguments and their default values
    default_kwargs = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    # Merge the default keyword arguments with the provided kwargs
    return {**default_kwargs, **kwargs}
