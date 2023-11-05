"""CRLB optimization utils."""

import torch

def calculate_crlb(grad, W=None, weight=1.0):
    """
    Calculate Cramer-Rao-Lower-Bound, given input
    gradient of shape (nparams, nechoes).
    
    Args:
        grad (tensor): gradient wrt tissue parameters.
        W (tensor): weighting matrix for the different parameters.
        weight (float): global cost scale.
        
    Returns:
        (tensor): weighted trace of crlb matrix.

    """
    if len(grad.shape) == 1:
        grad = grad[None, :]
        
    if W is None:
        W = torch.eye(grad.shape[0], dtype=grad.dtype, device=grad.device)
        
    J = torch.stack((grad.real, grad.imag), axis=0) # (nparams, nechoes)
    J = J.permute(2, 1, 0)
    
    # calculate Fischer information matrix
    In = torch.einsum("bij,bjk->bik", J, J.permute(0, 2, 1))
    I = In.sum(axis=0) # (nparams, nparams)

    # Invert
    return torch.trace(torch.linalg.inv(I) * W).real * weight

