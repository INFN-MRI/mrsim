"""
Levenberg-Marquardt related routines.
"""
__all__ = ["lmdif"]

import torch
from torch.func import vmap

# %% utils
bdiag = vmap(torch.diag)

def solve(A, B, p):
    try:
        return torch.linalg.solve(A, B)
    except:
        return torch.nan * p
    
def lmdif(fun, initial_guess, tau=1e-2, eps1=1e-6, eps2=1e-6, maxiter=20):
    """Implementation of the Levenberg-Marquardt algorithm in pure Python. Solves the normal equations.
    https://gist.github.com/geggo/92c77159a9b8db5aae73
    
    Args:
        fun: function computing residuals of the fitting model: 
                fun(pars, (predictors, observations) = observations - model(pars, predictors)
        pars: fitting parameters
        args: tuple with (predictors, observations)
    """
    p = initial_guess.clone()
    f, J = fun(p)
    
    A = torch.einsum("...ij,...jk->...ik", J, J.permute(0, 2, 1))  # J @ J.t() (natoms, np, np)
    g = torch.einsum("...ij,...j->...i", J, f)  # J @ J.t()
    mu = tau * bdiag(A).max(axis=-1)[0]

    I = torch.eye(p.shape[-1], dtype=f.dtype, device=f.device)[None, ...] 
    
    # initialize parameters
    niter = 0
    nu = 2.0 * torch.ones(mu.shape[0], dtype=mu.dtype, device=mu.device)
    stop = torch.linalg.norm(g, torch.inf, axis=-1) < eps1
    
    # aux
    one_third = 1.0 / 3 * torch.ones(p.shape[0], dtype=p.dtype, device=p.device)
    
    while not niter < maxiter:
        niter += 1

        # try and solve
        d = solve(A + mu * I, -g) # batch of vectors
        
        # get todo
        todo = torch.logical_not(torch.isnan(d))
        small = torch.linalg.norm(d, axis=-1) < eps2 * (torch.linalg.norm(d, axis=-1) + eps2) 
        
        # replace singular and small
        p[torch.logical_not(todo)] *= 0
        
        # get indices
        idx1 = torch.logical_and(todo, torch.logical_not(small))
                
        # update
        pnew = p[idx1] + d[idx1]
        fnew, Jnew = fun(pnew)
        
        div =  torch.einsum("...i,...i->...", d[idx1], (mu[idx1] * d[idx1] - g[idx1]))
        rho = (torch.linalg.norm(f[idx1], axis=-1)**2 - torch.linalg.norm(fnew[idx1], axis=-1)**2) / div
        
        # update
        idx2 = rho > 0
        
        # case 1
        p[idx1][idx2] = pnew[idx2]
        A[idx1][idx2] = torch.einsum("...ij,...jk->...ik", Jnew[idx2], Jnew[idx2].permute(0, 2, 1))        
        g[idx1][idx2] = torch.einsum("...ij,...j->...i", Jnew[idx2], fnew[idx2])
        f[idx1][idx2] = fnew[idx2]
        J[idx1][idx2] = Jnew[idx2]
        mu[idx1][idx2] = mu[idx1][idx2] * torch.stack((one_third[idx1][idx2], 1.0 - 1.0 - (2 * rho[idx2] - 1)**3), axis=0).max(axis=0)
        nu[idx1][idx2] = 2.0
        
        # case 2
        mu[idx1][torch.logical_not(idx2)] *= nu[idx1][torch.logical_not(idx2)]
        nu[idx1][torch.logical_not(idx2)] *= 2
    
    # replace
    p[stop] = initial_guess[stop]
    
    return p

