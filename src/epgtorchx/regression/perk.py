"""Main regression routines."""

__all__ = ["perk_train", "perk_eval"]

import numpy as np

import torch

def perk_eval(input, train, v=None, nu=None, reg=2**-41):
    ishape = input.shape[1:] # (nz, ny, nx)
    
    # reshape
    print(input.shape)
    input = input.reshape(input.shape[0], -1)    
    if nu is not None:
        nu = nu.reshape(nu.shape[0], -1)    
        input = torch.cat((input, nu), axis=0)
        
    # transpose
    print(input.shape)
    input = input.t() # (nechoes, nvoxels)
    
    # compress
    if v is not None:
        input = input @ v
    
    # unwind phase
    ph0 = input[0]
    ph0 = ph0 / (abs(ph0) + 0.000000000001) # keep only phase
    input = input * ph0.conj()
    
    # normalize
    norm = (input * input.conj()).sum(axis=0)**0.5
    input = input / (norm + 0.000000000001)
    
    # get real part only
    input = input.real
    
    # feature maps
    print(input.shape)
    z = rff_map(input, train["H"], train["freq"], train["ph"])
    
    # % kernel ridge regression
    tmp = z - train["mean"]["z"]
    tmp = train["cov"]["zz"]
    tmp = torch.linalg.solve(train["cov"]["zz"] + reg * torch.eye(train["H"]), tmp) # (H, V)
    
    output = train["cov"]["zz"].t() @ tmp + train["mean"]["x"][:, None]
    return output.reshape(-1, *ishape)


def perk_train(train_x, train_y, train_nu=None, H=10**3, lamda=2**-1.5, c=2**0.6):
    """
    Perform PERK training.

    Args:
        rff (dict): Random Fourier features object with the following fields:
            - 'snr' (float): Estimate of the maximum signal for unity m0.
            - 'std' (float): Noise standard deviation in the training data.
            - 'len' (Tensor): Kernel input length scales of shape [D+N].
            - 'c' (float): Global kernel length scale parameter.
            - 'H' (int): Embedding dimension.
            - 'K' (int): Number of training samples.
        dist (dict): Sampling distribution object (ignored if x0 is set) with the following fields:
            - 'x' (List): List of latent parameter objects, each with fields:
                - 'supp' (Tensor): Distribution support [lb, ub].
                - 'prior' (str): Distribution type ('unif', 'logunif').
            - 'nu' (List): List of known parameter objects, each with fields:
                - 'supp' (Tensor): Distribution support [lb, ub].
        y (Tensor): Coil-combined image data of shape [DV].
        w (Tensor): Dataset weights of shape [D].
        nu (List): List of known parameters.
        P (dict): Scan parameters with the following fields:
            - 1st field: data type (ir, se, sp, de)
            - 2nd field: scan parameter, as appropriate
            - 'tr' (List): Repetition times in milliseconds.
            - 'ti' (List): Inversion times in milliseconds (only for 'ir' data).
            - 'te' (List): Echo times in milliseconds.
            - 'ainv' (List): Nominal effective flip angle of inversion in radians (only for 'ir' data).
            - 'aex' (List): Nominal flip angle of excitation in radians.
            - 'aref' (List): Nominal flip angle of refocusing in radians.
        dim (dict): Object containing dimension info.
        bool (dict): Boolean variables with the following fields:
            - 'reset' (bool): Reset random number generator during training.
            - 'mag.*' (bool): Using magnitude (spgr, dess) data.
            - 'chat' (bool): Verbosity.
            - 'rfftst' (bool): Show kernel approximation.
            - 'nuclip' (bool): (PERK) Clip nu sampling distribution.

    Returns:
        dict: PERK training parameter object (if empty, training will be performed) with the following fields:
            - 'mean.z' (Tensor): Sample mean of feature maps.
            - 'mean.x' (Tensor): Sample mean of x.
            - 'cov.zz' (Tensor): Sample auto-covariance of feature maps.
            - 'cov.xz' (Tensor): Sample cross-covariance between x and feature maps.
            - 'cov.xx' (Tensor): Sample auto-covariance of latent parameters x.
            - 'freq' (Tensor): Random 'frequency' vector of shape [H, D+N].
            - 'ph' (Tensor): Random phase vector of shape [H].

    Version Control:
        - 1.1 (2017-06-06): Adapted from mri_multicomp_map.
        - 1.2 (2017-06-12): rff.snr now controls m0 distribution sampling.
        - 1.3 (2017-09-14): Added PERK nu distribution clipping.
    """
    # prepare output
    train = {"mean": {"z": None, "x": None}, 
             "cov": {"zz": None, "xz": None, "xx": None}, 
             "freq": None,
             "ph": None,
             "H": None}
    
    # Training inputs
    if train_nu is not None:
        train_y = torch.cat((train_y, train_nu), axis=-1) # (ntrain, nechoes+nknown)
                
    # get lengthscales
    lengthscale = lamda * train_y.mean(axis=0) # (nechoes+nknown,)
    Q = lengthscale.shape[0]
    K = train_y.shape[0]

    # Random Fourier Features
    # To approximate Gaussian kernel N(0, Sigma):
    # 1. Construct rff.cov = inv((2*pi)^2*Sigma)
    # 1. Draw rff.freq from N(0, rff.cov)
    # 2. Draw rff.ph from unif(0, 1)
    
    tmp = lengthscale * (2 * np.pi * c)
    tmp = 1.0 / (tmp**2 + 0.000000000000001)
    cov = torch.diag(tmp) # (nechoes+nknown, nechoes+nknown)
    freq = torch.randn(H, Q) @ torch.linalg.cholesky(cov)
    ph = torch.rand(H)

    # Feature maps
    z = rff_map(train_y, H, freq, ph) # (ntrain, H)

    # Sample means
    train["mean"]["z"] = z.mean(axis=0) # (H,)
    train["mean"]["x"] = train_x.mean(axis=0) # (nparams,)

    # Sample covariances
    tmp = train_x - train["mean"]["x"] # (ntrain, nparams) - (nparams,)
    z = z - train["mean"]["z"]
    train["cov"]["zz"] = (z.t() @ z) / K # (H, H)
    train["cov"]["xz"] = (tmp.t() @ z) / K # (nparams, H)
    train["cov"]["xx"] = (tmp.t() @ tmp) / K # (nparams, nparams)
    
    train["freq"] = freq
    train["ph"] = ph
    train["H"] = H

    return train


# %% local utils
def rff_map(train_y, H, freq, ph):
    """
    Feature mapping via random Fourier features.

    Args:
        q (Tensor): Input data of shape [K, D+N].
        rff (dict): Random Fourier features object with the following fields:
            - 'std' (float): Noise standard deviation in training data.
            - 'len' (Tensor): Kernel input length scales of shape [D+N].
            - 'H' (int): Embedding dimension.
            - 'K' (int): Number of training samples.
        w (Tensor): Dataset weights of shape [D].
        freq (Tensor): Random 'frequency' vector of shape [H, D+N].
        ph (Tensor): Random phase vector of shape [H].
        dim (dict): Object containing dimension info.

    Returns:
        Tensor: Higher-dimensional features of shape [K, H].

    Version Control:
        - 1.1 (2016-09-02): Original
    """
    # Check freq and ph dimensions
    if freq.shape[0] != H or ph.shape[0] != H:
        raise ValueError('Length mismatch: freq and/or ph not of length H!?')

    # Random Fourier Features
    tmp = freq @ train_y.t()
    tmp = tmp + ph[:, None]
    tmp = torch.cos(2 * np.pi * tmp)
    z = (2 / H)**0.5 * tmp

    return z.t()