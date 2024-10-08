"""bSSFP MR Fingerprinting simulator"""

__all__ = ["bssfpmrf"]

import warnings
import numpy as np

import dacite
from dacite import Config

from .. import blocks
from .. import ops
from . import base


def bssfpmrf(
    flip,
    TR,
    T1,
    T2,
    sliceprof=False,
    DE=False,
    diff=None,
    device="cpu",
    TI=None,
    **kwargs,
):
    """
    Simulate an inversion-prepared bSSFP sequence with variable flip angles.

    Args:
        flip (float, array-like): Flip angle in [deg] of shape (npulses,) or (npulses, nmodes).
        TR (float): Repetition time in [ms].
        T1 (float, array-like): Longitudinal relaxation time in [ms].
        T2 (float, array-like): Transverse relaxation time in [ms].
        sliceprof (optional, bool or array-like): excitation slice profile (i.e., flip angle scaling across slice).
            If False, pulse are non selective. If True, pulses are selective but ideal profile is assumed.
            If array, flip angle scaling along slice is simulated. Defaults to False.
        DE (optional, bool): If True, simulation is repeated 2 times to mimick Driven Equilibrium acquisition.
            Defaults to  False.
        diff (optional, str, tuple[str]): Arguments to get the signal derivative with respect to.
            Defaults to None (no differentation).
        device (optional, str): Computational device. Defaults to "cpu".
        TI (optional, float): Inversion time in [ms]. Defaults to None (no preparation).

    Kwargs (simulation):
        nstates (optional, int): Maximum number of EPG states to be retained during simulation.
            High numbers improve accuracy but decrease performance. Defaults to 10.
        max_chunk_size (optional, int): Maximum number of atoms to be simulated in parallel.
            High numbers increase speed and memory footprint. Defaults to natoms.
        nlocs (optional, int): Maximum number of spatial locations to be simulated (i.e., for slice profile effects).
            Defaults to 15 for slice selective and 1 for non-selective / ideal profile acquisitions.
        verbose (optional, bool): If true, prints execution time for signal (and gradient) calculations.

    Kwargs (sequence):
        TE (optional, float): Echo time in [ms]. Defaults to 0.0.
        B1sqrdTau (float): pulse energies in [uT**2 * ms] when flip = 1 [deg].

        global_inversion (bool): assume nonselective (True) or selective (False) inversion. Defaults to True.
        inv_B1sqrdTau (float): inversion pulse energy in [uT**2 * ms] when flip = 1 [deg].

     Kwargs (System):
         B1 (optional, float, array-like): flip angle scaling factor (1.0 := nominal flip angle).
             Defaults to None (nominal flip angle).
         B0 (optional, float, array-like): Bulk off-resonance in [Hz]. Defaults to None.
         B1Tx2 (optional, Union[float, npt.NDArray[float], torch.FloatTensor]): flip angle scaling factor for secondary RF mode (1.0 := nominal flip angle). Defaults to None.
         B1phase (optional, Union[float, npt.NDArray[float], torch.FloatTensor]): B1 relative phase in [deg]. (0.0 := nominal rf phase). Defaults to None.

    Kwargs (Main pool):
        chemshift (optional, float): chemical shift for main pool in [Hz]. Defaults to None.

    Kwargs (Bloch-McConnell):
        T1bm (optional, float, array-like): longitudinal relaxation time for secondary pool in [ms]. Defaults to None.
        T2bm (optional, float, array-like): transverse relaxation time for main secondary in [ms]. Defaults to None.
        kbm (optional, float, array-like). Nondirectional exchange between main and secondary pool in [Hz]. Defaults to None.
        weight_bm (optional, float, array-like): relative secondary pool fraction. Defaults to None.
        chemshift_bm (optional, float): chemical shift for secondary pool in [Hz]. Defaults to None.

    Kwargs (Magnetization Transfer):
        kmt (optional, float, array-like). Nondirectional exchange between free and bound pool in [Hz].
            If secondary pool is defined, exchange is between secondary and bound pools (i.e., myelin water and macromolecular), otherwise
            exchange is between main and bound pools. Defaults to None.
        weight_mt (optional, float, array-like): relative bound pool fraction. Defaults to None.

    """
    # constructor
    init_params = {
        "flip": flip,
        "TR": TR,
        "T1": T1,
        "T2": T2,
        "diff": diff,
        "device": device,
        "TI": TI,
        **kwargs,
    }

    # get TE
    if "TE" not in init_params:
        TE = 0.0
    else:
        TE = init_params["TE"]

    # get verbosity
    if "verbose" in init_params:
        verbose = init_params["verbose"]
    else:
        verbose = False

    # get verbosity
    if "asnumpy" in init_params:
        asnumpy = init_params["asnumpy"]
    else:
        asnumpy = True

    # get selectivity:
    if sliceprof:
        selective_exc = True
    else:
        selective_exc = False

    # check for global inversion
    if "global_inversion" in init_params:
        selective_inv = not (init_params["global_inversion"])
    else:
        selective_inv = False

    # check for conflicts in inversion selectivity
    if selective_exc is False and selective_inv is True:
        warnings.warn("3D acquisition - forcing inversion pulse to global.")
        selective_inv = False

    # inversion pulse properties
    if TI is None:
        inv_props = {}
    else:
        inv_props = {"slice_selective": selective_inv}

    if "inv_B1sqrdTau" in kwargs:
        inv_props["b1rms"] = kwargs["inv_B1sqrdTau"] ** 0.5
        inv_props["duration"] = 1.0

    # check conflicts in inversion settings
    if TI is None:
        if inv_props:
            warnings.warn(
                "Inversion not enabled - ignoring inversion pulse properties."
            )
            inv_props = {}

    # excitation pulse properties
    rf_props = {"slice_selective": selective_exc}
    if "B1sqrdTau" in kwargs:
        inv_props["b1rms"] = kwargs["B1sqrdTau"] ** 0.5
        inv_props["duration"] = 1.0

    if np.isscalar(sliceprof) is False:
        rf_props["slice_profile"] = kwargs["sliceprof"]

    # get nlocs
    if "nlocs" in init_params:
        nlocs = init_params["nlocs"]
    else:
        if selective_exc:
            nlocs = 15
        else:
            nlocs = 1

    # interpolate slice profile:
    if "slice_profile" in rf_props:
        nlocs = min(nlocs, len(rf_props["slice_profile"]))
    else:
        nlocs = 1

    # assign nlocs
    init_params["nlocs"] = nlocs

    # put all properties together
    props = {"inv_props": inv_props, "rf_props": rf_props, "DE": DE}

    # initialize simulator
    simulator = dacite.from_dict(
        bSSFPMRF, init_params, config=Config(check_types=False)
    )

    # run simulator
    if diff:
        # actual simulation
        sig, dsig = simulator(flip=flip, TR=TR, TI=TI, TE=TE, props=props)

        # post processing
        if asnumpy:
            sig = sig.detach().cpu().numpy()
            dsig = dsig.detach().cpu().numpy()

        # prepare info
        info = {"trun": simulator.trun, "tgrad": simulator.tgrad}
        if verbose:
            return sig, dsig, info
        else:
            return sig, dsig
    else:
        # actual simulation
        sig = simulator(flip=flip, TR=TR, TI=TI, TE=TE, props=props)

        # post processing
        if asnumpy:
            sig = sig.cpu().numpy()

        # prepare info
        info = {"trun": simulator.trun}
        if verbose:
            return sig, info
        else:
            return sig


# %% utils
class bSSFPMRF(base.BaseSimulator):
    """Class to simulate inversion-prepared (variable flip angle) bSSFP."""

    @staticmethod
    def sequence(
        flip,
        TR,
        TI,
        TE,
        props,
        T1,
        T2,
        B1,
        df,
        weight,
        k,
        chemshift,
        states,
        signal,
    ):
        # parsing pulses and grad parameters
        inv_props = props["inv_props"]
        rf_props = props["rf_props"]
        driven_equilibrium = props["DE"]

        # get number of repetitions
        if driven_equilibrium:
            nreps = 2
        else:
            nreps = 1

        # get number of frames and echoes
        npulses = flip.shape[0]

        # define preparation
        Prep = blocks.InversionPrep(TI, T1, T2, weight, k, inv_props)

        # prepare RF pulse
        RF = blocks.ExcPulse(states, B1, rf_props)

        # prepare free precession period
        Xpre, Xpost = blocks.bSSFPFidStep(states, TE, TR, T1, T2, weight, k, chemshift)

        for r in range(nreps):
            # magnetization prep
            states = Prep(states)

            # actual sequence loop
            for n in range(npulses):
                # apply pulse
                states = RF(states, flip[n])

                # relax, recover and record signal for each TE
                states = Xpre(states)
                signal[n] = ops.observe(states, RF.phi)

                # relax, recover and spoil
                states = Xpost(states)

        return signal
