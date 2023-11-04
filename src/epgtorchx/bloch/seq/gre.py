"""
"""
__all__ = ["ssfp_sim"]

import warnings

import dacite
from dacite import Config

from . import blocks

DEBUG = False

def ssfp_sim(flip, TR, T1, T2, diff=None, device="cpu", TI=None, **kwargs):
    """
    Simulate an (inversion-prepared) SSFP sequence (with variable flip angles).

    Args:
        flip (float, array-like): Flip angle in [deg] of shape (npulses,) or (npulses, nmodes).
        TR (float): Repetition time in [ms].
        T1 (float, array-like): Longitudinal relaxation time in [ms].
        T2 (float, array-like): Transverse relaxation time in [ms].
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
            Defaults to 15 for slice selective and 1 for non-selective acquisitions.
        verbose (optional, bool): If true, prints execution time for signal (and gradient) calculations.

    Kwargs (sequence):
        TE (optional, float, array-like): Echo time(s) in [ms]. Defaults to 0.0.
        slice_selective_exc (optional, bool): If True, rf pulse affects only in-slice spins and suffers from slice_profile effect.
            If false, ignores slice profile and excites moving spins as well.
        slice_selective_prep (optional, bool): If True, inversion pulse affects only in-slice spins.
            If false, invert moving spins as well.
        sliceprof (optional, array-like): excitation slice profile (i.e., flip angle scaling across slice).
            Ignored if pulses are non-selective. Defaults to None.
        slice_orient (optionl, str, array-like): slice orientation ("x", "y", "z" or versor).
            Ignored if pulses are non-selective. Defaults to "z".
        voxelsize (optional, str, array-like): voxel size (dx, dy, dz) in [mm]. If scalar, assume isotropic voxel.
            Defaults to "None".
        inv_tau (float): inversion pulse duration in [ms].
        inv_b1rms (float): inversion pulse root-mean-squared B1 when flip = 1 [deg].
        inv_df (optional, float): pulse frequency offset in [Hz]. Defaults to 0.
        rf_tau (float): pulse duration in [ms].
        rf_b1rms (float): pulse root-mean-squared B1 when flip = 1 [deg].
        rf_df (optional, float): pulse frequency offset in [Hz]. Defaults to 0.
        rf_envelope (array_like): rf time envelope when flip = 1 [deg].
            Together with "duration", it is Used to compute slice profile
            and b1rms if these are not provided.
        grad_tau (float): gradient lobe duration in [ms].
        grad_dephasing (optional, float): Total gradient-induced dephasing across a voxel (in grad direction).
            If gradient_amplitude is not provided, this is used to compute diffusion and flow effects.
        grad_amplitude (optional, float): gradient amplitude along unbalanced direction in [mT / m].
            If total_dephasing is not provided, this is used to compute diffusion and flow effects.
        grad_orient (optional, str, array-like): gradient orientation ("x", "y", "z" or versor). Defaults to "z".

     Kwargs (System):
         B1 (optional, float, array-like): flip angle scaling factor (1.0 := nominal flip angle).
             Defaults to None (nominal flip angle).
         B0 (optional, float, array-like): Bulk off-resonance in [Hz]. Defaults to None.
         inversio_efficiency (optional, float, array-like): Inversion efficiency.
             Defaults to None (perfect inversion).
         B1Tx2 (optional, Union[float, npt.NDArray[float], torch.FloatTensor]): flip angle scaling factor for secondary RF mode (1.0 := nominal flip angle). Defaults to None.
         B1phase (optional, Union[float, npt.NDArray[float], torch.FloatTensor]): B1 relative phase in [deg]. (0.0 := nominal rf phase). Defaults to None.

    Kwargs (Main pool):
        T2star  (optional, float, array-like): effective relaxation time for main pool in [ms]. Defaults to None.
        D  (optional, float, array-like): apparent diffusion coefficient in [um**2 / ms]. Defaults to None.
        v  (optional, float, array-like): spin velocity [cm / s]. Defaults to None.
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
    init_params = {"flip": flip, "TR": TR, "T1": T1, "T2": T2, "diff": diff, "device": device, "TI": TI, **kwargs}

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

    # check if it is 2D
    if "sliceprof" in init_params:
        slice_selective_exc = True
    else:
        slice_selective_exc = None

    # check for conflicts
    if slice_selective_exc is None:
        if "slice_selective_exc" in init_params:
            slice_selective_exc = init_params["slice_selective_exc"]
        else:
            slice_selective_exc = False
    else:
        if "slice_selective_exc" in init_params:
            if slice_selective_exc != init_params["slice_selective_exc"]:
                warnings.warn("Slice profile was provided but acquisition is non-selective - setting to selective")
    
    # add moving pool if required
    if slice_selective_exc and "v" in init_params:
        init_params["moving"] = True

    # check for global inversion
    if "slice_selective_inv" in init_params:
        slice_selective_inv = init_params["slice_selective_inv"]
    else:
        slice_selective_inv = False

    # check for conflicts in inversion selectivity
    if slice_selective_exc is False and slice_selective_inv is True:
        warnings.warn('3D acquisition - forcing inversion pulse to global.')
        slice_selective_inv = False
    
    # inversion pulse properties
    if TI is None:
        inv_props = {}
    else:
        inv_props = {"slice_selective": slice_selective_inv}

    if "inv_tau" in kwargs:
        inv_props["duration"] = kwargs["inv_tau"]
    if "inv_df" in kwargs:
        inv_props["freq_offset"] = kwargs["inv_df"]
    if "inv_b1rms" in kwargs:
        inv_props["b1rms"] = kwargs["inv_b1rms"]

    # check conflicts in inversion settings
    if TI is None:
        if inv_props:
            warnings.warn('Inversion not enabled - ignoring inversion pulse properties.')
            inv_props = {}

    # excitation pulse properties
    rf_props = {"slice_selective": slice_selective_exc}
    if "rf_tau" in kwargs:
        rf_props["duration"] = kwargs["rf_tau"]
    if "rf_df" in kwargs:
        rf_props["freq_offset"] = kwargs["rf_df"]
    if "rf_b1rms" in kwargs:
        rf_props["b1rms"] = kwargs["rf_b1rms"]
    if "rf_envelope" in kwargs:
        rf_props["rf_envelope"] = kwargs["rf_envelope"]
    if "rf_envelope" in kwargs:
        rf_props["rf_envelope"] = kwargs["rf_envelope"]
    if "sliceprof" in kwargs:
        rf_props["slice_profile"] = kwargs["sliceprof"]

    # check for possible inconsistencies:
    if "sliceprof" in rf_props and "rf_envelope" in rf_props:
        warnings.warn("Both slice profile and rf_envelope are provided - using the first")
    if "b1rms" in rf_props and "rf_envelope" in rf_props:
        warnings.warn("Both b1rms and rf_envelope are provided - using the first")
        
    # get nlocs
    if "nlocs" in init_params:
        nlocs = init_params["nlocs"]
    else:
        if slice_selective_exc:
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
    
    # unbalanced gradient properties
    grad_props = {}
    if "grad_tau" in kwargs:
        grad_props["duration"] = kwargs["grad_tau"]
    if "grad_dephasing" in kwargs:
        grad_props["total_dephasing"] = kwargs["grad_dephasing"]
    if "voxelsize" in kwargs:
        grad_props["voxelsize"] = kwargs["voxelsize"]
    if "grad_amplitude" in kwargs:
        grad_props["grad_amplitude"] = kwargs["grad_amplitude"]
    if "grad_orient" in kwargs:
        grad_props["grad_direction"] = kwargs["grad_orient"]
    if "slice_orient" in kwargs:
        grad_props["slice_direction"] = kwargs["slice_orient"]

    # check for possible inconsistencies:
    if "total_dephasing" in rf_props and "grad_amplitude" in rf_props:
        warnings.warn("Both total_dephasing and grad_amplitude are provided - using the first")

    # put all properties together
    props = {"inv_props": inv_props, "rf_props": rf_props, "grad_props": grad_props}

    # initialize simulator
    if slice_selective_exc:
        simulator = dacite.from_dict(SSFP2D, init_params, config=Config(check_types=False))
    else:
        simulator = dacite.from_dict(SSFP3D, init_params, config=Config(check_types=False))
            
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
            print(f"Elapsed time for simulation: {info['trun']} s")
            print(f"Elapsed time for gradient computation: {info['tgrad']} s")

        return sig, dsig, info
    else:
        # actual simulation
        sig = simulator(flip=flip, TR=TR, TI=TI, TE=TE, props=props)
        # post processing
        if asnumpy:
            sig = sig.cpu().numpy()

            # prepare info
            info = {"trun": simulator.trun}
            if verbose:
                print(f"Elapsed time for simulation: {info['trun']} s")

        return sig, info["trun"]


#%% utils
spin_defaults = {"T2star": None, "D": None, "v": None}

class SSFP2D(base.BaseSimulator):
    """Class to simulate slice-selective inversion-prepared (variable flip angle) SSFP."""

    @staticmethod
    def sequence(
        flip,
        phase
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
        D,
        v,
        states,
        signal,
    ):
        # parsing pulses and grad parameters
        inv_props = props["inv_props"]
        rf_props = props["rf_props"]
        grad_props = props["grad_props"]

        # get number of frames and echoes
        npulses = flip.shape[0]
        nechoes = len(TE)

        # define preparation
        Prep = blocks.InversionPrep(TI, T1, T2, weight, k, inv_props)

        # prepare RF pulse
        RF = blocks.ExcPulse(states, B1, rf_props)

        # prepare free precession period
        XTE, XSTETR = blocks.SSFPFidStep(
            states, TE, TR, T1, T2, weight, k, chemshift, D, v, grad_props
        )

        # magnetization prep
        states = Prep(states)

        # actual sequence loop
        for n in range(npulses):
            # apply pulse
            states = RF(states, flip[n])

            # relax, recover and record signal for each TE
            if nechoes == 1:  # single-echo case
                states = SSFPFidTE(states)
                signal[n] = ops.observe(states, RF.phi)
            else:
                for e in range(len(TE)):
                    states = SSFPFidTE[e](states)
                    signal[n, e] = ops.observe(states, RF.phi)

            # relax, recover and spoil
            states = SSFPFidTETR(states)

        return signal


class SSFP3D(base.BaseSimulator):
    """Class to simulate non-selective inversion-prepared (variable flip angle) SSFP."""

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
        D,
        v,
        states,
        signal,
    ):
        # parsing pulses and grad parameters
        inv_props = props["inv_props"]
        rf_props = props["rf_props"]
        grad_props = props["grad_props"]

        # get number of frames and echoes
        npulses = flip.shape[0]
        nechoes = len(TE)

        # define preparation
        Prep = InversionPrep(TI, T1, T2, weight, k, inv_props)

        # prepare RF pulse
        RF = ExcPulse(states, B1, rf_props)

        # prepare free precession period
        SSFPFidTE, SSFPFidTETR = SSFPFidStep(
            states, TE, TR, T1, T2, weight, k, chemshift, D, v, grad_props
        )

        for r in range(2):
            # magnetization prep
            states = Prep(states)

            # actual sequence loop
            for n in range(npulses):
                # apply pulse
                states = RF(states, flip[n])

                # # relax, recover and record signal for each TE
                if nechoes == 1:  # single-echo case
                    states = SSFPFidTE(states)
                    signal[n] = ops.observe(states, RF.phi)
                else:
                    for e in range(len(TE)):
                        states = SSFPFidTE[e](states)
                        signal[n, e] = ops.observe(states, RF.phi)

                # # relax, recover and spoil
                states = SSFPFidTETR(states)

        return ops.susceptibility(signal, TE, df)
