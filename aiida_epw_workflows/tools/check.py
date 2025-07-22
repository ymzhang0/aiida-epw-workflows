from aiida import orm

def check_convergence_allen_dynes_tc(
    a2f_conv_workchains: list[orm.WorkChainNode],
    convergence_threshold: float
    ) -> tuple[bool, str]:
    """Check if the convergence is reached."""

    try:
        prev_allen_dynes = a2f_conv_workchains[-2].outputs.output_parameters['Allen_Dynes_Tc']
        new_allen_dynes = a2f_conv_workchains[-1].outputs.output_parameters['Allen_Dynes_Tc']
        is_converged = (
            abs(prev_allen_dynes - new_allen_dynes) / new_allen_dynes
            < convergence_threshold
        )
        return (
            is_converged,
            f'Checking convergence: old {prev_allen_dynes}; new {new_allen_dynes} -> Converged = {is_converged}')
    except (AttributeError, IndexError, KeyError):
        return (False, 'Not enough data to check convergence.')

def check_convergence_iso_tc(
    iso_conv_workchains: list[orm.WorkChainNode],
    convergence_threshold: float
    ) -> tuple[bool, str]:
    """Check if the convergence is reached."""

    try:
        prev_iso_tc = iso_conv_workchains[-2].outputs.Tc_iso
        new_iso_tc = iso_conv_workchains[-1].outputs.Tc_iso
        is_converged = (
            abs(prev_iso_tc - new_iso_tc) / new_iso_tc
            < convergence_threshold
        )
        return (
            is_converged,
            f'Checking convergence: old {prev_iso_tc}; new {new_iso_tc} -> Converged = {is_converged}')
    except (AttributeError, IndexError, KeyError):
        return (False, 'Not enough data to check convergence.')

def check_stability_ph_base(
    workchain: orm.WorkChainNode,
    min_freq: float = -5.0 # cm^{-1}
    ) -> tuple[bool, str]:
    """Check if the phonon band structure is stable."""

    is_stable = True
    number_of_qpoints = workchain.outputs.output_parameters.get('number_of_qpoints')
    message = ''
    for iq in range(2, 1+number_of_qpoints):
        frequencies = workchain.outputs.output_parameters.get(f'dynamical_matrix_{iq}').get('frequencies')
        q_point = workchain.outputs.output_parameters.get(f'dynamical_matrix_{iq}').get('q_point')
        if any(f < min_freq for f in frequencies):
            is_stable = False
            message += f'Phonon at {iq}th qpoint {q_point} is unstable\n'

    return (is_stable, message)

def check_stability_epw_bands(
    bands_workchain: orm.WorkChainNode,
    min_freq: float # meV ~ 8.1 cm-1
    ) -> tuple[bool, str]:
    """Check if the phonon band structure is stable."""

    import numpy
    ph_bands = bands_workchain.outputs.bands.ph_band_structure.get_bands()
    min_freq = numpy.min(ph_bands)
    max_freq = numpy.max(ph_bands)

    if min_freq < min_freq:
        return (False, max_freq)
    else:
        return (True, max_freq)
