
from math import e
from aiida import orm
import re
import tempfile
from aiida.common.exceptions import NotExistentAttributeError
from ..data.constants import THZ_TO_MEV

def get_qpoints_and_frequencies(
    wc: orm.WorkChainNode,
    success_code = [0],
    ):

    if wc.process_label == 'EpwWorkChain':
        ph_base = get_phonon_wc_from_epw_wc(wc)
    elif wc.process_label == 'PhBaseWorkChain':
        ph_base = wc
    else:
        raise ValueError('Invalid input workchain')

    if not (ph_base.exit_status in success_code):
        raise ValueError('The PhBaseWorkChain failed')

    nqpoints = ph_base.outputs.output_parameters.get('number_of_qpoints')
    frequencies = [
        ph_base.outputs.output_parameters.get(f'dynamical_matrix_{iq}').get('frequencies') # unit: cm^{-1} (THz)
        for iq in range(1, 1+nqpoints)
    ]
    q_points = [
        ph_base.outputs.output_parameters.get(f'dynamical_matrix_{iq}').get('q_point')
        for iq in range(1, 1+nqpoints)
    ]
    return q_points, frequencies


def get_phonon_wc_from_epw_wc(
    epw_wc: orm.WorkChainNode
    ) -> orm.WorkChainNode:

    if epw_wc.process_label == 'EpwWorkChain':
        try:
            ph_base_wc = epw_wc.base.links.get_outgoing(link_label_filter='ph_base').first().node
            return ph_base_wc
        except Exception as e:
            try:
                epw_calcjob = epw_wc.base.links.get_outgoing(link_label_filter='epw').first().node
                ph_base_wc = epw_calcjob.inputs.parent_folder_ph.creator.caller
                return ph_base_wc
            except Exception as e:
                raise ValueError(f"Failed to get phonon workchain from EpwWorkChain {epw_wc.pk}: {e}")
    else:
        raise ValueError('Invalid input workchain')


def check_stability_ph_base(
    ph_base: orm.WorkChainNode,
    tolerance: float = -5.0 # cm^{-1}
    ) -> tuple[bool, str]:
    """Check if the phonon on the coarse grid is stable."""
    is_stable = True
    message = ''
    negative_freqs = {}

    qpoints, frequencies = get_qpoints_and_frequencies(ph_base)

    neg_freq0 = [f for f in frequencies[0] if f < 0]
    if len(neg_freq0) > 3:
        is_stable = False
        negative_freqs[1] = neg_freq0

    for iq, freq in enumerate(frequencies[1:]):
        neg_freq = [f for f in freq if f < tolerance]
        if len(neg_freq) > 0:
            is_stable = False
            negative_freqs[iq+2] = neg_freq

    if is_stable:
        message = 'Phonon is stable from `ph_base`.'
    else:
        message = f'Phonon is unstable from `ph_base`.\n'
        for iq, freqs in negative_freqs.items():
            q_points_str = ', '.join(map(str, qpoints[iq-1]))
            negative_freqs_str = ', '.join(map(str, freqs))
            message += f'{iq}th qpoint [{q_points_str}] has negative frequencies: {negative_freqs_str} cm^{-1}\n'

    return (is_stable, message)

def check_stability_matdyn_base(
    workchain: orm.WorkChainNode,
    tolerance: float = -5.0 # cm^{-1}
    ) -> tuple[bool, str]:
    from ..data.constants import THZ_TO_CM
    import numpy
    """Check if the matdyn.x interpolated phonon band structure is stable."""

    bands = workchain.outputs.output_phonon_bands.get_bands() * THZ_TO_CM # unit: THz
    min_freq = numpy.min(bands)
    if min_freq < tolerance:
        return (
            False,
            f'The phonon from `matdyn_base` is unstable.\nWith the minimum frequency {min_freq:.2f} cm^{-1}.')
    else:
        return (True, f'The phonon from `matdyn_base` is stable, with the minimum frequency {min_freq:.2f} cm^{-1}.')

def check_stability_epw_bands(
    workchain: orm.WorkChainNode,
    tolerance: float = -5.0 # meV ~ 8.1 cm-1
    ) -> tuple[bool, str, float]:
    """Check if the epw.x interpolated phonon band structure is stable."""
    import numpy
    ph_bands = workchain.outputs.bands.ph_band_structure.get_bands()
    min_freq = numpy.min(ph_bands)
    max_freq = numpy.max(ph_bands)

    if min_freq < tolerance:
        return (False, f'The phonon from `epw_bands` is unstable, with the minimum frequency {min_freq:.2f} cm^{-1}.', max_freq)
    else:
        return (True, f'The phonon from `epw_bands` is stable, with the minimum frequency {min_freq:.2f} cm^{-1}.', max_freq)
