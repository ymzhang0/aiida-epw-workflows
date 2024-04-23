# encoding: utf-8
"""Temporary home for various plotting tools."""
from aiida import orm
from typing import Tuple

def create_xticks(seekpath_params):
    """Create xticks and xtick_labels for a band structure plot from the Seek-path parameters."""

    def transform_gamma(label):
        if label == 'GAMMA':
            return r'$\Gamma$'
        return label

    path = seekpath_params['path']
    xtick_labels = [transform_gamma(path[0][0]), ] 

    for segment_number, segment_labels in enumerate(path):
        try:
            if segment_labels[1] == path[segment_number + 1][0]:
                xtick_labels.append(transform_gamma(segment_labels[1]))
            else:
                xtick_labels.append(f'{transform_gamma(segment_labels[1])}|{path[segment_number + 1][0]}')
        except IndexError:
            xtick_labels.append(transform_gamma(segment_labels[1]))

    explicit_segments = seekpath_params['explicit_segments']
    xticks = [explicit_segments[0][0]]

    for explicit_segment in explicit_segments:
        xticks.append(explicit_segment[1])

    return xticks, xtick_labels


def create_xticks_bands(bands: orm.BandsData) -> Tuple[list, list]:
    """Create xticks and xtick_labels for a band structure plot.
    
    Takes a BandsData object and returns a tuple of xticks and xtick_labels. The script takes care
    of two things:

    1. If the last label of a segment is not the same as the first label of the next segment, a
       vertical line is added between the two symmetry point labels.
    2. In case the label is "GAMMA", it is replaced with the greek capital letter gamma, as is the
       convention.

    """
    def transform_gamma(label):
        if label == 'GAMMA':
            return r'$\Gamma$'
        return label

    labels = bands.attributes['labels']
    label_numbers = bands.attributes['label_numbers']

    xticks = [label_numbers[0], ]
    xtick_labels = [transform_gamma(labels[0]), ]

    for label, label_number in zip(labels[1:], label_numbers[1:]):

        if label_number - xticks[-1] == 1:
            xtick_labels.append(f'{xtick_labels.pop()}|{transform_gamma(label)}')
        else:
            xtick_labels.append(transform_gamma(label))
            xticks.append(label_number)

    return xticks, xtick_labels


def plot_bands(bands: orm.BandsData, axis=None, reference_energy=0, seekpath_params=None, **kwargs):
    """Plot a band structure."""

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
        fig.patch.set_facecolor('white')
    else:
        ax = axis

    if seekpath_params is not None:
        xticks, xtick_labels = create_xticks(seekpath_params)
    else:
        try:
            xticks, xtick_labels = create_xticks_bands(bands)
        except IndexError:
            xticks = []
            xtick_labels = []

    if len(xticks) > 0:
        ax.set_xticks(xticks, xtick_labels)

    for tick in xticks:
        ax.axvline(tick, color='k')

    for band in bands.get_bands().transpose():
        ax.plot(band - reference_energy, **kwargs)
        
    ax.axhline(0, color='k', linestyle='--')

    if axis is None:
        return plt


def plot_bands_comparison(bands_qe, bands_w90, fermi_qe, fermi_w90, axis=None):

    xticks, xtick_labels = create_xticks_bands(bands_w90)

    if axis is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
        fig.patch.set_facecolor('white')
    else:
        ax = axis

    ax.plot([], '-r')
    ax.plot([], 'b.')
    ax.legend(['Quantum ESPRESSO', 'W90'])

    try:
        bands_qe = bands_qe.get_bands()
        bands_qe -= fermi_qe

        for band_qe in bands_qe.transpose():
            ax.plot(band_qe, '-r')
    except KeyError:
        pass

    try:
        bands_w90 = bands_w90.get_bands()
        bands_w90 -= fermi_w90

        for band_w90 in bands_w90.transpose():
            ax.plot(band_w90, 'b.', markersize=2)
    except KeyError:
        pass

    for tick in xticks:
        ax.axvline(tick)

    ax.set_xticks(xticks, xtick_labels)
    ax.axhline(0, color='k', linestyle='--')

    ax.set_ylabel('Energy (eV)')
    ax.set_yticks([-10, -5, 0, 5, 10, 15], [-10, -5, '$E_F$', 5, 10, 15])
    ax.set_ylim([-10, 10])

    if axis is None:
        return plt


def check_wannier_optimize(w90_optimize_workchain, filename=None):

    bands_qe = w90_optimize_workchain.inputs.optimize_reference_bands
    bands_w90 = w90_optimize_workchain.outputs.band_structure
    fermi_qe = bands_qe.creator.outputs.output_parameters.get_dict()['fermi_energy']
    fermi_w90 = w90_optimize_workchain.outputs.nscf.output_parameters.get_dict()['fermi_energy']

    plt = plot_bands_comparison(bands_qe, bands_w90, fermi_qe, fermi_w90)

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()


def check_wannier_bands(w90_bands_workchain, bands_workchain_qe, filename=None):

    bands_w90 = w90_bands_workchain.outputs.band_structure
    bands_qe = bands_workchain_qe.outputs.band_structure
    fermi_qe = bands_workchain_qe.outputs.band_parameters.get_dict()['fermi_energy']
    fermi_w90 = w90_bands_workchain.outputs.nscf.output_parameters.get_dict()['fermi_energy']

    plt = plot_bands_comparison(bands_qe, bands_w90, fermi_qe, fermi_w90)

    if filename is not None:
        plt.savefig(filename, dpi=300)
        plt.close()
