# -*- coding: utf-8 -*-
"""Parsers for the various superconductivity outputs."""
import numpy

from .calculators import allen_dynes

Ry2eV =  13.605662285137

def parse_epw_a2f(file_content):

    parsed_data = {}

    a2f, footer = file_content.split('\n Integrated el-ph coupling')

    a2f_array = numpy.array([l.split() for l in a2f.split('\n')], dtype=float)
    parsed_data['frequency'] = a2f_array[:, 0]
    parsed_data['a2f'] = a2f_array[:, 1:]

    footer = footer.split('\n')
    parsed_data['lambda'] = numpy.array(footer[1].strip('# ').split(), dtype=float)
    parsed_data['phonon_smearing'] = numpy.array(footer[3].strip('# ').split(), dtype=float)

    key_property_dict = {
        'Electron smearing (eV)': 'electron_smearing',
        'Fermi window (eV)': 'fermi_window',
        'Summed el-ph coupling': 'summed_elph_coupling'
    }
    for line in footer:
        for key, property in key_property_dict.items():
            if key in line:
                parsed_data[property] = float(line.split()[-1])

    return parsed_data


def parse_a2F(file_content):
    """Parse a Quantum ESPRESSO ``.a2F`` file containing the spectral function."""

    a2F = []

    for lines in file_content.split('\n'):

        split_line = lines.split()

        if (len(split_line) < 1):
            continue
        if (split_line[0] == '#'):
            continue
        if (split_line[0] == 'lambda'):
            lambo = float(split_line[2])
        else:
            a2F.append([float(split_line[0]) *  Ry2eV, float(split_line[1])])

    a2F = numpy.array(a2F)

    return lambo, a2F

def parse_lambda(file_content):
    """Parse a Quantum ESPRESSO ``lambda`` file."""

    elph_data = []

    for line in file_content.split('\n'):
        if 'Broadening' in line:
            split = line.split()

            elph_data.append(
                {
                    'broadening': float(split[1]),
                    'lambda': float(split[3]),
                    'omega_log': float(split[8]),
                    'Tc': (
                        allen_dynes(float(split[3]), float(split[8]), 0.1),
                        allen_dynes(float(split[3]), float(split[8]), 0.16),
                    )
                }
            )

    return elph_data
