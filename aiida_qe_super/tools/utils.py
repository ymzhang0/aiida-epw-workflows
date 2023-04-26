# -*- coding: utf-8 -*-
"""Utility methods for the superconductivity analysis."""

import numpy
from numpy.typing import ArrayLike
import scipy

eV_to_Kelvin = 11604.518121550082
Ry2eV =  13.605662285137

def allen_dynes(lambo, omega_log, mu_star):
    """Calculate the Allen-Dynes critical temperature Tc."""
    if lambo - mu_star * (1 + 0.62 * lambo) < 0:
        return 0
    else:
        return omega_log * numpy.exp(-1.04 * (1 + lambo) / (lambo - mu_star * (1 + 0.62 * lambo))) / 1.2

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


def calculate_lambda_omega(frequency: ArrayLike, spectrum: ArrayLike) -> tuple:
    """Calculate lambda and omega_log from the parsed a2F spectrum.

    :param frequency: Frequency array on which the a2F spectrum is defined.
    :param spectrum: a2F spectral function values.

    :returns: Tuple of the calculated lambda and omega_log values.
    """
    lambda_ = 2 * scipy.integrate.trapezoid(spectrum / frequency, frequency)  # unitless
    omega_log = numpy.exp(2 / lambda_ * scipy.integrate.trapezoid(spectrum / frequency * numpy.log(frequency), frequency))  # eV
    omega_log = omega_log * eV_to_Kelvin

    return lambda_, omega_log
