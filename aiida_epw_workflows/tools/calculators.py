# -*- coding: utf-8 -*-
"""Calculators for the superconductivity analysis."""

import numpy
from numpy.typing import ArrayLike
import scipy
from aiida.engine import calcfunction
from aiida.orm import ArrayData, XyData, Float
from scipy.interpolate import interp1d

meV_to_Kelvin = 11.604518121550082


def allen_dynes(lambo, omega_log, mu_star):
    """Calculate the Allen-Dynes critical temperature Tc."""
    if lambo - mu_star * (1 + 0.62 * lambo) < 0:
        return 0
    else:
        return omega_log * numpy.exp(-1.04 * (1 + lambo) / (lambo - mu_star * (1 + 0.62 * lambo))) / 1.2


def calculate_lambda_omega(frequency: ArrayLike, spectrum: ArrayLike) -> tuple:
    """Calculate lambda and omega_log from the parsed a2F spectrum.

    :param frequency: Frequency array on which the a2F spectrum is defined [meV].
    :param spectrum: a2F spectral function values.

    :returns: Tuple of the calculated lambda and omega_log values.
    """
    lambda_ = 2 * scipy.integrate.trapezoid(spectrum / frequency, frequency)  # unitless
    omega_log = numpy.exp(2 / lambda_ * scipy.integrate.trapezoid(spectrum / frequency * numpy.log(frequency), frequency))  # eV
    omega_log = omega_log * meV_to_Kelvin

    return lambda_, omega_log

@calcfunction
def calculate_Allen_Dynes_tc(a2f: ArrayData, mustar = 0.13) -> Float:
    w        = a2f.get_array('frequency')
    # Here we preassume that there are 10 smearing values for a2f calculation
    spectral = a2f.get_array('a2f')[:, 9]
    mev2K    = 11.604525006157

    _lambda  = 2*numpy.trapz(numpy.divide(spectral, w), x=w)

    # wlog =  np.exp(np.average(np.divide(alpha, w), weights=np.log(w)))
    wlog     =  numpy.exp(2/_lambda*numpy.trapz(numpy.multiply(numpy.divide(spectral, w), numpy.log(w)), x=w))

    Tc = wlog/1.2*numpy.exp(-1.04*(1+_lambda)/(_lambda-mustar*(1+0.62*_lambda))) * mev2K


    return Float(Tc)

@calcfunction
def calculate_iso_tc(max_eigenvalue: XyData) -> Float:
    me_array = max_eigenvalue.get_array('max_eigenvalue')
    if me_array[:, 1].max() < 1.0:
        return Float(0.0)
    else:
        return Float(float(interp1d(me_array[:, 1], me_array[:, 0])(1.0)))

