# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 11:36:40 2021

@author: Nicolas Kawahala
"""

import numpy as np
from scipy.special import digamma

def f(x):
    '''
    Compute values for the f function defined in the article \"Weak Localization
    corrections to the conductivity of double quantum wells\" [O E Raichev and 
    Vasilopoulos, J. Phys.: Condens. Matter 12 (2000) 589-600].
    
                    f(x) = digamma((1/2) + (1/x)) + ln(x)
    
    Parameters
    ----------
    x : ndarray
        An array containing the x points used to compute the f function.

    Returns
    -------
    ndarray
        An array with the values of f(x).

    '''
    return digamma(.5 + (1/x)) + np.log(x)

def delta_sigma(B, Hf, Hso, t):
    '''
    Compute values for Deltasigma(B) in units of e^2/(2*pi^2*hbar).
    
    Requires the function f(x).

    Parameters
    ----------
    B : ndarray
        Magnetic field B.
    Hf : float
        Parameter for the characteristic field H_phi, in the same units as B.
    Hso : float
        Parameter for the spin-orbit field H_so, in the same units as B.
    t : float
        Parameter for tau_phi/tau_t, giving a ratio between the average
        inelastic scattering time (tau_phi) and the tunneling time (tau_t).

    Returns
    -------
    ndarray
        An array with delta_sigma as a function of B, for the given Hf (H_phi),
        Hso (H_so) and t (tau_phi/tau_t) parameters.
    '''
    return f(B/(Hf+Hso)) + .5*f(B/(Hf+2*Hso)) - .5*(f(B/Hf) + f(B/(Hf*(1+2*t))))

def delta_sigma_old(B, Hf, Hso):
    '''
    Compute values for the Deltasigma(B) in units of e^2/(2*pi^2*hbar),
    without the corrections due to the double quantum well.
    
    Requires the function f(x).

    Parameters
    ----------
    B : ndarray
        Magnetic field B.
    Hf : float
        Parameter for the characteristic field H_phi, in the same units as B.
    Hso : float
        Parameter for the spin-orbit field H_so, in the same units as B.

    Returns
    -------
    ndarray
        An array with delta_sigma as a function of B, for the given Hf (H_phi),
        and Hso (H_so) parameters.

    '''
    return f(B/(Hf+Hso)) + .5*f(B/(Hf+2*Hso)) - .5*f(B/Hf)