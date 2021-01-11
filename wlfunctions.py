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
        An array with the values of f(x), with f.

    '''
    return digamma(.5 + (1/x)) + np.ln(x)
