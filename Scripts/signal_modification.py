# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 23:22:09 2020

@author: lasse
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Synthetic Data Generation
# =============================================================================
def sinew(J = 18, freq = 1000, phase = 0):
    """
    Signal consisting of a single sine wave with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(1 , N+1)
    A = 2 * np.pi * t / N
    x = np.sin(A * freq + phase)
    return x


def fsinew(J = 18, freq1 = 13, freq2 = 20, freq3 = 40, freq4 = 1000, 
           phase1 = 0, phase2 = 0, phase3 = 0, phase4 = 0, phase5 = 0):
    """
    Signal consisting of four sine waves with specified 
    frequencies, phases, and amount of points.
    """
    N = 2**J
    t = np.arange(1, N+1)
    A = 2 * np.pi * t / N
    x1 = np.sin(A * freq1 + phase1)
    x2 = np.sin(A * freq2 + phase2)
    x3 = np.sin(A * freq3 + phase3)
    x4 = np.sin(A * freq4 + phase4)
    x_sum = x1 + x2 + x3 + x4
    return x_sum


# =============================================================================
# Signal Modification Functions
# =============================================================================


# =============================================================================
# Application of Signal Modifications
# =============================================================================


# =============================================================================
# Execution and Plotting
# =============================================================================