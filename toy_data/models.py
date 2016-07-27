"""
Define various mathematical models
From toy_data:
    Module for toy-data generation for ML experiments
"""

import numpy as np

def Sine(
        A=3,
        y_offset=0,
        phase=0,
        frequency=1):
    return lambda x: y_offset + A * np.sin(frequency * x + phase)
