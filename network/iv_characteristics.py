# network/iv_characteristics.py

import numpy as np


def ohmic(V_drop, g):
    """
    Linear Ohmic I-V characteristic: I = g * V
    
    Args:
        V_drop: Voltage difference (scalar or array)
        g: Conductance (scalar or array)
        
    Returns:
        Current through the element
    """
    return g * V_drop


def relu_iv(V_drop, g, V_th=0.1):
    """
    Rectified I-V with threshold: I = g * ReLU(|V| - V_th) * sign(V)
    
    Models memristive behavior with activation threshold.
    
    Args:
        V_drop: Voltage difference
        g: Conductance
        V_th: Threshold voltage (default 0.1)
        
    Returns:
        Current through the element
    """
    magnitude = np.maximum(np.abs(V_drop) - V_th, 0.0)
    return g * magnitude * np.sign(V_drop)


def sigmoid_iv(V_drop, g, steepness=10.0):
    """
    Smooth sigmoid-like I-V: I = g * tanh(steepness * V)
    
    Provides smooth nonlinearity without hard threshold.
    
    Args:
        V_drop: Voltage difference
        g: Conductance
        steepness: Controls transition sharpness (default 10.0)
        
    Returns:
        Current through the element
    """
    return g * np.tanh(steepness * V_drop)


def diode_iv(V_drop, g_forward, g_reverse=1e-6):
    """
    Asymmetric diode-like I-V characteristic.
    
    Args:
        V_drop: Voltage difference
        g_forward: Forward conductance (V > 0)
        g_reverse: Reverse conductance (V < 0), typically very small
        
    Returns:
        Current through the element
    """
    g_effective = np.where(V_drop > 0, g_forward, g_reverse)
    return g_effective * V_drop
