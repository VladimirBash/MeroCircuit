# tests/test_iv_characteristics.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from network.iv_characteristics import ohmic, relu_iv, sigmoid_iv, diode_iv


def test_iv_curves():
    """Визуально проверить все I-V кривые."""
    V = np.linspace(-1, 1, 200)
    g = 1.0
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Ohmic
    I_ohmic = ohmic(V, g)
    axes[0, 0].plot(V, I_ohmic, 'b-', linewidth=2)
    axes[0, 0].set_title('Ohmic')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color='k', linewidth=0.5)
    axes[0, 0].axvline(0, color='k', linewidth=0.5)
    
    # ReLU
    I_relu = relu_iv(V, g, V_th=0.2)
    axes[0, 1].plot(V, I_relu, 'r-', linewidth=2)
    axes[0, 1].set_title('ReLU (V_th=0.2)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color='k', linewidth=0.5)
    axes[0, 1].axvline(0, color='k', linewidth=0.5)
    
    # Sigmoid
    I_sigmoid = sigmoid_iv(V, g, steepness=10)
    axes[1, 0].plot(V, I_sigmoid, 'g-', linewidth=2)
    axes[1, 0].set_title('Sigmoid (steepness=10)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color='k', linewidth=0.5)
    axes[1, 0].axvline(0, color='k', linewidth=0.5)
    
    # Diode
    I_diode = diode_iv(V, g_forward=1.0, g_reverse=0.01)
    axes[1, 1].plot(V, I_diode, 'm-', linewidth=2)
    axes[1, 1].set_title('Diode (g_fwd=1.0, g_rev=0.01)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='k', linewidth=0.5)
    axes[1, 1].axvline(0, color='k', linewidth=0.5)
    
    for ax in axes.flat:
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (I)')
    
    plt.tight_layout()
    plt.savefig('test_iv_curves.png', dpi=100)
    print("✓ I-V curves saved to test_iv_curves.png")
    plt.show()


if __name__ == "__main__":
    test_iv_curves()
