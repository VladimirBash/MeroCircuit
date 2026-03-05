# tests/test_plasticity_simple.py

"""
Minimal plasticity test: 3-node chain.

Setup: 0 -- 1 -- 2
Clamp: V[0]=0, V[2]=1
Goal: Train V[1] to reach target (e.g., 0.68)

This requires plasticity to make R01 ≠ R12.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic
from training.plasticity import SimplePlasticity, weights_to_conductances, compute_Q_from_voltages


def test_three_node_chain():
    """
    Simplest possible test: 3-node chain.
    
    Free phase: V[1] settles based on current weights
    Clamped phase: Penalty pulls V[1] toward target
    Update: Weights should adjust to make V[1] closer to target in free phase
    """
    print("\n=== Test: 3-Node Chain Plasticity ===")
    print("Topology: 0 -- 1 -- 2")
    print("Boundary: V[0]=0, V[2]=1")
    print("Target: V[1]=0.68\n")
    
    # Topology
    adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    # Initialize with equal weights
    weights = np.array([
        [0, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0]
    ], dtype=float)
    
    g_min, g_max = 0.1, 1.0
    
    # Solver and plasticity
    solver = VoltageDynamics(adjacency, ohmic, capacitances=1.0)
    plasticity = SimplePlasticity(eta=0.1, gamma=0.001, tau_integrate=5.0)
    
    # Boundary and target
    clamped_nodes = np.array([0, 2])
    clamped_values = np.array([0.0, 1.0])
    target_V1 = 0.68
    
    # "Penalty pair" for node 1
    # We'll create a virtual "input" node for the target
    # Simpler: just use penalty to pull node 1 toward target
    
    # Training parameters
    n_cycles = 50
    dt = 0.001
    max_steps = 10000
    
    # History
    V1_free_history = []
    V1_clamped_history = []
    
    print("Training...")
    for cycle in range(n_cycles):
        conductances = weights_to_conductances(weights, g_min, g_max)
        
        # === FREE PHASE ===
        V_init = np.array([0.0, 0.5, 1.0])
        result_free = solver.relax_transient(
            conductances, V_init,
            clamped_nodes=clamped_nodes,
            clamped_values=clamped_values,
            dt=dt, max_steps=max_steps, tol=1e-10
        )
        
        V1_free = result_free['V_final'][1]
        V1_free_history.append(V1_free)
        
        Q_free = compute_Q_from_voltages(result_free['V_final'], adjacency)
        
        # === CLAMPED PHASE ===
        # Manually push V[1] toward target by temporarily clamping it
        # (This simulates penalty coupling effect)
        V_init_clamped = result_free['V_final'].copy()
        V_init_clamped[1] = target_V1  # Start closer to target
        
        result_clamped = solver.relax_transient(
            conductances, V_init_clamped,
            clamped_nodes=np.array([0, 1, 2]),  # Clamp middle node too
            clamped_values=np.array([0.0, target_V1, 1.0]),
            dt=dt, max_steps=max_steps, tol=1e-10
        )
        
        V1_clamped = result_clamped['V_final'][1]
        V1_clamped_history.append(V1_clamped)
        
        Q_clamped = compute_Q_from_voltages(result_clamped['V_final'], adjacency)
        
        # === WEIGHT UPDATE ===
        weights_old = weights.copy()
        weights = plasticity.update_weights(weights, Q_free, Q_clamped, adjacency, dt_plasticity=1.0)
        
        if (cycle + 1) % 10 == 0:
            dw = weights - weights_old
            error = abs(V1_free - target_V1)
            print(f"  Cycle {cycle+1}/{n_cycles}:")
            print(f"    V[1] free: {V1_free:.4f} (target: {target_V1}, error: {error:.4f})")
            print(f"    Weights: w[0,1]={weights[0,1]:.4f}, w[1,2]={weights[1,2]:.4f}")
            print(f"    dw: {dw[0,1]:.6f}, {dw[1,2]:.6f}")
    
    # Final result
    final_error = abs(V1_free_history[-1] - target_V1)
    initial_error = abs(V1_free_history[0] - target_V1)
    
    print(f"\nInitial error: {initial_error:.4f}")
    print(f"Final error: {final_error:.4f}")
    print(f"Improvement: {(initial_error - final_error)/initial_error*100:.1f}%")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # V[1] evolution
    cycles = np.arange(1, n_cycles + 1)
    ax1.plot(cycles, V1_free_history, 'b-', linewidth=2, label='V[1] free')
    ax1.axhline(target_V1, color='r', linestyle='--', linewidth=2, label='Target')
    ax1.set_xlabel('Cycle')
    ax1.set_ylabel('V[1]')
    ax1.set_title('Node 1 Voltage Evolution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Weight evolution
    # Extract weight history (need to track during training - simplified here)
    ax2.plot(cycles, np.abs(np.array(V1_free_history) - target_V1), 'g-', linewidth=2)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Absolute Error vs Target')
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('test_plasticity_simple.png', dpi=100)
    
    # Assertions
    assert final_error < initial_error, "Error did not decrease!"
    assert final_error < 0.1, f"Final error too large: {final_error:.4f}"
    
    print("✓ Test passed!")
    print("✓ Plot saved to test_plasticity_simple.png\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Simple 3-Node Chain Plasticity Test")
    print("=" * 70)
    
    test_three_node_chain()
    
    print("=" * 70)
    print("✓ Test completed!")
    print("=" * 70)