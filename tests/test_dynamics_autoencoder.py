# tests/test_dynamics_autoencoder.py

"""
Test voltage dynamics on autoencoder-like topologies.

Tests the core use case: input → hidden → output network
with penalty coupling for reconstruction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic


def test_simple_autoencoder():
    """
    Test: input → hidden → output autoencoder topology.
    
    Without penalty: outputs drift to symmetric state.
    With penalty: outputs pulled toward inputs.
    """
    # === PARAMETERS ===
    n_input = 4
    n_hidden = 4
    n_output = 4
    beta_free = 0.0
    beta_clamped = 100.0
    g_penalty = 10.0
    dt = 0.001
    tol = 1e-10
    max_steps = 20000
    
    print("\n=== Test: Simple Autoencoder Topology ===")
    print(f"Architecture: {n_input} input → {n_hidden} hidden → {n_output} output")
    print(f"Parameters: β_clamped={beta_clamped}, g_penalty={g_penalty}")
    print(f"Solver: dt={dt}, tol={tol}\n")
    
    # === BUILD TOPOLOGY ===
    n_total = n_input + n_hidden + n_output
    
    # Node indices
    input_nodes = np.arange(n_input)
    hidden_nodes = np.arange(n_input, n_input + n_hidden)
    output_nodes = np.arange(n_input + n_hidden, n_total)
    
    # Build adjacency: input ↔ hidden, hidden ↔ output
    adjacency = np.zeros((n_total, n_total), dtype=bool)
    
    # Input ↔ Hidden (fully connected)
    for i in input_nodes:
        for h in hidden_nodes:
            adjacency[i, h] = True
            adjacency[h, i] = True
    
    # Hidden ↔ Output (fully connected)
    for h in hidden_nodes:
        for o in output_nodes:
            adjacency[h, o] = True
            adjacency[o, h] = True
    
    # Random conductances (symmetric)
    rng = np.random.default_rng(42)
    conductances = adjacency.astype(float) * rng.uniform(0.5, 1.5, size=(n_total, n_total))
    conductances = (conductances + conductances.T) / 2  # Symmetrize
    
    # Penalty pairs: input[i] ↔ output[i]
    penalty_pairs = [(i, n_input + n_hidden + i) for i in range(n_input)]
    
    # Input pattern
    V_input = np.array([1.0, 0.0, 1.0, 0.0])
    
    # Initialize solver
    solver = VoltageDynamics(adjacency, ohmic, capacitances=1.0)
    
    # Initial voltages
    V_init = np.zeros(n_total)
    V_init[input_nodes] = V_input
    V_init[hidden_nodes] = rng.uniform(0, 1, size=n_hidden)
    V_init[output_nodes] = rng.uniform(0, 1, size=n_output)
    
    # === FREE PHASE (β=0) ===
    print(f"Free phase (β={beta_free})...")
    result_free = solver.relax_transient(
        conductances, V_init,
        clamped_nodes=input_nodes,
        clamped_values=V_input,
        penalty_pairs=penalty_pairs,
        beta=beta_free,
        dt=dt,
        tol=tol,
        max_steps=max_steps,
        record_history=True
    )
    
    V_output_free = result_free['V_final'][output_nodes]
    mse_free = np.mean((V_input - V_output_free)**2)
    
    print(f"  Input:  {V_input}")
    print(f"  Output: {V_output_free}")
    print(f"  MSE: {mse_free:.6f}")
    print(f"  Converged: {result_free['converged']}, Steps: {result_free['n_steps']}\n")
    
    # === CLAMPED PHASE (β>0) ===
    print(f"Clamped phase (β={beta_clamped})...")
    result_clamped = solver.relax_transient(
        conductances, V_init,
        clamped_nodes=input_nodes,
        clamped_values=V_input,
        penalty_pairs=penalty_pairs,
        beta=beta_clamped,
        g_penalty=g_penalty,
        dt=dt,
        tol=tol,
        max_steps=max_steps,
        record_history=True
    )
    
    V_output_clamped = result_clamped['V_final'][output_nodes]
    mse_clamped = np.mean((V_input - V_output_clamped)**2)
    
    print(f"  Input:  {V_input}")
    print(f"  Output: {V_output_clamped}")
    print(f"  MSE: {mse_clamped:.6f}")
    print(f"  Converged: {result_clamped['converged']}, Steps: {result_clamped['n_steps']}\n")
    
    # === VALIDATION ===
    print(f"Reconstruction improvement: {mse_free:.6f} → {mse_clamped:.6f}")
    print(f"Improvement factor: {mse_free/mse_clamped:.1f}x\n")
    
    assert result_free['converged'], "Free phase did not converge"
    assert result_clamped['converged'], "Clamped phase did not converge"
    assert mse_clamped < mse_free, "Penalty should improve reconstruction"
    assert mse_clamped < 0.01, f"Clamped MSE too high: {mse_clamped:.6f}"
    
    # === VISUALIZATION ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Free phase
    V_hist_free = result_free['V_history']
    time_free = np.arange(len(V_hist_free)) * dt
    
    ax = axes[0]
    for i in range(n_input):
        ax.plot(time_free, V_hist_free[:, input_nodes[i]], 'b-', 
                alpha=0.7, linewidth=1.5, label=f'Input {i}' if i == 0 else '')
    for i in range(n_output):
        ax.plot(time_free, V_hist_free[:, output_nodes[i]], 'r-', 
                alpha=0.7, linewidth=2, label=f'Output {i}' if i == 0 else '')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Voltage', fontsize=11)
    ax.set_title(f'Free Phase (β={beta_free}): Outputs drift', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(['Inputs', 'Outputs'], loc='best')
    
    # Clamped phase
    V_hist_clamped = result_clamped['V_history']
    time_clamped = np.arange(len(V_hist_clamped)) * dt
    
    ax = axes[1]
    for i in range(n_input):
        ax.plot(time_clamped, V_hist_clamped[:, input_nodes[i]], 'b-', 
                alpha=0.7, linewidth=1.5)
    for i in range(n_output):
        ax.plot(time_clamped, V_hist_clamped[:, output_nodes[i]], 'r-', 
                alpha=0.7, linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Voltage', fontsize=11)
    ax.set_title(f'Clamped Phase (β={beta_clamped}): Outputs track inputs', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_autoencoder_topology.png', dpi=100)
    
    print("✓ Test passed")
    print("✓ Plot saved to test_autoencoder_topology.png\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Autoencoder Topology")
    print("=" * 70)
    
    test_simple_autoencoder()
    
    print("=" * 70)
    print("✓ All autoencoder tests passed!")
    print("=" * 70)
