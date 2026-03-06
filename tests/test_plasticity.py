# tests/test_plasticity.py

"""
Test plasticity rules and single-pattern overfitting.

Validates that the learning mechanism works by training network
to reproduce a single pattern without using Trainer class.
"""

import sys
from pathlib import Path
from training.plasticity import compute_Q_from_voltages
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass

from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic, relu_iv
from training.plasticity import SimplePlasticity, weights_to_conductances, conductances_to_weights


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PlasticityTestConfig:
    """Configuration for plasticity tests."""
    # Network architecture
    n_input: int = 4
    n_hidden: int = 3
    n_output: int = 4
    
    # Plasticity parameters
    eta: float = 1.05
    gamma: float = 0.001
    tau_integrate: float = 20.0
    
    # Training parameters
    n_cycles: int = 300
    exposure_time_free: float = 10.0
    exposure_time_clamped: float = 10.0
    
    # Physics parameters
    beta_free: float = 0.0
    beta_clamped: float = 100.0
    g_penalty: float = 10.0
    
    # Solver parameters
    dt: float = 0.001
    tol: float = 1e-10
    dt_plasticity: float = 1.0
    
    # Weight initialization
    w_min: float = 0.1
    w_max: float = 0.9
    g_min: float = 0.01
    g_max: float = 1.0
    
    # Reproducibility
    random_seed: int = 42

CONFIG = PlasticityTestConfig()


# ============================================================
# TESTS
# ============================================================

def test_weight_conversion():
    """Test conversion between weights and conductances."""
    print("\n=== Test: Weight ↔ Conductance Conversion ===")
    
    w = np.array([[0.0, 0.5, 1.0]])
    
    # Forward
    g = weights_to_conductances(w, CONFIG.g_min, CONFIG.g_max)
    print(f"Weights: {w}")
    print(f"Conductances: {g}")
    
    # Backward
    w_recovered = conductances_to_weights(g, CONFIG.g_min, CONFIG.g_max)
    print(f"Recovered weights: {w_recovered}")
    
    assert np.allclose(w, w_recovered), "Weight conversion roundtrip failed"
    
    print("✓ Passed\n")

def test_observable_integration():
    """Test EMA integration of observables."""
    print("=== Test: Observable Integration (EMA) ===")
    
    plasticity = SimplePlasticity(
        eta=CONFIG.eta, 
        gamma=CONFIG.gamma, 
        tau_integrate=CONFIG.tau_integrate
    )
    
    # Simple case: constant voltage history
    n_steps = 1000
    dt = 0.001
    T_total = n_steps * dt
    V_history = np.ones((n_steps, 3)) * np.array([1.0, 0.5, 0.0])
    
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    
    Q_avg = plasticity.integrate_observable_ema(V_history, adjacency, dt)
    
    print(f"Voltage pattern: [1.0, 0.5, 0.0]")
    print(f"Integration time: {T_total}s, tau_integrate: {plasticity.tau_integrate}s")
    print(f"Q_avg[0,1]: {Q_avg[0, 1]:.6f}")
    print(f"Asymptotic value (linear Q): ΔV = -0.5")  # V[1] - V[0] = 0.5 - 1.0 = -0.5
    
    # After time T ≈ tau, EMA should reach ~63% of asymptotic value
    expected_fraction = 1 - np.exp(-T_total / plasticity.tau_integrate)
    expected_Q = -0.5 * expected_fraction  # Linear observable: Q = ΔV
    
    print(f"Expected after {T_total}s: {expected_Q:.6f} ({expected_fraction*100:.1f}% of asymptotic)")
    
    # Check that EMA behaves correctly (within 10% of expected)
    assert np.abs(Q_avg[0, 1] - expected_Q) / np.abs(expected_Q) < 0.1, \
        f"EMA convergence incorrect: got {Q_avg[0,1]:.6f}, expected {expected_Q:.6f}"
    
    print("✓ EMA convergence correct\n")

#def test_observable_integration():
#    """Test EMA integration of observables."""
#    print("=== Test: Observable Integration (EMA) ===")
#    
#    plasticity = SimplePlasticity(
#        eta=CONFIG.eta, 
#        gamma=CONFIG.gamma, 
#        tau_integrate=CONFIG.tau_integrate
#    )
    
#    # Simple case: constant voltage history
#    n_steps = 1000
#    dt = 0.001
#    T_total = n_steps * dt
#    V_history = np.ones((n_steps, 3)) * np.array([1.0, 0.5, 0.0])
#    
#    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
#    
#    Q_avg = plasticity.integrate_observable_ema(V_history, adjacency, dt)
#    
#    print(f"Voltage pattern: [1.0, 0.5, 0.0]")
#    print(f"Integration time: {T_total}s, tau_integrate: {plasticity.tau_integrate}s")
#    print(f"Q_avg[0,1]: {Q_avg[0, 1]:.6f}")
#    print(f"Asymptotic value: (0.5)² = 0.25")
    
    # After time T ≈ tau, EMA should reach ~63% of asymptotic value
#    expected_fraction = 1 - np.exp(-T_total / plasticity.tau_integrate)
#    expected_Q = 0.25 * expected_fraction
    
#    print(f"Expected after {T_total}s: {expected_Q:.6f} ({expected_fraction*100:.1f}% of asymptotic)")
    
    # Check that EMA behaves correctly (within 10% of expected)
#    assert np.abs(Q_avg[0, 1] - expected_Q) / expected_Q < 0.1, \
#        f"EMA convergence incorrect: got {Q_avg[0,1]:.6f}, expected {expected_Q:.6f}"
    
#    print("✓ EMA convergence correct\n")

def test_single_pattern_overfitting():
    """
    Test: Can the network learn to reproduce a single pattern?
    
    This is the simplest test of plasticity. Train on [1, 0, 1, 0],
    verify that MSE decreases over cycles.
    """
    print("=== Test: Single Pattern Overfitting ===")
    print(f"Training {CONFIG.n_input}→{CONFIG.n_hidden}→{CONFIG.n_output} autoencoder on pattern [1, 0, 1, 0]")
    print(f"Parameters: eta={CONFIG.eta}, beta_clamped={CONFIG.beta_clamped}, g_penalty={CONFIG.g_penalty}")
    print(f"Exposure times: T_free={CONFIG.exposure_time_free}s, T_clamped={CONFIG.exposure_time_clamped}s")
    print("Expected: MSE should decrease significantly over cycles\n")
    
    # Build autoencoder
    n_total = CONFIG.n_input + CONFIG.n_hidden + CONFIG.n_output
    
    input_nodes = np.arange(CONFIG.n_input)
    hidden_nodes = np.arange(CONFIG.n_input, CONFIG.n_input + CONFIG.n_hidden)
    output_nodes = np.arange(CONFIG.n_input + CONFIG.n_hidden, n_total)
    
    adjacency = np.zeros((n_total, n_total), dtype=bool)
    for i in input_nodes:
        for h in hidden_nodes:
            adjacency[i, h] = adjacency[h, i] = True
    for h in hidden_nodes:
        for o in output_nodes:
            adjacency[h, o] = adjacency[o, h] = True
    
    # Initialize solver and plasticity
    solver = VoltageDynamics(adjacency, relu_iv, capacitances=1.0)
    plasticity = SimplePlasticity(
        eta=CONFIG.eta,
        gamma=CONFIG.gamma,
        tau_integrate=CONFIG.tau_integrate
    )
    
    penalty_pairs = [(i, CONFIG.n_input + CONFIG.n_hidden + i) for i in range(CONFIG.n_input)]
    
    # Initialize weights
    rng = np.random.default_rng(CONFIG.random_seed)
    weights = rng.uniform(CONFIG.w_min, CONFIG.w_max, size=(n_total, n_total))
    weights = weights * adjacency
    
    # Target pattern
    pattern = np.array([1.0, 0.0, 1.0, 0.0])
    
    # Compute max steps
    max_steps_free = int(CONFIG.exposure_time_free / CONFIG.dt)
    max_steps_clamped = int(CONFIG.exposure_time_clamped / CONFIG.dt)
    
    # Training history
    mse_free_history = []
    mse_clamped_history = []
    
    # Training loop
    print("Training...")
    for cycle in range(CONFIG.n_cycles):
        # Convert weights to conductances
        conductances = weights_to_conductances(weights, CONFIG.g_min, CONFIG.g_max)
        
        # Initial voltages
        V_init = np.zeros(n_total)
        V_init[input_nodes] = pattern
        
        # === FREE PHASE ===
        result_free = solver.relax_transient(
            conductances, V_init,
            clamped_nodes=input_nodes,
            clamped_values=pattern,
            penalty_pairs=penalty_pairs,
            beta=CONFIG.beta_free,
            dt=CONFIG.dt,
            tol=CONFIG.tol,
            max_steps=max_steps_free,
            record_history=True
        )
        
        # Integrate observables
        #Q_free = plasticity.integrate_observable_ema(
        #    result_free['V_history'], adjacency, CONFIG.dt
        #)
        Q_free = compute_Q_from_voltages(result_free['V_final'], adjacency)
        
        # Compute MSE
        V_output_free = result_free['V_final'][output_nodes]
        mse_free = np.mean((pattern - V_output_free) ** 2)
        mse_free_history.append(mse_free)
        
        # === CLAMPED PHASE ===
        result_clamped = solver.relax_transient(
            conductances, result_free['V_final'],
            clamped_nodes=input_nodes,
            clamped_values=pattern,
            penalty_pairs=penalty_pairs,
            beta=CONFIG.beta_clamped,
            g_penalty=CONFIG.g_penalty,
            dt=CONFIG.dt,
            tol=CONFIG.tol,
            max_steps=max_steps_clamped,
            record_history=True
        )
        
        # Integrate observables
        #Q_clamped = plasticity.integrate_observable_ema(
        #    result_clamped['V_history'], adjacency, CONFIG.dt
        #)
        Q_clamped = compute_Q_from_voltages(result_clamped['V_final'], adjacency)

        # Compute MSE
        V_output_clamped = result_clamped['V_final'][output_nodes]
        mse_clamped = np.mean((pattern - V_output_clamped) ** 2)
        mse_clamped_history.append(mse_clamped)
        
        # === WEIGHT UPDATE ===
        weights_old = weights.copy()

        weights = plasticity.update_weights(
            weights, Q_free, Q_clamped, adjacency, CONFIG.dt_plasticity
        )
        
        if (cycle + 1) % 10 == 0:
            delta_Q = Q_clamped - Q_free
            dw = weights - weights_old
            
            # Берём только существующие рёбра
            delta_Q_edges = delta_Q[adjacency]
            dw_edges = dw[adjacency]
            
            # === НОВОЕ: вычислить Q вручную из финальных состояний ===
            Q_free_manual = np.zeros_like(adjacency, dtype=float)
            Q_clamped_manual = np.zeros_like(adjacency, dtype=float)
    
            V_free_final = result_free['V_final']
            V_clamped_final = result_clamped['V_final']

            V_hidden_free = result_free['V_final'][hidden_nodes]
            V_hidden_clamped = result_clamped['V_final'][hidden_nodes]    
   
            for i in range(len(adjacency)):
                for j in range(len(adjacency)):
                    if adjacency[i, j]:
                        Q_free_manual[i, j] = (V_free_final[j] - V_free_final[i])**2
                        Q_clamped_manual[i, j] = (V_clamped_final[j] - V_clamped_final[i])**2
    
            delta_Q_manual = Q_clamped_manual - Q_free_manual
            
            print(f"  Cycle {cycle+1}/{CONFIG.n_cycles}:")
            print(f"    Free:    MSE={mse_free:.6f}, Output={np.round(V_output_free, 3)}, Hidden={np.round(V_hidden_free, 3)}")
            print(f"    Clamped: MSE={mse_clamped:.6f}, Output={np.round(V_output_clamped, 3)}, , Hidden={np.round(V_hidden_clamped, 3)}")
            print(f"    delta_Q (EMA):    mean={delta_Q_edges.mean():.6f}, max={np.abs(delta_Q_edges).max():.6f}")
            print(f"    delta_Q (manual): mean={delta_Q_manual[adjacency].mean():.6f}, max={np.abs(delta_Q_manual[adjacency]).max():.6f}")
            print(f"    dw:               mean={dw_edges.mean():.6f}, max={np.abs(dw_edges).max():.6f}")
    
    # Check improvement
    mse_initial = mse_free_history[0]
    mse_final = mse_free_history[-1]
    improvement = (mse_initial - mse_final) / mse_initial * 100
    
    print(f"\nInitial MSE: {mse_initial:.6f}")
    print(f"Final MSE: {mse_final:.6f}")
    print(f"Improvement: {improvement:.1f}%")
    
    # Assertions
    assert mse_final < mse_initial, "MSE did not decrease!"
    assert improvement > 10, f"Improvement too small: {improvement:.1f}%"
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Learning curve
    ax = axes[0]
    cycles = np.arange(1, len(mse_free_history) + 1)
    ax.plot(cycles, mse_free_history, 'b-', linewidth=2, label='Free phase')
    ax.plot(cycles, mse_clamped_history, 'r-', linewidth=2, label='Clamped phase')
    ax.set_xlabel('Training Cycle', fontsize=11)
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Learning Curve', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Final reconstruction
    ax = axes[1]
    conductances_final = weights_to_conductances(weights, CONFIG.g_min, CONFIG.g_max)
    V_init = np.zeros(n_total)
    V_init[input_nodes] = pattern
    
    result_final = solver.relax_transient(
        conductances_final, V_init,
        clamped_nodes=input_nodes,
        clamped_values=pattern,
        dt=CONFIG.dt,
        max_steps=max_steps_free,
        tol=CONFIG.tol
    )
    
    V_output = result_final['V_final'][output_nodes]
    
    x = np.arange(CONFIG.n_input)
    width = 0.35
    ax.bar(x - width/2, pattern, width, label='Input', alpha=0.7)
    ax.bar(x + width/2, V_output, width, label='Output (final)', alpha=0.7)
    ax.set_xlabel('Node', fontsize=11)
    ax.set_ylabel('Voltage', fontsize=11)
    ax.set_title('Final Reconstruction', fontsize=12)
    ax.set_xticks(x)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('test_plasticity_overfitting.png', dpi=100)
    
    print("✓ Test passed")
    print("✓ Plot saved to test_plasticity_overfitting.png\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Plasticity Rules")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Network: {CONFIG.n_input}→{CONFIG.n_hidden}→{CONFIG.n_output}")
    print(f"  Plasticity: eta={CONFIG.eta}, gamma={CONFIG.gamma}, tau={CONFIG.tau_integrate}s")
    print(f"  Physics: beta_clamped={CONFIG.beta_clamped}, g_penalty={CONFIG.g_penalty}")
    print(f"  Training: {CONFIG.n_cycles} cycles, T_free={CONFIG.exposure_time_free}s, T_clamped={CONFIG.exposure_time_clamped}s")
    print()
    
    test_weight_conversion()
    test_observable_integration()
    test_single_pattern_overfitting()
    
    print("=" * 70)
    print("✓ All plasticity tests passed!")
    print("=" * 70)
