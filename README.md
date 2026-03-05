# MeroCircuit

A minimal theoretical framework for autonomous learning in physical memristive networks, based on equilibrium propagation and autoencoder principles.

## Overview

This project implements a simulation of a fully autonomous neuromorphic learning system where:
- **Inference** corresponds to electrical relaxation under fixed boundary conditions
- **Learning** emerges from local adaptation driven by differences between two physical regimes (free and clamped)
- **Loss function** is embedded directly in the physics through penalty coupling

The system operates without external digital controllers or explicit gradient computation.

## Installation

### Requirements
- Python ≥ 3.9
- NumPy ≥ 1.20.0
- SciPy ≥ 1.7.0
- Matplotlib ≥ 3.4.0
- NetworkX ≥ 2.6.0
- Pillow ≥ 9.0.0 (for animations)
- pytest ≥ 7.0.0 (for testing)

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/MeroCircuit.git
cd MeroCircuit

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Project Structure

```
MeroCircuit/
├── network/                    # Network dynamics and topology
│   ├── dynamics.py            # VoltageDynamics: transient voltage relaxation solver
│   ├── iv_characteristics.py  # I-V curves (Ohmic, ReLU, Sigmoid, Diode)
│   └── topology/              # Network topology generators
│
├── training/                   # Learning algorithms
│   └── plasticity.py          # SimplePlasticity: contrastive Hebbian learning
│
├── datasets/                   # Training datasets
│   ├── bars_stripes.py        # BarsAndStripes dataset generator
│   └── generate.py            # Dataset utilities
│
├── visualization/              # Visualization tools
│   ├── plotting.py            # Basic plotting (patterns, reconstruction, loss)
│   └── dynamics_viz.py        # Network visualization and animations
│
└── tests/                      # Unit tests and examples
    ├── test_dynamics_basic.py        # Basic circuit tests
    ├── test_dynamics_autoencoder.py  # Autoencoder topology tests
    ├── test_bars_stripes.py          # Dataset tests
    └── test_visualization.py         # Visualization tests
```

## Key Concepts

### Voltage Dynamics
The `VoltageDynamics` class solves transient electrical relaxation:
```
C_i · dV_i/dt = Σ_j I_ij(V_j - V_i, g_ij) + I_penalty
```
- Implements capacitive dynamics with configurable time constants
- Supports multiple I-V characteristics (Ohmic, ReLU, Sigmoid, Diode)
- Handles penalty coupling for autoencoder learning

### I-V Characteristics
Multiple current-voltage relationships available in `network/iv_characteristics.py`:
- **Ohmic**: Linear resistive behavior `I = g·V`
- **ReLU**: Threshold-activated `I = g·ReLU(|V| - V_th)·sign(V)`
- **Sigmoid**: Smooth nonlinearity `I = g·tanh(k·V)`
- **Diode**: Asymmetric rectification for directional current flow

### Learning Protocol (Equilibrium Propagation)

For each training pattern:

1. **Free phase** (β=0): System relaxes with inputs clamped, outputs free
   ```python
   result_free = solver.relax_transient(
       conductances, V_init, clamped_nodes, clamped_values,
       penalty_pairs=pairs, beta=0.0, record_history=True
   )
   ```

2. **Clamped phase** (β>0): Penalty coupling weakly biases outputs toward inputs
   ```python
   result_clamped = solver.relax_transient(
       conductances, V_init, clamped_nodes, clamped_values,
       penalty_pairs=pairs, beta=1.0, g_penalty=1.0
   )
   ```

3. **Weight update**: Contrastive Hebbian plasticity
   ```
   dw = η · (Q_clamped - Q_free) · (1 - w) - γ · w
   ```
   where `Q_ij = (V_j - V_i)²` is the local observable (quadratic in voltage drop)

### Penalty Coupling
Physical implementation of autoencoder reconstruction loss:
- Virtual resistive links between input and output nodes
- Activated only during clamped phase (β > 0)
- Creates weak bias toward input-output matching without hard clamping
- Enables autonomous learning without external error signals

### Bars & Stripes Dataset

A benchmark dataset for testing autoencoder learning:
- N×N binary grid patterns
- Valid patterns: horizontal stripes OR vertical bars (no mixing)
- For 4×4 grid: 30 unique patterns (2⁴ + 2⁴ - 2)
- Provides information bottleneck for hidden layer compression

```python
from datasets.bars_stripes import BarsAndStripes

dataset = BarsAndStripes(N=4, voltage_on=1.0, voltage_off=0.0)
print(f"Total patterns: {dataset.n_patterns}")  # 30 for N=4

patterns = dataset.sample(n_samples=10, random_state=42)
voltages = dataset.to_voltages(patterns)
```

## Design Principles

### Separation of Timescales
- **Fast**: Electrical relaxation (inference) ~ milliseconds, governed by RC time constants
- **Slow**: Conductance adaptation (learning) ~ seconds, governed by plasticity parameters

### Locality
- No global error signals or backpropagation
- Each synapse updates based only on local voltage drops
- Penalty coupling is physical (virtual resistive links), not algorithmic

### Physical Plausibility
- All operations correspond to material processes
- Capacitive charging dynamics with realistic time constants
- Loss function embedded in circuit topology through penalty links
- Learning driven by energy landscape differences between phases

## Basic Usage

### Example: Simple voltage relaxation
```python
import numpy as np
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic

# Define 3-node resistor chain: 0 -- 1 -- 2
adjacency = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
])
conductances = adjacency.astype(float)

# Create solver
solver = VoltageDynamics(adjacency, ohmic, capacitances=1.0)

# Relax with boundary conditions V[0]=1, V[2]=0
V_init = np.array([1.0, 0.5, 0.0])
result = solver.relax_transient(
    conductances, V_init,
    clamped_nodes=np.array([0, 2]),
    clamped_values=np.array([1.0, 0.0]),
    dt=0.01, tol=1e-6
)

print(f"Final voltages: {result['V_final']}")
print(f"Converged in {result['n_steps']} steps")
```

### Example: Contrastive plasticity
```python
from training.plasticity import SimplePlasticity

plasticity = SimplePlasticity(eta=0.01, gamma=0.001, tau_integrate=1.0)

# Integrate observables from voltage history
Q_free = plasticity.integrate_observable_ema(V_history_free, adjacency, dt=0.01)
Q_clamped = plasticity.integrate_observable_ema(V_history_clamped, adjacency, dt=0.01)

# Update weights
w_new = plasticity.update_weights(w, Q_free, Q_clamped, adjacency, dt_plasticity=1.0)
```

## Running Tests

The test suite includes both unit tests and visualization examples:

```bash
# Run all tests
pytest tests/

# Run specific test with visualization output
python tests/test_dynamics_basic.py        # Basic circuit tests
python tests/test_dynamics_autoencoder.py  # Autoencoder topology
python tests/test_visualization.py         # Generates plots and animations
```

Test outputs include:
- Voltage evolution plots
- Network state visualizations
- Animated relaxation dynamics (GIF format)

## Visualization

The package includes comprehensive visualization tools:

```python
from visualization.dynamics_viz import plot_voltage_evolution, animate_relaxation

# Static plot of voltage evolution
fig = plot_voltage_evolution(
    V_history,
    node_groups={'Input': [0,1], 'Hidden': [2,3], 'Output': [4,5]},
    dt=0.01
)

# Animated relaxation with network graph
anim = animate_relaxation(
    V_history, adjacency, conductances, iv_function,
    node_groups={'Input': [0,1], 'Hidden': [2,3], 'Output': [4,5]},
    output_file='relaxation.gif', fps=30
)
```

## References

Based on theoretical proposal: *Fully autonomous neuromorphic element based on autoencoder learning principles* (Bagrov, Bashmakov, Kravchenko, 2025)

Key concepts:
- Equilibrium propagation (Scellier & Bengio, 2017)
- Physical autoencoder learning
- Memristive networks

## Contributors

- Andrey Bagrov
- Vladimir Bashmakov
- Anna Kravchenko

## License

(TBD)
