# MeroCircuit

Neuromorphic autoencoder implementation using memristive networks and equilibrium propagation principles.

## Project Overview

This project implements a theoretical framework for autonomous learning in physical memristive networks. The system uses equilibrium propagation: learning occurs through local adaptation driven by differences in electrical activity between free and clamped phases, without requiring external backpropagation.

### Key Features

- **Voltage dynamics solver**: Transient relaxation of voltages in resistive/memristive networks
- **Autoencoder topology**: Input → Hidden → Output architecture with penalty coupling
- **Multiple I-V characteristics**: Ohmic, ReLU, sigmoid, and diode models
- **Comprehensive visualization**: Network states, current flows, and animated relaxation dynamics
- **Experimental protocol**: Realistic simulation with fixed exposure times per phase

## Installation

### Requirements

- Python ≥ 3.9
- pip and venv

### Setup

1. **Clone the repository:**
```bash
   git clone https://github.com/BagrovAndrey/NeuromorphicAE.git
   cd NeuromorphicAE
```

2. **Create and activate virtual environment:**
```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
```

4. **Run tests:**
```bash
   pytest tests/
```

## Project Structure
```
MeroCircuit/
├── network/                 # Voltage dynamics and I-V characteristics
│   ├── dynamics.py         # VoltageDynamics solver (Euler method)
│   ├── iv_characteristics.py  # Ohmic, ReLU, sigmoid, diode I-V curves
│   └── legacy.py           # Volodya's original R_Network (preserved)
├── datasets/               # Bars & Stripes pattern generation
│   ├── bars_stripes.py    # BarsAndStripes class
│   └── generate.py        # Dataset generation script
├── visualization/          # Plotting and animation tools
│   ├── plotting.py        # Static plots for patterns
│   └── dynamics_viz.py    # Network graphs, current flows, animations
├── tests/                  # Test suite
│   ├── test_dynamics_basic.py      # Simple circuits (chains, dividers)
│   ├── test_dynamics_autoencoder.py  # Autoencoder topology tests
│   └── test_network_visualization.py  # Visualization tests
├── training/               # [In development] Plasticity and learning
└── memristor/              # [Future] Memristor models (Anya's work)
```

## Current Implementation Status

### ✅ Completed

**Voltage Dynamics Solver** (`network/dynamics.py`)
- Transient relaxation via explicit Euler method
- Penalty coupling for autoencoder reconstruction
- Configurable I-V characteristics and capacitances
- Free phase (β=0) and clamped phase (β>0) support

**I-V Characteristics** (`network/iv_characteristics.py`)
- Ohmic (linear)
- ReLU with threshold
- Sigmoid (smooth nonlinearity)
- Diode (asymmetric)

**Datasets** (`datasets/`)
- Bars & Stripes pattern generator for N×N grids
- Voltage encoding/decoding
- Train/test splitting
- Pattern type classification

**Visualization** (`visualization/`)
- Time series plots of voltage evolution
- Network graphs with voltage-coded nodes
- Current flow visualization with arrows
- Animated relaxation (free → clamped phase transition)
- Side-by-side phase comparison

**Testing**
- Basic circuit tests (resistor chains, voltage dividers)
- Autoencoder topology with realistic experimental protocol
- Penalty coupling validation
- Convergence tests with different capacitances

### 🚧 In Development

**Plasticity Rules** (`training/`)
- Contrastive Hebbian learning
- Time-averaged observables with EMA
- Weight update dynamics

**Training Loop**
- Multi-cycle free/clamped iteration
- MSE tracking over epochs
- Weight evolution visualization

### 📋 Planned

**Memristor Models** (Anya's module)
- Physical memristor characteristics
- State-dependent conductance
- Integration with dynamics solver

**Network Topology** (Volodya's module)
- Advanced network builders
- Custom connectivity patterns

## Usage Examples

### Generate Bars & Stripes Dataset
```python
from datasets.bars_stripes import BarsAndStripes

# Create dataset
ds = BarsAndStripes(N=4, voltage_on=1.0, voltage_off=0.0)

# Get all patterns
patterns = ds.get_all_patterns()  # Shape: (n_patterns, 4, 4)

# Sample random batch
batch = ds.sample(n_samples=8)

# Split for training
train_patterns, test_patterns = ds.split_train_test(test_fraction=0.2)
```

### Run Voltage Relaxation
```python
from network.dynamics import VoltageDynamics
from network.iv_characteristics import ohmic
import numpy as np

# Simple 3-node chain: 0 -- 1 -- 2
adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
conductances = adjacency.astype(float)

solver = VoltageDynamics(adjacency, ohmic, capacitances=1.0)

# Relax with boundary conditions
V_init = np.array([1.0, 0.5, 0.0])
result = solver.relax_transient(
    conductances, V_init,
    clamped_nodes=np.array([0, 2]),
    clamped_values=np.array([1.0, 0.0]),
    dt=0.01, max_steps=1000, record_history=True
)

print(f"Final voltages: {result['V_final']}")
print(f"Converged: {result['converged']} in {result['n_steps']} steps")
```

### Autoencoder with Penalty Coupling
```python
# Build 4→3→4 autoencoder
n_input, n_hidden, n_output = 4, 3, 4
n_total = n_input + n_hidden + n_output

# ... build adjacency and conductances ...

penalty_pairs = [(i, n_input + n_hidden + i) for i in range(n_input)]

# Free phase (β=0)
result_free = solver.relax_transient(
    conductances, V_init,
    clamped_nodes=input_nodes,
    clamped_values=V_input,
    beta=0.0, dt=0.001, max_steps=10000
)

# Clamped phase (β>0)
result_clamped = solver.relax_transient(
    conductances, result_free['V_final'],
    clamped_nodes=input_nodes,
    clamped_values=V_input,
    penalty_pairs=penalty_pairs,
    beta=1.0, g_penalty=1.0,
    dt=0.001, max_steps=10000
)
```

### Visualize Network Dynamics
```python
from visualization.dynamics_viz import plot_voltage_evolution, animate_relaxation

# Plot voltage evolution
fig = plot_voltage_evolution(
    result['V_history'],
    node_groups={'Input': [0, 3], 'Hidden': [1, 2]},
    dt=0.01
)
fig.savefig('voltage_evolution.png')

# Create animation (free → clamped)
V_combined = np.vstack([result_free['V_history'], result_clamped['V_history']])
anim = animate_relaxation(
    V_combined, adjacency, conductances, ohmic,
    nodes_to_plot=[0, 4, 8],  # Selected nodes
    phase_transition_frame=len(result_free['V_history']),
    output_file='relaxation.gif'
)
```

## Running Tests
```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_dynamics_autoencoder.py -v

# With coverage
pytest tests/ --cov=network --cov=datasets --cov=visualization
```

## Key Parameters

### Solver Configuration
- `dt`: Time step (0.001 for stable clamped phase with high β)
- `tol`: Convergence tolerance (1e-10 for high precision)
- `max_steps`: Maximum iterations per phase

### Physical Parameters
- `beta`: Penalty coupling strength (0 = free, >0 = clamped)
- `g_penalty`: Penalty link conductance
- `capacitances`: Node capacitances (affects relaxation speed)

### Experimental Protocol
- `exposure_time_free`: Duration of free phase
- `exposure_time_clamped`: Duration of clamped phase
- Both measured in seconds, independent of solver timestep

## Team

- **Andrey Bagrov**
- **Anna Kravchenko**
- **Vladimir Bashmakov**

## License

[To be determined]

## References

[Key papers and theoretical background to be added]