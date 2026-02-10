# Neuromorphic Autoencoder

A minimal theoretical framework for autonomous learning in physical memristive networks, based on equilibrium propagation and autoencoder principles.

## Overview

This project implements a simulation of a fully autonomous neuromorphic learning system where:
- **Inference** corresponds to electrical relaxation under fixed boundary conditions
- **Learning** emerges from local adaptation driven by differences between two physical regimes (free and clamped)
- **Loss function** is embedded directly in the physics through penalty coupling

The system operates without external digital controllers or explicit gradient computation.

## Project Structure
```
neuromorphic_autoencoder/
├── memristor/              # Memristor components
│   ├── memristor.py       # Main Memristor class
│   ├── iv_curves.py       # Current-voltage characteristics (Ohmic, ReLU, etc.)
│   ├── plasticity_rules.py # Learning rules (Contrastive Hebbian, etc.)
│   └── observables.py     # Local observables for plasticity (quadratic, etc.)
│
├── network/               # Network structure and dynamics
│   ├── network.py        # Network class (graph + memristors)
│   ├── dynamics.py       # Voltage relaxation and penalty coupling
│   └── topology/         
│       └── builders.py   # Network topology generators (sparse, feedforward, etc.)
│
├── datasets/              # Training datasets
│   └── bars_stripes.py   # Bars & Stripes benchmark dataset
│
├── training/              # Training orchestration
│   └── trainer.py        # Main training loop (free/clamped phases)
│
├── visualization/         # Plotting utilities
│   └── plotting.py       # Visualization tools
│
└── tests/                 # Unit tests
```

## Key Concepts

### Memristor
The fundamental building block. Each memristor has:
- **Internal state** `w ∈ [0,1]` controlling conductance
- **I-V characteristic** determining current flow
- **Plasticity rule** for learning from local electrical activity

### Network
A graph where:
- Nodes represent electrical potentials
- Edges are memristors with adaptive conductances
- Input/output nodes are designated for the autoencoder task

### Learning Protocol (Equilibrium Propagation)

For each training pattern:

1. **Free phase** (β=0): System relaxes with inputs clamped, outputs free
2. **Clamped phase** (β>0): Penalty coupling weakly biases outputs toward inputs
3. **Weight update**: Each memristor adapts based on the difference in local activity between phases
```
w_ij += η · (Q_ij^clamped - Q_ij^free) · (1 - w_ij) - γ · w_ij
```

### Minimal Benchmark: Bars & Stripes

A 4×4 grid autoencoder task with informational bottleneck:
- **Input**: 16 nodes (4×4 pixels)
- **Hidden**: 5 nodes (4 bits for line selection + 1 bit for orientation)
- **Output**: 16 nodes (reconstruction)

Valid patterns: horizontal or vertical stripes only (30 total patterns)

## Design Principles

### Separation of Timescales
- **Fast**: Electrical relaxation (inference) ~ milliseconds
- **Slow**: Conductance adaptation (learning) ~ seconds

### Locality
- No global error signals
- Each memristor updates based only on local voltage drops
- Penalty coupling is physical (virtual resistive links), not algorithmic

### Physical Plausibility
- All operations correspond to material processes
- Loss function embedded in circuit topology
- Learning driven by energy landscape differences

## Dependencies
```
numpy
scipy
networkx
matplotlib
```

## Getting Started

(Coming soon: basic usage example)

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
