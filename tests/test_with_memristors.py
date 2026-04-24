import numpy as np
import matplotlib.pyplot as plt

from memristor.memristor import Memristor
from network.dynamics_improved import VoltageDynamics

# Assume Memristor and VoltageDynamics already imported

print("STARTING SCRIPT")

# -----------------------
# 1. Define functions
# -----------------------

def g_func(w):
    return 0.1 + 0.9 * w  # conductance range

#def f_func(V):
#    return V  # Ohmic

def f_func(V):
    if V>0:
        return V
    else:
        return 0 # ReLU

def observable(V, I):
    return V * I  # power


# -----------------------
# 2. Build network
# -----------------------

n_nodes = 5

# Node indices
a, b, c, d, gnd = 0, 1, 2, 3, 4

# Fully connected for simplicity (you can change this)
adj = np.ones((n_nodes, n_nodes), dtype=bool)
np.fill_diagonal(adj, False)

# Create memristors
memristors = np.empty((n_nodes, n_nodes), dtype=object)

for i in range(n_nodes):
    for j in range(n_nodes):
        if adj[i, j]:
            memristors[i, j] = Memristor(
                eta=0.05,
                Q0=0.0,
                window_time=1.0,
                dt=0.01,
                w0=np.random.rand(),
                g_func=g_func,
                f_func=f_func,
                observable_func=observable,
                gamma=0.01
            )
        else:
            memristors[i, j] = None


# -----------------------
# 3. Voltage solver
# -----------------------

solver = VoltageDynamics(adj)

# Initial voltages
V = np.zeros(n_nodes)

# Clamp ground
ground_nodes = np.array([gnd])
ground_values = np.array([0.0])


# -----------------------
# 4. Training loop
# -----------------------

n_cycles = 101
steps_per_phase = 50

Va = 1 # input a
Vb = 2 # input b

W_history = []

def extract_W(memristors, adj):
    n = len(memristors)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if adj[i, j]:
                W[i, j] = memristors[i, j].w
    return W


for cycle in range(n_cycles):

    # -----------------------
    # FREE PHASE
    # -----------------------

    # Clamp inputs + ground
    clamped_nodes = np.array([a, b, gnd])
    clamped_values = np.array([
        Va,  
        Vb,  
        0.0
    ])

    result = solver.relax_transient(
        memristors=memristors,
        V_init=V,
        clamped_nodes=clamped_nodes,
        clamped_values=clamped_values,
        beta=0.0,  # no coupling
        dt=0.05,
        max_steps=500
    )

    V = result["V_final"]

    # Evolve memristors for a while (slow dynamics)
    for _ in range(steps_per_phase):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj[i, j]:
                    V_drop = V[j] - V[i]
                    memristors[i, j].step(V_drop)


    # -----------------------
    # CLAMPED PHASE (short circuit)
    # -----------------------

    penalty_pairs = [(a, c), (b, d)]

    result = solver.relax_transient(
        memristors=memristors,
        V_init=V,
        clamped_nodes=clamped_nodes,
        clamped_values=clamped_values,
        penalty_pairs=penalty_pairs,
        beta=5.0,          # strong coupling
        g_penalty=5.0,
        dt=0.05,
        max_steps=500
    )

    V = result["V_final"]

    # Evolve memristors again
    for _ in range(steps_per_phase):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj[i, j]:
                    V_drop = V[j] - V[i]
                    memristors[i, j].step(V_drop)

    # -----------------------
    # Monitoring
    # -----------------------

    if cycle % 10 == 0:
        avg_w = np.mean([
            memristors[i, j].w
            for i in range(n_nodes)
            for j in range(n_nodes)
            if adj[i, j]
        ])
        print(f"Cycle {cycle}: avg w = {avg_w:.3f}, V =  {V}")

   
    
    W = extract_W(memristors, adj)
    W_history.append(W.copy())

print("Done with leraning. Exam time.")

Va = 1
Vb = 2

clamped_nodes = np.array([a, b, gnd])
clamped_values = np.array([
    Va,  
    Vb,  
    0.0
])

result = solver.relax_transient(
    memristors=memristors,
    V_init=V,
    clamped_nodes=clamped_nodes,
    clamped_values=clamped_values,
    beta=0.0,  # no coupling
    dt=0.05,
    max_steps=500
)

V = result["V_final"]

print(f"V = {V}")

W_history = np.array(W_history)  # shape: (n_cycles, n_nodes, n_nodes)

plt.figure()

for t in range(len(W_history)):
    plt.clf()
    plt.imshow(W_history[t], vmin=0, vmax=1)
    plt.colorbar(label="w")
    plt.title(f"Cycle {t}")
    plt.pause(0.1)

plt.show()