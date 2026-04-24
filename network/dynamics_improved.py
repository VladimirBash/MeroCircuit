# network/dynamics.py
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))


from memristor.memristor import Memristor
from grid.grid import Grid
import numpy as np
from typing import Optional, Tuple, List


class VoltageDynamics:
    """
    Solver for voltage relaxation in memristive networks.
    
    Implements transient dynamics:
        C_i * dV_i/dt = Σ_j I_ij + I_penalty
    
    IMPORTANT:
    - Memristors are treated as STATIC during this relaxation.
    - Their internal state evolution must be handled externally.
    """
    
    def __init__(self,
                 grid,
                 memristors: np.ndarray,
                 ):
        """
        Initialize voltage dynamics solver.
        
        Args:
            1. An instance of a Grid class
            2. An array of memristors:
        """
        self.memristors = memristors
        self.grid = grid
        
        self.n_nodes = len(grid.adjacency)
    
    


    def relax_transient(self,
                    V_init: np.ndarray,
                    penalty_pairs=None,
                    beta: float = 0.0,
                    g_penalty: float = 1.0,
                    max_steps: int = 1000,
                    tol: float = 1e-6,
                    dt: float = 0.01,
                    record_history: bool = False):

        V = V_init.copy()

        # enforce clamped nodes
        V[self.grid.clamped_nodes] = self.grid.clamped_values

        free_nodes = np.setdiff1d(np.arange(self.n_nodes),
                             self.grid.clamped_nodes)
        
        if record_history:
            V_history = [V.copy()]

        for step in range(max_steps):

            dV_dt = self._compute_time_derivative(
                V, penalty_pairs, beta, g_penalty
            )

            V[free_nodes] += dt * dV_dt[free_nodes]

            if record_history:
                V_history.append(V.copy())

            if np.max(np.abs(dt * dV_dt[free_nodes])) < tol:
                result = {
                    'V_final': V,
                    'converged': True,
                    'n_steps': step + 1,
                }
                if record_history:
                    result['V_history'] = np.array(V_history)
                
                return result

            

        result = {
            'V_final': V,
            'converged': False,
            'n_steps': max_steps,
        }
        if record_history:
            result['V_history'] = np.array(V_history)

        return result
    
    def _compute_time_derivative(self,
                             V,
                             penalty_pairs,
                             beta,
                             g_penalty):
        
        dV_dt = np.zeros(self.n_nodes)
        
        clamped_mask = np.zeros(self.n_nodes, dtype=bool)
        clamped_mask[self.grid.clamped_nodes] = True

        for i in range(self.n_nodes):
            if clamped_mask[i]:
                continue
            I_neighbors = 0.0

            for j in range(self.n_nodes):
                if self.grid.adjacency[i, j]:
                    mem = self.memristors[i, j]
                    if mem is not None:
                        V_drop = V[j] - V[i]
                        I_neighbors += mem.current(V_drop)

            I_penalty = 0.0
        
            if penalty_pairs is not None and beta > 0:
                I_penalty = self._compute_penalty_current(
                    i, V, penalty_pairs, beta, g_penalty
                )

            dV_dt[i] = (I_neighbors + I_penalty) / self.grid.C[i]


        return dV_dt
    
    def _compute_penalty_current(self,
                                 node_idx: int,
                                 V: np.ndarray,
                                 penalty_pairs: List[Tuple[int, int]],
                                 beta: float,
                                 g_penalty: float) -> float:
        """
        Compute penalty current for autoencoder coupling.
        """
        I_pen = 0.0
        
        for in_node, out_node in penalty_pairs:
            if node_idx == out_node:
                V_drop = V[in_node] - V[out_node]
                I_pen += beta * g_penalty * V_drop
            elif node_idx == in_node:
                V_drop = V[in_node] - V[out_node]
                I_pen -= beta * g_penalty * V_drop
        
        return I_pen
    
    
    def compute_currents(self,
                        V: np.ndarray,
                        memristors: np.ndarray) -> np.ndarray:
        """
        Compute currents through all edges.
        
        Returns:
            I[i,j] = current flowing from j → i
        """
        I = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.memristors[i, j] is not None:
                    V_drop = V[j] - V[i]
                    I[i, j] = memristors[i, j].current(V_drop)
        
        return I