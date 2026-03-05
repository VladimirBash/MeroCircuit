# network/dynamics.py

import numpy as np
from typing import Callable, Optional, Tuple, List


class VoltageDynamics:
    """
    Solver for voltage relaxation in memristive networks.
    
    Implements transient dynamics: C_i * dV_i/dt = Σ_j I_ij + I_penalty
    Works with simple numpy arrays, independent of Network/Memristor classes.
    """
    
    def __init__(self, 
                 adjacency: np.ndarray,
                 iv_function: Callable,
                 capacitances: Optional[np.ndarray] = None):
        """
        Initialize voltage dynamics solver.
        
        Args:
            adjacency: (n_nodes, n_nodes) boolean array of connections
            iv_function: callable f(V_drop, g) -> current
            capacitances: (n_nodes,) array or scalar (default: all 1.0)
        """
        self.adj = adjacency.astype(bool)
        self.n_nodes = len(adjacency)
        self.iv_func = iv_function
        
        # Setup capacitances
        if capacitances is None:
            self.C = np.ones(self.n_nodes)
        elif np.isscalar(capacitances):
            self.C = np.full(self.n_nodes, capacitances)
        else:
            self.C = np.array(capacitances)
    
    
    def relax_transient(self,
                       conductances: np.ndarray,
                       V_init: np.ndarray,
                       clamped_nodes: np.ndarray,
                       clamped_values: np.ndarray,
                       penalty_pairs: Optional[List[Tuple[int, int]]] = None,
                       beta: float = 0.0,
                       g_penalty: float = 1.0,
                       dt: float = 0.01,
                       max_steps: int = 1000,
                       tol: float = 1e-6,
                       record_history: bool = False) -> dict:
        """
        Relax voltages to steady state via transient dynamics.
        
        Args:
            conductances: (n_nodes, n_nodes) conductance matrix G[i,j]
            V_init: (n_nodes,) initial voltages
            clamped_nodes: array of node indices to fix
            clamped_values: voltages for clamped nodes
            penalty_pairs: list of (input_idx, output_idx) for autoencoder coupling
            beta: penalty coupling strength (0 = free phase, >0 = clamped phase)
            g_penalty: conductance of penalty links
            dt: time step
            max_steps: maximum iterations
            tol: convergence tolerance on max(|dV/dt|)
            record_history: if True, save V at each step
            
        Returns:
            dict with:
                'V_final': (n_nodes,) steady-state voltages
                'converged': bool
                'n_steps': number of iterations
                'V_history': (n_steps, n_nodes) if record_history=True
        """
        V = V_init.copy()
        V[clamped_nodes] = clamped_values
    
        free_nodes = np.setdiff1d(np.arange(self.n_nodes), clamped_nodes)
    
        # Early exit if all nodes are clamped
        if len(free_nodes) == 0:
            result = {
                'V_final': V,
                'converged': True,
                'n_steps': 0,
            }
            if record_history:
                result['V_history'] = np.array([V])
    
            return result
    
        if record_history:
            V_history = [V.copy()]        
  
        max_dV = np.inf
  
        for step in range(max_steps):
            # Compute dV/dt for free nodes
            dV_dt = self._compute_time_derivative(
                V, conductances, free_nodes, penalty_pairs, beta, g_penalty
            )
            
            # Euler step
            V[free_nodes] += dt * dV_dt[free_nodes]
            
            if record_history:
                V_history.append(V.copy())
            
            # Check convergence
            max_dV = np.max(np.abs(dV_dt[free_nodes]))
            if max_dV < tol:
                result = {
                    'V_final': V,
                    'converged': True,
                    'n_steps': step + 1,
                }
                if record_history:
                    result['V_history'] = np.array(V_history)
                return result
        
        # Did not converge
        #print(f"Warning: Did not converge in {max_steps} steps (max_dV = {max_dV:.2e})") #Temporarily commented out
        result = {
            'V_final': V,
            'converged': False,
            'n_steps': max_steps,
        }
        if record_history:
            result['V_history'] = np.array(V_history)
        return result
    
    
    def _compute_time_derivative(self,
                                 V: np.ndarray,
                                 conductances: np.ndarray,
                                 free_nodes: np.ndarray,
                                 penalty_pairs: Optional[List],
                                 beta: float,
                                 g_penalty: float) -> np.ndarray:
        """
        Compute dV/dt = (I_neighbors + I_penalty) / C for each node.
        
        Only computes for free nodes to save time.
        """
        dV_dt = np.zeros(self.n_nodes)
        
        for i in free_nodes:
            # Current from neighbors via memristors
            I_neighbors = 0.0
            for j in range(self.n_nodes):
                if self.adj[i, j]:
                    V_drop = V[j] - V[i]
                    g_ij = conductances[i, j]
                    I_neighbors += self.iv_func(V_drop, g_ij)
            
            # Penalty current for autoencoder
            I_penalty = 0.0
            if penalty_pairs is not None and beta > 0:
                I_penalty = self._compute_penalty_current(
                    i, V, penalty_pairs, beta, g_penalty
                )
            
            dV_dt[i] = (I_neighbors + I_penalty) / self.C[i]
        
        return dV_dt
    
    
    def _compute_penalty_current(self,
                                 node_idx: int,
                                 V: np.ndarray,
                                 penalty_pairs: List[Tuple[int, int]],
                                 beta: float,
                                 g_penalty: float) -> float:
        """
        Compute penalty current for autoencoder coupling.
        
        Penalty links weakly connect input-output pairs during clamped phase.
        """
        I_pen = 0.0
        
        for in_node, out_node in penalty_pairs:
            if node_idx == out_node:
                # Current flows into output from corresponding input
                V_drop = V[in_node] - V[out_node]
                I_pen += beta * g_penalty * V_drop
            elif node_idx == in_node:
                # Equal and opposite current at input
                V_drop = V[in_node] - V[out_node]
                I_pen -= beta * g_penalty * V_drop
        
        return I_pen
    
    
    def compute_currents(self,
                        V: np.ndarray,
                        conductances: np.ndarray) -> np.ndarray:
        """
        Compute currents through all edges for given voltage configuration.
        
        Args:
            V: (n_nodes,) voltage array
            conductances: (n_nodes, n_nodes) conductance matrix
            
        Returns:
            (n_nodes, n_nodes) current matrix where I[i,j] is current from j to i
        """
        I = np.zeros((self.n_nodes, self.n_nodes))
        
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self.adj[i, j]:
                    V_drop = V[j] - V[i]
                    I[i, j] = self.iv_func(V_drop, conductances[i, j])
        
        return I
