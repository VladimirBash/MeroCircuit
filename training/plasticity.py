# training/plasticity.py

"""
Plasticity rules for memristive networks.

Implements local learning rules based on contrastive Hebbian updates
driven by differences in electrical activity between free and clamped phases.
"""

import numpy as np
from typing import Callable


class SimplePlasticity:
    """
    Simple contrastive Hebbian plasticity rule.
    
    Weight updates driven by contrast between time-averaged observables
    in free and clamped phases, with saturation and spontaneous decay.
    
    Update rule:
        dw/dt = η · ΔQ · (1 - w) - γ · w
    
    where ΔQ = ⟨Q⟩_clamped - ⟨Q⟩_free
    """
    
    def __init__(self, 
                 eta: float = 0.01,
                 gamma: float = 0.001,
                 tau_integrate: float = 1.0):
        """
        Initialize plasticity rule.
        
        Args:
            eta: Learning rate (growth timescale)
            gamma: Decay rate (spontaneous relaxation)
            tau_integrate: Integration time window for observables (seconds)
        """
        self.eta = eta
        self.gamma = gamma
        self.tau_integrate = tau_integrate
    
    
    def compute_observable(self, V_drop: float) -> float:
        """
        Compute instantaneous local observable.
        
        Args:
            V_drop: Voltage difference V_j - V_i
            
        Returns:
            Q_ij = (V_drop)²
        """
        return V_drop ** 2
    
    
    def integrate_observable_ema(self,
                                 V_history: np.ndarray,
                                 adjacency: np.ndarray,
                                 dt: float) -> np.ndarray:
        """
        Integrate observables over time using exponential moving average.
        
        Physical interpretation: Memristor integrates local activity with
        characteristic relaxation time tau_integrate.
        
        Args:
            V_history: (n_steps, n_nodes) voltage history
            adjacency: (n_nodes, n_nodes) adjacency matrix
            dt: Time step
            
        Returns:
            Q_avg: (n_nodes, n_nodes) time-averaged observables
        """
        n_steps, n_nodes = V_history.shape
        Q_avg = np.zeros((n_nodes, n_nodes))
        
        # EMA decay factor
        alpha = dt / self.tau_integrate
        
        for step in range(n_steps):
            V = V_history[step]
            
            # Compute instantaneous observables
            Q_instant = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adjacency[i, j]:
                        V_drop = V[j] - V[i]
                        Q_instant[i, j] = self.compute_observable(V_drop)
            
            # Update EMA
            Q_avg = (1 - alpha) * Q_avg + alpha * Q_instant
        
        return Q_avg
    
    
    def update_weights(self,
                      w: np.ndarray,
                      Q_free: np.ndarray,
                      Q_clamped: np.ndarray,
                      adjacency: np.ndarray,
                      dt_plasticity: float = 1.0) -> np.ndarray:
        """
        Update conductance weights based on contrastive rule.
        
        Args:
            w: (n_nodes, n_nodes) current weight matrix (0 to 1)
            Q_free: (n_nodes, n_nodes) time-averaged Q in free phase
            Q_clamped: (n_nodes, n_nodes) time-averaged Q in clamped phase
            adjacency: (n_nodes, n_nodes) which edges exist
            dt_plasticity: Time step for plasticity update
            
        Returns:
            w_new: Updated weight matrix
        """
        # Contrastive signal
        delta_Q = Q_clamped - Q_free
        
        # Update rule: dw = η·ΔQ·(1-w) - γ·w
        dw = self.eta * delta_Q * (1 - w) - self.gamma * w
        
        # Apply only to existing edges
        dw = dw * adjacency
        
        # Update and clip to [0, 1]
        w_new = w + dw * dt_plasticity
        w_new = np.clip(w_new, 0, 1)
        
        return w_new
    
    
    def __repr__(self) -> str:
        return (f"SimplePlasticity(eta={self.eta}, gamma={self.gamma}, "
                f"tau_integrate={self.tau_integrate})")


# Helper function to convert weights to conductances
def weights_to_conductances(w: np.ndarray, 
                            g_min: float = 0.01, 
                            g_max: float = 1.0) -> np.ndarray:
    """
    Convert normalized weights (0-1) to conductances.
    
    Args:
        w: Weight matrix [0, 1]
        g_min: Minimum conductance
        g_max: Maximum conductance
        
    Returns:
        g: Conductance matrix
    """
    return g_min + (g_max - g_min) * w


def conductances_to_weights(g: np.ndarray,
                            g_min: float = 0.01,
                            g_max: float = 1.0) -> np.ndarray:
    """
    Convert conductances to normalized weights (0-1).
    
    Args:
        g: Conductance matrix
        g_min: Minimum conductance
        g_max: Maximum conductance
        
    Returns:
        w: Weight matrix [0, 1]
    """
    return (g - g_min) / (g_max - g_min)