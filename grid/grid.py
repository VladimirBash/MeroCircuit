import numpy as np
from typing import Optional


class Grid:
    def __init__(self,
                 adjacency: np.ndarray,
                 clamped_nodes: np.ndarray,
                 clamped_values: np.ndarray,
                 capacitances: Optional[np.ndarray] = None):
        
        self.capacitances = capacitances
        self.adjacency = adjacency
        self.n_nodes = len(self.adjacency)
        self.clamped_nodes = clamped_nodes
        self.clamped_values = clamped_values
        
        # Setup capacitances
        
        if capacitances is None:
            self.C = np.ones(self.n_nodes)
        elif np.isscalar(capacitances):
            self.C = np.full(self.n_nodes, capacitances)
        else:
            if len(capacitances) == self.n_nodes:
                self.C = np.array(capacitances)
            else:
                raise ValueError("Wrong capacitances dimension")

        # Checking the clamped nodes
        if len(self.clamped_nodes) > self.n_nodes:
            raise ValueError("Too many clamped nodes!")

        if len(clamped_nodes) != len(clamped_values):
            raise ValueError("The number of clamped nodes and the number of clamped values must match")

        
        # Check of the adjucency matrix
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency matrix must be square")
