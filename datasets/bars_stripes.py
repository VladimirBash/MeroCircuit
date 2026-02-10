# neuromorphic_autoencoder/datasets/bars_stripes.py

import numpy as np
from typing import List, Tuple, Optional
from itertools import product


class BarsAndStripes:
    """
    Bars & Stripes dataset generator for autoencoder tasks.
    
    Generates binary patterns on an N×N grid where each pattern is either:
    - Bars: vertical stripes (columns fully on or off)
    - Stripes: horizontal stripes (rows fully on or off)
    
    Mixed patterns (both horizontal and vertical structures) are excluded.
    
    Total number of valid patterns: 2^N + 2^N - 2
    (minus 2 because all-zeros and all-ones appear in both classes)
    """
    
    def __init__(self, N: int, voltage_on: float = 1.0, voltage_off: float = 0.0):
        """
        Initialize Bars & Stripes dataset.
        
        Args:
            N: Grid size (N×N)
            voltage_on: Voltage value for active pixels
            voltage_off: Voltage value for inactive pixels
        """
        if N < 2:
            raise ValueError(f"Grid size must be at least 2, got {N}")
        
        self.N = N
        self.voltage_on = voltage_on
        self.voltage_off = voltage_off
        
        # Generate all valid patterns once
        self._patterns = self._generate_all_patterns()
        
    def _generate_all_patterns(self) -> np.ndarray:
        """
        Generate all valid Bars & Stripes patterns.
        
        Returns:
            Array of shape (n_patterns, N, N) with binary patterns
        """
        patterns = []
        
        # Generate all bars (vertical stripes)
        for col_mask in product([0, 1], repeat=self.N):
            pattern = np.zeros((self.N, self.N), dtype=int)
            for col_idx, is_active in enumerate(col_mask):
                if is_active:
                    pattern[:, col_idx] = 1
            patterns.append(pattern)
        
        # Generate all stripes (horizontal stripes)
        for row_mask in product([0, 1], repeat=self.N):
            pattern = np.zeros((self.N, self.N), dtype=int)
            for row_idx, is_active in enumerate(row_mask):
                if is_active:
                    pattern[row_idx, :] = 1
            patterns.append(pattern)
        
        # Remove duplicates (all-zeros and all-ones appear twice)
        patterns_array = np.array(patterns)
        unique_patterns = np.unique(patterns_array.reshape(len(patterns), -1), axis=0)
        
        return unique_patterns.reshape(-1, self.N, self.N)
    
    @property
    def n_patterns(self) -> int:
        """Total number of valid patterns."""
        return len(self._patterns)
    
    def get_all_patterns(self) -> np.ndarray:
        """
        Get all valid patterns.
        
        Returns:
            Array of shape (n_patterns, N, N)
        """
        return self._patterns.copy()
    
    def get_all_flattened(self) -> np.ndarray:
        """
        Get all patterns as flattened vectors.
        
        Returns:
            Array of shape (n_patterns, N*N)
        """
        return self._patterns.reshape(self.n_patterns, -1)
    
    def sample(self, n_samples: int, replace: bool = True, 
               random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample random patterns from the dataset.
        
        Args:
            n_samples: Number of patterns to sample
            replace: Whether to sample with replacement
            random_state: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, N, N)
        """
        rng = np.random.default_rng(random_state)
        indices = rng.choice(self.n_patterns, size=n_samples, replace=replace)
        return self._patterns[indices]
    
    def sample_flattened(self, n_samples: int, replace: bool = True,
                        random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample random patterns as flattened vectors.
        
        Args:
            n_samples: Number of patterns to sample
            replace: Whether to sample with replacement
            random_state: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, N*N)
        """
        return self.sample(n_samples, replace, random_state).reshape(n_samples, -1)
    
    def to_voltages(self, patterns: np.ndarray) -> np.ndarray:
        """
        Convert binary patterns to voltage values.
        
        Args:
            patterns: Binary patterns, shape (..., N, N) or (..., N*N)
            
        Returns:
            Voltage patterns with same shape as input
        """
        return np.where(patterns, self.voltage_on, self.voltage_off)
    
    def from_voltages(self, voltages: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Convert voltage values back to binary patterns.
        
        Args:
            voltages: Voltage patterns
            threshold: Binarization threshold (default: midpoint between on/off)
            
        Returns:
            Binary patterns with same shape as input
        """
        if threshold is None:
            threshold = (self.voltage_on + self.voltage_off) / 2
        
        return (voltages > threshold).astype(int)
    
    def get_pattern_type(self, pattern: np.ndarray) -> str:
        """
        Classify a pattern as 'bar', 'stripe', 'both', or 'invalid'.
        
        Args:
            pattern: Binary pattern of shape (N, N) or flattened (N*N,)
            
        Returns:
            'bar', 'stripe', 'both' (all-zeros/all-ones), or 'invalid'
        """
        if pattern.ndim == 1:
            pattern = pattern.reshape(self.N, self.N)
        
        # Check if it's a valid bar (each column is either all 0 or all 1)
        is_bar = all(len(np.unique(pattern[:, col])) == 1 for col in range(self.N))
        
        # Check if it's a valid stripe (each row is either all 0 or all 1)
        is_stripe = all(len(np.unique(pattern[row, :])) == 1 for row in range(self.N))
        
        if is_bar and is_stripe:
            return 'both'  # all-zeros or all-ones
        elif is_bar:
            return 'bar'
        elif is_stripe:
            return 'stripe'
        else:
            return 'invalid'
    
    def split_train_test(self, test_fraction: float = 0.2, 
                         random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split dataset into train and test sets.
        
        Args:
            test_fraction: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            (train_patterns, test_patterns)
        """
        rng = np.random.default_rng(random_state)
        n_test = max(1, int(self.n_patterns * test_fraction))
        
        indices = rng.permutation(self.n_patterns)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return self._patterns[train_indices], self._patterns[test_indices]
    
    def __len__(self) -> int:
        """Number of patterns in dataset."""
        return self.n_patterns
    
    def __repr__(self) -> str:
        return (f"BarsAndStripes(N={self.N}, n_patterns={self.n_patterns}, "
                f"voltage_range=[{self.voltage_off}, {self.voltage_on}])")


# Convenience function for quick usage
def generate_bars_stripes(N: int, n_samples: Optional[int] = None, 
                         as_voltages: bool = False,
                         random_state: Optional[int] = None) -> np.ndarray:
    """
    Quick generation of Bars & Stripes patterns.
    
    Args:
        N: Grid size
        n_samples: Number of samples (None = all patterns)
        as_voltages: Return as voltages instead of binary
        random_state: Random seed
        
    Returns:
        Patterns array
    """
    dataset = BarsAndStripes(N)
    
    if n_samples is None:
        patterns = dataset.get_all_flattened()
    else:
        patterns = dataset.sample_flattened(n_samples, random_state=random_state)
    
    if as_voltages:
        patterns = dataset.to_voltages(patterns)
    
    return patterns
