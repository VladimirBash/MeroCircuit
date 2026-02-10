# neuromorphic_autoencoder/tests/test_bars_stripes.py

import numpy as np
import pytest
from datasets.bars_stripes import BarsAndStripes, generate_bars_stripes


def test_pattern_count():
    """Test correct number of patterns for different N."""
    for N in [2, 3, 4, 5]:
        dataset = BarsAndStripes(N)
        expected = 2**N + 2**N - 2  # Formula from paper
        assert dataset.n_patterns == expected, f"N={N}: expected {expected}, got {dataset.n_patterns}"


def test_no_invalid_patterns():
    """Ensure all generated patterns are valid bars or stripes."""
    dataset = BarsAndStripes(N=4)
    
    for pattern in dataset.get_all_patterns():
        pattern_type = dataset.get_pattern_type(pattern)
        assert pattern_type in ['bar', 'stripe', 'both'], f"Invalid pattern found: {pattern_type}"


def test_voltage_conversion():
    """Test binary <-> voltage conversion."""
    dataset = BarsAndStripes(N=3, voltage_on=1.5, voltage_off=-0.5)
    
    binary = dataset.get_all_patterns()
    voltages = dataset.to_voltages(binary)
    recovered = dataset.from_voltages(voltages)
    
    np.testing.assert_array_equal(binary, recovered)


def test_sampling():
    """Test random sampling."""
    dataset = BarsAndStripes(N=4)
    
    # Sample with replacement
    samples = dataset.sample(100, replace=True, random_state=42)
    assert samples.shape == (100, 4, 4)
    
    # Sample without replacement (should fail if n > total)
    with pytest.raises(ValueError):
        dataset.sample(1000, replace=False)


def test_train_test_split():
    """Test dataset splitting."""
    dataset = BarsAndStripes(N=4)
    train, test = dataset.split_train_test(test_fraction=0.2, random_state=42)
    
    assert len(train) + len(test) == dataset.n_patterns
    assert len(test) >= 1  # At least one test sample


def test_pattern_classification():
    """Test pattern type detection."""
    dataset = BarsAndStripes(N=3)
    
    # All-zeros (both)
    assert dataset.get_pattern_type(np.zeros((3, 3))) == 'both'
    
    # All-ones (both)
    assert dataset.get_pattern_type(np.ones((3, 3))) == 'both'
    
    # Vertical bar
    bar = np.array([[1, 0, 1],
                    [1, 0, 1],
                    [1, 0, 1]])
    assert dataset.get_pattern_type(bar) == 'bar'
    
    # Horizontal stripe
    stripe = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [1, 1, 1]])
    assert dataset.get_pattern_type(stripe) == 'stripe'
    
    # Invalid (mixed)
    invalid = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    assert dataset.get_pattern_type(invalid) == 'invalid'


def test_convenience_function():
    """Test quick generation function."""
    patterns = generate_bars_stripes(N=3, n_samples=10, as_voltages=True, random_state=42)
    assert patterns.shape == (10, 9)  # 3*3 flattened
    assert np.all((patterns == 0.0) | (patterns == 1.0))  # Binary voltages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
