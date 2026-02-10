# tests/test_visualization.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for automated testing
import matplotlib.pyplot as plt

from datasets.bars_stripes import BarsAndStripes
from visualization.plotting import plot_patterns, plot_reconstruction, plot_loss


def test_plot_patterns():
    """Test pattern plotting with real dataset."""
    dataset = BarsAndStripes(N=4)
    patterns = dataset.sample(12, random_state=42)
    
    # Create plot with pattern type labels
    titles = [dataset.get_pattern_type(p) for p in patterns]
    fig = plot_patterns(patterns, titles=titles, n_cols=4)
    
    assert fig is not None
    assert len(fig.axes) >= 12
    
    # Save for visual inspection
    fig.savefig('test_output_patterns.png', dpi=100)
    print("✓ Pattern plot saved to test_output_patterns.png")


def test_plot_all_dataset():
    """Plot entire Bars & Stripes dataset."""
    dataset = BarsAndStripes(N=4)
    patterns = dataset.get_all_patterns()
    
    titles = [dataset.get_pattern_type(p) for p in patterns]
    fig = plot_patterns(patterns, titles=titles)
    
    assert fig is not None
    assert len(fig.axes) >= dataset.n_patterns
    
    fig.savefig('test_output_full_dataset.png', dpi=100)
    print(f"✓ Full dataset ({dataset.n_patterns} patterns) saved to test_output_full_dataset.png")


def test_plot_reconstruction():
    """Test reconstruction comparison with synthetic data."""
    # Create mock input/output
    inputs = np.random.randint(0, 2, (8, 16))
    outputs = inputs.copy().astype(float)
    
    # Add some reconstruction errors
    error_mask = np.random.rand(8, 16) < 0.1  # 10% error rate
    outputs[error_mask] = 1 - outputs[error_mask]
    
    fig = plot_reconstruction(inputs, outputs, n_samples=8)
    
    assert fig is not None
    assert len(fig.axes) == 16  # 2 rows × 8 columns
    
    fig.savefig('test_output_reconstruction.png', dpi=100)
    print("✓ Reconstruction comparison saved to test_output_reconstruction.png")


def test_plot_loss():
    """Test training loss curve."""
    # Mock training loss (decreasing with noise)
    losses = [1.0 - 0.1 * i + np.random.randn() * 0.05 for i in range(20)]
    losses = np.maximum(losses, 0.1)  # Floor at 0.1
    
    fig = plot_loss(losses)
    
    assert fig is not None
    
    fig.savefig('test_output_loss.png', dpi=100)
    print("✓ Loss curve saved to test_output_loss.png")


if __name__ == "__main__":
    """Run tests and display results interactively."""
    print("Running visualization tests...\n")
    
    # Switch to interactive backend for display
    matplotlib.use('TkAgg')
    
    test_plot_patterns()
    test_plot_all_dataset()
#    test_plot_reconstruction()
#    test_plot_loss()
    
    print("\n✓ All tests passed!")
    print("Displaying plots... (close windows to continue)")
    
    # Show all figures
    plt.show()
