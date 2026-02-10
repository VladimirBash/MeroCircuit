# visualization/plotting.py

import numpy as np
import matplotlib.pyplot as plt


def plot_patterns(patterns, titles=None, n_cols=8, figsize=None):
    """Plot grid of binary patterns."""
    n = len(patterns)
    
    # Reshape if flattened
    if patterns.ndim == 2:
        N = int(np.sqrt(patterns.shape[1]))
        patterns = patterns.reshape(n, N, N)
    
    n_rows = (n + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (n_cols * 1.5, n_rows * 1.5)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    
    for i in range(n):
        axes[i].imshow(patterns[i], cmap='binary', vmin=0, vmax=1)
        axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(titles[i], fontsize=10)
    
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_reconstruction(inputs, outputs, n_samples=8):
    """Plot input vs output side-by-side."""
    n = min(n_samples, len(inputs))
    
    if inputs.ndim == 2:
        N = int(np.sqrt(inputs.shape[1]))
        inputs = inputs[:n].reshape(n, N, N)
        outputs = outputs[:n].reshape(n, N, N)
    
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    
    for i in range(n):
        axes[0, i].imshow(inputs[i], cmap='binary')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(outputs[i], cmap='binary')
        axes[1, i].axis('off')
        
        error = np.mean(np.abs(inputs[i] - outputs[i]))
        axes[1, i].set_title(f'{error:.2f}', fontsize=9)
    
    axes[0, 0].set_ylabel('Input', fontsize=11)
    axes[1, 0].set_ylabel('Output', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_loss(losses):
    """Plot training loss."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses, 'b-o', linewidth=2, markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt.gcf()
