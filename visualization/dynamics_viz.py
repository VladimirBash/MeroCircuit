# visualization/dynamics_viz.py

"""
Visualization tools for voltage dynamics in memristive networks.

Provides static plots, network graphs, and animations of relaxation dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
import networkx as nx
from typing import Optional, List, Callable, Tuple


# ============================================================
# TIME SERIES PLOTS
# ============================================================

def plot_voltage_evolution(V_history: np.ndarray,
                           node_labels: Optional[List[str]] = None,
                           node_groups: Optional[dict] = None,
                           dt: float = 0.01,
                           figsize: tuple = (10, 6),
                           title: str = 'Voltage Evolution') -> plt.Figure:
    """
    Plot voltage evolution over time for all nodes.
    
    Args:
        V_history: (n_steps, n_nodes) voltage array
        node_labels: Optional labels for each node
        node_groups: Optional dict {group_name: [node_indices]} for color coding
        dt: Time step
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure
    """
    n_steps, n_nodes = V_history.shape
    time = np.arange(n_steps) * dt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if node_groups is None:
        # Plot all nodes with default colors
        for i in range(n_nodes):
            label = node_labels[i] if node_labels else f'Node {i}'
            ax.plot(time, V_history[:, i], alpha=0.7, linewidth=1.5, label=label)
    else:
        # Plot by groups with distinct colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(node_groups)))
        
        for (group_name, node_indices), color in zip(node_groups.items(), colors):
            for i in node_indices:
                label = f'{group_name} {i}' if len(node_indices) > 1 else group_name
                ax.plot(time, V_history[:, i], color=color, alpha=0.7, 
                       linewidth=2 if 'input' in group_name.lower() else 1.5,
                       label=label if i == node_indices[0] else '')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Voltage (V)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    
    if n_nodes <= 20:  # Only show legend if not too many nodes
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
    
    return fig


# ============================================================
# NETWORK GRAPHS
# ============================================================

def plot_network_graph(adjacency: np.ndarray,
                      V: np.ndarray,
                      conductances: Optional[np.ndarray] = None,
                      node_labels: Optional[List[str]] = None,
                      node_groups: Optional[dict] = None,
                      figsize: tuple = (10, 8),
                      layout: str = 'spring') -> plt.Figure:
    """
    Plot network as graph with voltage-coded node colors.
    
    Args:
        adjacency: (n_nodes, n_nodes) adjacency matrix
        V: (n_nodes,) current voltages
        conductances: Optional (n_nodes, n_nodes) conductance matrix for edge widths
        node_labels: Optional node labels
        node_groups: Optional dict {group_name: [node_indices]} for positioning
        figsize: Figure size
        layout: 'spring', 'circular', 'layered', or 'manual'
        
    Returns:
        matplotlib Figure
    """
    n_nodes = len(V)
    G = nx.from_numpy_array(adjacency)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'layered' and node_groups is not None:
        pos = _layered_layout(node_groups, n_nodes)
    else:
        pos = nx.spring_layout(G)
    
    # Node colors based on voltage
    node_colors = V
    vmin, vmax = V.min(), V.max()
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                   node_size=500, cmap='coolwarm',
                                   vmin=vmin, vmax=vmax, ax=ax)
    
    # Draw edges with width based on conductance
    if conductances is not None:
        edges = G.edges()
        weights = [conductances[u, v] for u, v in edges]
        weights_normalized = np.array(weights) / np.max(weights) * 3
        nx.draw_networkx_edges(G, pos, width=weights_normalized, alpha=0.5, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    
    # Draw labels
    if node_labels:
        labels = {i: node_labels[i] for i in range(n_nodes)}
    else:
        labels = {i: f'{i}\n{V[i]:.2f}V' for i in range(n_nodes)}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Voltage (V)', fontsize=11)
    
    ax.set_title('Network Graph (Voltage Coded)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def plot_current_flows(adjacency: np.ndarray,
                      V: np.ndarray,
                      conductances: np.ndarray,
                      iv_function: Callable,
                      node_groups: Optional[dict] = None,
                      threshold: float = 0.001,
                      figsize: tuple = (10, 8)) -> plt.Figure:
    """
    Plot network with current flows shown as arrows.
    
    Args:
        adjacency: (n_nodes, n_nodes) adjacency matrix
        V: (n_nodes,) voltages
        conductances: (n_nodes, n_nodes) conductance matrix
        iv_function: I-V characteristic function
        node_groups: Optional grouping for layered layout
        threshold: Minimum current to display (relative to max)
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    n_nodes = len(V)
    G = nx.from_numpy_array(adjacency)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    if node_groups is not None:
        pos = _layered_layout(node_groups, n_nodes)
    else:
        pos = nx.spring_layout(G, k=1)
    
    # Compute currents
    currents = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency[i, j]:
                V_drop = V[j] - V[i]
                currents[i, j] = iv_function(V_drop, conductances[i, j])
    
    max_current = np.abs(currents).max()
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=V, node_size=500, 
                          cmap='coolwarm', ax=ax)
    
    # Draw current arrows (only for upper triangle to avoid duplicates)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adjacency[i, j]:
                # Current from i to j
                I_ij = currents[i, j]
                
                if np.abs(I_ij) > threshold * max_current:
                    # Arrow direction: current flows from i to j if I_ij > 0
                    if I_ij > 0:
                        start, end = i, j
                    else:
                        start, end = j, i
                    
                    x_start, y_start = pos[start]
                    x_end, y_end = pos[end]
                    
                    # Arrow properties based on current magnitude
                    width = np.abs(I_ij) / max_current * 5
                    alpha = min(np.abs(I_ij) / max_current * 2, 1.0)
                    
                    arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                          arrowstyle='->', mutation_scale=20,
                                          linewidth=width, alpha=alpha,
                                          color='black', zorder=1)
                    ax.add_patch(arrow)
    
    ax.set_title('Current Flows (Arrow thickness ∝ current)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def _layered_layout(node_groups: dict, n_nodes: int) -> dict:
    """Create layered layout for autoencoder topology."""
    pos = {}
    
    group_names = list(node_groups.keys())
    n_layers = len(group_names)
    
    for layer_idx, (group_name, node_indices) in enumerate(node_groups.items()):
        x = layer_idx / (n_layers - 1) if n_layers > 1 else 0.5
        n_in_layer = len(node_indices)
        
        for i, node_idx in enumerate(node_indices):
            y = (i - (n_in_layer - 1) / 2) / max(n_in_layer, 3)
            pos[node_idx] = (x, y)
    
    return pos


# ============================================================
# ANIMATIONS
# ============================================================

def animate_relaxation(V_history: np.ndarray,
                      adjacency: np.ndarray,
                      conductances: np.ndarray,
                      iv_function: Callable,
                      node_groups: Optional[dict] = None,
                      nodes_to_plot: Optional[List[int]] = None,
                      phase_transition_frame: Optional[int] = None,
                      dt: float = 0.01,
                      max_frames: int = 200,
                      skip_frames: int = 10,
                      figsize: tuple = (12, 5),
                      output_file: Optional[str] = None,
                      fps: int = 30) -> animation.FuncAnimation:
    """
    Create animation of voltage relaxation.
    
    Args:
        V_history: (n_steps, n_nodes) voltage history
        adjacency: (n_nodes, n_nodes) adjacency matrix
        conductances: (n_nodes, n_nodes) conductance matrix
        iv_function: I-V characteristic
        node_groups: Optional node grouping for layout
        nodes_to_plot: List of node indices to show in time series (default: all)
        phase_transition_frame: Optional frame number where phase changes (draws vertical line)
        dt: Time step
        max_frames: Maximum number of frames in animation (auto-adjusts skip_frames)
        skip_frames: Initial skip factor (will be increased if needed)
        figsize: Figure size
        output_file: Optional filename to save (e.g., 'relaxation.gif' or '.mp4')
        fps: Frames per second for saved animation
        
    Returns:
        FuncAnimation object
    """
    n_steps, n_nodes = V_history.shape
    
    # Auto-adjust skip_frames to stay under max_frames
    actual_skip = skip_frames
    while n_steps // actual_skip > max_frames:
        actual_skip += 5
    
    if actual_skip != skip_frames:
        print(f"  Adjusted skip_frames: {skip_frames} → {actual_skip} (to fit {max_frames} frames)")
    
    V_history = V_history[::actual_skip]  # Subsample
    n_frames = len(V_history)
    
    # Adjust phase transition frame for subsampling
    if phase_transition_frame is not None:
        phase_transition_frame = phase_transition_frame // actual_skip
    
    # Default: plot all nodes
    if nodes_to_plot is None:
        nodes_to_plot = list(range(n_nodes))
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Layout for network graph
    G = nx.from_numpy_array(adjacency)
    if node_groups is not None:
        pos = _layered_layout(node_groups, n_nodes)
    else:
        pos = nx.spring_layout(G, k=1, seed=42)
    
    # Initialize time series plot
    lines = []
    colors = plt.cm.tab10(np.linspace(0, 1, len(nodes_to_plot)))
    
    for idx, (node_idx, color) in enumerate(zip(nodes_to_plot, colors)):
        line, = ax1.plot([], [], color=color, linewidth=2, 
                        label=f'Node {node_idx}', alpha=0.8)
        lines.append((node_idx, line))
    
    time_marker, = ax1.plot([], [], 'ro', markersize=6)
    
    ax1.set_xlim(0, n_steps * dt)
    ax1.set_ylim(V_history[:, nodes_to_plot].min() - 0.1, 
                 V_history[:, nodes_to_plot].max() + 0.1)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Voltage (V)', fontsize=11)
    ax1.set_title('Voltage Evolution', fontsize=12)
    ax1.grid(alpha=0.3)
    
    # Add phase transition line if specified
    if phase_transition_frame is not None:
        transition_time = phase_transition_frame * dt * actual_skip
        ax1.axvline(transition_time, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label='Phase transition')
    
    ax1.legend(loc='upper right', fontsize=9)
    
    # Network graph elements
    vmin, vmax = V_history.min(), V_history.max()
    
    node_collection = nx.draw_networkx_nodes(G, pos, node_color=[0]*n_nodes,
                                            node_size=500, cmap='coolwarm',
                                            vmin=vmin, vmax=vmax, ax=ax2)
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax2)
    
    # Node labels
    labels = {i: str(i) for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax2)
    
    ax2.set_title('Network State', fontsize=12)
    ax2.axis('off')
    
    time_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                        verticalalignment='top', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        for _, line in lines:
            line.set_data([], [])
        time_marker.set_data([], [])
        return [line for _, line in lines] + [time_marker, node_collection, time_text]
    
    def animate_frame(frame):
        # Update time series for selected nodes
        t = np.arange(frame + 1) * dt * actual_skip
        
        for node_idx, line in lines:
            V_node = V_history[:frame+1, node_idx]
            line.set_data(t, V_node)
        
        # Update marker (use first plotted node)
        if len(lines) > 0:
            node_idx, _ = lines[0]
            time_marker.set_data([t[-1]], [V_history[frame, node_idx]])
        
        # Update network graph
        V_current = V_history[frame]
        node_collection.set_array(V_current)
        
        # Update time text with phase info
        current_time = frame * dt * actual_skip
        if phase_transition_frame and frame < phase_transition_frame:
            phase_label = "Free (β=0)"
        elif phase_transition_frame and frame >= phase_transition_frame:
            phase_label = "Clamped (β>0)"
        else:
            phase_label = ""
        
        time_text.set_text(f't = {current_time:.3f} s\n{phase_label}\nFrame {frame+1}/{n_frames}')
        
        return [line for _, line in lines] + [time_marker, node_collection, time_text]
    
    anim = animation.FuncAnimation(fig, animate_frame, init_func=init,
                                  frames=n_frames, interval=1000/fps,
                                  blit=True, repeat=True)
    
    if output_file:
        print(f"  Saving animation with {n_frames} frames...")
        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps)
        elif output_file.endswith('.mp4'):
            anim.save(output_file, writer='ffmpeg', fps=fps)
        print(f"  Animation saved to {output_file}")
    
    return anim
