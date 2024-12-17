import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional


def plot_simulation_results(
        simulation_data: dict,
        param_set_idx: int = 0,
        figsize: tuple = None,
        save_path: Optional[str] = None,
        share_y_by_circuit: bool = True
) -> plt.Figure:
    """
    Plot simulation results vs experimental data using pre-computed simulation results

    Args:
        simulation_data: Output from CircuitFitter.simulate_parameters()
        param_set_idx: Index of the parameter set to plot
        figsize: Figure size (width, height). If None, automatically calculated
        save_path: Optional path to save the figure
        share_y_by_circuit: Whether to share y-axis across conditions within each circuit
    """
    # Calculate dimensions
    n_circuits = len(simulation_data)
    max_conditions = max(len(data['config'].condition_params)
                         for data in simulation_data.values())

    # Set figure size if not provided
    if figsize is None:
        figsize = (4 * max_conditions, 3 * n_circuits)

    # Create figure with a grid of subplots
    fig = plt.figure(figsize=figsize)

    # Setup style
    plt.style.use('seaborn')
    colors = sns.color_palette("husl", 3)

    # Track y-axis limits for shared axes
    circuit_y_limits = {}

    # First pass to determine y-axis limits if sharing
    if share_y_by_circuit:
        for circuit_idx, data in simulation_data.items():
            y_min, y_max = float('inf'), float('-inf')

            for condition_name in data['config'].condition_params.keys():
                # Get experimental data y limits
                exp_data = data['config'].experimental_data[
                    data['config'].experimental_data['condition'] == condition_name
                    ]
                y_min = min(y_min, exp_data['fluorescence'].min())
                y_max = max(y_max, exp_data['fluorescence'].max())

                # Get simulation y limits
                condition_mask = data['combined_params']['condition'] == condition_name
                sim_indices = data['combined_params'].index[condition_mask]
                param_indices = data['combined_params'].loc[condition_mask, 'param_set_idx']
                param_sim_idx = sim_indices[param_indices == param_set_idx][0]
                sim_values = data['simulation_results'].observables[param_sim_idx]['obs_Protein_GFP'] * 100
                y_min = min(y_min, np.min(sim_values))
                y_max = max(y_max, np.max(sim_values))

            # Add some padding
            y_range = y_max - y_min
            y_min -= 0.05 * y_range
            y_max += 0.05 * y_range
            circuit_y_limits[circuit_idx] = (y_min, y_max)

    # Create and plot subplots
    for circuit_idx, data in simulation_data.items():
        config = data['config']
        combined_params = data['combined_params']
        simulation_results = data['simulation_results']

        for condition_idx, condition_name in enumerate(config.condition_params.keys()):
            # Create subplot
            ax = plt.subplot(n_circuits, max_conditions,
                             circuit_idx * max_conditions + condition_idx + 1)

            # Get experimental data
            exp_data = config.experimental_data[
                config.experimental_data['condition'] == condition_name
                ]

            # Plot experimental replicates as scatter points
            ax.scatter(exp_data['time'], exp_data['fluorescence'],
                       color=colors[0], alpha=0.6, s=30,
                       label='Experimental data')

            # Get simulation results
            condition_mask = combined_params['condition'] == condition_name
            sim_indices = combined_params.index[condition_mask]
            param_indices = combined_params.loc[condition_mask, 'param_set_idx']
            param_sim_idx = sim_indices[param_indices == param_set_idx][0]
            sim_values = simulation_results.observables[param_sim_idx]['obs_Protein_GFP'] * 100

            # Plot simulation
            ax.plot(config.tspan, sim_values,
                    color=colors[1], label='Simulation',
                    linestyle='-', linewidth=2)

            # Set y-axis limits if sharing
            if share_y_by_circuit:
                ax.set_ylim(circuit_y_limits[circuit_idx])

            # Customize subplot
            ax.set_title(condition_name)
            ax.set_xlabel('Time')
            if condition_idx == 0:  # Only show y-label for first condition in each circuit
                ax.set_ylabel('Fluorescence')
                # Add circuit name to the left of the row
                ax.text(-0.5, 0.5, config.name,  # Changed from f'Circuit {config.model.name}'
                        rotation=90, transform=ax.transAxes,
                        verticalalignment='center',
                        horizontalalignment='right',
                        fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Adjust layout to make room for circuit names
    plt.subplots_adjust(left=0.15)

    # Save only if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()

    return fig