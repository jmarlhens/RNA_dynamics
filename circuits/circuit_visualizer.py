import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


class CircuitVisualizer:
    """
    A class for visualizing circuit simulation results.
    Handles both single and multiple simulation cases correctly.
    """

    @staticmethod
    def plot_simulation_results(result, circuit_name, t_span, pulse_config=None):
        """
        Plot simulation results for a circuit.

        Parameters:
        -----------
        result : SimulationResult
            The PySB simulation results object
        circuit_name : str
            Name of the circuit for plot titles
        t_span : array
            Time span used for simulation
        pulse_config : dict, optional
            Pulse configuration if the circuit uses pulses

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # Determine if we're dealing with a single simulation or multiple simulations
        if isinstance(result.observables, list):
            # Multiple simulations
            all_observable_names = result.observables[0].dtype.names
        else:
            # Single simulation
            all_observable_names = result.observables.dtype.names

        # Filter for protein and RNA observables
        protein_observables = [name for name in all_observable_names if name.startswith('obs_Protein_')]
        rna_observables = [name for name in all_observable_names if name.startswith('obs_RNA_')]

        # Determine if we have a pulse configuration
        if pulse_config is not None:
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Create protein plot
            CircuitVisualizer._plot_observables(axs[0], result, protein_observables, t_span)
            axs[0].set_ylabel('Protein concentration')
            axs[0].set_title(f'{circuit_name} - Protein Concentrations')
            axs[0].grid(True)

            # Create RNA plot
            CircuitVisualizer._plot_observables(axs[1], result, rna_observables, t_span)
            axs[1].set_ylabel('RNA concentration')
            axs[1].set_title('RNA Concentrations')
            axs[1].grid(True)

            # Create pulse profile plot
            CircuitVisualizer._plot_pulse_profile(axs[2], t_span, pulse_config)
            axs[2].set_ylabel('Concentration')
            axs[2].set_xlabel('Time')
            axs[2].set_title('Plasmid Pulse Profile')
            axs[2].grid(True)

        else:
            fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Create protein plot
            CircuitVisualizer._plot_observables(axs[0], result, protein_observables, t_span)
            axs[0].set_ylabel('Protein concentration')
            axs[0].set_title(f'{circuit_name} - Protein Concentrations')
            axs[0].grid(True)

            # Create RNA plot
            CircuitVisualizer._plot_observables(axs[1], result, rna_observables, t_span)
            axs[1].set_ylabel('RNA concentration')
            axs[1].set_xlabel('Time')
            axs[1].set_title('RNA Concentrations')
            axs[1].grid(True)

        plt.tight_layout()
        plt.show()

        return fig

    @staticmethod
    def _plot_observables(ax, result, observable_names, t_span):
        """
        Helper method to plot observables.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        result : SimulationResult
            The simulation results
        observable_names : list
            List of observable names to plot
        t_span : array
            Time span used for simulation
        """
        # Use a colormap for multiple simulations
        colors = plt.cm.tab10.colors

        # Check for multiple simulations
        multiple_simulations = isinstance(result.observables, list)

        for obs_idx, obs_name in enumerate(observable_names):
            # Choose color for this observable
            base_color = colors[obs_idx % len(colors)]

            if multiple_simulations:
                # For multiple simulations, extract data from each simulation
                n_sims = len(result.observables)

                # For many simulations, use alpha to prevent overcrowding
                alpha = max(0.7 / np.sqrt(n_sims), 0.1)

                # Plot each simulation with the same color but varying alpha
                for i in range(n_sims):
                    sim_data = result.observables[i][obs_name]
                    if i == 0:  # Only add label for the first one
                        ax.plot(t_span, sim_data, color=base_color, alpha=alpha,
                                label=CircuitVisualizer._get_display_name(obs_name))
                    else:
                        ax.plot(t_span, sim_data, color=base_color, alpha=alpha)

                # Add a thicker line for the mean if we have many simulations
                if n_sims > 3:
                    # Calculate mean across all simulations
                    mean_data = np.mean([result.observables[i][obs_name] for i in range(n_sims)], axis=0)
                    ax.plot(t_span, mean_data, color=base_color, linewidth=2,
                            label=f"{CircuitVisualizer._get_display_name(obs_name)} (mean)")
            else:
                # Single simulation
                ax.plot(t_span, result.observables[obs_name], color=base_color,
                        label=CircuitVisualizer._get_display_name(obs_name))

        ax.legend()

    @staticmethod
    def _get_display_name(obs_name):
        """Helper method to get a display name for the legend."""
        if obs_name.startswith('obs_Protein_'):
            return obs_name.replace('obs_Protein_', '')
        elif obs_name.startswith('obs_RNA_'):
            return f"RNA_{obs_name.replace('obs_RNA_', '')}"
        else:
            return obs_name

    @staticmethod
    def _plot_pulse_profile(ax, t_span, pulse_config):
        """
        Helper method to plot pulse profile.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        t_span : array
            Time span used for simulation
        pulse_config : dict
            Pulse configuration
        """
        pulse_profile = []
        for t in t_span:
            if t < pulse_config["pulse_start"] or t > pulse_config["pulse_end"]:
                pulse_profile.append(pulse_config["base_concentration"])
            else:
                pulse_profile.append(pulse_config["pulse_concentration"])

        ax.plot(t_span, pulse_profile, label='Pulse Profile', color='red', linewidth=2)
        ax.legend()

    @staticmethod
    def plot_parameter_comparison(result, t_span, observable, param_values, param_name,
                                  title=None, xlabel='Time', ylabel=None):
        """
        Create a comparison plot for parameter sweeps.

        Parameters:
        -----------
        result : SimulationResult
            The PySB simulation results object
        t_span : array
            Time span used for simulation
        observable : str
            Name of the observable to plot
        param_values : dict or list
            Values of the parameter being varied
        param_name : str
            Name of the parameter being varied
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label

        Returns:
        --------
        fig : matplotlib Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract parameter values to display in legend
        if isinstance(param_values, dict):
            values = param_values[param_name]
        else:
            values = param_values

        # Create colormap for different parameter values
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=min(values), vmax=max(values))

        # Check if we have multiple simulations
        if isinstance(result.observables, list):
            # Multiple simulations case
            n_sims = len(result.observables)

            # Plot each simulation with color based on parameter value
            for i, value in enumerate(values):
                if i < n_sims:  # Ensure we don't go out of bounds
                    sim_data = result.observables[i][observable]
                    color = cmap(norm(value))
                    ax.plot(t_span, sim_data, color=color,
                            label=f'{param_name}={value}')
        else:
            # Single simulation case - just plot it
            ax.plot(t_span, result.observables[observable],
                    label=CircuitVisualizer._get_display_name(observable))

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel else CircuitVisualizer._get_display_name(observable))
        ax.set_title(title if title else f'Effect of {param_name} on {CircuitVisualizer._get_display_name(observable)}')

        # Add a colorbar instead of overcrowded legend if we have many parameter values
        if len(values) > 10:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(param_name)
        else:
            ax.legend()

        ax.grid(True)
        plt.tight_layout()
        plt.show()

        return fig

    @staticmethod
    def plot_parameter_sweep_heatmap(result, observable, param_values1, param_name1,
                                     param_values2, param_name2, metric='max',
                                     title=None, cmap='viridis'):
        """
        Create a heatmap for a 2D parameter sweep.

        Parameters:
        -----------
        result : SimulationResult
            The PySB simulation results object
        observable : str
            Name of the observable to analyze
        param_values1 : array
            Values for the first parameter (y-axis)
        param_name1 : str
            Name of the first parameter
        param_values2 : array
            Values for the second parameter (x-axis)
        param_name2 : str
            Name of the second parameter
        metric : str, optional
            Metric to use: 'max', 'mean', 'auc', etc.
        title : str, optional
            Plot title
        cmap : str, optional
            Colormap name

        Returns:
        --------
        fig : matplotlib Figure
            The generated figure
        """
        # Calculate metric values
        n_values1 = len(param_values1)
        n_values2 = len(param_values2)
        total_sims = n_values1 * n_values2

        # Check if we have multiple simulations
        if isinstance(result.observables, list):
            # Extract values based on the metric for multiple simulations
            if metric == 'max':
                values = np.array([np.max(result.observables[i][observable])
                                   for i in range(min(total_sims, len(result.observables)))])
            elif metric == 'mean':
                values = np.array([np.mean(result.observables[i][observable])
                                   for i in range(min(total_sims, len(result.observables)))])
            elif metric == 'auc':
                values = np.array([np.trapz(result.observables[i][observable])
                                   for i in range(min(total_sims, len(result.observables)))])
            else:
                raise ValueError(f"Unknown metric: {metric}")
        else:
            # Single simulation case (unlikely in a parameter sweep)
            if metric == 'max':
                values = np.array([np.max(result.observables[observable])])
            elif metric == 'mean':
                values = np.array([np.mean(result.observables[observable])])
            elif metric == 'auc':
                values = np.array([np.trapz(result.observables[observable])])
            else:
                raise ValueError(f"Unknown metric: {metric}")

        # Reshape for heatmap - ensure we have the right number of values
        if len(values) == total_sims:
            value_matrix = values.reshape(n_values1, n_values2)
        else:
            # Handle case where we have fewer values than expected
            print(f"Warning: Expected {total_sims} values but got {len(values)}.")
            missing = total_sims - len(values)
            padded_values = np.pad(values, (0, missing), mode='constant', constant_values=np.nan)
            value_matrix = padded_values.reshape(n_values1, n_values2)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(value_matrix, cmap=cmap, origin='lower', aspect='auto',
                       extent=[min(param_values2), max(param_values2),
                               min(param_values1), max(param_values1)])

        # Add colorbar
        metric_label = {'max': 'Maximum', 'mean': 'Mean', 'auc': 'Area Under Curve'}
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric_label.get(metric, metric)} {CircuitVisualizer._get_display_name(observable)}')

        # Set labels and title
        ax.set_xlabel(param_name2)
        ax.set_ylabel(param_name1)
        ax.set_title(title if title else
                     f'Effect of {param_name1} and {param_name2} on {CircuitVisualizer._get_display_name(observable)}')

        plt.tight_layout()
        plt.show()

        return fig