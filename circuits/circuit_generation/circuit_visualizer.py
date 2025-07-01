import numpy as np
import matplotlib.pyplot as plt


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
        protein_observables = [
            name for name in all_observable_names if name.startswith("obs_Protein_")
        ]
        rna_observables = [
            name for name in all_observable_names if name.startswith("obs_RNA_")
        ]

        # Determine if we have a pulse configuration
        if pulse_config is not None:
            fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex="all")

            # Create protein plot
            CircuitVisualizer._plot_observables(
                axs[0], result, protein_observables, t_span
            )
            axs[0].set_ylabel("Protein concentration")
            axs[0].set_title(f"{circuit_name} - Protein Concentrations")
            axs[0].grid(True)

            # Create RNA plot
            CircuitVisualizer._plot_observables(axs[1], result, rna_observables, t_span)
            axs[1].set_ylabel("RNA concentration")
            axs[1].set_title("RNA Concentrations")
            axs[1].grid(True)

            # Create pulse profile plot
            CircuitVisualizer._plot_pulse_profile(axs[2], t_span, pulse_config)
            axs[2].set_ylabel("Concentration")
            axs[2].set_xlabel("Time")
            axs[2].set_title("Plasmid Pulse Profile")
            axs[2].grid(True)

        else:
            fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex="all")

            # Create protein plot
            CircuitVisualizer._plot_observables(
                axs[0], result, protein_observables, t_span
            )
            axs[0].set_ylabel("Protein concentration")
            axs[0].set_title(f"{circuit_name} - Protein Concentrations")
            axs[0].grid(True)

            # Create RNA plot
            CircuitVisualizer._plot_observables(axs[1], result, rna_observables, t_span)
            axs[1].set_ylabel("RNA concentration")
            axs[1].set_xlabel("Time")
            axs[1].set_title("RNA Concentrations")
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
                        ax.plot(
                            t_span,
                            sim_data,
                            color=base_color,
                            alpha=alpha,
                            label=CircuitVisualizer._get_display_name(obs_name),
                        )
                    else:
                        ax.plot(t_span, sim_data, color=base_color, alpha=alpha)

                # Add a thicker line for the mean if we have many simulations
                if n_sims > 3:
                    # Calculate mean across all simulations
                    mean_data = np.mean(
                        [result.observables[i][obs_name] for i in range(n_sims)], axis=0
                    )
                    ax.plot(
                        t_span,
                        mean_data,
                        color=base_color,
                        linewidth=2,
                        label=f"{CircuitVisualizer._get_display_name(obs_name)} (mean)",
                    )
            else:
                # Single simulation
                ax.plot(
                    t_span,
                    result.observables[obs_name],
                    color=base_color,
                    label=CircuitVisualizer._get_display_name(obs_name),
                )

        ax.legend()

    @staticmethod
    def _get_display_name(obs_name):
        """Helper method to get a display name for the legend."""
        if obs_name.startswith("obs_Protein_"):
            return obs_name.replace("obs_Protein_", "")
        elif obs_name.startswith("obs_RNA_"):
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

        ax.plot(t_span, pulse_profile, label="Pulse Profile", color="red", linewidth=2)
        ax.legend()

    @staticmethod
    def plot_parameter_comparison(
        result,
        t_span,
        observable,
        param_values,
        param_name,
        rna_observable_name=None,
        title=None,
        xlabel="Time",
        ylabel=None,
    ):
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
        rna_observable_name : str, optional
            Name of the RNA observable to plot (if applicable)
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
        # Create a new figure
        # Create subplots conditionally
        if rna_observable_name:
            fig, (ax_protein, ax_rna) = plt.subplots(
                2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [2, 2]}
            )
        else:
            fig, ax_protein = plt.subplots(figsize=(8, 6))

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
                    ax_protein.plot(
                        t_span, sim_data, color=color, label=f"{param_name}={value}"
                    )
        else:
            # Single simulation case - just plot it
            ax_rna.plot(
                t_span,
                result.observables[observable],
                label=CircuitVisualizer._get_display_name(observable),
            )

        # Set labels and title
        ax_protein.set_xlabel(xlabel)
        ax_protein.set_ylabel(
            ylabel if ylabel else CircuitVisualizer._get_display_name(observable)
        )
        ax_protein.set_title(
            title
            if title
            else f"Effect of {param_name} on {CircuitVisualizer._get_display_name(observable)}"
        )

        # Add a colorbar instead of overcrowded legend if we have many parameter values
        if len(values) > 10:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_protein)
            cbar.set_label(param_name)
        else:
            ax_protein.legend()

        ax_protein.grid(True)

        # After protein plotting section, add RNA plotting:
        if rna_observable_name:
            # Validate RNA observable exists
            if isinstance(result.observables, list):
                available_obs = result.observables[0].dtype.names
            else:
                available_obs = result.observables.dtype.names

            if rna_observable_name not in available_obs:
                print(f"Warning: {rna_observable_name} not found in observables")
                print(
                    f"Available RNA observables: {[obs for obs in available_obs if 'RNA' in obs]}"
                )
            else:
                # Plot RNA with same logic as proteins
                if isinstance(result.observables, list):
                    for i, value in enumerate(values):
                        if i < len(result.observables):
                            rna_data = result.observables[i][rna_observable_name]
                            color = cmap(norm(value))
                            ax_rna.plot(
                                t_span,
                                rna_data,
                                color=color,
                                label=f"{param_name}={value}",
                            )
                else:
                    ax_rna.plot(
                        t_span,
                        result.observables[rna_observable_name],
                        label=rna_observable_name,
                    )

                ax_rna.set_xlabel(xlabel)
                ax_rna.set_ylabel("RNA Concentration")
                ax_rna.set_title(f"RNA: {rna_observable_name}")
                ax_rna.legend()
                ax_rna.grid(True)

        plt.tight_layout()
        plt.show()

        return fig
