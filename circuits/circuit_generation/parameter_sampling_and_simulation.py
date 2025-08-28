import numpy as np
import matplotlib.pyplot as plt
from circuits.circuit_generation.circuit_visualizer import CircuitVisualizer
from circuits.modules.base_modules import KineticsType
import math
import matplotlib.gridspec as gridspec


def _get_display_name(obs_name):
    """Helper method to get a display name for the legend."""
    if obs_name.startswith("obs_Protein_"):
        return obs_name.replace("obs_Protein_", "")
    elif obs_name.startswith("obs_RNA_"):
        return f"RNA_{obs_name.replace('obs_RNA_', '')}"
    else:
        return obs_name


def create_parameter_samples(param_name, param_values, log_scale=False):
    """
    Create parameter samples for sweeping.

    Parameters:
    -----------
    param_name : str
        Name of the parameter to sample
    param_values : list or tuple or range
        Values to sample for the parameter
    log_scale : bool, optional
        Whether the parameter should be sampled in log scale

    Returns:
    --------
    list of dict
        List of parameter dictionaries, one for each sample
    """
    samples = []

    # Convert param_values to list if it's a range
    param_values = list(param_values)

    # Generate samples
    for value in param_values:
        # If log scale, store as 10^value
        actual_value = 10**value if log_scale else value
        samples.append({param_name: actual_value})

    return samples


class ParameterSamplingManager:
    """
    Extension for CircuitManager that adds support for parameter sampling with colorcoding,
    pulse visualization, and protein degradation rate control.
    """

    def __init__(self, circuit_manager, visualizer=None):
        """
        Initialize the ParameterSamplingManager.

        Parameters:
        -----------
        circuit_manager : CircuitManager
            The circuit manager instance to extend
        visualizer : CircuitVisualizer, optional
            The circuit visualizer instance to use (creates one if None)
        """
        self.circuit_manager = circuit_manager
        self.visualizer = visualizer or CircuitVisualizer()

    def run_parameter_sweep(
        self,
        circuit_name,
        param_df,
        k_prot_deg=0.1,
        _pulse_config=None,
        kinetics_type=KineticsType.MICHAELIS_MENTEN,
        t_span=None,
        additional_params=None,
        pulse_plasmids=None,
    ):
        """
        Run a parameter sweep for a given circuit and parameter.

        Parameters:
        -----------
        circuit_name : str
            Name of the circuit to simulate
        param_df : DataFrame
            DataFrame containing parameter values to sweep
        k_prot_deg : float, optional
            Protein degradation rate (default: 0.1)
        pulse_config : dict, optional
            Pulse configuration if using pulses
        kinetics_type : KineticsType, optional
            Type of kinetics to use (default: MICHAELIS_MENTEN)
        t_span : array, optional
            Time span for simulation (creates default if None)
        additional_params : dict, optional
            Additional parameters to set for all simulations
        pulse_plasmids : list, optional
            List of plasmid names to pulse (replaces pulse_indices)

        Returns:
        --------
        tuple
            Tuple containing (simulation_results, t_span, param_values)
        """
        # Set up time span if not provided
        if t_span is None:
            t_span = np.linspace(0, 30, 301)

        # Set base parameters
        base_params = {"k_prot_deg": k_prot_deg}
        if additional_params:
            base_params.update(additional_params)

        # Use pulses if pulse_config is provided
        use_pulses = _pulse_config is not None

        # If pulse_plasmids is not provided but pulses are enabled, use backward compatibility
        # by getting the first plasmid name
        if use_pulses and pulse_plasmids is None:
            # Create a temporary circuit to get plasmid names
            _temp_circuit = self.circuit_manager.create_circuit(circuit_name)
            if len(_temp_circuit.plasmids) > 0:
                # Get the name of the first plasmid (index 0)
                first_plasmid_name = _temp_circuit.plasmids[0][0]
                pulse_plasmids = [first_plasmid_name]

        # Create the circuit with named plasmids for pulsing
        circuit = self.circuit_manager.create_circuit(
            circuit_name,
            parameters=base_params,
            use_pulses=use_pulses,
            pulse_config=_pulse_config,
            pulse_plasmids=pulse_plasmids,  # Use plasmid names instead of indices
            kinetics_type=kinetics_type,
        )

        # Run simulation with parameter sweep
        result, _ = circuit.simulate(t_span=t_span, param_values=param_df)

        return result, t_span, param_df, circuit

    def plot_parameter_sweep_with_pulse(
        self,
        circuit_name,
        param_df,
        k_prot_deg=0.1,
        pulse_configuration=None,
        kinetics_type=KineticsType.MICHAELIS_MENTEN,
        t_span=None,
        additional_params=None,
        observe_protein="obs_Protein_GFP",
        observe_rna_species=None,
        title=None,
        figure_size=(6, 10),
        save_path=None,
        show_protein=True,
        show_pulse=True,
        pulse_plasmids=None,
        scores=None,
        score_metric=None,
        use_statistical_summary=False,
        statistical_summary_type="median_percentiles",  # "median_percentiles" or "mean_std"
        percentile_bounds=(10, 90),
        ribbon_alpha=0.25,
    ):
        """
        Run parameter sweep and create visualization with optional statistical summaries.

        Parameters:
        -----------
        observe_rna_species : str or None
            Specific RNA observable to plot (e.g., 'obs_RNA_GFP'). None = no RNA subplot.
        statistical_summary_type : str
            "median_percentiles" or "mean_std" for summary statistics
        """

        # Execute parameter sweep simulation
        simulation_result, time_points, parameter_values, circuit_instance = (
            self.run_parameter_sweep(
                circuit_name,
                param_df,
                k_prot_deg,
                pulse_configuration,
                kinetics_type,
                t_span,
                additional_params,
                pulse_plasmids=pulse_plasmids,
            )
        )

        # Determine subplot configuration
        subplot_components = []
        if show_protein:
            subplot_components.append("protein")
        if observe_rna_species is not None:
            subplot_components.append("rna")
        if show_pulse:
            subplot_components.append("pulse")

        subplot_count = len(subplot_components)
        if subplot_count == 0:
            raise ValueError("At least one plot type must be shown")

        # Create figure with subplot arrangement
        figure, subplot_axes = plt.subplots(
            subplot_count, 1, figsize=figure_size, sharex=True
        )
        if subplot_count == 1:
            subplot_axes = [subplot_axes]

        current_subplot_index = 0

        # Extract observable field names from structured arrays
        observable_field_names = (
            simulation_result.observables[0].dtype.names
            if simulation_result.observables
            else []
        )

        # Compute statistical summaries for summary mode
        trajectory_statistics = None
        if use_statistical_summary:
            trajectory_statistics = (
                self._compute_structured_array_statistical_summaries(
                    simulation_result,
                    time_points,
                    statistical_summary_type,
                    percentile_bounds,
                    observe_rna_species,
                )
            )

        # Configure trajectory coloring for individual mode (no colorbar)
        trajectory_colors = [plt.cm.tab10(i % 10) for i in range(len(param_df))]

        # Render protein concentration subplot
        if show_protein:
            protein_axis = subplot_axes[current_subplot_index]

            if use_statistical_summary and "protein" in trajectory_statistics:
                protein_summary_data = trajectory_statistics["protein"]

                protein_axis.plot(
                    time_points,
                    protein_summary_data["central"],
                    color="blue",
                    linewidth=2,
                    label="Central",
                    zorder=3,
                )

                protein_axis.fill_between(
                    time_points,
                    protein_summary_data["lower_bound"],
                    protein_summary_data["upper_bound"],
                    alpha=ribbon_alpha,
                    color="blue",
                    label=self._get_summary_label(
                        statistical_summary_type, percentile_bounds
                    ),
                    zorder=1,
                )
                protein_axis.legend()
            else:
                # Render individual protein trajectories without colorbar
                for parameter_set_index, structured_observables_array in enumerate(
                    simulation_result.observables
                ):
                    for observable_field_name in observable_field_names:
                        if observable_field_name.startswith("obs_Protein_"):
                            protein_trajectory = structured_observables_array[
                                observable_field_name
                            ]
                            trajectory_color = (
                                trajectory_colors[parameter_set_index]
                                if parameter_set_index < len(trajectory_colors)
                                else plt.cm.tab10(parameter_set_index % 10)
                            )
                            protein_axis.plot(
                                time_points,
                                protein_trajectory,
                                color=trajectory_color,
                                alpha=0.15,
                                zorder=1,
                            )

            protein_axis.set_ylabel("Protein Concentration (nM)")
            protein_axis.set_title(f"Protein - {circuit_name}")
            protein_axis.grid(True, alpha=0.3)
            current_subplot_index += 1

        # Render RNA concentration subplot for specific species only
        if observe_rna_species is not None:
            rna_axis = subplot_axes[current_subplot_index]

            if use_statistical_summary and "rna" in trajectory_statistics:
                rna_summary_data = trajectory_statistics["rna"]

                rna_axis.plot(
                    time_points,
                    rna_summary_data["central"],
                    color="red",
                    linewidth=2,
                    label="Central",
                    zorder=3,
                )

                rna_axis.fill_between(
                    time_points,
                    rna_summary_data["lower_bound"],
                    rna_summary_data["upper_bound"],
                    alpha=ribbon_alpha,
                    color="red",
                    label=self._get_summary_label(
                        statistical_summary_type, percentile_bounds
                    ),
                    zorder=1,
                )
                rna_axis.legend()
            else:
                # Render individual RNA trajectories for specific species only
                for parameter_set_index, structured_observables_array in enumerate(
                    simulation_result.observables
                ):
                    if observe_rna_species in structured_observables_array.dtype.names:
                        rna_trajectory = structured_observables_array[
                            observe_rna_species
                        ]
                        trajectory_color = (
                            trajectory_colors[parameter_set_index]
                            if parameter_set_index < len(trajectory_colors)
                            else plt.cm.tab10(parameter_set_index % 10)
                        )
                        rna_axis.plot(
                            time_points,
                            rna_trajectory,
                            color=trajectory_color,
                            alpha=0.15,
                            zorder=1,
                        )

            rna_axis.set_ylabel("RNA Concentration (nM)")
            rna_axis.set_title(f"RNA ({observe_rna_species}) - {circuit_name}")
            rna_axis.grid(True, alpha=0.3)
            current_subplot_index += 1

        # Render pulse profile subplot
        if show_pulse and pulse_configuration:
            pulse_axis = subplot_axes[current_subplot_index]

            # Construct pulse concentration profile
            pulse_concentration_profile = np.zeros_like(time_points)
            pulse_start_time = pulse_configuration.get("pulse_start", 0)
            pulse_end_time = pulse_configuration.get("pulse_end", 10)
            pulse_active_concentration = pulse_configuration.get(
                "pulse_concentration", 1.0
            )
            pulse_baseline_concentration = pulse_configuration.get(
                "base_concentration", 0.0
            )

            # Apply pulse timing mask
            pulse_active_mask = (time_points >= pulse_start_time) & (
                time_points <= pulse_end_time
            )
            pulse_concentration_profile[pulse_active_mask] = pulse_active_concentration
            pulse_concentration_profile[~pulse_active_mask] = (
                pulse_baseline_concentration
            )

            pulse_axis.plot(
                time_points,
                pulse_concentration_profile,
                "g-",
                linewidth=3,
                label="Pulse",
            )
            pulse_axis.set_ylabel("Pulse Concentration (nM)")
            pulse_axis.set_title("Pulse Profile")
            pulse_axis.grid(True, alpha=0.3)
            pulse_axis.legend()

        # Configure axis labels and figure title
        subplot_axes[-1].set_xlabel("Time (min)")

        if title:
            figure.suptitle(title, fontsize=16, y=0.98)
        else:
            if use_statistical_summary:
                summary_description = self._get_summary_description(
                    statistical_summary_type, percentile_bounds
                )
                plt.suptitle(
                    f"{circuit_name} - {summary_description}", fontsize=14, y=0.98
                )
            else:
                plt.suptitle(f"{circuit_name}", fontsize=14, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return figure

    def _compute_structured_array_statistical_summaries(
        self,
        simulation_result,
        time_points,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        observe_rna_species=None,
    ):
        """
        Compute statistical summaries for structured array simulation trajectories.

        Parameters:
        -----------
        statistical_summary_type : str
            "median_percentiles" or "mean_std"
        observe_rna_species : str or None
            Specific RNA observable to include in summaries
        """
        protein_trajectory_collection = []
        rna_trajectory_collection = []

        if len(simulation_result.observables) > 0:
            observable_field_names = simulation_result.observables[0].dtype.names

            for structured_observables_array in simulation_result.observables:
                for observable_field_name in observable_field_names:
                    trajectory_data = structured_observables_array[
                        observable_field_name
                    ]

                    if observable_field_name.startswith("obs_Protein_"):
                        protein_trajectory_collection.append(trajectory_data)
                    elif observable_field_name == observe_rna_species:
                        rna_trajectory_collection.append(trajectory_data)

        statistical_summaries = {}

        if protein_trajectory_collection:
            protein_trajectory_matrix = np.array(protein_trajectory_collection)
            statistical_summaries["protein"] = self._compute_summary_statistics(
                protein_trajectory_matrix, statistical_summary_type, percentile_bounds
            )

        if rna_trajectory_collection:
            rna_trajectory_matrix = np.array(rna_trajectory_collection)
            statistical_summaries["rna"] = self._compute_summary_statistics(
                rna_trajectory_matrix, statistical_summary_type, percentile_bounds
            )

        return statistical_summaries

    def _compute_summary_statistics(
        self, trajectory_matrix, statistical_summary_type, percentile_bounds
    ):
        """Compute either median+percentiles or mean+std statistics."""
        if statistical_summary_type == "mean_std":
            trajectory_mean = np.mean(trajectory_matrix, axis=0)
            trajectory_std = np.std(trajectory_matrix, axis=0)
            return {
                "central": trajectory_mean,
                "lower_bound": trajectory_mean - trajectory_std,
                "upper_bound": trajectory_mean + trajectory_std,
            }
        else:  # median_percentiles
            lower_percentile_value, upper_percentile_value = percentile_bounds
            return {
                "central": np.median(trajectory_matrix, axis=0),
                "lower_bound": np.percentile(
                    trajectory_matrix, lower_percentile_value, axis=0
                ),
                "upper_bound": np.percentile(
                    trajectory_matrix, upper_percentile_value, axis=0
                ),
            }

    def _get_summary_label(self, statistical_summary_type, percentile_bounds):
        """Generate appropriate label for statistical summary."""
        if statistical_summary_type == "mean_std":
            return "Mean ± Std"
        else:
            return f"{percentile_bounds[0]}-{percentile_bounds[1]}% range"

    def _get_summary_description(self, statistical_summary_type, percentile_bounds):
        """Generate figure title description for statistical summary."""
        if statistical_summary_type == "mean_std":
            return "Mean ± Standard Deviation"
        else:
            return f"Median ± {percentile_bounds[0]}-{percentile_bounds[1]}% Range"

    # ADD to ParameterSamplingManager class in parameter_sampling_and_simulation.py

    def plot_all_circuits_pulse_grid(
        self,
        circuit_simulation_data,  # dict: {circuit_name: (result, t_span, param_df, pulse_plasmids)}
        pulse_configuration=None,
        observe_rna_species="obs_RNA_GFP",
        use_statistical_summary=True,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        ribbon_alpha=0.25,
        figure_size=(20, 24),
        save_path=None,
    ):
        """
        Create unified grid plot showing all circuits with their protein/RNA/pulse subplots.

        Parameters:
        -----------
        circuit_simulation_data : dict
            {circuit_name: (simulation_result, time_points, param_df, pulse_plasmids)}
        """
        import matplotlib.gridspec as gridspec

        circuit_names = list(circuit_simulation_data.keys())
        num_circuits = len(circuit_names)

        # Determine subplot structure: protein, RNA (optional), pulse
        subplot_types = ["protein"]
        if observe_rna_species is not None:
            subplot_types.append("rna")
        subplot_types.append("pulse")
        num_subplot_types = len(subplot_types)

        # Create figure with gridspec
        figure = plt.figure(figsize=figure_size)
        grid_spec = gridspec.GridSpec(
            num_circuits,
            num_subplot_types,
            figure=figure,
            hspace=0.3,
            wspace=0.3,
            top=0.95,
            bottom=0.05,
            left=0.08,
            right=0.95,
        )

        # Store all axes for potential return
        all_axes = {}

        # Process each circuit
        for circuit_idx, circuit_name in enumerate(circuit_names):
            simulation_result, time_points, param_df, _ = circuit_simulation_data[
                circuit_name
            ]

            # Extract observable field names
            observable_field_names = (
                simulation_result.observables[0].dtype.names
                if simulation_result.observables
                else []
            )

            # Compute statistical summaries if needed
            trajectory_statistics = None
            if use_statistical_summary:
                trajectory_statistics = (
                    self._compute_structured_array_statistical_summaries(
                        simulation_result,
                        time_points,
                        statistical_summary_type,
                        percentile_bounds,
                        observe_rna_species,
                    )
                )

            # Configure colors for individual trajectories
            trajectory_colors = [plt.cm.tab10(i % 10) for i in range(len(param_df))]

            circuit_axes = {}
            subplot_idx = 0

            # PROTEIN SUBPLOT
            protein_ax = figure.add_subplot(grid_spec[circuit_idx, subplot_idx])
            circuit_axes["protein"] = protein_ax

            if use_statistical_summary and "protein" in trajectory_statistics:
                protein_summary_data = trajectory_statistics["protein"]
                central_label = (
                    "Median"
                    if statistical_summary_type == "median_percentiles"
                    else "Mean"
                )

                protein_ax.plot(
                    time_points,
                    protein_summary_data["central"],
                    color="blue",
                    linewidth=2,
                    label=central_label,
                    zorder=3,
                )

                protein_ax.fill_between(
                    time_points,
                    protein_summary_data["lower_bound"],
                    protein_summary_data["upper_bound"],
                    alpha=ribbon_alpha,
                    color="blue",
                    label=self._get_summary_label(
                        statistical_summary_type, percentile_bounds
                    ),
                    zorder=1,
                )
                protein_ax.legend(fontsize=8)
            else:
                # Individual protein trajectories
                for parameter_set_index, structured_observables_array in enumerate(
                    simulation_result.observables
                ):
                    for observable_field_name in observable_field_names:
                        if observable_field_name.startswith("obs_Protein_"):
                            protein_trajectory = structured_observables_array[
                                observable_field_name
                            ]
                            trajectory_color = (
                                trajectory_colors[parameter_set_index]
                                if parameter_set_index < len(trajectory_colors)
                                else plt.cm.tab10(parameter_set_index % 10)
                            )
                            protein_ax.plot(
                                time_points,
                                protein_trajectory,
                                color=trajectory_color,
                                alpha=0.15,
                                zorder=1,
                            )

            protein_ax.set_ylabel("Protein (nM)", fontsize=10)
            protein_ax.set_title(
                f"{circuit_name} - Protein", fontsize=11, fontweight="bold"
            )
            protein_ax.grid(True, alpha=0.3)
            protein_ax.tick_params(labelsize=8)
            subplot_idx += 1

            # RNA SUBPLOT (if requested)
            if observe_rna_species is not None:
                rna_ax = figure.add_subplot(grid_spec[circuit_idx, subplot_idx])
                circuit_axes["rna"] = rna_ax

                if use_statistical_summary and "rna" in trajectory_statistics:
                    rna_summary_data = trajectory_statistics["rna"]
                    central_label = (
                        "Median"
                        if statistical_summary_type == "median_percentiles"
                        else "Mean"
                    )

                    rna_ax.plot(
                        time_points,
                        rna_summary_data["central"],
                        color="red",
                        linewidth=2,
                        label=central_label,
                        zorder=3,
                    )

                    rna_ax.fill_between(
                        time_points,
                        rna_summary_data["lower_bound"],
                        rna_summary_data["upper_bound"],
                        alpha=ribbon_alpha,
                        color="red",
                        label=self._get_summary_label(
                            statistical_summary_type, percentile_bounds
                        ),
                        zorder=1,
                    )
                    rna_ax.legend(fontsize=8)
                else:
                    # Individual RNA trajectories
                    for parameter_set_index, structured_observables_array in enumerate(
                        simulation_result.observables
                    ):
                        if (
                            observe_rna_species
                            in structured_observables_array.dtype.names
                        ):
                            rna_trajectory = structured_observables_array[
                                observe_rna_species
                            ]
                            trajectory_color = (
                                trajectory_colors[parameter_set_index]
                                if parameter_set_index < len(trajectory_colors)
                                else plt.cm.tab10(parameter_set_index % 10)
                            )
                            rna_ax.plot(
                                time_points,
                                rna_trajectory,
                                color=trajectory_color,
                                alpha=0.15,
                                zorder=1,
                            )

                rna_ax.set_ylabel("RNA (nM)", fontsize=10)
                rna_ax.set_title(
                    f"{circuit_name} - RNA ({observe_rna_species})",
                    fontsize=11,
                    fontweight="bold",
                )
                rna_ax.grid(True, alpha=0.3)
                rna_ax.tick_params(labelsize=8)
                subplot_idx += 1

            # PULSE SUBPLOT
            pulse_ax = figure.add_subplot(grid_spec[circuit_idx, subplot_idx])
            circuit_axes["pulse"] = pulse_ax

            if pulse_configuration:
                # Create pulse profile
                pulse_profile = np.zeros_like(time_points)
                pulse_start_time = pulse_configuration.get("pulse_start", 0)
                pulse_end_time = pulse_configuration.get("pulse_end", 10)
                pulse_active_concentration = pulse_configuration.get(
                    "pulse_concentration", 1.0
                )
                pulse_baseline_concentration = pulse_configuration.get(
                    "base_concentration", 0.0
                )

                pulse_active_mask = (time_points >= pulse_start_time) & (
                    time_points <= pulse_end_time
                )
                pulse_profile[pulse_active_mask] = pulse_active_concentration
                pulse_profile[~pulse_active_mask] = pulse_baseline_concentration

                pulse_ax.plot(
                    time_points, pulse_profile, "g-", linewidth=3, label="Pulse"
                )
                pulse_ax.legend(fontsize=8)

            pulse_ax.set_ylabel("Pulse (nM)", fontsize=10)
            pulse_ax.set_title(
                f"{circuit_name} - Pulse", fontsize=11, fontweight="bold"
            )
            pulse_ax.grid(True, alpha=0.3)
            pulse_ax.tick_params(labelsize=8)

            # Set x-label only for bottom row
            if circuit_idx == num_circuits - 1:
                for ax_name, ax in circuit_axes.items():
                    ax.set_xlabel("Time (min)", fontsize=10)

            all_axes[circuit_name] = circuit_axes

        # Overall figure title
        summary_description = (
            "Mean ± Std"
            if statistical_summary_type == "mean_std"
            else f"Median ± {percentile_bounds[0]}-{percentile_bounds[1]}%"
        )
        mode_description = (
            "Statistical Summary"
            if use_statistical_summary
            else "Individual Trajectories"
        )

        figure.suptitle(
            f"All Circuits Pulse Response - {mode_description} ({summary_description})",
            fontsize=16,
            fontweight="bold",
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Grid plot saved: {save_path}")

        # plt.show()
        return figure, all_axes

    def _apply_equilibrium_baseline_correction(
        self,
        observables_data,
        full_time_span,
        equilibration_time,
        target_observable_names=None,
    ):
        """
        Apply baseline correction by subtracting concentration at equilibration time.
        Preserves structured array format for compatibility with downstream code.

        Parameters:
        -----------
        observables_data : list or structured array
            Observable data from simulation results
        full_time_span : np.ndarray
            Complete time span including equilibration period
        equilibration_time : float
            Time point to use as baseline (typically end of equilibration)
        target_observable_names : list or None
            Specific observables to correct, None applies to all protein/RNA observables

        Returns:
        --------
        Baseline-corrected observables data in same structured array format as input
        """
        # Find equilibration time index
        equilibration_time_index = np.searchsorted(full_time_span, equilibration_time)
        equilibration_time_index = min(
            equilibration_time_index, len(full_time_span) - 1
        )

        if isinstance(observables_data, list):
            # Multiple parameter sets - each is a structured array
            baseline_corrected_observables = []

            for parameter_set_observables in observables_data:
                # Create copy to avoid modifying original
                corrected_observables_array = parameter_set_observables.copy()

                for observable_name in parameter_set_observables.dtype.names:
                    trajectory_data = parameter_set_observables[observable_name]

                    # Apply correction to protein and RNA observables
                    should_correct = False
                    if target_observable_names is None:
                        should_correct = observable_name.startswith(
                            "obs_Protein_"
                        ) or observable_name.startswith("obs_RNA_")
                    else:
                        should_correct = observable_name in target_observable_names

                    if should_correct:
                        baseline_value = trajectory_data[equilibration_time_index]
                        corrected_observables_array[observable_name] = (
                            trajectory_data - baseline_value
                        )

                baseline_corrected_observables.append(corrected_observables_array)

            return baseline_corrected_observables

        else:
            # Single parameter set - structured array
            corrected_observables_array = observables_data.copy()

            for observable_name in observables_data.dtype.names:
                trajectory_data = observables_data[observable_name]

                # Apply correction to protein and RNA observables
                should_correct = False
                if target_observable_names is None:
                    should_correct = observable_name.startswith(
                        "obs_Protein_"
                    ) or observable_name.startswith("obs_RNA_")
                else:
                    should_correct = observable_name in target_observable_names

                if should_correct:
                    baseline_value = trajectory_data[equilibration_time_index]
                    corrected_observables_array[observable_name] = (
                        trajectory_data - baseline_value
                    )

            return corrected_observables_array

    def _compute_baseline_corrected_statistical_summaries(
        self,
        observables_data,
        full_time_span,
        equilibration_time,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        observe_rna_species=None,
        subtract_equilibrium_baseline=False,
    ):
        """
        Compute statistical summaries with optional baseline correction.
        Handles structured arrays consistently.
        """
        # Apply baseline correction if requested
        if subtract_equilibrium_baseline:
            target_observables = None
            if observe_rna_species:
                target_observables = [observe_rna_species]

            corrected_observables = self._apply_equilibrium_baseline_correction(
                observables_data, full_time_span, equilibration_time, target_observables
            )
        else:
            corrected_observables = observables_data

        # Extract trajectories for statistical computation
        protein_trajectory_collection = []
        rna_trajectory_collection = []

        if isinstance(corrected_observables, list):
            # Multiple parameter sets - each is a structured array
            for observables_array in corrected_observables:
                for observable_name in observables_array.dtype.names:
                    if observable_name.startswith("obs_Protein_"):
                        protein_trajectory_collection.append(
                            observables_array[observable_name]
                        )
                    elif observable_name == observe_rna_species:
                        rna_trajectory_collection.append(
                            observables_array[observable_name]
                        )
        else:
            # Single parameter set - structured array
            for observable_name in corrected_observables.dtype.names:
                if observable_name.startswith("obs_Protein_"):
                    protein_trajectory_collection.append(
                        corrected_observables[observable_name]
                    )
                elif observable_name == observe_rna_species:
                    rna_trajectory_collection.append(
                        corrected_observables[observable_name]
                    )

        statistical_summaries = {}

        if protein_trajectory_collection:
            protein_trajectory_matrix = np.array(protein_trajectory_collection)
            statistical_summaries["protein"] = (
                self._compute_trajectory_statistical_measures(
                    protein_trajectory_matrix,
                    statistical_summary_type,
                    percentile_bounds,
                )
            )

        if rna_trajectory_collection:
            rna_trajectory_matrix = np.array(rna_trajectory_collection)
            statistical_summaries["rna"] = (
                self._compute_trajectory_statistical_measures(
                    rna_trajectory_matrix, statistical_summary_type, percentile_bounds
                )
            )

        return statistical_summaries

    def plot_parameter_sweep_with_pulse_focused_display(
        self,
        equilibrated_simulation_result,
        full_equilibration_time_span,
        circuit_name,
        pulse_configuration,
        pulse_plasmids,
        pre_pulse_display_minutes=10,
        post_pulse_display_minutes=30,
        observe_protein="obs_Protein_GFP",
        observe_rna_species="obs_RNA_GFP",
        use_statistical_summary=False,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        ribbon_alpha=0.25,
        figure_size=(10, 12),
        save_path=None,
        subtract_equilibrium_baseline=False,  # NEW PARAMETER
    ):
        """
        Plot parameter sweep results with focused display on pulse region.

        Parameters:
        -----------
        subtract_equilibrium_baseline : bool
            If True, subtract concentration at equilibration_time from all time points
        """

        # Extract pulse and equilibration timing
        pulse_start_time = pulse_configuration["pulse_start"]
        pulse_end_time = pulse_configuration["pulse_end"]
        equilibration_time = pulse_configuration.get("equilibration_time", 0)
        pulse_active_concentration = pulse_configuration["pulse_concentration"]
        pulse_baseline_concentration = pulse_configuration["base_concentration"]

        # Calculate focused display window
        display_window_start_time = pulse_start_time - pre_pulse_display_minutes
        display_window_end_time = pulse_end_time + post_pulse_display_minutes

        # Find display window indices
        display_start_index = np.searchsorted(
            full_equilibration_time_span, display_window_start_time
        )
        display_end_index = np.searchsorted(
            full_equilibration_time_span, display_window_end_time
        )

        # Extract focused time span
        pulse_focused_time_span = full_equilibration_time_span[
            display_start_index : display_end_index + 1
        ]

        # Apply baseline correction to full data before extracting focused window
        if subtract_equilibrium_baseline:
            baseline_corrected_observables = (
                self._apply_equilibrium_baseline_correction(
                    equilibrated_simulation_result.observables,
                    full_equilibration_time_span,
                    equilibration_time,
                )
            )
        else:
            baseline_corrected_observables = equilibrated_simulation_result.observables

        # Extract focused observables from baseline-corrected data
        if isinstance(baseline_corrected_observables, list):
            # Multiple parameter sets - each is a structured array
            pulse_focused_observables = []
            for observables_array in baseline_corrected_observables:
                focused_observables_slice = observables_array[
                    display_start_index : display_end_index + 1
                ]
                pulse_focused_observables.append(focused_observables_slice)

            observable_field_names = baseline_corrected_observables[0].dtype.names
        else:
            # Single parameter set - structured array
            pulse_focused_observables = baseline_corrected_observables[
                display_start_index : display_end_index + 1
            ]
            observable_field_names = baseline_corrected_observables.dtype.names

        # Create pulse concentration profile for focused window
        pulse_concentration_profile_focused = np.full_like(
            pulse_focused_time_span, pulse_baseline_concentration
        )
        pulse_active_time_mask = (pulse_focused_time_span >= pulse_start_time) & (
            pulse_focused_time_span <= pulse_end_time
        )
        pulse_concentration_profile_focused[pulse_active_time_mask] = (
            pulse_active_concentration
        )

        # Determine subplot configuration
        subplot_components = []
        if observe_protein:
            subplot_components.append("protein")
        if observe_rna_species is not None:
            subplot_components.append("rna")
        subplot_components.append("pulse")

        subplot_count = len(subplot_components)

        # Create figure
        figure, subplot_axes = plt.subplots(
            subplot_count, 1, figsize=figure_size, sharex=True
        )
        if subplot_count == 1:
            subplot_axes = [subplot_axes]

        current_subplot_index = 0

        # Generate colors for individual trajectories
        if isinstance(pulse_focused_observables, list):
            parameter_set_count = len(pulse_focused_observables)
        else:
            parameter_set_count = 1
        trajectory_colors = [plt.cm.tab10(i % 10) for i in range(parameter_set_count)]

        # Compute statistical summaries for focused data
        focused_trajectory_statistics = None
        if use_statistical_summary:
            # Note: baseline correction already applied, so pass False here
            focused_trajectory_statistics = (
                self._compute_baseline_corrected_statistical_summaries(
                    pulse_focused_observables,
                    pulse_focused_time_span,
                    equilibration_time,  # Not used since correction already applied
                    statistical_summary_type,
                    percentile_bounds,
                    observe_rna_species,
                    subtract_equilibrium_baseline=False,  # Already corrected
                )
            )

        # Plot protein concentration
        if observe_protein:
            protein_subplot_axis = subplot_axes[current_subplot_index]

            if (
                use_statistical_summary
                and focused_trajectory_statistics
                and "protein" in focused_trajectory_statistics
            ):
                protein_summary_data = focused_trajectory_statistics["protein"]

                central_line_label = (
                    "Median"
                    if statistical_summary_type == "median_percentiles"
                    else "Mean"
                )
                protein_subplot_axis.plot(
                    pulse_focused_time_span,
                    protein_summary_data["central_tendency"],
                    color="blue",
                    linewidth=2,
                    label=central_line_label,
                    zorder=3,
                )

                bounds_label = self._get_statistical_bounds_label(
                    statistical_summary_type, percentile_bounds
                )
                protein_subplot_axis.fill_between(
                    pulse_focused_time_span,
                    protein_summary_data["lower_bound"],
                    protein_summary_data["upper_bound"],
                    alpha=ribbon_alpha,
                    color="blue",
                    label=bounds_label,
                    zorder=1,
                )
                protein_subplot_axis.legend(fontsize=10)
            else:
                # Plot individual protein trajectories
                if isinstance(pulse_focused_observables, list):
                    for parameter_set_index, focused_observables_array in enumerate(
                        pulse_focused_observables
                    ):
                        for observable_field_name in observable_field_names:
                            if observable_field_name.startswith("obs_Protein_"):
                                protein_concentration_trajectory = (
                                    focused_observables_array[observable_field_name]
                                )
                                trajectory_color = trajectory_colors[
                                    parameter_set_index % len(trajectory_colors)
                                ]
                                protein_subplot_axis.plot(
                                    pulse_focused_time_span,
                                    protein_concentration_trajectory,
                                    color=trajectory_color,
                                    alpha=0.15,
                                    zorder=1,
                                )
                else:
                    for observable_field_name in observable_field_names:
                        if observable_field_name.startswith("obs_Protein_"):
                            protein_concentration_trajectory = (
                                pulse_focused_observables[observable_field_name]
                            )
                            protein_subplot_axis.plot(
                                pulse_focused_time_span,
                                protein_concentration_trajectory,
                                color="blue",
                                linewidth=2,
                                zorder=2,
                            )

            # Set labels - adjust for baseline correction
            ylabel = "Protein Concentration (nM)"
            title_suffix = "(Pulse Region)"
            if subtract_equilibrium_baseline:
                ylabel = "Δ Protein Concentration (nM)"
                title_suffix = "(Baseline Corrected)"

            protein_subplot_axis.set_ylabel(ylabel)
            protein_subplot_axis.set_title(f"Protein - {circuit_name} {title_suffix}")
            protein_subplot_axis.grid(True, alpha=0.3)
            current_subplot_index += 1

        # Plot RNA concentration (similar modifications)
        if observe_rna_species is not None:
            rna_subplot_axis = subplot_axes[current_subplot_index]

            if (
                use_statistical_summary
                and focused_trajectory_statistics
                and "rna" in focused_trajectory_statistics
            ):
                rna_summary_data = focused_trajectory_statistics["rna"]

                central_line_label = (
                    "Median"
                    if statistical_summary_type == "median_percentiles"
                    else "Mean"
                )
                rna_subplot_axis.plot(
                    pulse_focused_time_span,
                    rna_summary_data["central_tendency"],
                    color="red",
                    linewidth=2,
                    label=central_line_label,
                    zorder=3,
                )

                bounds_label = self._get_statistical_bounds_label(
                    statistical_summary_type, percentile_bounds
                )
                rna_subplot_axis.fill_between(
                    pulse_focused_time_span,
                    rna_summary_data["lower_bound"],
                    rna_summary_data["upper_bound"],
                    alpha=ribbon_alpha,
                    color="red",
                    label=bounds_label,
                    zorder=1,
                )
                rna_subplot_axis.legend(fontsize=10)
            else:
                # Plot individual RNA trajectories
                if isinstance(pulse_focused_observables, list):
                    for parameter_set_index, focused_observables_array in enumerate(
                        pulse_focused_observables
                    ):
                        if observe_rna_species in focused_observables_array.dtype.names:
                            rna_concentration_trajectory = focused_observables_array[
                                observe_rna_species
                            ]
                            trajectory_color = trajectory_colors[
                                parameter_set_index % len(trajectory_colors)
                            ]
                            rna_subplot_axis.plot(
                                pulse_focused_time_span,
                                rna_concentration_trajectory,
                                color=trajectory_color,
                                alpha=0.15,
                                zorder=1,
                            )
                else:
                    if observe_rna_species in pulse_focused_observables.dtype.names:
                        rna_concentration_trajectory = pulse_focused_observables[
                            observe_rna_species
                        ]
                        rna_subplot_axis.plot(
                            pulse_focused_time_span,
                            rna_concentration_trajectory,
                            color="red",
                            linewidth=2,
                            zorder=2,
                        )

            # Set labels - adjust for baseline correction
            ylabel = "RNA Concentration (nM)"
            title_suffix = "(Pulse Region)"
            if subtract_equilibrium_baseline:
                ylabel = "Δ RNA Concentration (nM)"
                title_suffix = "(Baseline Corrected)"

            rna_subplot_axis.set_ylabel(ylabel)
            rna_subplot_axis.set_title(
                f"RNA ({observe_rna_species}) - {circuit_name} {title_suffix}"
            )
            rna_subplot_axis.grid(True, alpha=0.3)
            current_subplot_index += 1

        # Plot pulse profile (unchanged)
        pulse_profile_subplot_axis = subplot_axes[current_subplot_index]
        pulse_plasmid_names_label = (
            ", ".join(pulse_plasmids) if pulse_plasmids else "Pulse"
        )
        pulse_profile_subplot_axis.plot(
            pulse_focused_time_span,
            pulse_concentration_profile_focused,
            "g-",
            linewidth=3,
            label=pulse_plasmid_names_label,
        )
        pulse_profile_subplot_axis.set_ylabel("Pulse Concentration (nM)")
        pulse_profile_subplot_axis.set_title("Pulse Profile (Focused)")
        pulse_profile_subplot_axis.grid(True, alpha=0.3)
        pulse_profile_subplot_axis.legend(fontsize=10)

        # Configure axis labels and figure title
        subplot_axes[-1].set_xlabel("Time (min)")

        # Adjust title for baseline correction
        title_modifier = ""
        if subtract_equilibrium_baseline:
            title_modifier = " (Baseline Corrected)"

        if use_statistical_summary:
            summary_description = self._get_statistical_summary_description(
                statistical_summary_type, percentile_bounds
            )
            figure.suptitle(
                f"{circuit_name} - {summary_description}{title_modifier}",
                fontsize=14,
                y=0.98,
            )
        else:
            figure.suptitle(
                f"{circuit_name} - Individual Trajectories{title_modifier}",
                fontsize=14,
                y=0.98,
            )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return figure

    def _compute_pulse_focused_statistical_summaries(
        self,
        pulse_focused_observables,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        observe_rna_species=None,
    ):
        """Compute statistical summaries for pulse-focused simulation data."""
        protein_trajectory_collection = []
        rna_trajectory_collection = []

        if isinstance(pulse_focused_observables, list):
            # Multiple parameter sets
            for focused_observables_dict in pulse_focused_observables:
                for (
                    observable_field_name,
                    trajectory_data,
                ) in focused_observables_dict.items():
                    if observable_field_name.startswith("obs_Protein_"):
                        protein_trajectory_collection.append(trajectory_data)
                    elif observable_field_name == observe_rna_species:
                        rna_trajectory_collection.append(trajectory_data)
        else:
            # Single parameter set
            for (
                observable_field_name,
                trajectory_data,
            ) in pulse_focused_observables.items():
                if observable_field_name.startswith("obs_Protein_"):
                    protein_trajectory_collection.append(trajectory_data)
                elif observable_field_name == observe_rna_species:
                    rna_trajectory_collection.append(trajectory_data)

        statistical_summaries = {}

        if protein_trajectory_collection:
            protein_trajectory_matrix = np.array(protein_trajectory_collection)
            statistical_summaries["protein"] = (
                self._compute_trajectory_statistical_measures(
                    protein_trajectory_matrix,
                    statistical_summary_type,
                    percentile_bounds,
                )
            )

        if rna_trajectory_collection:
            rna_trajectory_matrix = np.array(rna_trajectory_collection)
            statistical_summaries["rna"] = (
                self._compute_trajectory_statistical_measures(
                    rna_trajectory_matrix, statistical_summary_type, percentile_bounds
                )
            )

        return statistical_summaries

    def _compute_trajectory_statistical_measures(
        self, trajectory_matrix, statistical_summary_type, percentile_bounds
    ):
        """Compute either median+percentiles or mean+std statistics for trajectory matrix."""
        if statistical_summary_type == "mean_std":
            trajectory_mean = np.mean(trajectory_matrix, axis=0)
            trajectory_standard_deviation = np.std(trajectory_matrix, axis=0)
            return {
                "central_tendency": trajectory_mean,
                "lower_bound": trajectory_mean - trajectory_standard_deviation,
                "upper_bound": trajectory_mean + trajectory_standard_deviation,
            }
        else:  # median_percentiles
            lower_percentile_value, upper_percentile_value = percentile_bounds
            return {
                "central_tendency": np.median(trajectory_matrix, axis=0),
                "lower_bound": np.percentile(
                    trajectory_matrix, lower_percentile_value, axis=0
                ),
                "upper_bound": np.percentile(
                    trajectory_matrix, upper_percentile_value, axis=0
                ),
            }

    def _get_statistical_bounds_label(
        self, statistical_summary_type, percentile_bounds
    ):
        """Generate appropriate label for statistical summary bounds."""
        if statistical_summary_type == "mean_std":
            return "Mean ± Std"
        else:
            return f"{percentile_bounds[0]}-{percentile_bounds[1]}% range"

    def _get_statistical_summary_description(
        self, statistical_summary_type, percentile_bounds
    ):
        """Generate figure title description for statistical summary."""
        if statistical_summary_type == "mean_std":
            return "Mean ± Standard Deviation"
        else:
            return f"Median ± {percentile_bounds[0]}-{percentile_bounds[1]}% Range"

    def plot_circuits_protein_only_grid(
        self,
        circuit_simulation_data,  # dict: {circuit_name: (result, t_span, param_df, pulse_plasmids)}
        pulse_configuration=None,
        use_statistical_summary=True,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        ribbon_alpha=0.25,
        pulse_shading_alpha=0.2,
        pulse_shading_color="grey",
        grid_layout=None,  # tuple (rows, cols) or None for auto
        figure_size=None,  # auto-calculated if None
        save_path=None,
        use_focused_display=True,
        pre_pulse_display_minutes=10,
        post_pulse_display_minutes=30,
    ):
        """
        Create protein-only grid plot with pulse period background shading.

        Parameters:
        -----------
        circuit_simulation_data : dict
            {circuit_name: (simulation_result, time_points, param_df, pulse_plasmids)}
        grid_layout : tuple or None
            (rows, cols) for explicit layout or None for automatic square-ish arrangement
        use_focused_display : bool
            Whether to show focused pulse region or full simulation
        """

        circuit_names = list(circuit_simulation_data.keys())
        num_circuits = len(circuit_names)

        # Determine grid layout
        if grid_layout is None:
            grid_cols = math.ceil(math.sqrt(num_circuits))
            grid_rows = math.ceil(num_circuits / grid_cols)
        else:
            grid_rows, grid_cols = grid_layout

        if grid_rows * grid_cols < num_circuits:
            raise ValueError(
                f"Grid layout {grid_layout} insufficient for {num_circuits} circuits"
            )

        # Auto-calculate figure size if not provided
        if figure_size is None:
            subplot_width = 4
            subplot_height = 3
            figure_size = (grid_cols * subplot_width, grid_rows * subplot_height)

        # Extract pulse timing for shading
        pulse_start_time = (
            pulse_configuration.get("pulse_start", 0) if pulse_configuration else 0
        )
        pulse_end_time = (
            pulse_configuration.get("pulse_end", 10) if pulse_configuration else 10
        )

        # Create figure with gridspec
        figure = plt.figure(figsize=figure_size)
        grid_spec = gridspec.GridSpec(
            grid_rows,
            grid_cols,
            figure=figure,
            hspace=0.3,
            wspace=0.3,
            top=0.92,
            bottom=0.08,
            left=0.08,
            right=0.95,
        )

        circuit_axes_collection = {}

        # Process each circuit
        for circuit_index, circuit_name in enumerate(circuit_names):
            row_index = circuit_index // grid_cols
            col_index = circuit_index % grid_cols

            # Create subplot
            protein_axis = figure.add_subplot(grid_spec[row_index, col_index])
            circuit_axes_collection[circuit_name] = protein_axis

            # Extract simulation data
            simulation_result, time_points, param_dataframe, pulse_plasmids = (
                circuit_simulation_data[circuit_name]
            )

            # Apply focused display if requested
            if use_focused_display and pulse_configuration:
                display_time_points, focused_observables = (
                    self._extract_focused_display_data(
                        simulation_result,
                        time_points,
                        pulse_configuration,
                        pre_pulse_display_minutes,
                        post_pulse_display_minutes,
                    )
                )
            else:
                display_time_points = time_points
                focused_observables = simulation_result.observables

            # Extract observable field names
            if isinstance(focused_observables, list) and len(focused_observables) > 0:
                observable_field_names = focused_observables[0].dtype.names
            elif hasattr(focused_observables, "dtype"):
                observable_field_names = focused_observables.dtype.names
            else:
                continue

            # Compute statistical summaries if requested
            if use_statistical_summary:
                protein_trajectory_statistics = (
                    self._compute_protein_statistical_summaries(
                        focused_observables, statistical_summary_type, percentile_bounds
                    )
                )

            # Plot protein trajectories
            if use_statistical_summary and protein_trajectory_statistics:
                # Plot statistical summary
                central_tendency_label = (
                    "Median"
                    if statistical_summary_type == "median_percentiles"
                    else "Mean"
                )

                protein_axis.plot(
                    display_time_points,
                    protein_trajectory_statistics["central_tendency"],
                    color="blue",
                    linewidth=2,
                    label=central_tendency_label,
                    zorder=3,
                )

                bounds_label = self._get_statistical_bounds_label(
                    statistical_summary_type, percentile_bounds
                )
                protein_axis.fill_between(
                    display_time_points,
                    protein_trajectory_statistics["lower_bound"],
                    protein_trajectory_statistics["upper_bound"],
                    alpha=ribbon_alpha,
                    color="blue",
                    label=bounds_label,
                    zorder=2,
                )
            else:
                # Plot individual trajectories
                trajectory_colors = [
                    plt.cm.tab10(i % 10) for i in range(len(param_dataframe))
                ]

                if isinstance(focused_observables, list):
                    for trajectory_index, observables_array in enumerate(
                        focused_observables
                    ):
                        for observable_name in observable_field_names:
                            if observable_name.startswith("obs_Protein_"):
                                protein_concentration = observables_array[
                                    observable_name
                                ]
                                trajectory_color = trajectory_colors[
                                    trajectory_index % len(trajectory_colors)
                                ]
                                protein_axis.plot(
                                    display_time_points,
                                    protein_concentration,
                                    color=trajectory_color,
                                    alpha=0.15,
                                    zorder=1,
                                )
                else:
                    for observable_name in observable_field_names:
                        if observable_name.startswith("obs_Protein_"):
                            protein_concentration = focused_observables[observable_name]
                            protein_axis.plot(
                                display_time_points,
                                protein_concentration,
                                color="blue",
                                linewidth=2,
                                zorder=2,
                            )

            # Add pulse period background shading
            if pulse_configuration:
                protein_axis.axvspan(
                    pulse_start_time,
                    pulse_end_time,
                    alpha=pulse_shading_alpha,
                    color=pulse_shading_color,
                    zorder=0,
                    label="Pulse Period",
                )

            # Configure subplot appearance
            protein_axis.set_title(circuit_name, fontsize=11, fontweight="bold")
            protein_axis.set_ylabel("Protein (nM)", fontsize=10)
            protein_axis.grid(True, alpha=0.3)
            protein_axis.tick_params(labelsize=9)

            # Add legend only to first subplot to avoid clutter
            if circuit_index == 0:
                protein_axis.legend(fontsize=8, loc="upper right")

            # Set x-label for bottom row
            if row_index == grid_rows - 1:
                protein_axis.set_xlabel("Time (min)", fontsize=10)

        # # Generate figure title
        # summary_description = self._get_statistical_summary_description(
        #     statistical_summary_type, percentile_bounds
        # )
        # mode_description = (
        #     "Statistical Summary"
        #     if use_statistical_summary
        #     else "Individual Trajectories"
        # )
        # display_description = (
        #     "Pulse Region" if use_focused_display else "Full Simulation"
        # )

        # figure.suptitle(
        #     f"Protein Response - {mode_description} ({display_description})",
        #     fontsize=14,
        #     fontweight="bold",
        # )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Protein grid plot saved: {save_path}")

        # plt.show()
        return figure, circuit_axes_collection

    def _extract_focused_display_data(
        self,
        simulation_result,
        time_points,
        pulse_configuration,
        pre_pulse_display_minutes,
        post_pulse_display_minutes,
    ):
        """Extract focused time window data for pulse display."""
        pulse_start_time = pulse_configuration["pulse_start"]
        pulse_end_time = pulse_configuration["pulse_end"]

        display_window_start_time = pulse_start_time - pre_pulse_display_minutes
        display_window_end_time = pulse_end_time + post_pulse_display_minutes

        # Find corresponding indices
        display_start_index = np.searchsorted(time_points, display_window_start_time)
        display_end_index = np.searchsorted(time_points, display_window_end_time)

        # Extract focused time span
        focused_time_points = time_points[display_start_index : display_end_index + 1]

        # Extract focused observables using direct slicing
        if isinstance(simulation_result.observables, list):
            focused_observables = []
            for observables_array in simulation_result.observables:
                focused_observables.append(
                    observables_array[display_start_index : display_end_index + 1]
                )
        else:
            focused_observables = simulation_result.observables[
                display_start_index : display_end_index + 1
            ]

        return focused_time_points, focused_observables

    def _compute_protein_statistical_summaries(
        self, observables_data, statistical_summary_type, percentile_bounds
    ):
        """Compute statistical summaries specifically for protein observables."""
        protein_trajectory_collection = []

        if isinstance(observables_data, list):
            # Multiple parameter sets
            for observables_array in observables_data:
                for observable_name in observables_array.dtype.names:
                    if observable_name.startswith("obs_Protein_"):
                        protein_trajectory_collection.append(
                            observables_array[observable_name]
                        )
        else:
            # Single parameter set
            for observable_name in observables_data.dtype.names:
                if observable_name.startswith("obs_Protein_"):
                    protein_trajectory_collection.append(
                        observables_data[observable_name]
                    )

        if not protein_trajectory_collection:
            return None

        protein_trajectory_matrix = np.array(protein_trajectory_collection)
        return self._compute_trajectory_statistical_measures(
            protein_trajectory_matrix, statistical_summary_type, percentile_bounds
        )
