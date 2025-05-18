import numpy as np
import matplotlib.pyplot as plt
from circuits.circuit_generation.circuit_manager import CircuitManager
from circuits.circuit_generation.circuit_visualizer import CircuitVisualizer
from circuits.modules.base_modules import KineticsType


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
        title=None,
        figure_size=(6, 10),
        save_path=None,
        show_protein=True,
        show_rna=True,
        show_pulse=True,
        pulse_plasmids=None,
        scores=None,
        score_metric=None,
    ):
        """
        Run a parameter sweep and create a visualization with color coding based on scores.

        Parameters:
        -----------
        (previous parameters...)
        scores : pandas.Series, optional
            Score values to use for color coding (e.g., posterior, likelihood)
        score_metric : str, optional
            Name of the score metric (for title/legend)
        """
        # Run parameter sweep
        result, t_span, param_values, circuit = self.run_parameter_sweep(
            circuit_name,
            param_df,
            k_prot_deg,
            pulse_configuration,
            kinetics_type,
            t_span,
            additional_params,
            pulse_plasmids=pulse_plasmids,
        )

        # Determine which plots to show
        plots_to_show = []
        if show_protein:
            plots_to_show.append("protein")
        if show_rna:
            plots_to_show.append("rna")
        if show_pulse:
            plots_to_show.append("pulse")

        number_of_plots = len(plots_to_show)
        if number_of_plots == 0:
            raise ValueError("At least one plot type must be shown")

        # Create figure with subplots
        fig, axs = plt.subplots(number_of_plots, 1, figsize=figure_size, sharex=True)
        if number_of_plots == 1:
            axs = [axs]

        plot_idx = 0

        # Create a colormap for scores if provided
        if scores is not None:
            cmap = plt.cm.viridis
            # Normalize scores for colormap
            norm = plt.Normalize(scores.min(), scores.max())
            # Create colors based on scores
            colors = [cmap(norm(score)) for score in scores]
        else:
            # Default colors if no scores
            colors = [plt.cm.tab10(i % 10) for i in range(len(param_df))]

        # Plot protein observables if requested
        if show_protein:
            for i in range(len(param_df)):
                if i < len(result.observables):
                    if observe_protein in result.observables[i].dtype.names:
                        # Use color from our color list
                        color = colors[i] if i < len(colors) else "blue"
                        axs[plot_idx].plot(
                            t_span, result.observables[i][observe_protein], color=color
                        )

            axs[plot_idx].set_ylabel("Protein concentration")
            if score_metric:
                axs[plot_idx].set_title(f"Protein Dynamics (colored by {score_metric})")
            else:
                axs[plot_idx].set_title("Protein Dynamics")
            axs[plot_idx].grid(True)
            plot_idx += 1

        # Plot RNA observables if requested
        if show_rna:
            rna_observables = [
                name
                for name in result.observables[0].dtype.names
                if name.startswith("obs_RNA_")
            ]

            for rna_obs in rna_observables:
                for i in range(len(param_df)):
                    if i < len(result.observables):
                        # Use same color scheme as protein
                        color = colors[i] if i < len(colors) else plt.cm.tab10(i % 10)
                        axs[plot_idx].plot(
                            t_span,
                            result.observables[i][rna_obs],
                            label=_get_display_name(rna_obs) if i == 0 else None,
                            color=color,
                            alpha=0.7,
                        )

            axs[plot_idx].set_ylabel("RNA concentration")
            if score_metric:
                axs[plot_idx].set_title(f"RNA Dynamics (colored by {score_metric})")
            else:
                axs[plot_idx].set_title("RNA Dynamics")
            axs[plot_idx].grid(True)
            axs[plot_idx].legend(loc="best")
            plot_idx += 1

        # Plot pulse profile if requested
        if show_pulse:
            if pulse_configuration:
                pulse_profile = []
                for t in t_span:
                    if (
                        t < pulse_configuration["pulse_start"]
                        or t > pulse_configuration["pulse_end"]
                    ):
                        pulse_profile.append(pulse_configuration["base_concentration"])
                    else:
                        pulse_profile.append(pulse_configuration["pulse_concentration"])

                # Display the pulsed plasmid names in the label
                pulse_label = ", ".join(pulse_plasmids) if pulse_plasmids else "Pulse"

                axs[plot_idx].plot(
                    t_span, pulse_profile, label=pulse_label, color="red", linewidth=2
                )
                axs[plot_idx].set_ylabel("Input concentration")
                axs[plot_idx].set_title("Pulse Profile")
                axs[plot_idx].grid(True)
                axs[plot_idx].legend(loc="best")
            else:
                # If no pulse, show parameter effect visualization
                # Create a heatmap-like visualization
                param_matrix = np.tile(param_values, (len(t_span), 1)).T
                axs[plot_idx].imshow(
                    param_matrix, aspect="auto", extent=[min(t_span), max(t_span), 0, 1]
                )

        # Set overall labels and title
        axs[-1].set_xlabel("Time (min)")
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        else:
            plt.suptitle(f"{circuit_name}", fontsize=14, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        return fig


# Example usage
if __name__ == "__main__":
    # Create a circuit manager
    manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors_with_mass_action.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # Create a parameter sampling manager
    sampling_manager = ParameterSamplingManager(manager)

    # Define pulse configuration
    pulse_config = {
        "use_pulse": True,
        "pulse_start": 4,
        "pulse_end": 15,
        "pulse_concentration": 5.0,
        "base_concentration": 0.0,
    }

    # Create a temporary circuit to get plasmid names
    temp_circuit = manager.create_circuit("toehold_trigger")
    plasmid_names = [plasmid[0] for plasmid in temp_circuit.plasmids]

    # Use the first plasmid for pulsing
    pulse_plasmid = plasmid_names[1] if len(plasmid_names) > 1 else plasmid_names[0]

    # Run a parameter sweep for toehold_trigger circuit with named plasmid
    sampling_manager.plot_parameter_sweep_with_pulse(
        circuit_name="toehold_trigger",
        param_df={"k_Trigger3_concentration": [0, 1, 2, 3, 4, 5]},
        k_prot_deg=0.1,
        pulse_configuration=pulse_config,
        pulse_plasmids=[pulse_plasmid],  # Use named plasmid
        save_path="toehold_parameter_sweep.png",
    )
