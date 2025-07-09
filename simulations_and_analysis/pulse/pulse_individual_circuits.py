import os
import pandas as pd
import numpy as np
from circuits.circuit_generation.circuit_manager import CircuitManager
from circuits.circuit_generation.parameter_sampling_and_simulation import (
    ParameterSamplingManager,
)
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)

# Define pulse plasmid mapping - matches exact plasmid names from circuits.json
PULSE_PLASMID_MAPPING = {
    "sense_star_6": ["pr-star6_plasmid"],
    "cffl_type_1": ["star6_expression"],
    "cascade": ["star6_plasmid"],
    "toehold_trigger": ["trigger3_plasmid"],
    "star_antistar_1": ["star1_plasmid"],
    "trigger_antitrigger": ["trigger3_plasmid"],  # Main input for pulse
}

# Define pulse configuration
PULSE_CONFIGURATION = {
    "use_pulse": True,
    "pulse_start": 30.1,  # minutes
    "pulse_end": 40.1,  # minutes
    "pulse_concentration": 5.0,
    "base_concentration": 0.0,
}


def setup_calibration():
    """Set up GFP calibration parameters"""
    calibration_data = pd.read_csv("../../utils/calibration_gfp/gfp_Calibration.csv")
    calibration_results = fit_gfp_calibration(
        calibration_data,
        concentration_col="GFP Concentration (nM)",
        fluorescence_pattern="F.I. (a.u)",
    )
    brightness_correction, _ = get_brightness_correction_factor("avGFP", "sfGFP")

    return {
        "slope": calibration_results["slope"],
        "intercept": calibration_results["intercept"],
        "brightness_correction": brightness_correction,
    }


def convert_mcmc_to_parameter_dataframe(
    mcmc_samples, parameters_to_fit, burn_in_fraction=0.4
):
    """Convert MCMC samples to parameter DataFrame for ParameterSamplingManager"""
    # Filter chain 0 and apply burn-in
    chain_zero_samples = mcmc_samples[mcmc_samples["chain"] == 0]
    burn_in_cutoff = chain_zero_samples["iteration"].max() * burn_in_fraction
    post_burnin_samples = chain_zero_samples[
        chain_zero_samples["iteration"] > burn_in_cutoff
    ]

    # Extract parameter columns and convert from log10 to linear scale
    parameter_columns = [
        col for col in parameters_to_fit if col in post_burnin_samples.columns
    ]
    linear_parameters = 10 ** post_burnin_samples[parameter_columns]

    # Add likelihood scores for color coding
    if "likelihood" in post_burnin_samples.columns:
        linear_parameters["likelihood"] = post_burnin_samples["likelihood"]

    return linear_parameters


def get_pulse_plasmids_for_circuit(circuit_name):
    """Get plasmid names to pulse for specific circuit"""
    return PULSE_PLASMID_MAPPING.get(circuit_name, [f"{circuit_name}_plasmid_0"])


def simulate_circuit_pulse_batch(
    circuit_name,
    mcmc_samples,
    parameters_to_fit,
    circuit_manager,
    protein_degradation_rate=0.1,
    sample_count=50,
    output_directory=None,
    use_statistical_summary=False,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    observe_rna_species="obs_RNA_GFP",
):
    """
    Batch pulse simulation with optional statistical summaries.

    Parameters:
    -----------
    statistical_summary_type : str
        "median_percentiles" or "mean_std" for summary computation
    observe_rna_species : str or None
        Specific RNA observable to plot. None = no RNA subplot.
    """
    # Convert MCMC samples to parameter DataFrame
    parameter_dataframe = convert_mcmc_to_parameter_dataframe(
        mcmc_samples, parameters_to_fit
    )

    # Sample parameters for simulation
    if len(parameter_dataframe) > sample_count:
        sampled_parameters = parameter_dataframe.sample(n=sample_count, random_state=42)
    else:
        sampled_parameters = parameter_dataframe.copy()

    # Get pulse plasmids for this circuit
    pulse_plasmids = get_pulse_plasmids_for_circuit(circuit_name)

    # Create ParameterSamplingManager
    sampling_manager = ParameterSamplingManager(circuit_manager)

    # Define time span for pulse simulation
    pulse_time_span = np.linspace(0, 120, 601)

    # Extract likelihood scores and filter parameters
    likelihood_scores = sampled_parameters.get("likelihood")
    parameter_subset = sampled_parameters.drop(columns=["likelihood"], errors="ignore")

    parameter_subset_filtered = parameter_subset[
        (parameter_subset > parameter_subset.quantile(0.05))
        & (parameter_subset < parameter_subset.quantile(0.95))
    ].dropna()

    # Construct unified output path in same directory
    output_path = None
    if output_directory:
        mode_descriptor = "summary" if use_statistical_summary else "individual"
        summary_type_descriptor = (
            f"_{statistical_summary_type}" if use_statistical_summary else ""
        )
        percentile_descriptor = (
            f"_{percentile_bounds[0]}_{percentile_bounds[1]}"
            if use_statistical_summary
            and statistical_summary_type == "median_percentiles"
            else ""
        )

        filename = f"{circuit_name}_pulse_{mode_descriptor}{summary_type_descriptor}{percentile_descriptor}.png"
        output_path = os.path.join(output_directory, filename)

    # Determine figure size based on number of subplots
    subplot_count = sum(
        [True, observe_rna_species is not None, True]
    )  # protein, rna, pulse
    figure_size = (8, 3 * subplot_count + 1)

    # Execute pulse simulation with visualization
    sampling_manager.plot_parameter_sweep_with_pulse(
        circuit_name=circuit_name,
        param_df=parameter_subset_filtered,
        k_prot_deg=protein_degradation_rate,
        pulse_configuration=PULSE_CONFIGURATION,
        pulse_plasmids=pulse_plasmids,
        t_span=pulse_time_span,
        figure_size=figure_size,
        save_path=output_path,
        show_protein=True,
        observe_rna_species=observe_rna_species,
        show_pulse=True,
        scores=likelihood_scores,
        score_metric="likelihood",
        use_statistical_summary=use_statistical_summary,
        statistical_summary_type=statistical_summary_type,
        percentile_bounds=percentile_bounds,
        ribbon_alpha=0.25,
    )

    return len(sampled_parameters)


def create_all_circuits_grid_plots(
    mcmc_results_by_circuit,
    output_directory=".",
    sample_count=100,
    protein_degradation_rate=0.1,
    observe_rna_species="obs_RNA_GFP",
):
    """
    Generate unified grid plots showing all circuits together.
    """
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    parameters_to_fit = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()

    sampling_manager = ParameterSamplingManager(circuit_manager)

    # Prepare simulation data for all circuits
    circuit_simulation_data = {}

    print("Preparing simulation data for all circuits...")
    for circuit_name, mcmc_samples in mcmc_results_by_circuit.items():
        print(f"Processing {circuit_name}...")

        # Convert and sample MCMC parameters
        parameter_dataframe = convert_mcmc_to_parameter_dataframe(
            mcmc_samples, parameters_to_fit
        )

        if len(parameter_dataframe) > sample_count:
            sampled_parameters = parameter_dataframe.sample(
                n=sample_count, random_state=42
            )
        else:
            sampled_parameters = parameter_dataframe.copy()

        # Get pulse plasmids
        pulse_plasmids = get_pulse_plasmids_for_circuit(circuit_name)

        # Filter parameters
        parameter_subset = sampled_parameters.drop(
            columns=["likelihood"], errors="ignore"
        )
        parameter_subset_filtered = parameter_subset[
            (parameter_subset > parameter_subset.quantile(0.05))
            & (parameter_subset < parameter_subset.quantile(0.95))
        ].dropna()

        # Run simulation
        simulation_result, time_points, param_df, circuit_instance = (
            sampling_manager.run_parameter_sweep(
                circuit_name,
                parameter_subset_filtered,
                protein_degradation_rate,
                PULSE_CONFIGURATION,
                pulse_plasmids=pulse_plasmids,
                t_span=np.linspace(0, 120, 601),
            )
        )

        circuit_simulation_data[circuit_name] = (
            simulation_result,
            time_points,
            param_df,
            pulse_plasmids,
        )

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Generate grid plots for different modes
    plot_configs = [
        {
            "use_statistical_summary": False,
            "filename": "all_circuits_pulse_grid_individual.png",
            "description": "Individual Trajectories",
        },
        {
            "use_statistical_summary": True,
            "statistical_summary_type": "median_percentiles",
            "percentile_bounds": (10, 90),
            "filename": "all_circuits_pulse_grid_summary_median_10_90.png",
            "description": "Median ± 10-90%",
        },
        {
            "use_statistical_summary": True,
            "statistical_summary_type": "median_percentiles",
            "percentile_bounds": (25, 75),
            "filename": "all_circuits_pulse_grid_summary_median_25_75.png",
            "description": "Median ± 25-75%",
        },
        {
            "use_statistical_summary": True,
            "statistical_summary_type": "mean_std",
            "filename": "all_circuits_pulse_grid_summary_mean_std.png",
            "description": "Mean ± Std",
        },
    ]

    generated_plots = []

    for config in plot_configs:
        print(f"Generating grid plot: {config['description']}")

        save_path = os.path.join(output_directory, config["filename"])

        figure, axes = sampling_manager.plot_all_circuits_pulse_grid(
            circuit_simulation_data,
            pulse_configuration=PULSE_CONFIGURATION,
            observe_rna_species=observe_rna_species,
            use_statistical_summary=config["use_statistical_summary"],
            statistical_summary_type=config.get(
                "statistical_summary_type", "median_percentiles"
            ),
            percentile_bounds=config.get("percentile_bounds", (10, 90)),
            figure_size=(
                18,
                4 * len(circuit_simulation_data),
            ),  # Dynamic height based on circuit count
            save_path=save_path,
        )

        generated_plots.append((config["description"], save_path))

    print(f"\nGenerated {len(generated_plots)} grid plots:")
    for description, path in generated_plots:
        print(f"  - {description}: {path}")

    return generated_plots


def plot_fits_with_batch_pulse_simulation(
    mcmc_results_by_circuit,
    output_directory=".",
    sample_count=60,
    protein_degradation_rate=0.1,
    use_statistical_summary=False,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    observe_rna_species="obs_RNA_GFP",
):
    """
    Generate pulse simulation plots with unified directory structure.

    Parameters:
    -----------
    statistical_summary_type : str
        "median_percentiles" or "mean_std" for summary computation
    observe_rna_species : str or None
        Specific RNA observable to plot. None = no RNA subplot.
    """
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    parameters_to_fit = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()

    # Create unified output directory
    os.makedirs(output_directory, exist_ok=True)

    for circuit_name, mcmc_raw_samples in mcmc_results_by_circuit.items():
        print(f"Simulating {circuit_name} with batch pulse processing...")

        sample_count_used = simulate_circuit_pulse_batch(
            circuit_name=circuit_name,
            mcmc_samples=mcmc_raw_samples,
            parameters_to_fit=parameters_to_fit,
            circuit_manager=circuit_manager,
            protein_degradation_rate=protein_degradation_rate,
            sample_count=sample_count,
            output_directory=output_directory,
            use_statistical_summary=use_statistical_summary,
            statistical_summary_type=statistical_summary_type,
            percentile_bounds=percentile_bounds,
            observe_rna_species=observe_rna_species,
        )

        print(
            f"✓ Completed {circuit_name}: {sample_count_used} parameter sets processed"
        )

    print(f"All pulse simulations completed. Output: {output_directory}")


def main():
    subfolder = "/10000_steps_updated"
    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_visualization_directory = (
        "../../figures/individual_circuits_pulse" + subfolder
    )

    # Load MCMC results from individual circuit fits
    mcmc_results = load_individual_circuit_results(input_directory)

    # Generate individual circuit plots (existing functionality)
    print("Generating individual circuit plots...")

    plot_fits_with_batch_pulse_simulation(
        mcmc_results,
        output_visualization_directory,
        sample_count=300,
        protein_degradation_rate=0.1,
        use_statistical_summary=False,
        observe_rna_species="obs_RNA_GFP",
    )

    plot_fits_with_batch_pulse_simulation(
        mcmc_results,
        output_visualization_directory,
        sample_count=300,
        protein_degradation_rate=0.1,
        use_statistical_summary=True,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        observe_rna_species="obs_RNA_GFP",
    )

    plot_fits_with_batch_pulse_simulation(
        mcmc_results,
        output_visualization_directory,
        sample_count=300,
        protein_degradation_rate=0.1,
        use_statistical_summary=True,
        statistical_summary_type="mean_std",
        observe_rna_species="obs_RNA_GFP",
    )

    # Generate unified grid plots (NEW functionality)
    print("\nGenerating unified grid plots...")

    create_all_circuits_grid_plots(
        mcmc_results,
        output_visualization_directory,
        sample_count=100,  # Smaller sample for grid plots to manage figure complexity
        protein_degradation_rate=0.1,
        observe_rna_species=None,
    )

    print("\nAll plots completed!")


if __name__ == "__main__":
    main()
