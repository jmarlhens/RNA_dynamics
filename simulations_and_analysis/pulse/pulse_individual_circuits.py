import os
import pandas as pd
import numpy as np
from circuits.circuit_generation.circuit_manager import CircuitManager
from circuits.circuit_generation.parameter_sampling_and_simulation import (
    ParameterSamplingManager,
)
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)
from data.circuits.circuit_configs import get_circuit_conditions

# REUSE existing functions from individual_circuits_simulations.py
from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data

# Define pulse plasmid mapping - matches exact plasmid names from circuits.json
PULSE_PLASMID_MAPPING = {
    "sense_star_6": ["pr-star6_plasmid"],
    "cffl_type_1": ["star6_expression"],
    "cascade": ["star6_plasmid"],
    "toehold_trigger": ["trigger3_plasmid"],
    "star_antistar_1": ["star1_plasmid"],
    "trigger_antitrigger": ["trigger3_plasmid"],
    "inhibited_cascade": ["star6_plasmid"],
    "inhibited_incoherent_cascade": ["star6_plasmid"],
    "or_gate_c1ffl": ["star6_plasmid"],
    "iffl_1": ["star6_trigger3_plasmid"],
}


def get_pulse_plasmids_for_circuit(circuit_name):
    """Get plasmid names to pulse for specific circuit"""
    return PULSE_PLASMID_MAPPING.get(circuit_name, [f"{circuit_name}_plasmid_0"])


def get_circuit_pulse_configuration_with_equilibration(
    circuit_name, equilibration_time=120
):
    """Get circuit-specific pulse parameters with pre-equilibration period"""
    circuit_specific_concentrations = {
        "sense_star_6": 15.0,
        "cffl_type_1": 15.0,
        "cascade": 15.0,
        "toehold_trigger": 15.0,
        "star_antistar_1": 15.0,
        "trigger_antitrigger": 15.0,
        "iffl_1": 15.0,
    }

    pulse_concentration = circuit_specific_concentrations.get(circuit_name, 15.0)

    return {
        "use_pulse": True,
        "pulse_start": equilibration_time + 0.5,  # Start pulse after equilibration
        "pulse_end": equilibration_time + 200.5,  # End pulse 10 minutes later
        "pulse_concentration": pulse_concentration,
        "base_concentration": 0.0,
        "equilibration_time": equilibration_time,
    }


def create_extended_time_span(
    equilibration_time=120, pulse_duration=200, observation_time=200
):
    """Create extended time span including equilibration, pulse, and observation periods"""
    total_simulation_time = equilibration_time + pulse_duration + observation_time
    # Use higher resolution for better temporal accuracy
    time_points_count = int(total_simulation_time * 5) + 1  # 5 points per minute
    return np.linspace(0, total_simulation_time, time_points_count)


def extract_plasmid_to_parameter_mapping(circuit_manager, circuit_name):
    """Extract systematic mapping from plasmids to their concentration parameters"""
    circuit_configuration = circuit_manager.get_circuit_config(circuit_name)
    plasmid_to_parameter_mapping = {}

    for plasmid_name, tx_control, tl_control, cds_list in circuit_configuration[
        "plasmids"
    ]:
        # Extract component names produced by this plasmid
        produced_components = [
            component_name for is_protein, component_name in cds_list
        ]

        # Map to concentration parameters based on circuit's default parameters
        circuit_default_parameters = circuit_configuration["default_parameters"]

        for parameter_name in circuit_default_parameters.keys():
            if parameter_name.endswith("_concentration"):
                # Check if parameter contains any component produced by this plasmid
                for component_name in produced_components:
                    if component_name in parameter_name:
                        plasmid_to_parameter_mapping[plasmid_name] = parameter_name
                        break

    return plasmid_to_parameter_mapping


def create_pulse_circuit_simulation_data(
    circuit_name,
    mcmc_raw_samples,
    circuit_manager,
):
    """
    Modified version of create_circuit_simulation_data that incorporates pulse configuration
    and experimental baseline concentrations for non-pulsed plasmids.
    """
    # REUSE existing MCMC processing logic
    mcmc_processed_result = process_mcmc_data(
        mcmc_raw_samples, burn_in=0.6, chain_idx=0
    )
    mcmc_filtered_samples = mcmc_processed_result["processed_data"]

    # Get experimental baseline concentrations for non-pulsed plasmids
    circuit_experimental_conditions = get_circuit_conditions(circuit_name)
    first_experimental_condition = list(circuit_experimental_conditions.keys())[0]
    experimental_baseline_concentrations = circuit_experimental_conditions[
        first_experimental_condition
    ]

    pulse_plasmids = get_pulse_plasmids_for_circuit(circuit_name)
    pulse_configuration = get_circuit_pulse_configuration_with_equilibration(
        circuit_name
    )

    # SYSTEMATIC parameter identification using circuit definition
    plasmid_to_parameter_mapping = extract_plasmid_to_parameter_mapping(
        circuit_manager, circuit_name
    )

    # Identify pulsed parameters using systematic mapping
    pulsed_concentration_parameters = set()
    for pulsed_plasmid_name in pulse_plasmids:
        if pulsed_plasmid_name in plasmid_to_parameter_mapping:
            pulsed_parameter_name = plasmid_to_parameter_mapping[pulsed_plasmid_name]
            pulsed_concentration_parameters.add(pulsed_parameter_name)

    # Separate pulsed vs non-pulsed concentrations
    non_pulsed_concentrations = {}
    for (
        parameter_name,
        concentration_value,
    ) in experimental_baseline_concentrations.items():
        if parameter_name not in pulsed_concentration_parameters:
            non_pulsed_concentrations[parameter_name] = concentration_value

    return {
        "mcmc_processed_samples": mcmc_filtered_samples,
        "pulse_configuration": pulse_configuration,
        "pulse_plasmids": pulse_plasmids,
        "non_pulsed_concentrations": non_pulsed_concentrations,
        "experimental_baseline_concentrations": experimental_baseline_concentrations,
    }


def execute_circuit_pulse_simulation_with_equilibration(
    circuit_name,
    mcmc_raw_samples,
    parameters_to_fit,
    circuit_manager,
    protein_degradation_rate=0.1,
    sample_count=50,
    equilibration_time=120,
    pulse_duration=200,
    observation_time=200,
):
    """
    Execute pulse simulation for single circuit and return simulation data.
    Separated from plotting to enable data reuse for both individual and grid plots.
    """
    # Create extended time span
    extended_time_span = create_extended_time_span(
        equilibration_time, pulse_duration, observation_time
    )

    # Get pulse configuration with equilibration timing
    pulse_configuration = get_circuit_pulse_configuration_with_equilibration(
        circuit_name, equilibration_time
    )

    # Process MCMC data
    pulse_simulation_data = create_pulse_circuit_simulation_data(
        circuit_name,
        mcmc_raw_samples,
        circuit_manager,
    )

    mcmc_filtered_samples = pulse_simulation_data["mcmc_processed_samples"]
    pulse_plasmids = pulse_simulation_data["pulse_plasmids"]
    non_pulsed_concentrations = pulse_simulation_data["non_pulsed_concentrations"]

    # Sample parameters for pulse simulation
    final_sample_size = min(sample_count, len(mcmc_filtered_samples))
    sampled_mcmc_parameters = (
        mcmc_filtered_samples.sample(n=final_sample_size, random_state=42)
        if len(mcmc_filtered_samples) > final_sample_size
        else mcmc_filtered_samples.copy()
    )

    # Extract parameter values (assuming parameters are in log10 space in MCMC results)
    parameter_columns = [
        col for col in parameters_to_fit if col in sampled_mcmc_parameters.columns
    ]
    linear_parameter_values = 10 ** sampled_mcmc_parameters[parameter_columns]

    # Filter outliers
    parameter_subset_filtered = linear_parameter_values[
        (linear_parameter_values > linear_parameter_values.quantile(0.05))
        & (linear_parameter_values < linear_parameter_values.quantile(0.95))
    ].dropna()

    # Prepare additional parameters: protein degradation + non-pulsed concentrations
    additional_simulation_parameters = {"k_prot_deg": protein_degradation_rate}
    additional_simulation_parameters.update(non_pulsed_concentrations)

    # Create ParameterSamplingManager and run pulse simulation
    sampling_manager = ParameterSamplingManager(circuit_manager)

    # Execute pulse simulation
    (
        equilibrated_simulation_result,
        time_points,
        parameter_dataframe,
        circuit_instance,
    ) = sampling_manager.run_parameter_sweep(
        circuit_name=circuit_name,
        param_df=parameter_subset_filtered,
        k_prot_deg=protein_degradation_rate,
        _pulse_config=pulse_configuration,
        t_span=extended_time_span,
        additional_params=additional_simulation_parameters,
        pulse_plasmids=pulse_plasmids,
    )

    return {
        "simulation_result": equilibrated_simulation_result,
        "time_span": time_points,
        "parameter_dataframe": parameter_dataframe,
        "pulse_plasmids": pulse_plasmids,
        "pulse_configuration": pulse_configuration,
        "sampling_manager": sampling_manager,
        "processed_sample_count": len(sampled_mcmc_parameters),
    }


def generate_individual_pulse_plot_from_simulation_data(
    circuit_name,
    simulation_data,
    output_directory,
    use_statistical_summary=False,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    observe_rna_species="obs_RNA_GFP",
    subtract_equilibrium_baseline=False,
):
    """Generate individual circuit pulse plot from precomputed simulation data"""

    # Determine subplot count for figure sizing
    subplot_count = sum([True, observe_rna_species is not None, True])
    figure_size = (10, 3 * subplot_count + 1)

    # Construct output filename - include baseline correction in filename
    mode_descriptor = "summary" if use_statistical_summary else "individual"
    summary_type_descriptor = (
        f"_{statistical_summary_type}" if use_statistical_summary else ""
    )
    percentile_descriptor = (
        f"_{percentile_bounds[0]}_{percentile_bounds[1]}"
        if use_statistical_summary and statistical_summary_type == "median_percentiles"
        else ""
    )
    equilibration_time = simulation_data["pulse_configuration"]["equilibration_time"]
    equilibration_descriptor = f"_eq{equilibration_time}min"
    baseline_descriptor = "_baselined" if subtract_equilibrium_baseline else ""

    filename = f"{circuit_name}_pulse_extended{equilibration_descriptor}_{mode_descriptor}{summary_type_descriptor}{percentile_descriptor}{baseline_descriptor}.png"
    output_path = os.path.join(output_directory, filename)

    # Generate focused display plot with baseline correction option
    figure = simulation_data[
        "sampling_manager"
    ].plot_parameter_sweep_with_pulse_focused_display(
        equilibrated_simulation_result=simulation_data["simulation_result"],
        full_equilibration_time_span=simulation_data["time_span"],
        circuit_name=circuit_name,
        pulse_configuration=simulation_data["pulse_configuration"],
        pulse_plasmids=simulation_data["pulse_plasmids"],
        pre_pulse_display_minutes=40,
        post_pulse_display_minutes=40,
        observe_protein="obs_Protein_GFP",
        observe_rna_species=observe_rna_species,
        use_statistical_summary=use_statistical_summary,
        statistical_summary_type=statistical_summary_type,
        percentile_bounds=percentile_bounds,
        ribbon_alpha=0.25,
        figure_size=figure_size,
        save_path=output_path,
        subtract_equilibrium_baseline=subtract_equilibrium_baseline,  # PASS PARAMETER
    )

    return figure


def generate_unified_grid_plot_from_simulation_data(
    circuit_simulation_data_collection,
    output_directory,
    use_statistical_summary=False,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    observe_rna_species="obs_RNA_GFP",
):
    """Generate unified grid plot from precomputed simulation data for all circuits"""

    # Get pulse configuration from first circuit (all should be identical)
    first_circuit_data = list(circuit_simulation_data_collection.values())[0]
    pulse_configuration = first_circuit_data["pulse_configuration"]
    sampling_manager = first_circuit_data["sampling_manager"]

    # Transform simulation data to format expected by plot_all_circuits_pulse_grid
    grid_plot_simulation_data = {}
    for circuit_name, simulation_data in circuit_simulation_data_collection.items():
        grid_plot_simulation_data[circuit_name] = (
            simulation_data["simulation_result"],
            simulation_data["time_span"],
            simulation_data["parameter_dataframe"],
            simulation_data["pulse_plasmids"],
        )

    # Construct output filename
    mode_descriptor = "summary" if use_statistical_summary else "individual"
    summary_type_descriptor = (
        f"_{statistical_summary_type}" if use_statistical_summary else ""
    )
    percentile_descriptor = (
        f"_{percentile_bounds[0]}_{percentile_bounds[1]}"
        if use_statistical_summary and statistical_summary_type == "median_percentiles"
        else ""
    )
    equilibration_time = pulse_configuration["equilibration_time"]
    equilibration_descriptor = f"_eq{equilibration_time}min"

    unified_filename = f"all_circuits_pulse_grid{equilibration_descriptor}_{mode_descriptor}{summary_type_descriptor}{percentile_descriptor}.png"
    unified_output_path = os.path.join(output_directory, unified_filename)

    # Generate unified grid plot
    figure, all_axes = sampling_manager.plot_all_circuits_pulse_grid(
        circuit_simulation_data=grid_plot_simulation_data,
        pulse_configuration=pulse_configuration,
        observe_rna_species=observe_rna_species,
        use_statistical_summary=use_statistical_summary,
        statistical_summary_type=statistical_summary_type,
        percentile_bounds=percentile_bounds,
        ribbon_alpha=0.25,
        figure_size=(20, 24),
        save_path=unified_output_path,
    )
    return figure, all_axes


def generate_both_individual_and_unified_pulse_plots(
    mcmc_results_by_circuit,
    output_directory=".",
    sample_count=60,
    protein_degradation_rate=0.1,
    equilibration_time=120,
    pulse_duration=10,
    observation_time=200,
    use_statistical_summary=False,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    observe_rna_species="obs_RNA_GFP",
    subtract_equilibrium_baseline=True,
):
    """
    Generate both individual circuit plots and unified grid plot.
    Modified to include baseline correction option.
    """
    prior_file = "../../data/prior/model_parameters_priors_092025_correction.csv"

    circuit_manager = CircuitManager(
        parameters_file=prior_file,
        json_file="../../data/circuits/circuits.json",
    )

    model_priors = pd.read_csv(prior_file)
    parameters_to_fit = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()

    os.makedirs(output_directory, exist_ok=True)

    # Execute simulations once for all circuits and collect data
    circuit_simulation_data_collection = {}

    for circuit_name, mcmc_raw_samples in mcmc_results_by_circuit.items():
        simulation_data = execute_circuit_pulse_simulation_with_equilibration(
            circuit_name=circuit_name,
            mcmc_raw_samples=mcmc_raw_samples,
            parameters_to_fit=parameters_to_fit,
            circuit_manager=circuit_manager,
            protein_degradation_rate=protein_degradation_rate,
            sample_count=sample_count,
            equilibration_time=equilibration_time,
            pulse_duration=pulse_duration,
            observation_time=observation_time,
        )

        circuit_simulation_data_collection[circuit_name] = simulation_data

    # Generate individual plots from simulation data
    # Generate individual plots from simulation data with baseline correction
    for circuit_name, simulation_data in circuit_simulation_data_collection.items():
        generate_individual_pulse_plot_from_simulation_data(
            circuit_name=circuit_name,
            simulation_data=simulation_data,
            output_directory=output_directory,
            use_statistical_summary=use_statistical_summary,
            statistical_summary_type=statistical_summary_type,
            percentile_bounds=percentile_bounds,
            observe_rna_species=observe_rna_species,
            subtract_equilibrium_baseline=subtract_equilibrium_baseline,  # PASS PARAMETER
        )

    # Generate unified grid plot from simulation data
    generate_unified_grid_plot_from_simulation_data(
        circuit_simulation_data_collection=circuit_simulation_data_collection,
        output_directory=output_directory,
        use_statistical_summary=use_statistical_summary,
        statistical_summary_type=statistical_summary_type,
        percentile_bounds=percentile_bounds,
        observe_rna_species=observe_rna_species,
    )

    return circuit_simulation_data_collection


def generate_protein_only_grid_from_simulation_data(
    circuit_simulation_data_collection,
    output_directory,
    pulse_configuration,
    use_statistical_summary=True,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    grid_layout=None,
    use_focused_display=True,
):
    """
    Generate protein-only grid plot from precomputed simulation data.
    Wrapper function that integrates with existing workflow.
    """
    # Get sampling manager from first circuit
    first_circuit_data = list(circuit_simulation_data_collection.values())[0]
    sampling_manager = first_circuit_data["sampling_manager"]

    # Transform simulation data to format expected by protein grid plot
    grid_plot_simulation_data = {}
    for circuit_name, simulation_data in circuit_simulation_data_collection.items():
        grid_plot_simulation_data[circuit_name] = (
            simulation_data["simulation_result"],
            simulation_data["time_span"],
            simulation_data["parameter_dataframe"],
            simulation_data["pulse_plasmids"],
        )

    # Construct output filename
    mode_descriptor = "summary" if use_statistical_summary else "individual"
    summary_type_descriptor = (
        f"_{statistical_summary_type}" if use_statistical_summary else ""
    )
    percentile_descriptor = (
        f"_{percentile_bounds[0]}_{percentile_bounds[1]}"
        if use_statistical_summary and statistical_summary_type == "median_percentiles"
        else ""
    )
    display_descriptor = "_focused" if use_focused_display else "_full"
    equilibration_time = pulse_configuration["equilibration_time"]
    equilibration_descriptor = f"_eq{equilibration_time}min"

    protein_grid_filename = f"protein_only_grid{equilibration_descriptor}{display_descriptor}_{mode_descriptor}{summary_type_descriptor}{percentile_descriptor}.png"
    protein_grid_output_path = os.path.join(output_directory, protein_grid_filename)

    # Generate protein-only grid plot
    figure, axes_collection = sampling_manager.plot_circuits_protein_only_grid(
        circuit_simulation_data=grid_plot_simulation_data,
        pulse_configuration=pulse_configuration,
        use_statistical_summary=use_statistical_summary,
        statistical_summary_type=statistical_summary_type,
        percentile_bounds=percentile_bounds,
        grid_layout=grid_layout,
        use_focused_display=use_focused_display,
        save_path=protein_grid_output_path,
    )
    return figure, axes_collection


def main_both_individual_and_unified_plots():
    """Main function to generate both individual and unified pulse plots efficiently"""
    subfolder = "/2025-11-11_100000_steps_adaptive_new_priors_part_1"
    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_visualization_directory = (
        "../../figures/individual_circuits_pulse" + subfolder
    )

    _mcmc_results = load_individual_circuit_results(input_directory, prefix="results_")

    # Define processing order and inclusion
    circuit_processing_sequence = [
        "sense_star_6",
        "toehold_trigger",
        "cascade",
        "cffl_type_1",
        "or_gate_c1ffl",
        # 'star_antistar_1',
        # 'trigger_antitrigger',
        # 'inhibited_incoherent_cascade',
        # 'inhibited_cascade',
        "iffl_1",
    ]

    filtered_mcmc_results = {
        circuit_name: _mcmc_results[circuit_name]
        for circuit_name in circuit_processing_sequence
        if circuit_name in _mcmc_results
    }

    # Parameters for extended simulation
    equilibration_time = 200  # Minutes to reach steady state
    pulse_duration = 200  # Pulse duration in minutes
    observation_time = 100  # Time to observe after pulse
    protein_degradation_rate = 0.1  # Protein degradation rate in min^-1
    sample_count = 1000  # Number of samples to simulate

    # Generate individual trajectory plots (both individual and grid)
    individual_simulation_data_collection = (
        generate_both_individual_and_unified_pulse_plots(
            filtered_mcmc_results,
            output_visualization_directory,
            sample_count=sample_count,
            protein_degradation_rate=protein_degradation_rate,
            equilibration_time=equilibration_time,
            pulse_duration=pulse_duration,
            observation_time=observation_time,
            use_statistical_summary=False,
            observe_rna_species="obs_RNA_GFP",
            subtract_equilibrium_baseline=True,
        )
    )

    # Generate statistical summary plots (both individual and grid)
    summary_simulation_data_collection = (
        generate_both_individual_and_unified_pulse_plots(
            filtered_mcmc_results,
            output_visualization_directory,
            sample_count=sample_count,
            protein_degradation_rate=protein_degradation_rate,
            equilibration_time=equilibration_time,
            pulse_duration=pulse_duration,
            observation_time=observation_time,
            use_statistical_summary=True,
            statistical_summary_type="median_percentiles",
            percentile_bounds=(10, 90),
            observe_rna_species="obs_RNA_GFP",
            subtract_equilibrium_baseline=True,
        )
    )

    # Generate protein-only grid plots (both individual and statistical)
    pulse_configuration = {
        "pulse_start": equilibration_time + 0.5,
        "pulse_end": equilibration_time + 200.5,
        "equilibration_time": equilibration_time,
    }

    # Individual trajectories protein grid
    generate_protein_only_grid_from_simulation_data(
        individual_simulation_data_collection,
        output_visualization_directory,
        pulse_configuration=pulse_configuration,
        use_statistical_summary=False,
    )

    # Statistical summary protein grid
    generate_protein_only_grid_from_simulation_data(
        summary_simulation_data_collection,
        output_visualization_directory,
        pulse_configuration=pulse_configuration,
        use_statistical_summary=True,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
    )


if __name__ == "__main__":
    main_both_individual_and_unified_plots()
