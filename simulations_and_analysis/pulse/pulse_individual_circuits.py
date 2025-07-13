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
}


def get_pulse_plasmids_for_circuit(circuit_name):
    """Get plasmid names to pulse for specific circuit"""
    return PULSE_PLASMID_MAPPING.get(circuit_name, [f"{circuit_name}_plasmid_0"])


def get_circuit_pulse_configuration(circuit_name):
    """Get circuit-specific pulse parameters"""
    circuit_specific_concentrations = {
        "sense_star_6": 15.0,
        "cffl_type_1": 15.0,
        "cascade": 15.0,
        "toehold_trigger": 15.0,
        "star_antistar_1": 15.0,
        "trigger_antitrigger": 15.0,
    }

    return {
        "use_pulse": True,
        "pulse_start": 30.5,
        "pulse_end": 40.5,
        "pulse_concentration": circuit_specific_concentrations.get(circuit_name, 15.0),
        "base_concentration": 0.0,  # Only for pulsed plasmid
    }


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
        mcmc_raw_samples, burn_in=0.4, chain_idx=0
    )
    mcmc_filtered_samples = mcmc_processed_result["processed_data"]

    # Get experimental baseline concentrations for non-pulsed plasmids
    circuit_experimental_conditions = get_circuit_conditions(circuit_name)
    first_experimental_condition = list(circuit_experimental_conditions.keys())[0]
    experimental_baseline_concentrations = circuit_experimental_conditions[
        first_experimental_condition
    ]

    pulse_plasmids = get_pulse_plasmids_for_circuit(circuit_name)
    pulse_configuration = get_circuit_pulse_configuration(circuit_name)

    # SYSTEMATIC parameter identification using circuit definition
    plasmid_to_parameter_mapping = extract_plasmid_to_parameter_mapping(
        circuit_manager, circuit_name
    )

    # Log the configuration
    print(f"\n=== {circuit_name} Pulse Setup ===")
    print(f"Pulsed plasmids: {pulse_plasmids}")
    print(f"Pulse concentration: {pulse_configuration['pulse_concentration']} nM")
    print(f"Plasmid-to-parameter mapping: {plasmid_to_parameter_mapping}")
    print(
        f"Experimental baseline concentrations: {experimental_baseline_concentrations}"
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
        if parameter_name in pulsed_concentration_parameters:
            print(
                f"  PULSED: {parameter_name} = 0.0 nM (baseline) → {pulse_configuration['pulse_concentration']} nM (pulse)"
            )
        else:
            non_pulsed_concentrations[parameter_name] = concentration_value
            print(
                f"  NON-PULSED: {parameter_name} = {concentration_value} nM (maintained)"
            )
    print("=== End Configuration ===\n")

    return {
        "mcmc_processed_samples": mcmc_filtered_samples,
        "pulse_configuration": pulse_configuration,
        "pulse_plasmids": pulse_plasmids,
        "non_pulsed_concentrations": non_pulsed_concentrations,
        "experimental_baseline_concentrations": experimental_baseline_concentrations,
    }


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
    }

    pulse_concentration = circuit_specific_concentrations.get(circuit_name, 15.0)

    return {
        "use_pulse": True,
        "pulse_start": equilibration_time + 0.5,  # Start pulse after equilibration
        "pulse_end": equilibration_time + 10.5,  # End pulse 10 minutes later
        "pulse_concentration": pulse_concentration,
        "base_concentration": 0.0,
        "equilibration_time": equilibration_time,
    }


def create_extended_time_span(
    equilibration_time=120, pulse_duration=10, observation_time=60
):
    """Create extended time span including equilibration, pulse, and observation periods"""
    total_simulation_time = equilibration_time + pulse_duration + observation_time
    # Use higher resolution for better temporal accuracy
    time_points_count = int(total_simulation_time * 5) + 1  # 5 points per minute
    return np.linspace(0, total_simulation_time, time_points_count)


def simulate_circuit_pulse_batch_with_equilibration(
    circuit_name,
    mcmc_raw_samples,
    parameters_to_fit,
    circuit_manager,
    protein_degradation_rate=0.1,
    sample_count=50,
    equilibration_time=120,
    pulse_duration=10,
    observation_time=60,
    output_directory=None,
    use_statistical_summary=False,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    observe_rna_species="obs_RNA_GFP",
):
    """
    Enhanced batch pulse simulation with pre-equilibration period
    """
    # Create extended time span
    extended_time_span = create_extended_time_span(
        equilibration_time, pulse_duration, observation_time
    )

    # Get pulse configuration with equilibration timing
    pulse_configuration = get_circuit_pulse_configuration_with_equilibration(
        circuit_name, equilibration_time
    )

    # REUSE existing MCMC processing logic
    pulse_simulation_data = create_pulse_circuit_simulation_data(
        circuit_name,
        mcmc_raw_samples,
        circuit_manager,
    )

    mcmc_filtered_samples = pulse_simulation_data["mcmc_processed_samples"]
    pulse_plasmids = pulse_simulation_data["pulse_plasmids"]
    non_pulsed_concentrations = pulse_simulation_data["non_pulsed_concentrations"]

    print(
        f"{circuit_name}: {len(mcmc_raw_samples)} → {len(mcmc_filtered_samples)} samples after burn-in"
    )
    print(
        f"Equilibration: {equilibration_time} min, Pulse: {pulse_configuration['pulse_start']:.1f}-{pulse_configuration['pulse_end']:.1f} min"
    )

    # Sample parameters for pulse simulation
    final_sample_size = min(sample_count, len(mcmc_filtered_samples))
    print(f"Sampling {final_sample_size} parameters for extended pulse simulation...")
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

    # Add likelihood scores for visualization
    if "likelihood" in sampled_mcmc_parameters.columns:
        linear_parameter_values["likelihood"] = sampled_mcmc_parameters["likelihood"]

    # Filter outliers (REUSE existing filtering logic)
    likelihood_scores = linear_parameter_values.get("likelihood")
    parameter_subset = linear_parameter_values.drop(
        columns=["likelihood"], errors="ignore"
    )
    parameter_subset_filtered = parameter_subset[
        (parameter_subset > parameter_subset.quantile(0.05))
        & (parameter_subset < parameter_subset.quantile(0.95))
    ].dropna()

    # Prepare additional parameters: protein degradation + non-pulsed concentrations
    additional_simulation_parameters = {"k_prot_deg": protein_degradation_rate}
    additional_simulation_parameters.update(non_pulsed_concentrations)

    # Create ParameterSamplingManager and run extended pulse simulation
    sampling_manager = ParameterSamplingManager(circuit_manager)

    # Construct output path
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
        equilibration_descriptor = f"_eq{equilibration_time}min"
        filename = f"{circuit_name}_pulse_extended{equilibration_descriptor}_{mode_descriptor}{summary_type_descriptor}{percentile_descriptor}.png"
        output_path = os.path.join(output_directory, filename)

    # Determine figure size (accounting for longer time series)
    subplot_count = sum([True, observe_rna_species is not None, True])
    figure_size = (10, 3 * subplot_count + 1)  # Wider for extended time series

    # Execute extended pulse simulation
    sampling_manager.plot_parameter_sweep_with_pulse(
        circuit_name=circuit_name,
        param_df=parameter_subset_filtered,
        k_prot_deg=protein_degradation_rate,
        pulse_configuration=pulse_configuration,
        pulse_plasmids=pulse_plasmids,
        t_span=extended_time_span,
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
        additional_params=additional_simulation_parameters,
    )

    return len(sampled_mcmc_parameters)


def plot_fits_with_extended_pulse_simulation(
    mcmc_results_by_circuit,
    output_directory=".",
    sample_count=60,
    protein_degradation_rate=0.1,
    equilibration_time=120,
    pulse_duration=10,
    observation_time=60,
    use_statistical_summary=False,
    statistical_summary_type="median_percentiles",
    percentile_bounds=(10, 90),
    observe_rna_species="obs_RNA_GFP",
):
    """Generate extended pulse simulation plots with pre-equilibration"""
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    model_priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    parameters_to_fit = model_priors[
        model_priors["Parameter"] != "k_prot_deg"
    ].Parameter.tolist()

    os.makedirs(output_directory, exist_ok=True)

    for circuit_name, mcmc_raw_samples in mcmc_results_by_circuit.items():
        print(f"Simulating {circuit_name} with extended pulse processing...")

        sample_count_used = simulate_circuit_pulse_batch_with_equilibration(
            circuit_name=circuit_name,
            mcmc_raw_samples=mcmc_raw_samples,
            parameters_to_fit=parameters_to_fit,
            circuit_manager=circuit_manager,
            protein_degradation_rate=protein_degradation_rate,
            sample_count=sample_count,
            equilibration_time=equilibration_time,
            pulse_duration=pulse_duration,
            observation_time=observation_time,
            output_directory=output_directory,
            use_statistical_summary=use_statistical_summary,
            statistical_summary_type=statistical_summary_type,
            percentile_bounds=percentile_bounds,
            observe_rna_species=observe_rna_species,
        )

        print(
            f"✓ Completed {circuit_name}: {sample_count_used} parameter sets processed"
        )

    print(f"All extended pulse simulations completed. Output: {output_directory}")


# Modified main function to use extended pulse simulation
def main_with_equilibration():
    subfolder = "/tighter_3"
    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_visualization_directory = (
        "../../figures/individual_circuits_pulse" + subfolder
    )

    mcmc_results = load_individual_circuit_results(input_directory)

    print("Generating individual circuit plots with 120min pre-equilibration...")

    # Parameters for extended simulation
    equilibration_time = 300  # Minutes to reach steady state
    pulse_duration = 10  # Pulse duration in minutes
    observation_time = 60  # Time to observe after pulse

    plot_fits_with_extended_pulse_simulation(
        mcmc_results,
        output_visualization_directory,
        sample_count=300,
        protein_degradation_rate=0.1,
        equilibration_time=equilibration_time,
        pulse_duration=pulse_duration,
        observation_time=observation_time,
        use_statistical_summary=False,
        observe_rna_species="obs_RNA_GFP",
    )

    plot_fits_with_extended_pulse_simulation(
        mcmc_results,
        output_visualization_directory,
        sample_count=300,
        protein_degradation_rate=0.1,
        equilibration_time=equilibration_time,
        pulse_duration=pulse_duration,
        observation_time=observation_time,
        use_statistical_summary=True,
        statistical_summary_type="median_percentiles",
        percentile_bounds=(10, 90),
        observe_rna_species="obs_RNA_GFP",
    )

    print("\nAll extended pulse plots completed!")


if __name__ == "__main__":
    main_with_equilibration()
