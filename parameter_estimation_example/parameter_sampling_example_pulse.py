"""
Parameter Sampling and Visualization for Synthetic Circuits

This module provides functions for loading and visualizing parameter sets
from circuit fitting results, and comparing circuit behavior with different
parameter configurations.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from circuits.circuit_generation.circuit_manager import CircuitManager
from circuits.circuit_generation.parameter_sampling_and_simulation import (
    ParameterSamplingManager,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default configuration settings
DEFAULT_CONFIG = {
    "results_dir": "../data/fit_data/individual_circuits",
    "parameters_file": "../data/prior/model_parameters_priors.csv",
    "circuits_file": "../data/circuits/circuits.json",
    "output_dir": "../figures/parameter_sampling_plots",
    "pulse_config": {
        "use_pulse": True,
        "pulse_start": 30,  # Start pulse at 30 minutes
        "pulse_end": 40,  # End pulse at 40 minutes
        "pulse_concentration": 5.0,
        "base_concentration": 0.0,
    },
    "k_prot_deg_values": [0.1, 0.2, 0.5],
    "random_n_params": 50,
    "figsize": (6, 10),
    # Plot configuration options
    "show_protein": True,
    "show_rna": False,
    "show_pulse": True,
}

# Define which plasmids to pulse for each circuit
# This mapping specifies exactly which plasmid to pulse for each circuit type
CIRCUIT_PULSE_PLASMIDS = {
    # Circuit name -> List of plasmid names to pulse
    "sense_star_6": ["pr-star6_plasmid"],
    "and_gate": ["star6_expression_plasmid", "trigger3_expression_plasmid"],
    "cffl_type_1": ["star6_expression"],
    "cascade": ["star6_plasmid"],
    "toehold_trigger": ["trigger3_plasmid"],
    "star_antistar_1": ["star1_plasmid"],
}


def load_circuit_results(results_dir=None):
    """
    Load all circuit fitting results from CSV files.

    Parameters:
    -----------
    results_dir : str, optional
        Directory containing result CSV files. Defaults to config value.

    Returns:
    --------
    dict
        Dictionary mapping circuit names to their results DataFrames
    """
    results_dir = results_dir or DEFAULT_CONFIG["results_dir"]

    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return {}

    results = {}
    pattern = os.path.join(results_dir, "results_*.csv")
    result_files = glob.glob(pattern)

    if not result_files:
        logger.warning(f"No result files found matching pattern: {pattern}")
        return {}

    for file_path in result_files:
        try:
            # Extract circuit name from filename
            filename = os.path.basename(file_path)
            circuit_name = "_".join(filename.split("_")[1:-2])

            # Load results
            df = pd.read_csv(file_path)
            results[circuit_name] = df

            logger.info(f"Loaded {circuit_name} results from {filename}")
            logger.info(f"  Number of samples: {len(df)}")
            logger.info(f"  Best likelihood: {df['likelihood'].max():.2f}")

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")

    return results


def extract_parameters_from_results(df, random_n=10, convert_from_log=True):
    """
    Extract parameters from results DataFrame for visualization.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing circuit fitting results
    random_n : int, optional
        Number of random parameters to extract
    convert_from_log : bool, optional
        Whether to convert parameters from log10 scale to linear scale

    Returns:
    --------
    pandas.DataFrame
        DataFrame with extracted parameter values
    """

    # only keep chain 0
    df = df[df["chain"] == 0]

    # burn the first 50% of the samples (steps should be > than 50% of the total steps)
    df = df[df["iteration"] > df["iteration"].max() / 1.5]

    # Sample random parameters if requested
    if random_n:
        random_indices = np.random.choice(df.index, size=random_n, replace=False)
        random_params = df.loc[random_indices]

    # # only select the 5 best likelihood samples
    # random_params = df.nlargest(random_n, 'likelihood')

    # remove iteration, walker, chain, likelihood, posterior, prior, step_accepted columns
    cols_to_remove = [
        "iteration",
        "walker",
        "chain",
        "likelihood",
        "posterior",
        "prior",
        "step_accepted",
    ]
    random_params = random_params.drop(columns=cols_to_remove, errors="ignore")

    # take 10**values
    if convert_from_log:
        random_params = 10**random_params

    return random_params


def get_pulse_plasmids_for_circuit(circuit_name):
    """
    Get the plasmid names to pulse for a specific circuit.

    Parameters:
    -----------
    circuit_name : str
        Name of the circuit

    Returns:
    --------
    list
        List of plasmid names to pulse
    """
    # Get plasmid names to pulse from the mapping
    if circuit_name in CIRCUIT_PULSE_PLASMIDS:
        return CIRCUIT_PULSE_PLASMIDS[circuit_name]
    else:
        logger.warning(
            f"No pulse plasmids defined for circuit '{circuit_name}'. Using fallback mechanism."
        )
        # Fallback: use a generic name that likely exists in the circuit
        return [f"{circuit_name}_plasmid_0"]


def visualize_fits_with_sampler(
    circuit_name,
    parameters,
    circuit_manager,
    output_dir=None,
    pulse_config=None,
    show_protein=True,
    show_rna=True,
    show_pulse=True,
):
    """
    Visualize circuit fits using the ParameterSamplingManager with configurable plot options.

    Parameters:
    -----------
    circuit_name : str
        Name of the circuit to visualize
    parameters : dict
        Dictionary with parameter values (from extract_parameters_from_results)
    circuit_manager : CircuitManager
        Circuit manager instance
    output_dir : str, optional
        Directory to save output plots
    pulse_config : dict, optional
        Pulse configuration for visualization
    show_protein : bool, optional
        Whether to show protein dynamics in the plots (default: True)
    show_rna : bool, optional
        Whether to show RNA dynamics in the plots (default: True)
    show_pulse : bool, optional
        Whether to show pulse profile in the plots (default: True)
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create parameter sampling manager
    sampling_manager = ParameterSamplingManager(circuit_manager)

    # Create time span for simulation
    t_span = np.linspace(0, 480, 501)  # 0 to 480 minutes with 501 points

    # Fixed protein degradation rate
    k_prot_deg = 0.1
    logger.info(f"Visualizing circuit {circuit_name} with k_prot_deg={k_prot_deg}")

    # Get plasmid names to pulse for this circuit
    pulse_plasmids = get_pulse_plasmids_for_circuit(circuit_name)
    logger.info(f"Using plasmids {pulse_plasmids} for pulsing")

    # Define save path
    save_path = None
    if output_dir:
        save_path = os.path.join(output_dir, f"{circuit_name}_parameter_sweep.png")

    # Adjust figure size based on number of plots shown
    num_plots = sum([show_protein, show_rna, show_pulse])
    figsize = (6, 3 * num_plots)  # Allocate roughly 3 inches of height per plot

    # Run the parameter sweep visualization with the selected plot options
    sampling_manager.plot_parameter_sweep_with_pulse(
        circuit_name=circuit_name,
        param_df=parameters,
        k_prot_deg=k_prot_deg,
        pulse_configuration=pulse_config,
        pulse_plasmids=pulse_plasmids,  # Use specific plasmid names for this circuit
        t_span=t_span,
        figsize=figsize,
        save_path=save_path,
        show_protein=show_protein,
        show_rna=show_rna,
        show_pulse=show_pulse,
    )


def run_parameter_analysis(config=None):
    """
    Run a complete parameter analysis workflow.

    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary overriding default settings
    """
    # Merge default config with provided config
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    # Create output directory
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Load circuit results
    logger.info("Loading circuit results...")
    results = load_circuit_results(cfg["results_dir"])

    if not results:
        logger.error("No circuit results loaded. Exiting.")
        return

    # Initialize circuit manager
    logger.info("Initializing circuit manager...")
    circuit_manager = CircuitManager(
        parameters_file=cfg["parameters_file"], json_file=cfg["circuits_file"]
    )

    # Load parameters
    # priors = pd.read_csv(cfg["parameters_file"])
    # Exclude protein degradation rate since we'll set it manually
    # prior_parameters = priors[priors["Parameter"] != "k_prot_deg"]

    # Get circuit names to process
    circuit_names = sorted(list(results.keys()))
    logger.info(
        f"Found {len(circuit_names)} circuits to process: {', '.join(circuit_names)}"
    )

    # Get plot configuration options (with defaults if not provided)
    show_protein = cfg.get("show_protein", True)
    show_rna = cfg.get("show_rna", True)
    show_pulse = cfg.get("show_pulse", True)

    # Process each circuit
    for circuit_name in circuit_names[:]:
        if circuit_name not in results:
            logger.warning(f"No results found for circuit {circuit_name}")
            continue

        logger.info(f"Processing circuit {circuit_name}")

        # Extract parameters from results
        parameters = extract_parameters_from_results(
            results[circuit_name],
            random_n=cfg["random_n_params"],
            convert_from_log=True,
        )

        # Create circuit-specific output directory
        circuit_output_dir = os.path.join(output_dir, circuit_name)

        # Visualize with parameter sampling
        visualize_fits_with_sampler(
            circuit_name,
            parameters,
            circuit_manager,
            output_dir=circuit_output_dir,
            pulse_config=cfg["pulse_config"],
            show_protein=show_protein,
            show_rna=show_rna,
            show_pulse=show_pulse,
        )


def main():
    """Main function demonstrating the use of parameter sampling with fitted results"""
    # Run the full analysis with default configuration
    run_parameter_analysis()


if __name__ == "__main__":
    main()
