"""
Mixed Effects Model Analysis
Analyzes mixed effects MCMC results by splitting into shared (β) and hierarchical (α) components
"""

import os
import pandas as pd
from analysis_and_figures.hierarchical_pairplot_analysis import (
    create_hierarchical_histogram_grid,
)
from analysis_and_figures.process_hierarchical import (
    generate_hierarchical_prior_statistics,
)


def execute_mixed_effects_unified_analysis(
    mixed_effects_results_filepath,
    prior_parameters_filepath,
    fitted_parameter_names,
    output_visualization_directory,
    circuits_to_analyze=None,
    burn_in_fraction=0.5,
):
    """
    Execute complete mixed effects analysis pipeline by splitting into:
    1. β (shared/fixed) parameter analysis
    2. α (global hierarchical) parameter analysis
    """

    os.makedirs(output_visualization_directory, exist_ok=True)

    # Setup hierarchical model (reuse existing function)
    from simulations_and_analysis.hierarchical.hierarchial_design_from_file import (
        setup_hierarchical_model,
    )

    if circuits_to_analyze is None:
        circuits_to_analyze = [
            "trigger_antitrigger",
            "toehold_trigger",
            "sense_star_6",
            "cascade",
            "cffl_type_1",
            "star_antistar_1",
        ]

    hierarchical_fitter, _ = setup_hierarchical_model(circuits_to_analyze)

    # Load mixed effects results
    mixed_effects_dataframe = pd.read_csv(mixed_effects_results_filepath)

    # Handle column naming issue if present
    if (
        "walker" in mixed_effects_dataframe.columns
        and "iteration" in mixed_effects_dataframe.columns
    ):
        mixed_effects_dataframe = mixed_effects_dataframe.rename(
            columns={"walker": "iteration", "iteration": "walker"}
        )

    # Infer actual parameter structure from column names (ignore fitter configuration)
    actual_shared_parameters = []
    actual_hierarchical_parameters = []
    circuit_names = [config.name for config in hierarchical_fitter.configs]

    # Extract β parameters
    beta_columns = [
        col for col in mixed_effects_dataframe.columns if col.startswith("beta_")
    ]
    actual_shared_parameters = [col.replace("beta_", "") for col in beta_columns]

    # Extract α parameters
    alpha_columns = [
        col for col in mixed_effects_dataframe.columns if col.startswith("alpha_")
    ]
    actual_hierarchical_parameters = [
        col.replace("alpha_", "") for col in alpha_columns
    ]

    print(f"Loaded mixed effects results: {len(mixed_effects_dataframe)} samples")
    print(f"Available circuits: {circuit_names}")
    print(f"Actual shared parameters (β): {actual_shared_parameters}")
    print(f"Actual hierarchical parameters (α): {actual_hierarchical_parameters}")

    # Verify θ parameters exist for hierarchical parameters
    for circuit_name in circuit_names:
        for param_name in actual_hierarchical_parameters:
            theta_column = f"theta_{circuit_name}_{param_name}"
            if theta_column not in mixed_effects_dataframe.columns:
                print(f"WARNING: Missing θ parameter {theta_column}")

    # Override fitter parameter lists with actual structure
    hierarchical_fitter.shared_parameter_names = actual_shared_parameters
    hierarchical_fitter.hierarchical_parameter_names = actual_hierarchical_parameters
    hierarchical_fitter.n_shared_params = len(actual_shared_parameters)
    hierarchical_fitter.n_hierarchical_params = len(actual_hierarchical_parameters)

    # =========================================================================
    # ANALYSIS 1: β (SHARED/FIXED) PARAMETERS
    # =========================================================================

    print("\n=== ANALYZING SHARED (β) PARAMETERS ===")

    if hierarchical_fitter.n_shared_params > 0:
        shared_unified_dataset, shared_consensus_parameters = (
            convert_mixed_effects_shared_to_unified_format(
                mixed_effects_dataframe,
                hierarchical_fitter,
                burn_in_fraction=burn_in_fraction,
            )
        )

        # Generate prior statistics for shared parameters
        shared_prior_statistics, shared_available_prior_parameters = (
            generate_hierarchical_prior_statistics(
                prior_parameters_filepath,
                shared_consensus_parameters,
            )
        )

        # Combine datasets: MCMC data + prior statistics
        shared_comparison_dataset = pd.concat(
            [shared_unified_dataset, shared_prior_statistics], ignore_index=True
        )

        print(f"Shared parameter dataset: {len(shared_comparison_dataset)} samples")
        print(f"Shared parameters for visualization: {shared_consensus_parameters}")

        # Generate shared parameter visualizations
        print("Creating shared parameter histogram grid...")
        create_hierarchical_histogram_grid(
            shared_comparison_dataset,
            shared_consensus_parameters,
            output_visualization_directory + "/shared_beta_ridgeline_unified.png",
            plot_kind="ridge",
            ridge_offset=1.1,
            ridge_label_pad=-0.0,
        )

        create_hierarchical_histogram_grid(
            shared_comparison_dataset,
            shared_consensus_parameters,
            output_visualization_directory + "/shared_beta_kde_grid_unified.png",
            plot_kind="kde",
            share_y=True,
            legend_on_idx=1,
        )
    else:
        print("No shared parameters found in model")

    # =========================================================================
    # ANALYSIS 2: α (GLOBAL HIERARCHICAL) PARAMETERS
    # =========================================================================

    print("\n=== ANALYZING GLOBAL HIERARCHICAL (α) PARAMETERS ===")

    if hierarchical_fitter.n_hierarchical_params > 0:
        global_unified_dataset, global_consensus_parameters = (
            convert_mixed_effects_global_to_unified_format(
                mixed_effects_dataframe,
                hierarchical_fitter,
                burn_in_fraction=burn_in_fraction,
            )
        )

        # Generate prior statistics for global parameters
        global_prior_statistics, global_available_prior_parameters = (
            generate_hierarchical_prior_statistics(
                prior_parameters_filepath,
                global_consensus_parameters,
            )
        )

        # Combine datasets: MCMC data + prior statistics
        global_comparison_dataset = pd.concat(
            [global_unified_dataset, global_prior_statistics], ignore_index=True
        )

        print(f"Global parameter dataset: {len(global_comparison_dataset)} samples")
        print(f"Global parameters for visualization: {global_consensus_parameters}")

        # Generate global parameter visualizations
        print("Creating global parameter histogram grid...")
        create_hierarchical_histogram_grid(
            global_comparison_dataset,
            global_consensus_parameters,
            output_visualization_directory + "/global_alpha_ridgeline_unified.png",
            plot_kind="ridge",
            ridge_offset=1.1,
            ridge_label_pad=-0.0,
        )

        create_hierarchical_histogram_grid(
            global_comparison_dataset,
            global_consensus_parameters,
            output_visualization_directory + "/global_alpha_kde_grid_unified.png",
            plot_kind="kde",
            share_y=True,
            legend_on_idx=1,
        )
    else:
        print("No hierarchical parameters found in model")

    return {
        "shared_dataset": shared_comparison_dataset
        if hierarchical_fitter.n_shared_params > 0
        else None,
        "global_dataset": global_comparison_dataset
        if hierarchical_fitter.n_hierarchical_params > 0
        else None,
        "hierarchical_fitter": hierarchical_fitter,
    }


def convert_mixed_effects_shared_to_unified_format(
    mixed_effects_dataframe,
    hierarchical_fitter,
    burn_in_fraction=0.5,
):
    """
    Convert mixed effects β (shared) parameters to unified format for visualization
    """

    # Filter burn-in period
    max_iteration = mixed_effects_dataframe["iteration"].max()
    burn_in_cutoff = int(max_iteration * burn_in_fraction)
    filtered_dataframe = mixed_effects_dataframe[
        mixed_effects_dataframe["iteration"] > burn_in_cutoff
    ].copy()

    # Extract β parameter columns
    shared_parameter_columns = []
    for param_name in hierarchical_fitter.shared_parameter_names:
        beta_column = f"beta_{param_name}"
        if beta_column in filtered_dataframe.columns:
            shared_parameter_columns.append(beta_column)

    if not shared_parameter_columns:
        raise ValueError("No shared parameter columns found in dataframe")

    # Create unified dataset
    unified_records = []

    for _, mcmc_sample in filtered_dataframe.iterrows():
        record = {"Circuit": "Shared_Beta", "type": "Circuit"}

        # Add β parameter values
        for beta_column in shared_parameter_columns:
            param_name = beta_column.replace("beta_", "")
            record[param_name] = mcmc_sample[beta_column]

        unified_records.append(record)

    unified_dataset = pd.DataFrame(unified_records)

    # Get consensus parameter names (remove beta_ prefix)
    consensus_parameters = [
        col.replace("beta_", "") for col in shared_parameter_columns
    ]

    return unified_dataset, consensus_parameters


def convert_mixed_effects_global_to_unified_format(
    mixed_effects_dataframe,
    hierarchical_fitter,
    burn_in_fraction=0.5,
):
    """
    Convert mixed effects α (global) + θ (circuit-specific) parameters to unified format for visualization
    Similar to hierarchical case: shows global alpha + circuit-specific theta together
    """

    # Filter burn-in period
    max_iteration = mixed_effects_dataframe["iteration"].max()
    burn_in_cutoff = int(max_iteration * burn_in_fraction)
    filtered_dataframe = mixed_effects_dataframe[
        mixed_effects_dataframe["iteration"] > burn_in_cutoff
    ].copy()

    # Extract α parameter columns
    alpha_parameter_columns = []
    for param_name in hierarchical_fitter.hierarchical_parameter_names:
        alpha_column = f"alpha_{param_name}"
        if alpha_column in filtered_dataframe.columns:
            alpha_parameter_columns.append(alpha_column)

    # Extract θ parameter columns for each circuit
    circuit_names = [config.name for config in hierarchical_fitter.configs]
    theta_parameter_columns = []
    for circuit_name in circuit_names:
        for param_name in hierarchical_fitter.hierarchical_parameter_names:
            theta_column = f"theta_{circuit_name}_{param_name}"
            if theta_column in filtered_dataframe.columns:
                theta_parameter_columns.append(theta_column)

    if not alpha_parameter_columns and not theta_parameter_columns:
        raise ValueError("No hierarchical parameter columns found in dataframe")

    # Create unified dataset
    unified_records = []

    for _, mcmc_sample in filtered_dataframe.iterrows():
        # Add α (global) parameter record
        if alpha_parameter_columns:
            alpha_record = {"Circuit": "Global_Alpha", "type": "Global"}
            for alpha_column in alpha_parameter_columns:
                param_name = alpha_column.replace("alpha_", "")
                alpha_record[param_name] = mcmc_sample[alpha_column]
            unified_records.append(alpha_record)

        # Add θ (circuit-specific) parameter records
        for circuit_name in circuit_names:
            circuit_theta_columns = [
                col
                for col in theta_parameter_columns
                if col.startswith(f"theta_{circuit_name}_")
            ]
            if circuit_theta_columns:
                theta_record = {"Circuit": circuit_name, "type": "Circuit"}
                for theta_column in circuit_theta_columns:
                    param_name = theta_column.replace(f"theta_{circuit_name}_", "")
                    theta_record[param_name] = mcmc_sample[theta_column]
                unified_records.append(theta_record)

    unified_dataset = pd.DataFrame(unified_records)

    # Get consensus parameter names (hierarchical parameters only)
    consensus_parameters = hierarchical_fitter.hierarchical_parameter_names

    return unified_dataset, consensus_parameters


def main():
    """Execute mixed effects unified analysis"""

    mixed_effects_data_id = "hierarchical_results_20250703_063716"

    # Configuration
    mixed_effects_results_filepath = (
        "../../data/fit_data/mixed_effect/" + mixed_effects_data_id + ".csv"
    )
    prior_parameters_filepath = "../../data/prior/model_parameters_priors.csv"
    output_visualization_directory = (
        "../../figures/mixed_effects/" + mixed_effects_data_id
    )

    # Load parameter specifications
    prior_parameters = pd.read_csv(prior_parameters_filepath)
    fitted_parameter_names = prior_parameters[
        prior_parameters["Parameter"] != "k_prot_deg"
    ]["Parameter"].tolist()

    # Remove RNA degradation and km parameters if needed
    fitted_parameter_names = [
        p for p in fitted_parameter_names if p not in ["k_rna_deg", "k_rna_km"]
    ]

    print(f"Fitted parameters: {fitted_parameter_names}")

    # Execute mixed effects analysis pipeline
    _ = execute_mixed_effects_unified_analysis(
        mixed_effects_results_filepath=mixed_effects_results_filepath,
        prior_parameters_filepath=prior_parameters_filepath,
        fitted_parameter_names=fitted_parameter_names,
        output_visualization_directory=output_visualization_directory,
        burn_in_fraction=0.5,
    )

    print(
        f"Mixed effects unified analysis complete. Results saved to: {output_visualization_directory}"
    )


if __name__ == "__main__":
    main()
