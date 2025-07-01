import numpy as np
import pandas as pd
from analysis_and_figures.mcmc_analysis_hierarchical import (
    process_mcmc_data,
)


def convert_hierarchical_to_unified_format(
    hierarchical_results_dataframe,
    hierarchical_fitter,
    fitted_parameter_names,
    burn_in_fraction=0.5,
    post_burnin_samples_per_circuit=30000,
    include_global_parameters=True,
):
    """
    Convert hierarchical MCMC results to unified format combining:
    - Circuit-specific parameters (θ)
    - Global parameters (α)
    - Prior means for comparison

    Returns DataFrame in format compatible with individual circuit analysis functions
    """

    # Apply burn-in filtering using existing function
    processed_mcmc_result = process_mcmc_data(
        hierarchical_results_dataframe, burn_in=burn_in_fraction, chain_idx=0
    )
    hierarchical_filtered_samples = processed_mcmc_result["processed_data"]

    print(
        f"Hierarchical MCMC: {processed_mcmc_result['metadata']['n_samples_raw']} → "
        f"{processed_mcmc_result['metadata']['n_samples_processed']} samples after burn-in"
    )

    # Sample from filtered data
    hierarchical_final_samples = (
        hierarchical_filtered_samples.sample(
            n=post_burnin_samples_per_circuit, random_state=42
        )
        if len(hierarchical_filtered_samples) > post_burnin_samples_per_circuit
        else hierarchical_filtered_samples.copy()
    )

    unified_theta_samples = []
    circuit_names = [config.name for config in hierarchical_fitter.configs]

    # Identify available circuit-specific parameter columns
    available_circuit_parameters = []
    for circuit_name in circuit_names:
        circuit_theta_parameters = pd.DataFrame()
        circuit_available_parameters = []

        for parameter_name in fitted_parameter_names:
            theta_column_name = f"theta_{circuit_name}_{parameter_name}"
            if theta_column_name in hierarchical_final_samples.columns:
                circuit_theta_parameters[parameter_name] = hierarchical_final_samples[
                    theta_column_name
                ]
                circuit_available_parameters.append(parameter_name)

        if len(circuit_available_parameters) > 0:
            circuit_theta_parameters["sample_id"] = range(len(circuit_theta_parameters))
            circuit_theta_parameters["type"] = "Circuit"
            circuit_theta_parameters["circuit"] = circuit_name
            circuit_theta_parameters["Circuit"] = circuit_name

            unified_theta_samples.append(circuit_theta_parameters)
            if circuit_name == circuit_names[0]:  # Store once
                available_circuit_parameters = circuit_available_parameters
        else:
            print(f"Warning: No theta parameters found for circuit {circuit_name}")

    # Identify available global parameters (α)
    available_global_parameters = []
    if include_global_parameters:
        global_alpha_parameters = pd.DataFrame()

        for parameter_name in fitted_parameter_names:
            alpha_column_name = f"alpha_{parameter_name}"
            if alpha_column_name in hierarchical_final_samples.columns:
                global_alpha_parameters[parameter_name] = hierarchical_final_samples[
                    alpha_column_name
                ]
                available_global_parameters.append(parameter_name)

        if len(available_global_parameters) > 0:
            global_alpha_parameters["sample_id"] = range(len(global_alpha_parameters))
            global_alpha_parameters["type"] = "Global"
            global_alpha_parameters["circuit"] = "Global_Alpha"
            global_alpha_parameters["Circuit"] = "Global_Alpha"

            unified_theta_samples.append(global_alpha_parameters)
        else:
            print("Warning: No alpha parameters found in hierarchical results")

    # MODIFIED: Use union instead of intersection to include all available parameters
    consensus_available_parameters = list(
        set(available_circuit_parameters) | set(available_global_parameters)
    )

    # Filter datasets to only include consensus parameters
    filtered_unified_samples = []
    for dataset in unified_theta_samples:
        # Keep only consensus parameters plus metadata columns
        metadata_columns = ["sample_id", "type", "circuit", "Circuit"]
        parameter_columns = [
            p for p in consensus_available_parameters if p in dataset.columns
        ]
        columns_to_keep = metadata_columns + parameter_columns

        filtered_dataset = dataset[columns_to_keep].copy()
        filtered_unified_samples.append(filtered_dataset)

    print(f"Available parameters for analysis: {consensus_available_parameters}")

    if not filtered_unified_samples:
        raise ValueError("No valid parameter columns found in hierarchical results")

    return pd.concat(
        filtered_unified_samples, ignore_index=True
    ), consensus_available_parameters


def generate_hierarchical_prior_statistics(
    prior_parameters_filepath,
    fitted_parameter_names,
):
    """
    Generate prior statistics for vertical line rendering in hierarchical plots.
    Returns single-row DataFrame with prior means and std deviations in format expected by plotting functions.
    """
    prior_parameter_specifications = pd.read_csv(prior_parameters_filepath)

    # Prior means and std deviations
    prior_statistics_coordinates = {"sample_id": [0]}

    # Track available parameters
    available_prior_parameters = []
    for _, prior_specification_row in prior_parameter_specifications.iterrows():
        if prior_specification_row["Parameter"] in fitted_parameter_names:
            parameter_name = prior_specification_row["Parameter"]
            log10_mean = np.log10(prior_specification_row["Mean"])
            prior_statistics_coordinates[parameter_name] = [log10_mean]

            # Add std deviation column in format expected by plotting function
            log10_std_column_name = f"{parameter_name}_log10stdev"
            prior_statistics_coordinates[log10_std_column_name] = [
                prior_specification_row["log10stddev"]
            ]

            available_prior_parameters.append(parameter_name)

    # Validate parameter coverage
    missing_priors = set(fitted_parameter_names) - set(available_prior_parameters)
    if missing_priors:
        print(
            f"Warning: No prior specifications found for parameters: {missing_priors}"
        )
        print(f"Using only parameters with priors: {available_prior_parameters}")

    prior_statistics_coordinates.update(
        {"type": ["Prior"], "circuit": ["Prior"], "Circuit": ["Prior"]}
    )

    return pd.DataFrame(prior_statistics_coordinates), available_prior_parameters


def prepare_hierarchical_pairplot_data_processed(
    processed_df, param_names, circuit_names, n_samples=100
):
    """
    Prepare pairplot data from already-processed dataframe (no burn-in filtering)
    """

    # Sample for performance if needed
    if len(processed_df) > n_samples:
        df_sample = processed_df.sample(n=n_samples, random_state=42).reset_index(
            drop=True
        )
    else:
        df_sample = processed_df.reset_index(drop=True)

    print(f"Using {len(df_sample)} processed samples for pairplot")

    # Add sample_id for tracking
    df_sample["sample_id"] = df_sample.index

    # --- Global parameters ---
    global_cols = [
        f"alpha_{p}" for p in param_names if f"alpha_{p}" in df_sample.columns
    ]
    global_df = df_sample[["sample_id"] + global_cols].copy()
    global_df = global_df.rename(
        columns={
            f"alpha_{p}": p for p in param_names if f"alpha_{p}" in df_sample.columns
        }
    )
    global_df["type"] = "Global"
    global_df["circuit"] = "Global"

    # --- Circuit-specific parameters ---
    circuit_dfs = []
    for circuit in circuit_names:
        theta_cols = [
            f"theta_{circuit}_{p}"
            for p in param_names
            if f"theta_{circuit}_{p}" in df_sample.columns
        ]
        if not theta_cols:
            continue
        circuit_df = df_sample[["sample_id"] + theta_cols].copy()
        circuit_df = circuit_df.rename(
            columns={
                f"theta_{circuit}_{p}": p
                for p in param_names
                if f"theta_{circuit}_{p}" in df_sample.columns
            }
        )
        circuit_df["type"] = "Circuit"
        circuit_df["circuit"] = circuit
        circuit_dfs.append(circuit_df)

    # Combine all data
    pairplot_df = pd.concat([global_df] + circuit_dfs, ignore_index=True)
    pairplot_df["Circuit"] = pairplot_df["type"] + " - " + pairplot_df["circuit"]

    print(
        f"Created pairplot data with {len(pairplot_df)} rows ({len(df_sample)} samples × {1 + len(circuit_names)} parameter sets)"
    )

    return pairplot_df
