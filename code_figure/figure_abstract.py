import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your project-specific modules
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from likelihood_functions.hierarchical_likelihood.base_hierarchical import (
    HierarchicalCircuitFitter,
)
from likelihood_functions.hierarchical_likelihood.mcmc_analysis_hierarchical import (
    process_mcmc_data,
)


def setup_hierarchical_model(circuits_to_fit):
    """Setup the hierarchical model components"""
    circuit_manager = CircuitManager(
        parameters_file="../data/prior/model_parameters_priors.csv",
        json_file="../data/circuits/circuits.json",
    )

    available_circuits = circuit_manager.list_circuits()
    circuits_to_fit = [c for c in circuits_to_fit if c in available_circuits]

    circuit_configs = create_circuit_configs(
        circuit_manager, circuits_to_fit, min_time=30, max_time=210
    )

    priors = pd.read_csv("../data/prior/model_parameters_priors.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]
    parameters_to_fit = priors.Parameter.tolist()

    calibration_params = setup_calibration()

    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configs, parameters_to_fit, priors, calibration_params
    )

    return hierarchical_fitter, parameters_to_fit


def simulate_single_circuit_best_samples(
    hierarchical_fitter, processed_df, param_names, circuit_name, n_samples=60
):
    """
    Simulate only one specific circuit using its best samples
    Much more efficient than simulating all circuits
    """

    print(f"Simulating only {circuit_name} with {len(processed_df)} processed samples")

    # Find circuit index
    circuit_idx = None
    for i, config in enumerate(hierarchical_fitter.configs):
        if config.name == circuit_name:
            circuit_idx = i
            break

    if circuit_idx is None:
        raise ValueError(f"Circuit '{circuit_name}' not found")

    # Find circuit-specific likelihood column
    likelihood_col = None
    for col in processed_df.columns:
        if f"likelihood_{circuit_name}" in col or f"{circuit_name}_likelihood" in col:
            likelihood_col = col
            break

    # If no circuit-specific likelihood found, use total likelihood
    if likelihood_col is None:
        likelihood_col = "likelihood"
        print("  Using total likelihood (no circuit-specific found)")
    else:
        print(f"  Using circuit-specific likelihood: {likelihood_col}")

    # Select top n samples for this circuit based on likelihood
    if len(processed_df) > n_samples:
        top_indices = processed_df.nlargest(n_samples, likelihood_col).index
        sampled_df = processed_df.loc[top_indices].reset_index(drop=True)
    else:
        sampled_df = processed_df.reset_index(drop=True)

    # select random instead
    # sampled_df = processed_df.sample(n=n_samples, random_state=30).reset_index(drop=True)

    print(f"  Selected {len(sampled_df)} best samples for {circuit_name}")

    # Extract circuit-specific parameters (theta) for selected samples
    circuit_log_params = np.zeros((len(sampled_df), len(param_names)))
    for p, param in enumerate(param_names):
        col_name = f"theta_{circuit_name}_{param}"
        circuit_log_params[:, p] = sampled_df[col_name].values

    # Simulate only this specific circuit with its own parameters
    circuit_sim_data = hierarchical_fitter.simulate_single_circuit(
        circuit_idx, circuit_log_params
    )

    return circuit_sim_data


def plot_circuit_condition_abstract(
    circuit_sim_data,
    condition_name,
    exp_subsample_fraction=0.1,
    max_sim_lines=20,
    figsize=(5.5, 5),
    save_path=None,
):
    """
    Create abstract figure for a single circuit and condition
    """

    config = circuit_sim_data["config"]
    combined_params = circuit_sim_data["combined_params"]
    simulation_results = circuit_sim_data["simulation_results"]

    # Check if condition exists
    if condition_name not in config.condition_params.keys():
        raise ValueError(
            f"Condition '{condition_name}' not found. Available: {list(config.condition_params.keys())}"
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get experimental data for this condition
    exp_data = config.experimental_data[
        config.experimental_data["condition"] == condition_name
    ]

    # Subsample experimental data points
    n_exp_points = len(exp_data)
    n_keep = max(1, int(n_exp_points * exp_subsample_fraction))
    step = max(1, n_exp_points // n_keep)
    exp_subset = exp_data.iloc[::step]

    print(
        f"Experimental data: {n_exp_points} points -> {len(exp_subset)} points ({exp_subsample_fraction * 100:.0f}%)"
    )

    # Plot experimental data (scatter points)
    ax.scatter(
        exp_subset["time"],
        exp_subset["fluorescence"],
        color="red",
        alpha=0.8,
        s=50,
        zorder=3,
        label="Experimental data",
    )

    # Get simulation results for this condition
    condition_mask = combined_params["condition"] == condition_name
    sim_indices = combined_params.index[condition_mask]

    # Limit number of simulation lines for clarity
    n_sim_lines = min(max_sim_lines, len(sim_indices))
    selected_sim_indices = sim_indices[:n_sim_lines]

    print(f"Simulation lines: {len(sim_indices)} available -> {n_sim_lines} shown")

    # Plot simulation lines
    for i, sim_idx in enumerate(selected_sim_indices):
        sim_values = simulation_results.observables[sim_idx]["obs_Protein_GFP"]

        # Use different shades of blue for simulation lines
        alpha = max(0.3, 1.0 - i / n_sim_lines)  # Fade later lines

        ax.plot(
            config.tspan, sim_values, color="blue", alpha=alpha, linewidth=1.5, zorder=1
        )

        # Add label only for first line
        if i == 0:
            ax.plot(
                [],
                [],
                color="blue",
                alpha=0.7,
                linewidth=1.5,
                label=f"Model predictions (n={n_sim_lines})",
            )

    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Customize plot
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Fluorescence (nM)", fontsize=12)
    # ax.set_title(f'{config.name}\n{condition_name}', fontsize=14, fontweight='bold')

    # Set y-limits based on data
    all_sim_values = [
        simulation_results.observables[idx]["obs_Protein_GFP"]
        for idx in selected_sim_indices
    ]
    sim_max = max(np.max(vals) for vals in all_sim_values)
    exp_max = exp_subset["fluorescence"].max()
    y_max = max(sim_max, exp_max) * 1.1

    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    # Clean up the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        # save with transparent background
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved abstract plot to: {save_path}")

    return fig, ax


# Main execution code
def main():
    # Load and setup data
    results_file = (
        "../data/fit_data/hierarchical/hierarchical_results_20250521_082008.csv"
    )
    output_folder = "../figures/hierarchical_results_20250521_082008"
    burn_in = 0.7

    # Load data
    print("Loading results and setting up model...")
    df = pd.read_csv(results_file)
    df = df.rename(columns={"walker": "iteration", "iteration": "walker"})

    # Setup hierarchical model (only need cffl_type_1)
    circuits_to_fit = ["cffl_type_1"]  # Only cffl_type_1!
    hierarchical_fitter, parameters_to_fit = setup_hierarchical_model(circuits_to_fit)

    # Process MCMC data
    processed = process_mcmc_data(df, burn_in=burn_in, chain_idx=0)
    processed_df = processed["processed_data"]

    print(f"Processed samples: {len(processed_df)}")

    # Check available conditions for cffl_type_1
    print("\n=== Available conditions for cffl_type_1 ===")
    config = hierarchical_fitter.configs[0]  # Only one circuit
    print("CFFL Type 1 circuit conditions:")
    for condition_name, condition_params in config.condition_params.items():
        print(f"  - '{condition_name}': {condition_params}")

    # Simulate only cffl_type_1 with best samples
    print("\n=== Simulating cffl_type_1 ===")
    circuit_sim_data = simulate_single_circuit_best_samples(
        hierarchical_fitter,
        processed_df,
        parameters_to_fit,
        "cffl_type_1",
        n_samples=60,
    )

    # Choose your condition here - replace with actual condition name
    # You can see the available conditions printed above
    condition_to_plot = list(config.condition_params.keys())[0]  # Use first condition
    print(f"\n=== Plotting condition: {condition_to_plot} ===")

    # Create the plot
    fig, ax = plot_circuit_condition_abstract(
        circuit_sim_data,
        condition_name=condition_to_plot,
        exp_subsample_fraction=0.1,  # 10% of exp points
        max_sim_lines=60,
        save_path=os.path.join(
            output_folder,
            f"cffl_type_1_{condition_to_plot.replace(' ', '_')}_abstract.png",
        ),
    )

    plt.show()


if __name__ == "__main__":
    main()
