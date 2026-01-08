import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from circuits.circuit_generation.circuit_manager import CircuitManager
from utils.GFP_calibration import setup_calibration
from analysis_and_figures.mcmc_analysis_hierarchical import process_mcmc_data
from analysis_and_figures.plots_simulation import (
    plot_circuit_simulations,
    # plot_circuit_conditions_overlay,
    extract_trajectory_data,
)
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)
from simulations_and_analysis.individual.individual_circuits_simulations import (
    create_circuit_simulation_data,
    simulate_and_organize_parameter_sets,
)
import networkx as nx
import numpy as np


def build_parameter_dependency_graph(circuit_parameters):
    """
    Build a directed graph based on parameter subset relationships.

    If circuit A has all parameters of circuit B (params_A ⊇ params_B),
    then A can predict B, so we add edge A → B.

    Parameters:
    -----------
    circuit_parameters : dict
            Dictionary mapping circuit names to sets of parameter names

    Returns:
    --------
    G : networkx.DiGraph
            Directed graph of parameter dependencies
    edge_info : dict
            Information about each edge (shared params, extra params)
    """
    G = nx.DiGraph()
    edge_info = {}

    circuit_names = list(circuit_parameters.keys())

    # Add all circuits as nodes
    for name in circuit_names:
        G.add_node(name, n_params=len(circuit_parameters[name]))

    # Check all pairs
    for source in circuit_names:
        source_params = circuit_parameters[source]

        for target in circuit_names:
            if source == target:
                continue

            target_params = circuit_parameters[target]

            # Check if source contains all target parameters
            # (source can predict target if source_params ⊇ target_params)
            if target_params.issubset(source_params):
                G.add_edge(source, target)

                extra_params = source_params - target_params
                edge_info[(source, target)] = {
                    "shared_params": target_params,
                    "extra_params_in_source": extra_params,
                    "n_shared": len(target_params),
                    "n_extra": len(extra_params),
                }

    return G, edge_info


def visualize_dependency_dag(G, circuit_parameters, output_directory, edge_info=None):
    """
    Visualize the parameter dependency DAG.

    Parameters:
    -----------
    G : networkx.DiGraph
            The dependency graph
    circuit_parameters : dict
            Dictionary mapping circuit names to parameter sets
    output_path : str
            Path to save the figure
    edge_info : dict
            Optional edge information for labels
    """

    fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))

    # Use hierarchical layout based on number of parameters
    # Circuits with more parameters should be higher
    n_params = {name: len(params) for name, params in circuit_parameters.items()}

    # nice layout
    pos = nx.spring_layout(G, seed=42, k=1.1, iterations=20)
    # use something else, maybe nodes not too close
    # pos = nx.kamada_kawai_layout(G)

    # Node colors based on number of parameters
    node_colors = [n_params[node] for node in G.nodes()]

    # Draw the graph
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax1,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=2000,
        alpha=0.9,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax1,
        font_size=8,
        font_weight="bold",
    )

    # Draw edges with arrows
    # add some space between each edge and node, no curve
    # space between node and edge
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax1,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.0",
        width=1.5,
        min_source_margin=25,  # Add margin at the source node
        min_target_margin=25,  # Add margin at the target nod
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("Number of Parameters", fontsize=10)

    # ax1.set_title('Parameter Dependency DAG\n(A → B means A can predict B)', fontsize=14)
    ax1.axis("off")

    plt.tight_layout()

    # plt.show()

    plt.savefig(
        output_directory + "/parameter_dependency_dag.png", dpi=300, bbox_inches="tight"
    )
    # in pdf
    plt.savefig(output_directory + "/parameter_dependency_dag.pdf", bbox_inches="tight")

    plt.close()

    return fig


def print_parameter_summary(circuit_parameters):
    """Print a summary of parameters for each circuit"""
    print("\n" + "=" * 60)
    print("Parameter Summary by Circuit")
    print("=" * 60)

    # Sort by number of parameters
    sorted_circuits = sorted(
        circuit_parameters.items(), key=lambda x: len(x[1]), reverse=True
    )

    for circuit_name, params in sorted_circuits:
        print(f"\n{circuit_name} ({len(params)} parameters):")
        for param in sorted(params):
            print(f"  - {param}")


def print_dependency_analysis(G, edge_info, circuit_parameters):
    """Print analysis of the dependency graph"""
    print("\n" + "=" * 60)
    print("Dependency Analysis")
    print("=" * 60)

    # Find circuits that can predict the most others
    out_degrees = dict(G.out_degree())
    sorted_by_predictive = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)

    print("\nCircuits by predictive power (can predict N others):")
    for circuit, degree in sorted_by_predictive:
        print(f"  {circuit}: can predict {degree} circuits")

    # Find bidirectional relationships (same parameters)
    print("\nBidirectional relationships (same or equivalent parameters):")
    seen = set()
    for source, target in G.edges():
        if (target, source) in seen:
            continue
        if G.has_edge(target, source):
            seen.add((source, target))
            params_source = circuit_parameters[source]
            params_target = circuit_parameters[target]
            if params_source == params_target:
                print(f"  {source} ↔ {target} (identical parameters)")
            else:
                diff_s = params_source - params_target
                diff_t = params_target - params_source
                print(f"  {source} ↔ {target}")
                if diff_s:
                    print(f"    {source} has extra: {diff_s}")
                if diff_t:
                    print(f"    {target} has extra: {diff_t}")

    # Find isolated circuits (can't predict or be predicted by any)
    isolated = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    if isolated:
        print(f"\nIsolated circuits (unique parameters): {isolated}")


def plot_fits_with_dag(
    mcmc_results_by_circuit,
    output_directory=".",
    sample_count=60,
    time_bounds_max=None,
    time_bounds_min=None,
    priors_csv_path="../../data/prior/model_parameters_priors_092025_correction.csv",
):
    """Plot fits using DAG-based cross-validation"""

    circuit_manager = CircuitManager(
        parameters_file=priors_csv_path,
        json_file="../../data/circuits/circuits.json",
    )

    calibration_parameters = setup_calibration()

    # Step 1: Create circuit fitters and extract parameters
    print("=" * 60)
    print("Step 1: Creating circuit fitters and extracting parameters")
    print("=" * 60)

    circuit_fitters = {}
    circuit_configurations = {}
    circuit_parameters = {}  # For DAG construction

    for circuit_name in mcmc_results_by_circuit.keys():
        print(f"Creating fitter for circuit: {circuit_name}")

        circuit_configuration, circuit_fitter = create_circuit_simulation_data(
            circuit_name,
            circuit_manager,
            calibration_parameters,
            time_bounds_max,
            time_bounds_min,
            priors_csv_path,
        )

        circuit_fitters[circuit_name] = circuit_fitter
        circuit_configurations[circuit_name] = circuit_configuration
        circuit_parameters[circuit_name] = set(circuit_fitter.parameters_to_fit)

        print(
            f"  Parameters ({len(circuit_fitter.parameters_to_fit)}): {circuit_fitter.parameters_to_fit}"
        )

    # Step 2: Build and visualize the DAG
    print("\n" + "=" * 60)
    print("Step 2: Building parameter dependency DAG")
    print("=" * 60)

    G, edge_info = build_parameter_dependency_graph(circuit_parameters)

    # Print summary
    print_parameter_summary(circuit_parameters)
    print_dependency_analysis(G, edge_info, circuit_parameters)

    # Visualize
    visualize_dependency_dag(G, circuit_parameters, output_directory, edge_info)

    # Step 3: Cross-validate only valid pairs based on DAG
    print("\n" + "=" * 60)
    print("Step 3: Cross-validating based on DAG")
    print("=" * 60)

    # Initialize storage
    combined_random_simulation_data_all = {}
    combined_random_results_all = {}

    # Create matrix for best log posteriors
    circuit_names = list(mcmc_results_by_circuit.keys())
    best_likelihoods_matrix = pd.DataFrame(
        index=circuit_names,
        columns=circuit_names,
        dtype=float,
    )

    # Fill with NaN to indicate invalid pairs
    best_likelihoods_matrix[:] = np.nan

    final_sample_size = min(sample_count, 200)

    for posterior_source in circuit_names:
        mcmc_raw_samples = mcmc_results_by_circuit[posterior_source]

        print(f"\nProcessing posterior from: {posterior_source}")

        # Initialize storage
        combined_random_simulation_data_all[posterior_source] = {}
        combined_random_results_all[posterior_source] = []

        # Process MCMC samples
        chain_idx = mcmc_raw_samples["chain"].min()
        mcmc_processed = process_mcmc_data(
            mcmc_raw_samples, burn_in=0.4, chain_idx=chain_idx
        )
        mcmc_filtered_samples = mcmc_processed["processed_data"]

        actual_sample_size = min(final_sample_size, len(mcmc_filtered_samples))
        mcmc_final_samples = (
            mcmc_filtered_samples.sample(n=actual_sample_size, random_state=42)
            if len(mcmc_filtered_samples) > actual_sample_size
            else mcmc_filtered_samples.copy()
        )

        # Only test targets that this source can predict (based on DAG)
        valid_targets = [posterior_source]  # Always test self
        valid_targets.extend([t for t in G.successors(posterior_source)])

        print(f"  Valid targets based on DAG: {valid_targets}")

        for target_circuit in valid_targets:
            if target_circuit not in circuit_fitters:
                continue

            circuit_fitter = circuit_fitters[target_circuit]
            print(f"  Testing against: {target_circuit}")

            # Filter samples to target's parameters
            filtered_samples = mcmc_final_samples[
                [
                    p
                    for p in mcmc_final_samples.columns
                    if p in circuit_fitter.parameters_to_fit
                ]
            ]

            random_simulation_data, random_results_dataframe = (
                simulate_and_organize_parameter_sets(
                    filtered_samples,
                    circuit_fitter,
                    circuit_fitter.parameters_to_fit,
                )
            )

            best_posterior = random_results_dataframe[
                ("metrics", "log_posterior")
            ].max()
            best_likelihoods_matrix.at[posterior_source, target_circuit] = (
                best_posterior
            )
            print(f"    Best log posterior: {best_posterior:.2f}")

            combined_random_simulation_data_all[posterior_source][target_circuit] = {
                "config": circuit_configurations[target_circuit],
                "combined_params": random_simulation_data["combined_params"],
                "simulation_results": random_simulation_data["simulation_results"],
            }
            combined_random_results_all[posterior_source].append(
                random_results_dataframe
            )

        # Plot results for this posterior source
        if combined_random_results_all[posterior_source]:
            combined_df = pd.concat(
                combined_random_results_all[posterior_source], ignore_index=True
            )

            trajectory_data = extract_trajectory_data(
                combined_random_simulation_data_all[posterior_source], combined_df
            )

            plot_circuit_simulations(
                combined_random_simulation_data_all[posterior_source],
                trajectory_data,
                plot_mode="summary",
                summary_type="median_iqr",
                percentile_bounds=(10, 90),
                show_title=True,
            )

            plt.suptitle(f"Posterior from: {posterior_source}")
            plt.savefig(
                os.path.join(
                    output_directory, f"cross_val_{posterior_source}_posterior.png"
                ),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

    # Step 4: Plot the results matrix
    print("\n" + "=" * 60)
    print("Step 4: Generating results heatmap")
    print("=" * 60)

    plt.figure(figsize=(14, 12))

    # Create mask for NaN values
    mask = best_likelihoods_matrix.isna()

    sns.heatmap(
        best_likelihoods_matrix.astype(float),
        annot=True,
        fmt=".1f",
        cmap="viridis",
        cbar_kws={"label": "Best Log Posterior"},
        mask=mask,
        linewidths=0.5,
        linecolor="lightgray",
    )

    plt.title(
        "Cross-Validation Matrix (DAG-based)\nGray = Invalid pair (missing parameters)"
    )
    plt.xlabel("Target Circuit")
    plt.ylabel("Posterior Source")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, "best_likelihoods_matrix_dag.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # Save matrix
    best_likelihoods_matrix.to_csv(
        os.path.join(output_directory, "best_likelihoods_matrix_dag.csv")
    )

    print(f"Results saved to: {output_directory}")

    return G, best_likelihoods_matrix


def main():
    subfolder = "/2025-11-11_100000_steps_adaptive_new_priors_part_1"

    input_directory = "../../data/fit_data/individual_circuits" + subfolder
    output_visualization_directory = "../../figures/individual_circuits" + subfolder
    priors_csv_path = "../../data/prior/model_parameters_priors_092025_correction.csv"

    # Create output directory if it does not exist
    os.makedirs(output_visualization_directory, exist_ok=True)

    mcmc_results = load_individual_circuit_results(input_directory, prefix="results_")

    # Define processing order and inclusion
    circuit_processing_sequence = [
        "constitutive sfGFP",
        "constitutive sfGFP sim",
        "sense_star_6",
        "toehold_trigger",
        "cascade",
        "to1_cascade",
        "cffl_type_1",
        "or_gate_c1ffl",
        "star_antistar_1",
        "trigger_antitrigger",
        "inhibited_incoherent_cascade",
        "inhibited_cascade",
        "cffl_12",
        "iffl_1",
    ]

    filtered_mcmc_results = {
        circuit_name: mcmc_results[circuit_name]
        for circuit_name in circuit_processing_sequence
        if circuit_name in mcmc_results
    }

    # Generate cross-validation plots
    plot_fits_with_dag(
        filtered_mcmc_results,
        output_visualization_directory,
        sample_count=100,
        time_bounds_max=210,
        time_bounds_min=30,
        priors_csv_path=priors_csv_path,
    )

    # Optionally generate per-circuit plots
    # generate_per_circuit_individual_plots(
    #     filtered_mcmc_results,
    #     output_visualization_directory,
    #     sample_count=1,
    #     time_bounds_max=210,
    #     time_bounds_min=30,
    #     priors_csv_path=priors_csv_path,
    # )


if __name__ == "__main__":
    main()
