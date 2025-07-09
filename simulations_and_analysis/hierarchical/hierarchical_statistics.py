"""
Hierarchical Parameter Analysis Extensions
Adapts individual circuit analysis functions for hierarchical model parameters
"""

import os
import pandas as pd
from analysis_and_figures.hierarchical_pairplot_analysis import (
    create_hierarchical_histogram_grid,
)
from analysis_and_figures.process_hierarchical import (
    convert_hierarchical_to_unified_format,
    generate_hierarchical_prior_statistics,
)


def execute_hierarchical_unified_analysis(
    hierarchical_results_filepath,
    prior_parameters_filepath,
    fitted_parameter_names,
    output_visualization_directory,
    circuits_to_analyze=None,
    burn_in_fraction=0.5,
    include_global_parameters=True,
):
    """
    Execute complete hierarchical analysis pipeline combining individual circuit analysis approach
    with hierarchical model parameters
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

    # Load hierarchical results
    hierarchical_results_dataframe = pd.read_csv(hierarchical_results_filepath)

    # Handle column naming issue if present
    if (
        "walker" in hierarchical_results_dataframe.columns
        and "iteration" in hierarchical_results_dataframe.columns
    ):
        hierarchical_results_dataframe = hierarchical_results_dataframe.rename(
            columns={"walker": "iteration", "iteration": "walker"}
        )

    print(f"Loaded hierarchical results: {len(hierarchical_results_dataframe)} samples")
    print(
        f"Available circuits: {[config.name for config in hierarchical_fitter.configs]}"
    )

    # Convert to unified format
    unified_hierarchical_dataset, consensus_available_parameters = (
        convert_hierarchical_to_unified_format(
            hierarchical_results_dataframe,
            hierarchical_fitter,
            fitted_parameter_names,
            burn_in_fraction=burn_in_fraction,
            include_global_parameters=include_global_parameters,
        )
    )

    # Generate prior statistics for vertical line rendering
    hierarchical_prior_statistics, available_prior_parameters = (
        generate_hierarchical_prior_statistics(
            prior_parameters_filepath,
            consensus_available_parameters,  # Use consensus parameters only
        )
    )

    # Final parameter set for visualization
    visualization_parameter_names = consensus_available_parameters

    # Combine datasets: MCMC data + prior statistics metadata
    hierarchical_unified_comparison_dataset = pd.concat(
        [unified_hierarchical_dataset, hierarchical_prior_statistics], ignore_index=True
    )

    print(f"Unified dataset: {len(hierarchical_unified_comparison_dataset)} samples")
    print(f"Data groups: {hierarchical_unified_comparison_dataset['Circuit'].unique()}")
    print(f"Final parameters for visualization: {visualization_parameter_names}")

    # Generate visualizations using existing functions

    print("Creating hierarchical histogram grid...")
    create_hierarchical_histogram_grid(
        hierarchical_unified_comparison_dataset,
        visualization_parameter_names,
        output_visualization_directory + "/hierarchical_ridgeline_unified.png",
        plot_kind="ridge",
        ridge_offset=1.1,
        ridge_label_pad=-0.0,
    )

    create_hierarchical_histogram_grid(
        hierarchical_unified_comparison_dataset,
        visualization_parameter_names,
        output_visualization_directory + "/hierarchical_kde_grid_unified.png",
        plot_kind="kde",
        share_y=True,
        legend_on_idx=1,
    )

    # print("Creating hierarchical pairplot...")
    # create_circuit_prior_comparison_pairplot(
    #     hierarchical_unified_comparison_dataset,
    #     visualization_parameter_names,
    #     output_visualization_directory + "/hierarchical_pairplot_unified.png",
    #     diagonal_visualization_type="kde",
    #     offdiagonal_visualization_type="scatter",
    # )

    return {
        "unified_dataset": hierarchical_unified_comparison_dataset,
        "hierarchical_fitter": hierarchical_fitter,
    }


def main():
    """Execute hierarchical unified analysis"""

    data = "hierarchical_results_20250628_234739"

    # Configuration
    hierarchical_results_filepath = "../../data/fit_data/hierarchical/" + data + ".csv"
    prior_parameters_filepath = "../../data/prior/model_parameters_priors_updated.csv"
    output_visualization_directory = "../../figures/hierarchical/" + data

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

    # Execute unified analysis pipeline
    _ = execute_hierarchical_unified_analysis(
        hierarchical_results_filepath=hierarchical_results_filepath,
        prior_parameters_filepath=prior_parameters_filepath,
        fitted_parameter_names=fitted_parameter_names,
        output_visualization_directory=output_visualization_directory,
        burn_in_fraction=0.5,
        include_global_parameters=True,
    )

    print(
        f"Hierarchical unified analysis complete. Results saved to: {output_visualization_directory}"
    )


if __name__ == "__main__":
    main()
