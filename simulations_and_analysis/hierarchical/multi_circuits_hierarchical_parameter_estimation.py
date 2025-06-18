import numpy as np
from matplotlib import pyplot as plt

from likelihood_functions.hierarchical_likelihood.mcmc_adaption import (
    HierarchicalMCMCAdapter,
)
from likelihood_functions.hierarchical_likelihood.base_hierarchical import (
    HierarchicalCircuitFitter,
)
from datetime import datetime
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from analysis_and_figures.visualization_hierarchical import (
    plot_hierarchical_results,
)
from analysis_and_figures.mcmc_analysis_hierarchical import (
    analyze_hierarchical_mcmc_results,
)
from simulations_and_analysis.hierarchical.hierarchical_model_diagnostics import (
    analyze_individual_fits_main,
    extract_best_circuit_parameters,
    construct_hierarchical_parameters,
)
from simulations_and_analysis.individual.individual_circuits_analysis import (
    load_individual_circuit_results,
)


def fit_hierarchical_multiple_circuits(
    circuit_configs,
    parameters_to_fit,
    priors,
    calibration_params,
    n_samples=2000,
    n_walkers=5,
    n_chains=12,
):
    print("Loading individual circuit MCMC results for hyperparameter estimation...")
    individual_circuit_posterior_results = load_individual_circuit_results(
        individual_results_directory="../../../data/fit_data/individual_circuits/"
    )
    print(
        f"Loaded {len(individual_circuit_posterior_results)} individual circuit results for hyperparameter estimation"
    )
    """Fit multiple circuits with hierarchical Bayesian approach"""
    print("Setting up hierarchical model...")
    # Create hierarchical circuit fitter
    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configs,
        parameters_to_fit,
        priors,
        calibration_params,
        individual_circuit_posterior_results=individual_circuit_posterior_results,
    )

    print("Generating initial parameters...")
    # Create MCMC adapter
    adapter = HierarchicalMCMCAdapter(hierarchical_fitter)
    initial_parameters = adapter.get_initial_parameters()

    # Get best results from individual fits
    individual_results = analyze_individual_fits_main(parameters_to_fit, n_best=100)
    best_circuit_params = extract_best_circuit_parameters(
        individual_results["best_fits"], hierarchical_fitter.parameters_to_fit
    )
    individual_params, alpha, sigma = construct_hierarchical_parameters(
        best_circuit_params, hierarchical_fitter
    )
    initial_parameters = individual_params

    # individual_params.shape
    # Out[2]: (350,)

    print("Setting up parallel tempering...")
    # Setup parallel tempering
    pt = adapter.setup_hierarchical_parallel_tempering(
        n_walkers=n_walkers, n_chains=n_chains
    )

    print(f"Running MCMC sampling for {n_samples} iterations...")
    # Run sampling
    parameters, priors_out, likelihoods, step_accepts, swap_accepts = pt.run(
        initial_parameters=initial_parameters,
        n_samples=n_samples,
        target_acceptance_ratio=0.3,
        adaptive_temperature=True,
    )

    print("Analyzing results...")
    # Analyze results
    results = analyze_hierarchical_mcmc_results(
        parameters=parameters,
        priors=priors_out,
        likelihoods=likelihoods,
        step_accepts=step_accepts,
        swap_accepts=swap_accepts,
        hierarchical_fitter=hierarchical_fitter,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results["dataframe"].to_csv(f"hierarchical_results_{timestamp}.csv", index=False)

    print("Generating parameter distribution visualizations...")
    # Generate parameter distribution plots
    plot_hierarchical_results(results, hierarchical_fitter)

    # REPLACE THE OLD COMPARISON SECTION WITH THIS:
    print("Generating posterior comparison simulations...")
    # from likelihood_functions.hierarchical_likelihood.visualization import (
    #     simulate_hierarchical_comparison_posterior_sampling,
    #     plot_hierarchical_posterior_comparison,
    # )
    #
    # # Sample from posterior and create comparison
    # comparison_results = simulate_hierarchical_comparison_posterior_sampling(
    #     hierarchical_fitter,
    #     results,
    #     n_samples=100,  # Number of posterior samples to use
    #     burn_in=0.3  # Remove first 30% as burn-in
    # )
    #
    # # Create the posterior comparison plot
    # print("Creating posterior comparison plot...")
    # # Ensure figures directory exists
    # os.makedirs("../figures/analysis_plots", exist_ok=True)
    #
    # fig = plot_hierarchical_posterior_comparison(
    #     hierarchical_fitter, comparison_results
    # )
    #
    # # Save the plot
    # save_path = f"../figures/analysis_plots/hierarchical_posterior_comparison_{timestamp}.png"
    # fig.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)
    #
    # print(f"Posterior comparison plot saved to: {save_path}")
    #
    # return results, comparison_results

    from numpy.lib.stride_tricks import sliding_window_view
    step_accepts_sliding_window = np.transpose(sliding_window_view(step_accepts[:, :, :], 100, axis=0),
                                               axes=(0, 3, 1, 2))
    step_accepts_avg = np.mean(step_accepts_sliding_window, axis=1)

    swap_accepts_sliding_window = np.transpose(sliding_window_view(swap_accepts, 10, axis=0), axes=(0, 3, 1, 2))
    swap_accepts_avg = np.mean(swap_accepts_sliding_window, axis=1)

    for iWalker in range(n_walkers):
        fig, axes = plt.subplots(ncols=2)

        for iChain in range(step_accepts_avg.shape[-1]):
            axes[0].plot(np.arange(2) * (len(step_accepts_avg) - 1),
                         np.ones(2) * 0.4 + (n_chains - iChain - 1), "k--", alpha=0.5)
            axes[0].plot(np.arange(len(step_accepts_avg)),
                         step_accepts_avg[:, iWalker, iChain] + (n_chains - iChain - 1), label=iChain, alpha=1)

        for iChain in range(swap_accepts_avg.shape[-1]):
            axes[1].plot(np.arange(2) * (len(swap_accepts_avg) - 1), np.ones(2) * 0 + (n_chains - iChain - 1), "r--",
                         alpha=0.5)
            axes[1].plot(np.arange(len(swap_accepts_avg)),
                         swap_accepts_avg[:, iWalker, iChain] + (n_chains - iChain - 1), label=iChain, alpha=1)

        ylim = axes[0].get_ylim()
        axes[1].set_ylim(ylim)
        axes[0].legend()
        axes[1].legend()
        plt.show()


if __name__ == "__main__":
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # Define circuits to fit
    circuits_to_fit = [
        "trigger_antitrigger",
        "toehold_trigger",
        "sense_star_6",
        "cascade",
        "cffl_type_1",
        "star_antistar_1",
    ]

    # List available circuits
    available_circuits = circuit_manager.list_circuits()
    print(f"Available circuits: {available_circuits}")

    # Filter to only include available circuits
    circuits_to_fit = [c for c in circuits_to_fit if c in available_circuits]

    if not circuits_to_fit:
        print("Error: None of the specified circuits are available.")
        exit()

    # Create circuit configurations
    circuit_configs = create_circuit_configs(
        circuit_manager, circuits_to_fit, min_time=30, max_time=210
    )

    # Load priors
    import pandas as pd

    priors = pd.read_csv("../../data/prior/model_parameters_priors.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]  # Exclude protein degradation
    parameters_to_fit = priors.Parameter.tolist()

    # Setup calibration
    calibration_params = setup_calibration()

    # Run hierarchical parameter estimation
    results = fit_hierarchical_multiple_circuits(
        circuit_configs,
        parameters_to_fit,
        priors,
        calibration_params,
        n_samples=400,
        n_walkers=12,
        n_chains=8,
    )

    print("Hierarchical parameter estimation completed successfully!")
