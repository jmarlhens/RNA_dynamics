from likelihood_functions.hierarchical_likelihood.mcmc_adaption import (
    HierarchicalMCMCAdapter,
)
from likelihood_functions.hierarchical_likelihood.base_hierarchical import (
    HierarchicalCircuitFitter,
)
from datetime import datetime
from likelihood_functions.hierarchical_likelihood.mcmc_analysis import (
    analyze_hierarchical_mcmc_results,
)
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from likelihood_functions.hierarchical_likelihood.visualization import (
    plot_hierarchical_results,
)
import matplotlib.pyplot as plt


def fit_hierarchical_multiple_circuits(
    circuit_configs,
    parameters_to_fit,
    priors,
    calibration_params,
    n_samples=2000,
    n_walkers=5,
    n_chains=12,
):
    """Fit multiple circuits with hierarchical Bayesian approach"""
    print("Setting up hierarchical model...")
    # Create hierarchical circuit fitter
    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configs, parameters_to_fit, priors, calibration_params
    )

    print("Generating initial parameters...")
    # Create MCMC adapter
    adapter = HierarchicalMCMCAdapter(hierarchical_fitter)
    initial_parameters = adapter.get_initial_parameters()

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

    print("Generating visualizations...")
    # Generate visualizations
    plot_hierarchical_results(results, hierarchical_fitter)

    # After analyzing results and before saving
    print("Generating comparison simulations...")
    from likelihood_functions.hierarchical_likelihood.visualization import (
        simulate_hierarchical_comparison,
        plot_hierarchical_comparison,
    )

    comparison_results = simulate_hierarchical_comparison(
        hierarchical_fitter, results, n_best=3
    )

    # Plot for each parameter set
    for i in range(3):
        fig = plot_hierarchical_comparison(
            hierarchical_fitter, comparison_results, param_set_idx=i
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig.savefig(
            f"hierarchical_comparison_{i + 1}_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # [rest of existing code]

    return results, comparison_results  # Return comparison results too


if __name__ == "__main__":
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../data/prior/model_parameters_priors.csv",
        json_file="../data/circuits/circuits.json",
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

    priors = pd.read_csv("../data/prior/model_parameters_priors.csv")
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
        n_samples=10,
        n_walkers=10,
        n_chains=6,
    )

    print("Hierarchical parameter estimation completed successfully!")
