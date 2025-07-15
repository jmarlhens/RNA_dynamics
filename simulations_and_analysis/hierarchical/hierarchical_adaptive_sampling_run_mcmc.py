import numpy as np
from datetime import datetime
from likelihood_functions.hierarchical_likelihood.mcmc_adaption import (
    HierarchicalMCMCAdapter,
)
from likelihood_functions.hierarchical_likelihood.base_hierarchical import (
    HierarchicalCircuitFitter,
)
from likelihood_functions.hierarchical_likelihood.base_hierarchical_adaptive_proposal import (
    HierarchicalAdaptiveProposal,
)
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from analysis_and_figures.mcmc_analysis_hierarchical import (
    analyze_hierarchical_mcmc_results,
)
from analysis_and_figures.visualization_hierarchical import plot_hierarchical_results


def run_hierarchical_mcmc_with_adaptive_sampling():
    """Hierarchical MCMC with adaptive proposal mechanism"""

    print("Setting up hierarchical MCMC with adaptive sampling...")

    # Initialize circuit manager
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors_updated_tighter.csv",
        json_file="../../data/circuits/circuits.json",
    )

    # Define circuits
    target_circuits = [
        # "trigger_antitrigger",
        # "toehold_trigger",
        # "sense_star_6",
        "star_antistar_1",
        # "cascade",
        # "cffl_type_1",
        # "inhibited_incoherent_cascade",
        "inhibited_cascade",
        # "or_gate_c1ffl",
        # "iffl_1",
        # "cffl_12",
    ]

    available_circuits = circuit_manager.list_circuits()
    validated_circuits = [
        circuit for circuit in target_circuits if circuit in available_circuits
    ]

    print(f"Fitting circuits: {validated_circuits}")

    # Create circuit configurations
    circuit_configurations = create_circuit_configs(
        circuit_manager, validated_circuits, min_time=30, max_time=210
    )

    # Load parameter priors
    import pandas as pd

    model_priors = pd.read_csv(
        "../../data/prior/model_parameters_priors_updated_tighter.csv"
    )
    model_priors = model_priors[model_priors["Parameter"] != "k_prot_deg"]
    parameters_to_fit = model_priors.Parameter.tolist()

    # Setup calibration
    calibration_parameters = setup_calibration()

    # Create hierarchical circuit fitter
    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configurations, parameters_to_fit, model_priors, calibration_parameters
    )

    print(f"Total parameters: {hierarchical_fitter.n_total_params}")

    # Create MCMC adapter
    adapter = HierarchicalMCMCAdapter(hierarchical_fitter)

    # Create hierarchical adaptive proposal
    hierarchical_adaptive_proposal = HierarchicalAdaptiveProposal(hierarchical_fitter)

    # Setup parallel tempering with adaptive proposal
    from optimization.adaptive_parallel_tempering import ParallelTempering

    pt = ParallelTempering(
        log_likelihood=adapter.get_log_likelihood_function(),
        log_prior=adapter.get_log_prior_function(),
        n_dim=hierarchical_fitter.n_total_params,
        n_walkers=4,
        n_chains=12,
        proposal_function=hierarchical_adaptive_proposal,
    )

    # Get initial parameters
    initial_parameters = adapter.get_initial_parameters()

    print("Running MCMC with adaptive hierarchical sampling...")
    print(
        f"Chain configurations: {pt.n_chains} chains with {pt.n_walkers} walkers each"
    )

    # Run sampling
    parameters, priors, likelihoods, step_accepts, swap_accepts = pt.run(
        initial_parameters=initial_parameters,
        n_samples=1000,
        target_acceptance_ratio=0.4,
        adaptive_temperature=True,
    )

    print("MCMC completed!")

    # Analyze results
    results = analyze_hierarchical_mcmc_results(
        parameters=parameters,
        priors=priors,
        likelihoods=likelihoods,
        step_accepts=step_accepts,
        swap_accepts=swap_accepts,
        hierarchical_fitter=hierarchical_fitter,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results["dataframe"].to_csv(
        f"../../data/fit_data/hierarchical/adaptive_hierarchical_results_{timestamp}.csv",
        index=False,
    )

    # Generate plots
    plot_hierarchical_results(results, hierarchical_fitter)

    # Print acceptance rates
    mean_acceptance = np.mean(step_accepts, axis=0)
    print(f"Mean acceptance rates per chain: {np.mean(mean_acceptance, axis=0)}")

    return results


if __name__ == "__main__":
    results = run_hierarchical_mcmc_with_adaptive_sampling()
    print("Hierarchical MCMC with adaptive sampling completed successfully!")
