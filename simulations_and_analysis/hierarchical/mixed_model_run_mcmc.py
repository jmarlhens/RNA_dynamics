from likelihood_functions.hierarchical_likelihood.mcmc_adaption import (
    HierarchicalMCMCAdapter,
)
from likelihood_functions.hierarchical_likelihood.base_hierarchical_mixed_effect_models import (
    HierarchicalCircuitFitter,
)
from datetime import datetime
from circuits.circuit_generation.circuit_manager import CircuitManager
from likelihood_functions.circuit_utils import create_circuit_configs, setup_calibration
from analysis_and_figures.mcmc_analysis_hierarchical import (
    analyze_hierarchical_mcmc_results,
)
from simulations_and_analysis.individual.individual_circuits_statistics import (
    load_individual_circuit_results,
)


def fit_hierarchical_multiple_circuits(
    circuit_configurations,
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
    # Define shared parameters for mixed effects testing
    shared_parameter_names = [
        "k_star_act_reg",
        "k_tx_init",
        "k_trigger_binding",
        "k_tl_bound_toehold",
        "k_star_act",
        "k_star_stop_reg",
        "k_trigger_unbinding",
        "k_star_unbind",
        "k_sequestration_unbind",
        "k_star_bind",
        "k_star_stop",
        "k_tl_unbound_toehold",
        "k_sequestration_bind",
        "k_mat",
    ]
    print(
        f"Testing mixed effects: {len(shared_parameter_names)} shared, {len(parameters_to_fit) - len(shared_parameter_names)} hierarchical"
    )

    # Create hierarchical circuit fitter with mixed effects
    hierarchical_fitter = HierarchicalCircuitFitter(
        circuit_configurations,
        parameters_to_fit,
        priors,
        calibration_params,
        shared_parameter_names=shared_parameter_names,  # ADD THIS LINE
    )

    # Debug parameter structure
    print("Mixed effects parameter structure:")
    print(f"  Shared parameters (β): {hierarchical_fitter.shared_parameter_names}")
    print(
        f"  Hierarchical parameters (θ): {hierarchical_fitter.hierarchical_parameter_names}"
    )
    print(
        f"  Parameter vector dimensions: β={hierarchical_fitter.n_beta_params}, θ={hierarchical_fitter.n_theta_params}, α={hierarchical_fitter.n_alpha_params}, Σ={hierarchical_fitter.n_sigma_params}"
    )
    print(f"  Total parameters: {hierarchical_fitter.n_total_params}")

    # Debug parameter mapping
    print(f"  Shared parameter indices: {hierarchical_fitter.shared_param_indices}")
    print(
        f"  Hierarchical parameter indices: {hierarchical_fitter.hierarchical_param_indices}"
    )

    # beta_params = hierarchical_fitter.beta_prior_means
    # theta_params = hierarchical_fitter.alpha
    # # Debug: verify first parameter set reconstruction
    # if len(beta_params) > 0:
    #     first_beta = beta_params[0] if hierarchical_fitter.n_shared_params > 0 else np.array([])
    #     first_theta_circuit_0 = theta_params[0, 0]
    #     reconstructed_first_params = hierarchical_fitter._reconstruct_simulation_parameters(first_beta,
    #                                                                                         first_theta_circuit_0)
    #     print(f"Debug parameter reconstruction (sample 0, circuit 0): {reconstructed_first_params[:3]}...")

    print("Generating initial parameters...")
    # Create MCMC adapter
    adapter = HierarchicalMCMCAdapter(hierarchical_fitter)
    initial_parameters = adapter.get_initial_parameters()

    # # Get best results from individual fits
    # individual_results = analyze_individual_fits_main(parameters_to_fit, n_best=100)
    # best_circuit_params = extract_best_circuit_parameters(
    #     individual_results["best_fits"], hierarchical_fitter.parameters_to_fit
    # )
    # individual_params, alpha, sigma = construct_hierarchical_parameters(
    #     best_circuit_params, hierarchical_fitter
    # )
    # initial_parameters = individual_params

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
    # plot_hierarchical_results(results, hierarchical_fitter)

    # REPLACE THE OLD COMPARISON SECTION WITH THIS:
    print("Generating posterior comparison simulations...")


if __name__ == "__main__":
    # Initialize CircuitManager with existing circuits file
    circuit_manager = CircuitManager(
        parameters_file="../../data/prior/model_parameters_priors_updated.csv",
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

    priors = pd.read_csv("../../data/prior/model_parameters_priors_updated.csv")
    priors = priors[priors["Parameter"] != "k_prot_deg"]  # Exclude protein degradation
    priors = priors[~priors["Parameter"].isin(["k_rna_deg", "k_rna_km"])]

    parameters_to_fit = priors.Parameter.tolist()

    # Setup calibration
    calibration_params = setup_calibration()

    # Run hierarchical parameter estimation
    results = fit_hierarchical_multiple_circuits(
        circuit_configs,
        parameters_to_fit,
        priors,
        calibration_params,
        n_samples=6000,
        n_walkers=10,
        n_chains=8,
    )

    print("Hierarchical parameter estimation completed successfully!")
