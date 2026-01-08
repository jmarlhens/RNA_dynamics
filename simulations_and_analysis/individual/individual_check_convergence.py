import pandas as pd
import os
import glob
from optimization.mcmc_utils import convergence_test, plot_traces
from analysis_and_figures.mcmc_analysis import dataframe_to_mcmc_arrays


def load_parameter_names_from_prior(prior_filepath):
    """Load fitted parameter names from prior CSV file"""
    prior_dataframe = pd.read_csv(prior_filepath)
    fitted_parameter_names = prior_dataframe[
        prior_dataframe["Parameter"] != "k_prot_deg"
    ]["Parameter"].tolist()
    #
    # # Remove k_rna_deg, k_rna_km as per your individual_circuits_statistics.py
    # fitted_parameter_names = [p for p in fitted_parameter_names if p not in ["k_rna_deg", "k_rna_km"]]

    return fitted_parameter_names


def analyze_circuit_convergence(mcmc_csv_filepath, parameter_names, output_directory):
    """Run complete convergence analysis on single circuit"""

    os.makedirs(output_directory, exist_ok=True)

    mcmc_dataframe = pd.read_csv(mcmc_csv_filepath)
    mcmc_arrays = dataframe_to_mcmc_arrays(mcmc_dataframe, parameter_names)

    # R-hat convergence diagnostics
    r_hat_overall = convergence_test(
        mcmc_arrays["convergence"], per_parameter_test=False
    )
    r_hat_per_parameter = convergence_test(
        mcmc_arrays["convergence"], per_parameter_test=True
    )

    print(f"Overall R-hat: {r_hat_overall:.4f}")
    for param_name, r_hat_value in zip(parameter_names, r_hat_per_parameter):
        status = "✓" if r_hat_value < 1.2 else "⚠" if r_hat_value < 1.5 else "✗"
        print(f"  {param_name:20s}: {r_hat_value:.4f} {status}")

    plot_traces(
        mcmc_arrays["convergence"],
        f"{output_directory}/traces_walker.pdf",
        param_names=mcmc_arrays["parameter_names"],
    )

    return {
        "r_hat_overall": r_hat_overall,
        "r_hat_per_parameter": dict(zip(parameter_names, r_hat_per_parameter)),
    }


def main():
    """Analyze all circuit MCMC results"""

    results_directory = "../../data/fit_data/individual_circuits/20000_steps"
    results_directory = "../../data/fit_data/shared_parameters"
    prior_filepath = "../../data/prior/model_parameters_priors_updated.csv"

    # Load parameter names from prior
    fitted_parameter_names = load_parameter_names_from_prior(prior_filepath)
    print(f"Fitted parameters: {fitted_parameter_names}")

    # Process all circuit results
    circuit_files = glob.glob(f"{results_directory}/results_*.csv")
    convergence_summary = {}

    for circuit_filepath in circuit_files:
        circuit_name = os.path.basename(circuit_filepath).split("_")[1:-2]
        circuit_name = "_".join(circuit_name)

        print(f"\nAnalyzing {circuit_name}")

        circuit_results = analyze_circuit_convergence(
            mcmc_csv_filepath=circuit_filepath,
            parameter_names=fitted_parameter_names,
            output_directory=f"{results_directory}/{circuit_name}_convergence",
        )

        convergence_summary[circuit_name] = circuit_results["r_hat_overall"]

    for circuit_name, r_hat in convergence_summary.items():
        print(f"{circuit_name:25s}: {r_hat:.4f}")


if __name__ == "__main__":
    main()
