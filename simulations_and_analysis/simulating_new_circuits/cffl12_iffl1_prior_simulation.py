import numpy as np
import pandas as pd
from pathlib import Path
from circuits.circuit_generation.circuit_manager import CircuitManager
from data.circuits.circuit_configs import get_circuit_conditions
from likelihood_functions.config import CircuitConfig
from likelihood_functions.base import CircuitFitter
from prior_simulation_plotting import plot_prior_simulation_results


class PriorSamplingFFLSimulator:
    """Simulate FFL circuits using parameter samples from prior distributions"""

    def __init__(self, priors_updated_path: str, circuits_json_path: str):
        self.circuit_manager = CircuitManager(
            parameters_file=priors_updated_path, json_file=circuits_json_path
        )
        self.priors_dataframe = pd.read_csv(priors_updated_path)
        self.parameters_to_fit = self.priors_dataframe["Parameter"].tolist()

    def create_dummy_experimental_data(
        self, circuit_conditions: dict, simulation_timespan: np.ndarray
    ) -> pd.DataFrame:
        """Create minimal dummy experimental data to satisfy CircuitConfig requirements"""
        dummy_experimental_records = []

        for condition_name in circuit_conditions.keys():
            for time_point in simulation_timespan:
                dummy_experimental_records.append(
                    {
                        "condition": condition_name,
                        "time": time_point,
                        "fluorescence": 1.0,  # Dummy value - not used in prior simulation
                        "replicate": 1,
                    }
                )

        return pd.DataFrame(dummy_experimental_records)

    def setup_calibration_parameters(self) -> dict:
        """Setup minimal calibration parameters for CircuitConfig compatibility"""
        return {
            "slope": 1.0,  # Identity conversion
            "intercept": 0.0,
            "brightness_correction": 1.0,
        }

    def create_circuit_configuration(
        self, circuit_name: str, simulation_timespan: np.ndarray
    ) -> CircuitConfig:
        """Create CircuitConfig for specified circuit with dummy experimental data"""
        circuit_conditions = get_circuit_conditions(circuit_name)

        if not circuit_conditions:
            raise ValueError(f"No conditions defined for circuit '{circuit_name}'")

        # Create circuit instance for model extraction
        first_condition_parameters = next(iter(circuit_conditions.values()))
        circuit_instance = self.circuit_manager.create_circuit(
            circuit_name, parameters=first_condition_parameters
        )

        # Generate dummy experimental data
        dummy_experimental_data = self.create_dummy_experimental_data(
            circuit_conditions, simulation_timespan
        )
        calibration_parameters = self.setup_calibration_parameters()

        circuit_configuration = CircuitConfig(
            model=circuit_instance.model,
            name=circuit_name,
            condition_params=circuit_conditions,
            experimental_data=dummy_experimental_data,
            tspan=simulation_timespan,
            calibration_params=calibration_parameters,
        )

        return circuit_configuration

    def simulate_circuit_with_prior_samples(
        self,
        circuit_name: str,
        n_prior_samples: int = 50,
        simulation_hours: float = 9.0,
    ) -> dict:
        """Vectorized simulation of circuit across all conditions using prior parameter samples"""

        # Create simulation timespan
        simulation_minutes = simulation_hours * 60
        simulation_timespan = np.linspace(
            0, simulation_minutes, int(simulation_minutes * 2) + 1
        )  # 0.5 min resolution

        # Create circuit configuration
        circuit_configuration = self.create_circuit_configuration(
            circuit_name, simulation_timespan
        )

        # Setup CircuitFitter for vectorized simulation
        circuit_fitter = CircuitFitter(
            configs=[circuit_configuration],
            parameters_to_fit=self.parameters_to_fit,
            model_parameters_priors=self.priors_dataframe,
            calibration_data=self.setup_calibration_parameters(),
        )

        # Sample prior parameters in log space using existing CircuitFitter method
        prior_log_parameters = circuit_fitter.generate_test_parameters(n_prior_samples)

        # Execute vectorized simulation across all conditions and parameter sets
        simulation_results_dict = circuit_fitter.simulate_parameters(
            prior_log_parameters
        )

        # Convert log parameters to linear for storage
        prior_linear_parameters = circuit_fitter.log_to_linear_params(
            prior_log_parameters, self.parameters_to_fit
        )

        return {
            "circuit_name": circuit_name,
            "simulation_results": simulation_results_dict,
            "prior_parameters_linear": prior_linear_parameters,
            "prior_parameters_log": prior_log_parameters,
            "simulation_timespan": simulation_timespan,
            "n_samples": n_prior_samples,
        }

    def extract_gfp_concentration_trajectories(
        self, vectorized_simulation_results: dict
    ) -> pd.DataFrame:
        """Extract GFP concentration trajectories from vectorized simulation results"""
        circuit_simulation_data = vectorized_simulation_results["simulation_results"][
            0
        ]  # Single circuit
        combined_parameters_dataframe = circuit_simulation_data["combined_params"]
        pysb_simulation_results = circuit_simulation_data["simulation_results"]
        simulation_timespan = vectorized_simulation_results["simulation_timespan"]

        trajectory_records = []

        for simulation_index in range(len(pysb_simulation_results.observables)):
            parameter_set_index = combined_parameters_dataframe.iloc[simulation_index][
                "param_set_idx"
            ]
            condition_name = combined_parameters_dataframe.iloc[simulation_index][
                "condition"
            ]
            gfp_concentration_trajectory = pysb_simulation_results.observables[
                simulation_index
            ]["obs_Protein_GFP"]

            for time_index, time_point in enumerate(simulation_timespan):
                trajectory_records.append(
                    {
                        "parameter_set_idx": parameter_set_index,
                        "condition": condition_name,
                        "time_minutes": time_point,
                        "gfp_concentration_nM": gfp_concentration_trajectory[
                            time_index
                        ],
                    }
                )

        return pd.DataFrame(trajectory_records)

    def compute_trajectory_summary_statistics(
        self, trajectory_dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute summary statistics across parameter sets for each condition and timepoint"""
        summary_statistics_dataframe = (
            trajectory_dataframe.groupby(["condition", "time_minutes"])[
                "gfp_concentration_nM"
            ]
            .agg(
                [
                    "mean",
                    "std",
                    "count",
                    ("percentile_5", lambda x: np.percentile(x, 5)),
                    ("percentile_25", lambda x: np.percentile(x, 25)),
                    ("median", lambda x: np.percentile(x, 50)),
                    ("percentile_75", lambda x: np.percentile(x, 75)),
                    ("percentile_95", lambda x: np.percentile(x, 95)),
                ]
            )
            .reset_index()
        )

        return summary_statistics_dataframe

    def save_prior_simulation_results(
        self,
        circuit_name: str,
        vectorized_simulation_results: dict,
        output_directory: str,
    ) -> tuple:
        """Save prior simulation results and return trajectory DataFrames"""
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract trajectory data
        trajectory_dataframe = self.extract_gfp_concentration_trajectories(
            vectorized_simulation_results
        )

        # Compute summary statistics
        summary_statistics_dataframe = self.compute_trajectory_summary_statistics(
            trajectory_dataframe
        )

        # Save raw trajectories and summary statistics
        trajectory_dataframe.to_csv(
            output_path / f"{circuit_name}_prior_trajectories.csv", index=False
        )
        summary_statistics_dataframe.to_csv(
            output_path / f"{circuit_name}_trajectory_summary_stats.csv", index=False
        )

        # Save parameter samples used
        vectorized_simulation_results["prior_parameters_linear"].to_csv(
            output_path / f"{circuit_name}_prior_parameter_samples.csv", index=False
        )

        print(f"Prior simulation results saved to {output_path}")
        return trajectory_dataframe, summary_statistics_dataframe

    def simulate_cffl12_circuit(
        self,
        n_prior_samples: int = 50,
        output_directory: str = "output/cffl12_prior_simulation",
    ) -> tuple:
        """Complete prior simulation pipeline for CFFL-12 circuit"""
        vectorized_results = self.simulate_circuit_with_prior_samples(
            "cffl_12", n_prior_samples, simulation_hours=9.0
        )

        trajectory_dataframe, summary_stats_dataframe = (
            self.save_prior_simulation_results(
                "cffl_12", vectorized_results, output_directory
            )
        )

        return trajectory_dataframe, summary_stats_dataframe, vectorized_results

    def simulate_iffl1_circuit(
        self,
        n_prior_samples: int = 50,
        output_directory: str = "output/iffl1_prior_simulation",
    ) -> tuple:
        """Complete prior simulation pipeline for IFFL-1 circuit"""
        vectorized_results = self.simulate_circuit_with_prior_samples(
            "iffl_1", n_prior_samples, simulation_hours=9.0
        )

        trajectory_dataframe, summary_stats_dataframe = (
            self.save_prior_simulation_results(
                "iffl_1", vectorized_results, output_directory
            )
        )

        return trajectory_dataframe, summary_stats_dataframe, vectorized_results


def execute_prior_simulation_analysis():
    """Execute complete prior simulation analysis for both FFL circuit types"""

    prior_simulator = PriorSamplingFFLSimulator(
        priors_updated_path="data/prior/model_parameters_priors_updated.csv",
        circuits_json_path="data/circuits/circuits.json",
    )

    print("Executing CFFL-12 prior simulation...")
    cffl12_trajectories, cffl12_summary, cffl12_results = (
        prior_simulator.simulate_cffl12_circuit(n_prior_samples=50)
    )

    print("Executing IFFL-1 prior simulation...")
    iffl1_trajectories, iffl1_summary, iffl1_results = (
        prior_simulator.simulate_iffl1_circuit(n_prior_samples=50)
    )

    print(f"CFFL-12 analysis: {len(cffl12_trajectories)} trajectory points")
    print(f"IFFL-1 analysis: {len(iffl1_trajectories)} trajectory points")

    # Generate plots
    print("Generating visualization plots...")

    generated_figures = plot_prior_simulation_results(
        cffl12_trajectories,
        iffl1_trajectories,
        output_directory="output/prior_simulation_plots",
    )

    return (
        prior_simulator,
        (cffl12_trajectories, cffl12_summary),
        (iffl1_trajectories, iffl1_summary),
        generated_figures,
    )


if __name__ == "__main__":
    simulator, cffl12_data, iffl1_data = execute_prior_simulation_analysis()
