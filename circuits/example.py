from circuit_manager import CircuitManager
from circuit_visualizer import CircuitVisualizer
import numpy as np
import pandas as pd


def main():
    """
    Main entry point for the circuit simulation system.
    Contains example usage of the CircuitManager and Circuit classes with separated visualization.
    """
    # Create a circuit manager and visualizer
    manager = CircuitManager(parameters_file="../data/model_parameters_priors.csv")
    visualizer = CircuitVisualizer()

    # List available circuits
    print(f"Available circuits: {manager.list_circuits()}")

    # Example: create a constant circuit
    gfp_circuit = manager.create_circuit("gfp")

    # Simulate it with printing options
    result_constant, t_span = gfp_circuit.simulate(print_rules=True)

    # Now visualize the results separately
    visualizer.plot_simulation_results(result_constant, gfp_circuit.name, t_span)

    # Example: create a pulsed circuit
    pulse_config = {
        'use_pulse': True,
        'pulse_start': 4,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    }

    # Also adjust degradation rates for quicker response
    pulsed_params = {
        'k_prot_deg': 0.1,
        'k_rna_deg': 0.1
    }

    gfp_pulsed_circuit = manager.create_circuit(
        "gfp",
        parameters=pulsed_params,
        use_pulses=True,
        pulse_config=pulse_config,
        pulse_indices=[0]  # Pulse the GFP plasmid (first plasmid)
    )

    # Simulate it
    result_pulsed, t_span = gfp_pulsed_circuit.simulate()

    # Visualize pulsed results
    visualizer.plot_simulation_results(
        result_pulsed,
        gfp_pulsed_circuit.name,
        t_span,
        pulse_config=pulse_config
    )

    # Example: Parameter comparison with multiple simulations
    compare_parameters_with_param_values()

    # Parameter sweep example
    parameter_sweep()


def compare_parameters_with_param_values():
    """
    Example using param_values for multiple simulations with improved visualization.
    """
    # Create circuit manager and visualizer
    manager = CircuitManager(parameters_file="../data/model_parameters_priors.csv")
    visualizer = CircuitVisualizer()

    # Create a single circuit instance
    circuit = manager.create_circuit("star")

    # Define parameter values to compare
    param_values = {
        "k_Star6_concentration": [0.5, 1.0, 2.0, 5.0, 10.0]
    }

    # Time span for all simulations
    t_span = np.linspace(0, 30, 3001)

    # Run all simulations at once using param_values
    result, t_span = circuit.simulate(t_span=t_span, param_values=param_values)

    # Use the visualizer to create a parameter comparison plot
    visualizer.plot_parameter_comparison(
        result,
        t_span,
        'obs_Protein_GFP',
        param_values,
        'k_Star6_concentration',
        title='Effect of Star6 Concentration on GFP Expression',
        ylabel='GFP Concentration'
    )

    # Also plot all simulations without parameter-specific formatting
    visualizer.plot_simulation_results(result, circuit.name, t_span)

    # Example with more parameters
    # Create a grid of parameters
    many_params = {
        "k_Star6_concentration": np.linspace(0.5, 5.0, 10),
        "k_prot_deg": np.linspace(0.05, 0.15, 3)
    }

    # Create all combinations
    param_grid = []
    for conc in many_params["k_Star6_concentration"]:
        for deg in many_params["k_prot_deg"]:
            param_grid.append({
                "k_Star6_concentration": conc,
                "k_prot_deg": deg
            })

    param_df = pd.DataFrame(param_grid)

    # Run many simulations
    result_many, t_span = circuit.simulate(t_span=t_span, param_values=param_df)

    # Visualize all simulations
    visualizer.plot_simulation_results(result_many, f"{circuit.name} (30 parameter sets)", t_span)

    return result


def parameter_sweep():
    """
    Example of a parameter sweep with improved visualization.
    """
    # Create circuit manager and visualizer
    manager = CircuitManager(parameters_file="../data/model_parameters_priors.csv")
    visualizer = CircuitVisualizer()

    # Create a single circuit instance
    circuit = manager.create_circuit("star")

    # Define parameter grid
    star_concentrations = np.linspace(0.1, 5.0, 10)
    protein_degradation_rates = np.linspace(0.05, 0.15, 3)

    # Create parameter grid as a DataFrame directly
    param_grid = []
    for conc in star_concentrations:
        for deg_rate in protein_degradation_rates:
            param_grid.append({
                "k_Star6_concentration": conc,
                "k_prot_deg": deg_rate
            })

    param_df = pd.DataFrame(param_grid)

    # Time span for all simulations
    t_span = np.linspace(0, 30, 3001)

    # Run all simulations at once
    result, _ = circuit.simulate(t_span=t_span, param_values=param_df)

    # Use the visualizer to create a heatmap
    visualizer.plot_parameter_sweep_heatmap(
        result,
        'obs_Protein_GFP',
        star_concentrations,
        'Star6 Concentration',
        protein_degradation_rates,
        'Protein Degradation Rate',
        metric='max',
        title='Parameter Sweep: Effect on Max GFP Expression'
    )

    # Also plot with a different metric
    visualizer.plot_parameter_sweep_heatmap(
        result,
        'obs_Protein_GFP',
        star_concentrations,
        'Star6 Concentration',
        protein_degradation_rates,
        'Protein Degradation Rate',
        metric='auc',
        title='Parameter Sweep: Effect on Total GFP Expression (AUC)',
        cmap='plasma'
    )

    return result


if __name__ == "__main__":
    main()
