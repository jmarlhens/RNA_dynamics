from circuit_manager import CircuitManager
from circuit_visualizer import CircuitVisualizer
import numpy as np
from circuits.modules.base_modules import KineticsType


def main():
    """
    Main entry point for the circuit simulation system.
    Shows examples of using both Michaelis-Menten and mass action kinetics.
    """
    # Create circuit managers for both kinetics types
    manager_mm = CircuitManager(parameters_file="../data/model_parameters_priors_with_mass_action.csv")

    # Create a visualizer
    visualizer = CircuitVisualizer()

    # List available circuits
    print(f"Available circuits: {manager_mm.list_circuits()}")

    # Example: create and simulate a GFP circuit with Michaelis-Menten kinetics (default)
    gfp_circuit_mm = manager_mm.create_circuit("gfp")
    result_mm, t_span = gfp_circuit_mm.simulate(print_rules=True)

    # Visualize the Michaelis-Menten results
    visualizer.plot_simulation_results(
        result_mm,
        f"{gfp_circuit_mm.name} (Michaelis-Menten)",
        t_span
    )

    # Create and simulate the same circuit with mass action kinetics
    gfp_circuit_ma = manager_mm.create_circuit(
        "gfp",
        kinetics_type=KineticsType.MASS_ACTION
    )
    result_ma, t_span = gfp_circuit_ma.simulate(print_rules=True)

    # Visualize the mass action results
    visualizer.plot_simulation_results(
        result_ma,
        f"{gfp_circuit_ma.name} (Mass Action)",
        t_span
    )

    # Example: Compare pulsed circuit with both kinetics types
    pulse_config = {
        'use_pulse': True,
        'pulse_start': 4,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    }

    # Adjust degradation rates for quicker response
    pulsed_params = {
        'k_prot_deg': 0.1,
        'k_rna_deg': 0.1
    }

    # Create pulsed circuit with Michaelis-Menten kinetics
    pulsed_mm = manager_mm.create_circuit(
        "gfp",
        parameters=pulsed_params,
        use_pulses=True,
        pulse_config=pulse_config,
        pulse_indices=[0],  # Pulse the GFP plasmid
        kinetics_type=KineticsType.MICHAELIS_MENTEN
    )
    #
    # # Create pulsed circuit with mass action kinetics
    # pulsed_ma = manager_mm.create_circuit(
    #     "gfp",
    #     parameters=pulsed_params,
    #     use_pulses=True,
    #     pulse_config=pulse_config,
    #     pulse_indices=[0],  # Pulse the GFP plasmid
    #     kinetics_type=KineticsType.MASS_ACTION
    # )

    # Simulate both
    result_pulsed_mm, t_span = pulsed_mm.simulate()
    # result_pulsed_ma, t_span = pulsed_ma.simulate()

    # Visualize pulsed results
    visualizer.plot_simulation_results(
        result_pulsed_mm,
        f"{pulsed_mm.name} Pulsed (Michaelis-Menten)",
        t_span,
        pulse_config=pulse_config
    )

    # visualizer.plot_simulation_results(
    #     result_pulsed_ma,
    #     f"{pulsed_ma.name} Pulsed (Mass Action)",
    #     t_span,
    #     pulse_config=pulse_config
    # )

    # Compare parameters with both kinetics types
    compare_parameters_with_both_kinetics()


def compare_parameters_with_both_kinetics():
    """
    Example comparing parameter effects using both kinetics models.
    """
    # Create circuit manager and visualizer
    manager = CircuitManager(parameters_file="../data/model_parameters_priors_with_mass_action.csv")
    visualizer = CircuitVisualizer()

    # Create a circuit instance with Michaelis-Menten kinetics
    circuit_mm = manager.create_circuit(
        "star",
        kinetics_type=KineticsType.MICHAELIS_MENTEN
    )

    # Create the same circuit with mass action kinetics
    circuit_ma = manager.create_circuit(
        "star",
        kinetics_type=KineticsType.MASS_ACTION
    )

    # Define parameter values to compare
    param_values = {
        "k_Star6_concentration": [0.5, 1.0, 2.0, 5.0, 10.0]
    }

    # Time span for all simulations
    t_span = np.linspace(0, 30, 3001)

    # Run simulations for both kinetics types
    print("Michaelis-Menten rules:")
    print(circuit_mm.model.rules)
    print("Mass Action rules:")
    print(circuit_ma.model.rules)
    result_mm, _ = circuit_mm.simulate(t_span=t_span, param_values=param_values)
    result_ma, _ = circuit_ma.simulate(t_span=t_span, param_values=param_values)

    # Plot parameter comparisons
    visualizer.plot_parameter_comparison(
        result_mm,
        t_span,
        'obs_Protein_GFP',
        param_values,
        'k_Star6_concentration',
        title='Effect of Star6 Concentration (Michaelis-Menten)',
        ylabel='GFP Concentration'
    )

    visualizer.plot_parameter_comparison(
        result_ma,
        t_span,
        'obs_Protein_GFP',
        param_values,
        'k_Star6_concentration',
        title='Effect of Star6 Concentration (Mass Action)',
        ylabel='GFP Concentration'
    )

    return result_mm, result_ma


if __name__ == "__main__":
    main()
