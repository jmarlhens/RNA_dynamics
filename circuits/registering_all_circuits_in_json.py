from circuit_manager import CircuitManager


def register_all_circuits(manager=None, parameters_file=None):
    """
    Register all available circuits to the database.

    Parameters:
    -----------
    manager : CircuitManager, optional
        Existing manager instance to use. If None, a new one will be created.
    parameters_file : str, optional
        Path to the parameters file

    Returns:
    --------
    CircuitManager
        The manager with all circuits registered
    """
    if manager is None:
        manager = CircuitManager(parameters_file=parameters_file)

    # Register basic circuits (already in initialize_database, but included for completeness)
    register_basic_circuits(manager)

    return manager


def register_basic_circuits(manager):
    """Register basic circuits (GFP and STAR)"""

    # Star circuit
    star_plasmids = [
        (("Sense6", "Star6"), None, [(True, "GFP")]),
        (None, None, [(False, "Star6")]),
    ]
    manager.add_circuit(
        name="star",
        plasmids=star_plasmids,
        default_parameters={
            "k_Sense6_GFP_concentration": 1,
            "k_Star6_concentration": 1
        }
    )

    # GFP positive control circuit
    gfp_plasmids = [
        (None, None, [(True, "GFP")])
    ]
    manager.add_circuit(
        name="gfp",
        plasmids=gfp_plasmids,
        default_parameters={
            "k_GFP_concentration": 3
        }
    )

    plasmids = [
        (("Sense_6", "STAR_6"), ("Toehold_3", "Trigger_3"), [(True, "GFP")]),
        # AND gate: Sense6/STAR6 and Toehold/Trigger3
        (None, None, [(False, "STAR_6")]),  # Express STAR6
        (None, None, [(False, "Trigger_3")]),  # Express Trigger3
    ]

    manager.add_circuit(
        name="and_gate",
        plasmids=plasmids,
        default_parameters={
            "k_Sense_6_Toehold_3_GFP_concentration": 1,
            "k_STAR_6_concentration": 1,
            "k_Trigger_3_concentration": 1,
        }
    )

    """Register coherent feed-forward loop circuit"""
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (("Sense6", "Star6"), None, [(False, "Trigger3")]),  # Step 2: Sense6 (controlled by Star6) controls Trigger3
        (("Sense6", "Star6"), ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Step 3: Sense6 and Toehold3 control GFP
    ]

    manager.add_circuit(
        name="cffl_type_1",
        plasmids=plasmids,
        default_parameters={
            "k_Sense6_Trigger3_concentration": 1,
            "k_Star6_concentration": 1,
            "k_Sense6_Toehold3_GFP_concentration": 1,
        }
    )

    """Register cascade circuit"""
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (("Sense6", "Star6"), None, [(False, "Trigger3")]),  # Step 2: Sense6 (controlled by Star6) controls Trigger3
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Step 3: Trigger3 controls Toehold3 and GFP translation
    ]

    manager.add_circuit(
        name="cascade",
        plasmids=plasmids,
        default_parameters={
            "k_Star6_concentration": 1,
            "k_Sense6_Trigger3_concentration": 1,
            "k_Toehold3_GFP_concentration": 1,
        }
    )

    """Register CFFL-1,2 circuit"""
    plasmids = [
        (None, None, [(False, "Star1")]),  # Express Star1
        (None, None, [(False, "Star6")]),  # Express Star6
        (("Sense1", "Star1"), None, [(False, "aSTAR6")]),  # Express aSTAR6 under control of Sense1
        (("Sense6", "Star6"), None, [(False, "Trigger3")]),  # Express Trigger3 under control of Sense6
        (("Sense6", "Star6"), ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Control GFP with Toehold3 and Trigger3
    ]

    manager.add_circuit(
        name="cffl_12",
        plasmids=plasmids,
        default_parameters={
            "k_Star1_concentration": 1,
            "k_Star6_concentration": 1,
            "k_Sense1_aSTAR6_concentration": 1000,
            "k_Sense6_Trigger3_concentration": 1,
            "k_Sense6_Toehold3_GFP_concentration": 1,
        }
    )

    """Register cleaved transcription and translation circuit"""
    plasmids = [
        (None, None, [(False, "STAR"), (True, "GFP")])
    ]

    manager.add_circuit(
        name="cleaved_transcription",
        plasmids=plasmids,
        default_parameters={
            "k_STAR_GFP_concentration": 1,  # Concentration factor for STAR and GFP
        }
    )

    """Register incoherent feed-forward loop circuit"""
    plasmids = [
        (None, None, [(False, "Star6"), (False, "Trigger3")]),  # Step 1: Express Star6 and Trigger3, cleaved by Csy4
        (("Sense6", "Star6"), None, [(False, "aTrigger3")]),  # Step 2: Express aTrigger3 under control of Sense6
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Step 3: Use Toehold3 to control GFP translation
    ]

    manager.add_circuit(
        name="iffl_1",
        plasmids=plasmids,
        default_parameters={
            "k_Star6_Trigger3_concentration": 1,  # Concentration factor for Trigger3 production
            "k_Sense6_aTrigger3_concentration": 1,  # Concentration factor for aTrigger3 production
            "k_Toehold3_GFP_concentration": 1,  # Concentration factor for GFP translation
        }
    )

    """Register sequestration circuit"""
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (None, None, [(False, "aStar6")]),  # Step 2: Express aStar6
    ]

    manager.add_circuit(
        name="sequestration",
        plasmids=plasmids,
        default_parameters={
            'k_Star6_concentration': 1,  # Initial concentration of Star6
            'k_aStar6_concentration': 1,  # Initial concentration of aStar6
        }
    )

    """Register toehold circuit"""
    plasmids = [
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),
        (None, None, [(False, "Trigger3")]),
    ]

    manager.add_circuit(
        name="toehold",
        plasmids=plasmids,
        default_parameters={
            "k_Toehold3_GFP_concentration": 1,
            "k_Trigger3_concentration": 1
        }
    )


def main():
    """Initialize the database with all circuits"""
    # You can specify the parameters file path here
    parameters_file = "../data/model_parameters_priors.csv"

    # Create a circuit manager and register all circuits
    manager = CircuitManager(parameters_file=parameters_file)
    register_all_circuits(manager, parameters_file)

    # List all available circuits
    circuits = manager.list_circuits()
    print(f"Successfully registered {len(circuits)} circuits:")
    for circuit in circuits:
        print(f"  - {circuit}")

    # Example: Load and simulate a circuit
    example_circuit = "cascade"
    if example_circuit in circuits:
        circuit = manager.create_circuit(example_circuit)
        _ = circuit.simulate(plot=True)
        print(f"Successfully simulated {example_circuit} circuit")


if __name__ == "__main__":
    main()
