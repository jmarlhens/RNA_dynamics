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

    # Star circuit with named plasmids
    star_plasmids = [
        ("sense_gfp_plasmid", ("Sense6", "Star6"), None, [(True, "GFP")]),
        ("star6_plasmid", None, None, [(False, "Star6")]),
    ]
    manager.add_circuit(
        name="sense_star_6",  # Updated to match existing JSON
        plasmids=star_plasmids,
        default_parameters={
            "k_Sense6_GFP_concentration": 1,
            "k_Star6_concentration": 1,
        },
    )

    # GFP positive control circuit with named plasmid
    gfp_plasmids = [("gfp_plasmid", None, None, [(True, "GFP")])]
    manager.add_circuit(
        name="gfp", plasmids=gfp_plasmids, default_parameters={"k_GFP_concentration": 3}
    )

    # AND gate with named plasmids
    plasmids = [
        (
            "and_gate_plasmid",
            ("Sense_6", "STAR_6"),
            ("Toehold_3", "Trigger_3"),
            [(True, "GFP")],
        ),
        ("star6_expression_plasmid", None, None, [(False, "STAR_6")]),  # Express STAR6
        (
            "trigger3_expression_plasmid",
            None,
            None,
            [(False, "Trigger_3")],
        ),  # Express Trigger3
    ]

    manager.add_circuit(
        name="and_gate",
        plasmids=plasmids,
        default_parameters={
            "k_Sense_6_Toehold_3_GFP_concentration": 1,
            "k_STAR_6_concentration": 1,
            "k_Trigger_3_concentration": 1,
        },
    )

    # Coherent feed-forward loop circuit with named plasmids
    plasmids = [
        ("star6_expression", None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (
            "trigger3_expression",
            ("Sense6", "Star6"),
            None,
            [(False, "Trigger3")],
        ),  # Step 2: Sense6 controls Trigger3
        (
            "gfp_expression",
            ("Sense6", "Star6"),
            ("Toehold3", "Trigger3"),
            [(True, "GFP")],
        ),  # Step 3: Both control GFP
    ]

    manager.add_circuit(
        name="cffl_type_1",
        plasmids=plasmids,
        default_parameters={
            "k_Sense6_Trigger3_concentration": 1,
            "k_Star6_concentration": 1,
            "k_Sense6_Toehold3_GFP_concentration": 1,
        },
    )

    # Cascade circuit with named plasmids
    plasmids = [
        ("star6_plasmid", None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (
            "trigger3_plasmid",
            ("Sense6", "Star6"),
            None,
            [(False, "Trigger3")],
        ),  # Step 2: Sense6 controls Trigger3
        (
            "gfp_plasmid",
            None,
            ("Toehold3", "Trigger3"),
            [(True, "GFP")],
        ),  # Step 3: Trigger3 controls GFP translation
    ]

    manager.add_circuit(
        name="cascade",
        plasmids=plasmids,
        default_parameters={
            "k_Star6_concentration": 1,
            "k_Sense6_Trigger3_concentration": 1,
            "k_Toehold3_GFP_concentration": 1,
        },
    )

    # CFFL-1,2 circuit with named plasmids
    plasmids = [
        ("star1_plasmid", None, None, [(False, "Star1")]),  # Express Star1
        ("star6_plasmid", None, None, [(False, "Star6")]),  # Express Star6
        (
            "astar6_plasmid",
            ("Sense1", "Star1"),
            None,
            [(False, "aSTAR6")],
        ),  # Express aSTAR6 under Sense1
        (
            "trigger3_plasmid",
            ("Sense6", "Star6"),
            None,
            [(False, "Trigger3")],
        ),  # Express Trigger3 under Sense6
        (
            "gfp_plasmid",
            ("Sense6", "Star6"),
            ("Toehold3", "Trigger3"),
            [(True, "GFP")],
        ),  # GFP with both controls
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
        },
    )

    # Cleaved transcription and translation circuit with named plasmid
    plasmids = [("cleaved_plasmid", None, None, [(False, "STAR"), (True, "GFP")])]

    manager.add_circuit(
        name="cleaved_transcription",
        plasmids=plasmids,
        default_parameters={
            "k_STAR_GFP_concentration": 1,  # Concentration factor for STAR and GFP
        },
    )

    # Incoherent feed-forward loop circuit with named plasmids
    plasmids = [
        (
            "star6_trigger3_plasmid",
            None,
            None,
            [(False, "Star6"), (False, "Trigger3")],
        ),  # Express Star6 and Trigger3
        (
            "atrigger3_plasmid",
            ("Sense6", "Star6"),
            None,
            [(False, "aTrigger3")],
        ),  # Express aTrigger3 under Sense6
        (
            "gfp_plasmid",
            None,
            ("Toehold3", "Trigger3"),
            [(True, "GFP")],
        ),  # Toehold3 controls GFP translation
    ]

    manager.add_circuit(
        name="iffl_1",
        plasmids=plasmids,
        default_parameters={
            "k_Star6_Trigger3_concentration": 1,  # Concentration factor for Trigger3 production
            "k_Sense6_aTrigger3_concentration": 1,  # Concentration factor for aTrigger3 production
            "k_Toehold3_GFP_concentration": 1,  # Concentration factor for GFP translation
        },
    )

    # Sequestration circuit with named plasmids
    plasmids = [
        ("star6_plasmid", None, None, [(False, "Star6")]),  # Step 1: Express Star6
        ("astar6_plasmid", None, None, [(False, "aStar6")]),  # Step 2: Express aStar6
    ]

    manager.add_circuit(
        name="sequestration",
        plasmids=plasmids,
        default_parameters={
            "k_Star6_concentration": 1,  # Initial concentration of Star6
            "k_aStar6_concentration": 1,  # Initial concentration of aStar6
        },
    )

    # Toehold circuit with named plasmids
    plasmids = [
        ("gfp_plasmid", None, ("Toehold3", "Trigger3"), [(True, "GFP")]),
        ("trigger3_plasmid", None, None, [(False, "Trigger3")]),
    ]

    manager.add_circuit(
        name="toehold_trigger",  # Updated to match existing JSON
        plasmids=plasmids,
        default_parameters={
            "k_Toehold3_GFP_concentration": 1,
            "k_Trigger3_concentration": 1,
        },
    )


def register_star_antistar_1(manager):
    """Register STAR-STAR* circuit with named plasmids"""
    plasmids = [
        ("star1_plasmid", None, None, [(False, "Star1")]),  # Step 1: Express Star1
        ("astar1_plasmid", None, None, [(False, "aStar1")]),  # Step 2: Express aStar1
        (
            "gfp_plasmid",
            ("Sense1", "Star1"),
            None,
            [(True, "GFP")],
        ),  # Step 3: Sense1 controls GFP
    ]

    # Define binding between Star1 and aStar1
    bindings = [("Star1", "aStar1")]

    manager.add_circuit(
        name="star_antistar_1",
        plasmids=plasmids,
        default_parameters={
            "k_Star1_concentration": 1,  # Initial concentration of Star1
            "k_aStar1_concentration": 1,  # Initial concentration of aStar1
            "k_Sense1_GFP_concentration": 1,  # Initial concentration of GFP
        },
        bindings=bindings,  # Add bindings
    )


def main():
    """Initialize the database with all circuits"""
    # You can specify the parameters file path here
    parameters_file = "../../data/prior/model_parameters_priors.csv"

    # Create a circuit manager and register all circuits
    manager = CircuitManager(parameters_file=parameters_file)
    register_star_antistar_1(manager)
    register_all_circuits(manager)

    # List all available circuits
    circuits = manager.list_circuits()
    print(f"Successfully registered {len(circuits)} circuits:")
    for circuit in circuits:
        print(f"  - {circuit}")

    # Example: Load and simulate a circuit
    example_circuit = "star_antistar_1"
    if example_circuit in circuits:
        # Example of using named plasmids for pulsing
        circuit = manager.create_circuit(
            example_circuit,
            use_pulses=True,
            pulse_config={
                "use_pulse": True,
                "pulse_start": 5,
                "pulse_end": 15,
                "pulse_concentration": 5.0,
                "base_concentration": 0.0,
            },
            pulse_plasmids=["star1_plasmid"],  # Pulse by name instead of index
        )
        _ = circuit.simulate()
        print(
            f"Successfully simulated {example_circuit} circuit with pulse on star1_plasmid"
        )


if __name__ == "__main__":
    main()
