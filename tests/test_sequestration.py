import numpy as np
from simulation.simulate import setup_model, simulate_model, visualize_simulation

def test_sequestration():
    # Plasmid design for sequestration test:
    # 1. First plasmid expresses Star6
    # 2. Second plasmid expresses aStar6 (anti-Star6)
    plasmids = [
        (None, None, [(False, "Star6")]),  # Step 1: Express Star6
        (None, None, [(False, "aStar6")]),  # Step 2: Express aStar6
    ]

    # Define model parameters
    parameters = {
        "k_tx": 2,  # Transcription rate
        "k_rna_deg": 0.5,  # RNA degradation rate
        "k_sequestration_bind": 1,  # Binding rate for sequestration
        "k_sequestration_unbind": 0.1,  # Unbinding rate for sequestration
        'k_Star6_concentration': 1,  # Initial concentration of Star6
        'k_aStar6_concentration': 4,  # Initial concentration of aStar6
    }

    # Define sequestration reactions: [(species1, species2)]
    # Here, Star6 and aStar6 bind and unbind
    bindings = [
        ("RNA_Star6", "RNA_aStar6"),  # Sequestration between Star6 and aStar6
    ]

    # Setup the model with the plasmids, parameters, and bindings
    model = setup_model(plasmids, parameters, bindings=bindings)

    # Time span for simulation
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Run the simulation
    y_res = simulate_model(model, t)

    # Visualize results
    species_to_plot = list(model.observables.keys())
    visualize_simulation(t, y_res, species_to_plot=species_to_plot)

if __name__ == "__main__":
    test_sequestration()

