import numpy as np
from circuits.build_model import setup_model, simulate_model, visualize_simulation
import pandas as pd
from pysb.simulator import ScipyOdeSimulator
from matplotlib import pyplot as plt


def test_toehold(plot=False, parameters_plasmids={"k_Toehold3_GFP_concentration": 1, "k_Trigger3_concentration": 1}):
    # Plasmid design
    plasmids = [
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),
        (None, None, [(False, "Trigger3")]),
    ]

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters_priors.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)

    # Setup the model
    model = setup_model(plasmids, parameters)

    # Time span for build_simulate_analyse
    n_steps = 1000
    t = np.linspace(0, 20, n_steps)

    # Run the build_simulate_analyse
    y_res = simulate_model(model, t)

    if plot:
        # Visualize results
        species_to_plot = list(model.observables.keys())
        visualize_simulation(t, y_res, species_to_plot=species_to_plot)

    return model


def test_toehold_pulsed(plot=True, print_rules=False, print_odes=False, debug_observables=False):
    """
    Test toehold circuit with pulsed Trigger3 input

    Parameters:
    -----------
    plot : bool
        Whether to plot results
    print_rules : bool
        Whether to print model rules
    print_odes : bool
        Whether to print model ODEs

    Returns:
    --------
    model : PySB model
        The simulated model
    """
    # Plasmid design - we have two plasmids: a toehold device and a trigger
    plasmids = [
        (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),  # Plasmid 0: Toehold switch (constant)
        (None, None, [(False, "Trigger3")]),  # Plasmid 1: Trigger (to be pulsed)
    ]

    # Load parameters but DON'T add k_Trigger3_concentration for pulsed case
    parameters_df = pd.read_csv('../data/model_parameters_priors.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.pop('k_Trigger3_concentration', None)

    # Set Toehold concentration
    parameters['k_Toehold3_GFP_concentration'] = 5
    parameters['k_prot_deg'] = 0.6
    parameters['k_rna_deg'] = 0.6


    # Pulse configuration
    pulse_config = {
        'use_pulse': True,
        'pulse_start': 4,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    }

    # Setup the model with pulses only for the second plasmid (index 1)
    model = setup_model(
        plasmids,
        parameters,
        use_pulses=True,
        pulse_config=pulse_config,
        pulse_indices=[1]  # Only pulse the second plasmid (Trigger3)
    )

    # Time span for simulation
    t_span = np.linspace(0, 30, 3001)
    sim = ScipyOdeSimulator(model, tspan=t_span)
    result = sim.run()

    # Debug: Print available observables if requested
    if debug_observables:
        print("Available observables:")
        for obs_name in result.observables.dtype.names:
            print(f"  - {obs_name}")

    if plot:
        # Get Time values from simulation results
        time_values = result.observables['obs_Time']

        # Calculate Trigger3 concentration values
        trigger_values = [float(model.expressions['k_Trigger3_concentration'].expr.subs(
            model.observables['obs_Time'], t)) for t in t_span]

        # Find the RNA observable for our toehold construct
        rna_observable = None
        for obs_name in result.observables.dtype.names:
            if 'RNA' in obs_name and 'Toehold3' in obs_name and 'GFP' in obs_name:
                rna_observable = obs_name
                break

        if not rna_observable:
            print("Warning: Could not find RNA observable for Toehold3_GFP construct")
            # Look for any RNA observable as fallback
            for obs_name in result.observables.dtype.names:
                if 'RNA' in obs_name:
                    rna_observable = obs_name
                    print(f"Using {rna_observable} as fallback")
                    break

        # Create four subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 12), sharex=True)

        # Plot GFP protein concentration
        ax1.plot(t_span, result.observables['obs_Protein_GFP'], label='GFP')
        ax1.set_ylabel('GFP concentration')
        ax1.set_title('GFP Concentration Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot RNA concentration if available
        if rna_observable:
            ax2.plot(t_span, result.observables[rna_observable],
                     label=rna_observable.replace('obs_', ''), color='blue')
            ax2.set_ylabel('RNA concentration')
            ax2.set_title('RNA Concentration Over Time')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "No RNA observable found",
                     horizontalalignment='center', verticalalignment='center')
            ax2.set_title("RNA Plot (No Data Available)")
            ax2.grid(True)

        # Plot k_Trigger3_concentration values
        ax3.plot(t_span, trigger_values, label='k_Trigger3_concentration', color='red')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('k_Trigger3_concentration')
        ax3.set_title('Trigger3 Concentration Over Time')
        ax3.legend()
        ax3.grid(True)

        # Plot Time monomer values
        ax4.plot(t_span, time_values, label='Time', color='green')
        ax4.set_ylabel('Time value')
        ax4.set_title('Time Variable Over Time')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()

    if print_rules:
        print("Model Rules:")
        for rule in model.rules:
            print(rule)

    if print_odes:
        print("\nModel ODEs:")
        for ode in model.odes:
            print(ode)

    return model


if __name__ == "__main__":
    # Run the pulsed model
    model = test_toehold_pulsed(plot=True, print_rules=False, print_odes=False, debug_observables=True)