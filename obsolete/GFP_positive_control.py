import numpy as np
from circuits.build_model import setup_model
import pandas as pd
from utils.print_odes import find_ODEs_from_Pysb_model
from pysb.simulator import ScipyOdeSimulator
from matplotlib import pyplot as plt


def test_pos_control_constant(plot=False, print_rules=False, print_odes=False,
                              parameters_plasmids={"k_GFP_concentration": 3}):
    # Plasmid design
    plasmids = [
        (None, None, [(True, "GFP")]),
    ]

    # load and add parameters_plasmids
    parameters_df = pd.read_csv('../data/model_parameters_priors.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    parameters.update(parameters_plasmids)


    # Setup the model
    model = setup_model(plasmids, parameters)

    # Time span for simulation
    t_span = np.linspace(0, 30, 3001)
    sim = ScipyOdeSimulator(model, tspan=t_span)
    result = sim.run()

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # Plot GFP protein concentration
        ax1.plot(t_span, result.observables['obs_Protein_GFP'], label='GFP')
        ax1.set_ylabel('GFP concentration')
        ax1.set_title('GFP Concentration Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot RNA GFP concentration
        ax2.plot(t_span, result.observables['obs_RNA_GFP'], label='RNA_GFP', color='blue')
        ax2.set_ylabel('RNA GFP concentration')
        ax2.set_title('RNA GFP Concentration Over Time')
        ax2.legend()
        ax2.grid(True)

        # plot GFP protein concentration
        k_gfp_values = [float(model.parameters['k_GFP_concentration'].value) for t in t_span]
        ax3.plot(t_span, k_gfp_values, label='k_GFP_concentration', color='red')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('k_GFP_concentration')
        ax3.set_title('Plasmid Concentration Over Time')
        ax3.legend()
        ax3.grid(True)



    if print_rules:
        print("Model Rules:")
        for rule in model.rules:
            print(rule)

    if print_odes:
        print("\nModel ODEs:")
        for ode in model.odes:
            print(ode)
        equations = find_ODEs_from_Pysb_model(model)
        print(equations)

    return model


def test_pos_control_pulsed(plot=True, print_rules=True, print_odes=True):
    """
    Test GFP positive control with pulsed GFP plasmid

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
    # Plasmid design - just one plasmid for GFP
    plasmids = [
        (None, None, [(True, "GFP")]),  # Plasmid 0: GFP (to be pulsed)
    ]

    # Load parameters but DON'T add k_GFP_concentration for pulsed case
    parameters_df = pd.read_csv('../data/model_parameters_priors.csv')
    parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    # parameters.pop('k_GFP_concentration', None)

    # Increase k_prot_deg and k_rna_deg for quicker response
    parameters['k_prot_deg'] = 0.1
    parameters['k_rna_deg'] = 0.1

    # Pulse configuration
    pulse_config = {
        'use_pulse': True,
        'pulse_start': 4,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    }

    # Setup the model with pulses for the GFP plasmid (index 0)
    model = setup_model(
        plasmids,
        parameters,
        use_pulses=True,
        pulse_config=pulse_config,
        pulse_indices=[0]  # Pulse the first (and only) plasmid
    )

    # Time span for simulation
    t_span = np.linspace(0, 30, 3001)
    sim = ScipyOdeSimulator(model, tspan=t_span)
    result = sim.run()

    if plot:
        # Get Time values from simulation results
        time_values = result.observables['obs_Time']

        # Calculate k_GFP_concentration values
        k_gfp_values = [float(model.expressions['k_GFP_concentration'].expr.subs(
            model.observables['obs_Time'], t)) for t in t_span]

        # Create four subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 12), sharex=True)

        # Plot GFP protein concentration
        ax1.plot(t_span, result.observables['obs_Protein_GFP'], label='GFP')
        ax1.set_ylabel('GFP concentration')
        ax1.set_title('GFP Concentration Over Time')
        ax1.legend()
        ax1.grid(True)

        # Plot RNA GFP concentration
        ax2.plot(t_span, result.observables['obs_RNA_GFP'], label='RNA_GFP', color='blue')
        ax2.set_ylabel('RNA GFP concentration')
        ax2.set_title('RNA GFP Concentration Over Time')
        ax2.legend()
        ax2.grid(True)

        # Plot k_GFP_concentration values
        ax3.plot(t_span, k_gfp_values, label='k_GFP_concentration', color='red')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('k_GFP_concentration')
        ax3.set_title('Plasmid Concentration Over Time')
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
        equations = find_ODEs_from_Pysb_model(model)
        print(equations)

    return model


if __name__ == "__main__":
    #
    # test_pos_control_constant(plot=True, print_rules=False, print_odes=False)
    # Choose which test to run
    model = test_pos_control_pulsed(plot=True, print_rules=False, print_odes=False)
