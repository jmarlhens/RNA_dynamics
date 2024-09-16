import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator
from build_simulate_analyse.build_model import simulate_model

def dose_response(model, parameter_name, parameter_values, t, output_obs):
    """
    Simulate dose-response by varying a specified parameter over a range of values using PySB simulators.

    :param model: PySB model object to simulate.
    :param parameter_name: Name of the parameter to vary.
    :param parameter_values: Array of values for the parameter.
    :param t: Time array for simulation.
    :param observable_name: Name of the observable to track for dose-response.
    :return: None
    """
    # Get the default parameter values from the model
    param_sweep = {parameter_name: parameter_values}

    # Simulate the model with the parameter sweep
    simulator = ScipyOdeSimulator(model, tspan=t)
    simulation_result = simulator.run(param_values=param_sweep)

    # Extract the observable of interest at the final time point
    # simulation_result.observables[i][output_obs]
    dose_response_data = [simulation_result.observables[i][output_obs][-1] for i in range(len(parameter_values))]

    return dose_response_data

def plot_dose_response(parameter_values, dose_response_data, parameter_name, output_obs):
    """
    Plot the dose-response curve for a given parameter and observable.

    :param parameter_values: Array of values for the parameter.
    :param dose_response_data: Array of observable values for the dose-response.
    :param parameter_name: Name of the parameter varied.
    :param output_obs: Name of the observable tracked.
    :return: None
    """
    plt.figure()
    plt.plot(parameter_values, dose_response_data, 'o-')
    plt.xlabel(parameter_name)
    plt.ylabel(output_obs)
    # log
    plt.xscale('log')
    plt.title(f"Dose-response curve for {output_obs} vs {parameter_name}")
    plt.show()

# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Setup the model with the parameters and bindings
    from tests.test_star import test_star
    model = test_star(plot=False)

    # Define time points for simulation
    n_steps = 100
    t = np.linspace(0, 20, n_steps)

    # Parameter to vary and its range
    parameter_name = "k_Star1_concentration"
    parameter_values = np.logspace(-2, 1, 30).tolist()

    # Observable to track
    output_obs = "obs_Protein_GFP"

    # Run dose-response
    dose_response_data = dose_response(model, parameter_name, parameter_values, t, output_obs)

    # plot
    plot_dose_response(parameter_values, dose_response_data, parameter_name[0], output_obs)


    # Setup the model with the parameters and bindings
    from tests.test_toehold import test_toehold
    model = test_toehold(plot=False)

    # Parameter to vary and its range
    parameter_name = "k_Trigger1_concentration"

    # Run dose-response
    dose_response_data = dose_response(model, parameter_name, parameter_values, t, output_obs)

    # plot
    plot_dose_response(parameter_values, dose_response_data, parameter_name[0], output_obs)
