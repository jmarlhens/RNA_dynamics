import numpy as np
import matplotlib.pyplot as plt
from pysb.simulator import ScipyOdeSimulator
import pandas as pd
from matplotlib import cm
import re

def dose_response_2d(model, param1_name, param1_values, param2_name, param2_values, t, output_observables):
    """
    Simulate 2D dose-response by varying two specified parameters over a range of values using PySB simulators.

    :param model: PySB model object to simulate.
    :param param1_name: Name of the first parameter to vary.
    :param param1_values: Array of values for the first parameter.
    :param param2_name: Name of the second parameter to vary.
    :param param2_values: Array of values for the second parameter.
    :param t: Time array for simulation.
    :param output_observable: Name of the observable to track for dose-response.
    :return: 2D array of observable values at the final time point for each parameter pair.
    """
    # Create a meshgrid for the two parameters
    param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
    param_sweep = {
        param1_name: param1_grid.flatten().tolist(),
        param2_name: param2_grid.flatten().tolist(),
    }

    # Simulate the model with the parameter grid
    simulator = ScipyOdeSimulator(model, tspan=t)
    simulation_result = simulator.run(param_values=param_sweep)

    # Extract the observable of interest at the final time point
    dose_response_data = [
        [simulation_result.observables[i][output_obs][-1] for i in range(len(param1_values) * len(param2_values))]
        for output_obs in output_observables
    ]

    return dose_response_data

def plot_3d_dose_response(param1_values, param2_values, dose_response_values, param1_name, param2_name, output_observables, circuit_name=None, cmap=cm.viridis):
    """
    Plot a 3D surface for the dose-response curve with two parameters.

    :param param1_values: Array of values for the first parameter.
    :param param2_values: Array of values for the second parameter.
    :param dose_response_data: 2D array of observable values for the dose-response.
    :param param1_name: Name of the first parameter varied.
    :param param2_name: Name of the second parameter varied.
    :param output_observable: Name of the observable tracked.
    :return: None
    """
    param1_values, param2_values = np.meshgrid(param1_values, param2_values)

    for output_observable, dose_response_data in zip(output_observables, dose_response_values):
        fig = plt.figure(figsize=(8, 7))

        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(np.log(param1_values), np.log(param2_values), np.array(dose_response_data).reshape(len(param2_values), len(param1_values)), cmap=cmap)

        param1_name_plot = re.search(r'^k_(.+?)_concentration$', param1_name)
        param1_name_plot = "[p" + param1_name_plot.group(1) + "]"
        param1_name_plot = param1_name_plot.replace("_", "-")
        param2_name_plot = re.search(r'^k_(.+?)_concentration$', param2_name)
        param2_name_plot = "[p" + param2_name_plot.group(1) + "]"
        param2_name_plot = param2_name_plot.replace("_", "-")

        ax.set_xlabel(f" log({param1_name_plot})")
        ax.set_ylabel(f" log({param2_name_plot})")

        output_observable_plot = output_observable.split("_")[-1]
        ax.set_zlabel(output_observable_plot)
        # if circuit_name:
        #     ax.set_title(f"3D dose-response surface for {circuit_name}")
        # else:
        #     ax.set_title("3D dose-response surface")

        plt.show()

def run_2d_dose_response_tests(circuits, param1_values, param2_values, t, cmap = cm.viridis):
    for circuit in circuits:
        model = circuit['test_func'](plot=False)
        dose_response_data = dose_response_2d(
            model, circuit['param1_name'], param1_values,
            circuit['param2_name'], param2_values, t, circuit['output_observables']
        )
        plot_3d_dose_response(
            param1_values, param2_values, dose_response_data,
            circuit['param1_name'], circuit['param2_name'],
            circuit['output_observables'], circuit['name'],
            cmap=cmap
        )

def dose_response(model, parameter_name, parameter_values, t, output_observables):
    """
    Simulate dose-response by varying a specified parameter over a range of values using PySB simulators.

    :param model: PySB model object to simulate.
    :param parameter_name: Name of the parameter to vary.
    :param parameter_values: Array of values for the parameter.
    :param t: Time array for simulation.
    :param output_observables: Names of the observables to track for dose-response.
    :return: Array of observable values at the final time point for each parameter value.
    """
    # Prepare parameter sweep dictionary for the simulator
    param_sweep = {parameter_name: parameter_values}

    # Simulate the model with the parameter sweep
    simulator = ScipyOdeSimulator(model, tspan=t)
    simulation_result = simulator.run(param_values=param_sweep)

    # Extract the observable of interest at the final time point
    dose_response_data = [
        [simulation_result.observables[i][output_obs][-1] for i in range(len(parameter_values))]
        for output_obs in output_observables
    ]

    return dose_response_data

def plot_dose_response(parameter_values, dose_response_data, parameter_name, output_observables, circuit_name=None, color=(0, 0, 0)):
    """
    Plot the dose-response curve for a given parameter and observable.

    :param parameter_values: Array of values for the parameter.
    :param dose_response_data: Array of observable values for the dose-response.
    :param parameter_name: Name of the parameter varied.
    :param output_observables: Names of the observables tracked.
    :param circuit_name: Name of the circuit being tested.
    :return: None
    """
    plt.figure()

    for output_obs, dose_response_values in zip(output_observables, dose_response_data):
        output_obs_plot = output_obs.split("_")[-1]
        plt.plot(parameter_values, dose_response_values, label=output_obs_plot, marker='o', color=color)

    # change parameter_name: k_somename_concentration to [trigger]
    parameter_name_plot = re.search(r'^k_(.+?)_concentration$', parameter_name)
    parameter_name_plot = "[p" + parameter_name_plot.group(1) + "]"
    parameter_name_plot = parameter_name_plot.replace("_", "-")

    plt.xlabel(f"{parameter_name_plot}")
    plt.ylabel("Fluorescence Intensity (a.u.)")
    plt.legend()
    plt.xscale('log')
    # plt.title(f"Dose-response curve for {circuit_name}" if circuit_name else "Dose-response curve")
    # plt.grid(True)
    plt.show()

def run_dose_response_tests(circuits, parameter_values, t, color=(0, 0, 0)):
    """
    Run dose-response circuits for a series of circuits and plot the results.

    :param circuits: List of dictionaries containing circuit configurations.
    :param parameter_values: Array of values for the parameter to vary.
    :param t: Time array for simulation.
    :return: None
    """
    for circuit in circuits:
        model = circuit['test_func'](plot=False)
        dose_response_data = dose_response(model, circuit['parameter_name'], parameter_values, t, circuit['output_observables'])
        plot_dose_response(parameter_values, dose_response_data, circuit['parameter_name'], circuit['output_observables'], circuit['name'], color)

# Example usage
# Example usage
if __name__ == "__main__":
    import ast
    color_scheme = pd.read_excel("../Color scheme.xlsx", index_col=0, header=0)
    dark_blue = ast.literal_eval(color_scheme.loc["Dark blue", "Python color code"][1])

    from color_scheme import hex_to_rgb
    from matplotlib.colors import LinearSegmentedColormap
    cmap_colors = ["#FFFF00", "#FF0000"]
    cmap_colors = [hex_to_rgb(hex_val) for hex_val in cmap_colors]
    cmap_colors = np.array(cmap_colors)
    cmap = LinearSegmentedColormap.from_list("my_cmap", cmap_colors / 255)

    # Define time points for simulation
    n_steps = 100
    t = np.linspace(0, 6, n_steps)

    # Define parameter values for dose-response
    parameter_values = np.logspace(-2, 2, 60).tolist()

    # List of circuits with configurations for dose-response circuits
    circuits = [
        {
            'name': "STAR regulation",
            'test_func': lambda plot: __import__('circuits.test_star').test_star.test_star(plot=plot),
            'parameter_name': "k_Star1_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "Toehold regulation",
            'test_func': lambda plot: __import__('circuits.test_toehold').test_toehold.test_toehold(plot=plot),
            'parameter_name': "k_Trigger1_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "AND gate",
            'test_func': lambda plot: __import__('circuits.test_AND_gate').test_AND_gate.test_AND_gate(plot=plot),
            'parameter_name': "k_STAR_6_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "Cascade",
            'test_func': lambda plot: __import__('circuits.test_cascade').test_cascade.test_cascade(plot=plot),
            'parameter_name': "k_Star6_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "CFFL-1",
            'test_func': lambda plot: __import__('circuits.test_cffl_type_1').test_cffl_type_1.test_coherent_feed_forward_loop(plot=plot),
            'parameter_name': "k_Star6_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "IFFL-1",
            'test_func': lambda plot: __import__('circuits.test_iffl1').test_iffl1.test_iffl_1(plot=plot),
            'parameter_name': "k_Sense6_aTrigger3_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "IFFL-1",
            'test_func': lambda plot: __import__('circuits.test_iffl1').test_iffl1.test_iffl_1(plot=plot),
            'parameter_name': "k_Star6_Trigger3_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "CFFL-1,2",
            'test_func': lambda plot: __import__('circuits.c12ffl').c12ffl.test_cffl_12(plot=plot),
            'parameter_name': "k_Star1_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        {
            'name': "CFFL-1,2",
            'test_func': lambda plot: __import__('circuits.c12ffl').c12ffl.test_cffl_12(plot=plot),
            'parameter_name': "k_Star6_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
    ]

    # Run dose-response circuits for all circuits
    # run_dose_response_tests(circuits, parameter_values, t, color=dark_blue)


    ##################
    # 2D Dose-Response
    ##################
    # Define time points for simulation
    n_steps = 100
    t = np.linspace(0, 4, n_steps)

    param1_values = np.logspace(3, 10, 20)
    param2_values = np.logspace(3, 10, 20)

    circuits = [
        {
            'name': "CFFL-1",
            'test_func': lambda plot: __import__('circuits.test_cffl_type_1').test_cffl_type_1.test_coherent_feed_forward_loop(plot=plot),
            'param1_name': "k_Star6_concentration",
            'param2_name': "k_Sense6_Trigger3_concentration",
            'output_observables': ["obs_Protein_GFP"],
        },
        # {
        #     'name': "AND gate",
        #     'test_func': lambda plot: __import__('circuits.test_AND_gate').test_AND_gate.test_AND_gate(plot=plot),
        #     'param1_name': "k_Trigger_3_concentration",
        #     'param2_name': "k_STAR_6_concentration",
        #     'output_observables': ["obs_Protein_GFP"],
        # },
        # {
        #     'name': "IFFL-1",
        #     'test_func': lambda plot: __import__('circuits.test_iffl1').test_iffl1.test_iffl_1(plot=plot),
        #     'param1_name': "k_Sense6_aTrigger3_concentration",
        #     'param2_name': "k_Star6_Trigger3_concentration",
        #     'output_observables': ["obs_Protein_GFP", "obs_RNA_aTrigger3", "obs_RNA_Trigger3"],
        # },
        # {
        #     'name': "CFFL-1,2",
        #     'test_func': lambda plot: __import__('circuits.c12ffl').c12ffl.test_cffl_12(plot=plot),
        #     'param1_name': "k_Star1_concentration",
        #     'param2_name': "k_Star6_concentration",
        #     'output_observables': ["obs_Protein_GFP", "obs_RNA_aSTAR6", "obs_RNA_Trigger3"],
        # },
    ]


    run_2d_dose_response_tests(circuits, param1_values, param2_values, t, cmap = cmap)