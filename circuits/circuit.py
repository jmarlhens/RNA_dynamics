import numpy as np
import pandas as pd
from pathlib import Path
from circuits.build_model import setup_model, KineticsType
from utils.print_odes import find_ODEs_from_Pysb_model, convert_to_latex, write_to_file
from pysb.simulator import ScipyOdeSimulator
import itertools


class Circuit:
    """
    A class representing a specific circuit model.
    Handles model setup and simulation.
    """

    def __init__(self, name, plasmids, parameters=None,
                 use_pulses=False, pulse_config=None, pulse_indices=None,
                 parameters_file=None, kinetics_type=KineticsType.MICHAELIS_MENTEN):
        """
        Initialize a circuit with its model.

        Parameters:
        -----------
        (... other parameters as before ...)
        kinetics_type : KineticsType, optional
            Type of kinetics to use (Michaelis-Menten or mass action)
        """
        self.name = name
        self.plasmids = plasmids
        self.user_parameters = parameters or {}
        self.use_pulses = use_pulses
        self.pulse_config = pulse_config
        self.pulse_indices = pulse_indices or []
        self.parameters_file = parameters_file
        self.kinetics_type = kinetics_type

        # Load parameters from CSV and merge with user parameters
        self.parameters = self._load_parameters()

        # Setup the model during initialization
        self.model = self._setup_model()

    def _setup_model(self):
        """
        Internal method to setup the model using the provided configuration.
        """
        # Setup the model with kinetics_type
        if self.use_pulses and self.pulse_config:
            model = setup_model(
                self.plasmids,
                self.parameters,
                use_pulses=True,
                pulse_config=self.pulse_config,
                pulse_indices=self.pulse_indices,
                kinetics_type=self.kinetics_type
            )
        else:
            model = setup_model(
                self.plasmids,
                self.parameters,
                kinetics_type=self.kinetics_type
            )

        return model

    def _get_parameters_to_remove(self):
        """
        Identify parameters that should be removed when using pulses.
        These are typically the concentration parameters for pulsed plasmids.

        Returns:
        --------
        list
            List of parameter names to remove
        """
        if not self.use_pulses or not self.pulse_indices:
            return []

        to_remove = []
        for idx in self.pulse_indices:
            if idx < len(self.plasmids):
                plasmid = self.plasmids[idx]
                # For a plasmid (tx_control, tl_control, cds), handle each gene
                _, _, genes = plasmid
                for is_translated, gene_name in genes:
                    param_name = f"k_{gene_name}_concentration"
                    to_remove.append(param_name)

        return to_remove

    def _load_parameters(self):
        """
        Load parameters from CSV file and merge with user parameters.
        For pulsed plasmids, remove the corresponding concentration parameters.

        Returns:
        --------
        dict
            Merged parameters
        """
        # Find parameters file
        parameters_file = Path(self.parameters_file)

        # Load parameters from CSV
        parameters_df = pd.read_csv(parameters_file)
        model_parameters = dict(zip(parameters_df['Parameter'], parameters_df['Value']))

        # Update with user-provided parameters
        model_parameters.update(self.user_parameters)

        # If using pulses, remove the concentration parameters for pulsed plasmids
        if self.use_pulses:
            params_to_remove = self._get_parameters_to_remove()
            for param in params_to_remove:
                if param in model_parameters:
                    print(f"Removing {param} parameter for pulsed plasmid")
                    model_parameters.pop(param, None)

        return model_parameters

    def simulate(self, t_span=None, param_values=None, print_rules=False, print_odes=False):
        """
        Simulate the circuit.

        Parameters:
        -----------
        t_span : array, optional
            Time span for simulation
        param_values : dict, DataFrame, or list, optional
            Parameter values for multiple simulations. Can be:
            - dict mapping parameter names to lists of values
            - pandas DataFrame with parameter names as columns
            - list of dictionaries, each containing a set of parameter values
        print_rules : bool, optional
            Whether to print model rules
        print_odes : bool, optional
            Whether to print model ODEs

        Returns:
        --------
        tuple
            (SimulationResult, t_span) - The simulation results and the time span used
        """
        # Set up time span
        if t_span is None:
            t_span = np.linspace(0, 30, 3001)

        # Print rules if requested
        if print_rules:
            print("Model Rules:")
            for rule in self.model.rules:
                print(rule)

        # Print ODEs if requested
        if print_odes:
            print("\nModel ODEs:")
            for ode in self.model.odes:
                print(ode)
            equations = find_ODEs_from_Pysb_model(self.model)
            print(equations)

        # Create simulator
        sim = ScipyOdeSimulator(self.model, tspan=t_span)

        # If param_values is provided, run multiple simulations
        if param_values is not None:
            # Convert param_values to a format suitable for ScipyOdeSimulator
            if isinstance(param_values, dict):
                # Convert dict to DataFrame (handle single values as well as lists)
                param_sets = []
                keys = list(param_values.keys())
                values = list(param_values.values())

                # Handle the case where values are not all lists
                for i, val in enumerate(values):
                    if not isinstance(val, (list, tuple, np.ndarray)):
                        values[i] = [val]

                # Generate all combinations of parameter values
                for combo in itertools.product(*values):
                    param_set = dict(zip(keys, combo))
                    param_sets.append(param_set)

                param_df = pd.DataFrame(param_sets)
            elif isinstance(param_values, list):
                # Convert list of dicts to DataFrame
                param_df = pd.DataFrame(param_values)
            else:
                # Assume it's already a DataFrame
                param_df = param_values

            # Run simulation with param_values
            result = sim.run(param_values=param_df)
        else:
            # Run a single simulation
            result = sim.run()

        return result, t_span

    def print_latex_equations(self):
        """
        Print and save the ODEs in LaTeX format.

        Returns:
        --------
        str
            LaTeX formatted equations
        """
        equations = find_ODEs_from_Pysb_model(self.model)
        latex_equations = convert_to_latex(equations)
        write_to_file(latex_equations)
        return latex_equations
