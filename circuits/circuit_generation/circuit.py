import numpy as np
import pandas as pd
from pathlib import Path
from circuits.circuit_generation.build_model import setup_model, KineticsType
from utils.print_odes import find_ODEs_from_Pysb_model, convert_to_latex, write_to_file
from pysb.simulator import ScipyOdeSimulator


class Circuit:
    """
    A class representing a specific circuit model.
    Handles model setup and simulation.
    """

    def __init__(
        self,
        name,
        plasmids,
        plasmid_concentration_parameters=None,
        use_pulses=False,
        pulse_config=None,
        pulse_indices=None,
        pulse_plasmids=None,
        parameters_file=None,
        kinetics_type=KineticsType.MICHAELIS_MENTEN,
        bindings=None,
    ):
        self.name = name
        self.plasmids = plasmids
        self.plasmid_concentration_parameters = plasmid_concentration_parameters or {}
        self.use_pulses = use_pulses
        self.pulse_config = pulse_config

        # Store both pulse indices (for backward compatibility) and pulse_plasmids (new approach)
        self.pulse_indices = pulse_indices or []
        self.pulse_plasmids = pulse_plasmids or []

        # Create a mapping from plasmid name to index for faster lookups
        self.plasmid_name_to_index = {
            plasmid[0]: i for i, plasmid in enumerate(plasmids)
        }

        self.parameters_file = parameters_file
        self.kinetics_type = kinetics_type
        self.bindings = bindings or []

        # Load parameters from CSV and merge with user parameters
        self.parameters = self._load_parameters()

        # Set up the model during initialization
        self.model = self._setup_model()

    def _setup_model(self):
        """
        Internal method to set up the model using the provided configuration.
        """
        # Set up the model with kinetics_type and bindings
        if self.use_pulses and self.pulse_config:
            # For backward compatibility, use pulse_indices if provided
            # Otherwise, convert pulse_plasmids to indices
            pulse_indices = self.pulse_indices
            if not pulse_indices and self.pulse_plasmids:
                pulse_indices = [
                    self.plasmid_name_to_index[name]
                    for name in self.pulse_plasmids
                    if name in self.plasmid_name_to_index
                ]

            model = setup_model(
                self.plasmids,
                self.parameters,
                self.plasmid_concentration_parameters,
                bindings=self.bindings,
                use_pulses=True,
                pulse_config=self.pulse_config,
                pulse_indices=pulse_indices,
                pulse_plasmids=self.pulse_plasmids,  # Pass names too for future compatibility
                kinetics_type=self.kinetics_type,
            )
        else:
            model = setup_model(
                self.plasmids,
                self.parameters,
                self.plasmid_concentration_parameters,
                bindings=self.bindings,
                kinetics_type=self.kinetics_type,
            )

        return model

    def _get_parameters_to_remove(self):
        """
        Identify parameters that should be removed when using pulses.
        These are typically the concentration parameters for pulsed plasmids.

        Returns:
        --------
        list of parameter names to remove
        """
        if not self.use_pulses:
            return []

        to_remove = []

        # Get indices to pulse - either from direct indices or from plasmid names
        indices_to_pulse = self.pulse_indices.copy()

        # If we also have plasmid names, convert them to indices
        if self.pulse_plasmids:
            for plasmid_name in self.pulse_plasmids:
                if plasmid_name in self.plasmid_name_to_index:
                    idx = self.plasmid_name_to_index[plasmid_name]
                    if idx not in indices_to_pulse:
                        indices_to_pulse.append(idx)

        # If no indices specified, return empty list
        if not indices_to_pulse:
            return []

        # Process each plasmid index
        for idx in indices_to_pulse:
            if idx < len(self.plasmids):
                plasmid = self.plasmids[idx]
                # For a plasmid (name, tx_control, tl_control, cds), handle each gene
                _, _, _, genes = plasmid
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
        model_parameters = dict(zip(parameters_df["Parameter"], parameters_df["Mean"]))

        # Update with user-provided parameters
        # model_parameters.update(self.plasmid_concentration_parameters)

        # If using pulses, remove the concentration parameters for pulsed plasmids
        if self.use_pulses:
            params_to_remove = self._get_parameters_to_remove()
            for param in params_to_remove:
                if param in model_parameters:
                    print(f"Removing {param} parameter for pulsed plasmid")
                    self.plasmid_concentration_parameters.pop(param, None)
                    # the structure could now be made easier as we now take self.plasmid_concentration_parameters separately

        return model_parameters

    def simulate(
        self,
        t_span=None,
        param_values=None,
        solver_options=None,
        print_rules=False,
        print_odes=False,
        plot=False,
    ):
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
        solver_options : dict, optional
            Options for the ODE solver, e.g., {'rtol': 1e-6, 'atol': 1e-8}
        print_rules : bool, optional
            Whether to print model rules
        print_odes : bool, optional
            Whether to print model ODEs
        plot : bool, optional
            Whether to plot the results

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
        default_solver_options = {"rtol": 1e-6, "atol": 1e-8, "mxstep": 5000}
        final_solver_options = {**default_solver_options, **(solver_options or {})}

        if self.use_pulses and self.pulse_config:
            # Extract solver options from pulse_config if present
            pulse_solver_options = self.pulse_config.get("solver_options", {})
            final_solver_options.update(pulse_solver_options)

        sim = ScipyOdeSimulator(
            self.model,
            tspan=t_span,
            integrator="lsoda",
            integrator_options=final_solver_options,
        )

        # If param_values is provided, run multiple simulations
        if param_values is not None:
            param_df = {
                key: value.tolist() if hasattr(value, "tolist") else value
                for key, value in param_values.items()
            }

            # Run simulation with param_values
            result = sim.run(param_values=param_df)
        else:
            # Run a single simulation
            result = sim.run()

        # Optional plotting can be implemented here
        if plot:
            self._plot_results(result, t_span)

        return result, t_span

    def _plot_results(self, result, t_span):
        """
        Plot the simulation results.
        This is a placeholder for implementing visualization.

        Parameters:
        -----------
        result : SimulationResult
            The result from the simulation
        t_span : array
            Time span used for the simulation
        """
        try:
            import matplotlib.pyplot as plt

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot observables
            for obs_name, values in result.observables.items():
                if "Protein" in obs_name:
                    ax.plot(t_span, values, label=obs_name.replace("obs_Protein_", ""))

            # Add labels and legend
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.set_title(f"Simulation Results for {self.name} Circuit")
            ax.legend()
            plt.show()

        except ImportError:
            print(
                "Matplotlib not available for plotting. Install it with 'pip install matplotlib'"
            )

        except Exception as e:
            print(f"Error plotting results: {e}")

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

    def get_plasmid_index(self, plasmid_name):
        """
        Get the index of a plasmid by its name.

        Parameters:
        -----------
        plasmid_name : str
            Name of the plasmid

        Returns:
        --------
        int or None
            Index of the plasmid or None if not found
        """
        return self.plasmid_name_to_index.get(plasmid_name)

    def get_plasmid_by_name(self, plasmid_name):
        """
        Get a plasmid by its name.

        Parameters:
        -----------
        plasmid_name : str
            Name of the plasmid

        Returns:
        --------
        tuple or None
            The plasmid tuple or None if not found
        """
        idx = self.get_plasmid_index(plasmid_name)
        if idx is not None and idx < len(self.plasmids):
            return self.plasmids[idx]
        return None
