import json
from pathlib import Path
import pandas as pd
from circuits.circuit import Circuit
from circuits.modules.base_modules import KineticsType


class CircuitManager:
    """
    A class to manage circuit configurations stored in a JSON file.
    Handles loading, saving, and creating Circuit instances.
    """

    def __init__(self, json_file='../data/circuits/circuits.json', parameters_file=None):
        """
        Initialize the CircuitManager with a JSON file path.

        Parameters:
        -----------
        json_file : str
            Path to the JSON file for storing circuit configurations
        parameters_file : str, optional
            Path to the CSV file containing model parameters
        """
        self.json_file = json_file
        self.parameters_file = parameters_file

    def load_circuits(self):
        """
        Load circuit configurations from JSON file.

        Returns:
        --------
        dict
            Dictionary containing circuit configurations
        """
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"circuits": {}}

    def save_circuits(self, circuits):
        """
        Save circuit configurations to JSON file.

        Parameters:
        -----------
        circuits : dict
            Dictionary containing circuit configurations
        """
        with open(self.json_file, 'w') as f:
            json.dump(circuits, f, indent=2)

    def add_circuit(self, name, plasmids, default_parameters=None, bindings=None):
        """
        Add a new circuit configuration to the JSON file.

        Parameters:
        -----------
        name : str
            Name of the circuit
        plasmids : list of tuples
            List of plasmid tuples in the format [(tx_control, tl_control, cds), ...]
        default_parameters : dict, optional
            Default parameters for the circuit
        bindings : list, optional
            List of tuples specifying sequestration reactions between species
        """
        circuits = self.load_circuits()

        # Convert plasmids to JSON-serializable format
        json_plasmids = []
        for plasmid in plasmids:
            tx_control, tl_control, cds = plasmid
            json_tx_control = list(tx_control) if tx_control else None
            json_tl_control = list(tl_control) if tl_control else None
            json_cds = [[gene[0], gene[1]] for gene in cds]
            json_plasmids.append([json_tx_control, json_tl_control, json_cds])

        # Add circuit to the dictionary
        circuits["circuits"][name] = {
            "plasmids": json_plasmids,
            "default_parameters": default_parameters or {},
            "bindings": bindings or []  # Add bindings
        }

        # Save updated circuits to JSON file
        self.save_circuits(circuits)

    def get_circuit_config(self, name):
        """
        Get a specific circuit configuration from the JSON file.
        """
        circuits = self.load_circuits()
        circuit = circuits["circuits"].get(name)
        if not circuit:
            raise ValueError(f"Circuit '{name}' not found in {self.json_file}")

        # Convert JSON plasmids back to Python tuples
        plasmids = []
        for json_plasmid in circuit["plasmids"]:
            json_tx_control, json_tl_control, json_cds = json_plasmid
            tx_control = tuple(json_tx_control) if json_tx_control else None
            tl_control = tuple(json_tl_control) if json_tl_control else None
            cds = [(gene[0], gene[1]) for gene in json_cds]
            plasmids.append((tx_control, tl_control, cds))

        # Get default parameters and bindings
        default_parameters = circuit.get("default_parameters", {})
        bindings = circuit.get("bindings", [])

        return {
            "name": name,
            "plasmids": plasmids,
            "default_parameters": default_parameters,
            "bindings": bindings
        }

    def create_circuit(self, name, parameters=None, use_pulses=False,
                       pulse_config=None, pulse_indices=None,
                       kinetics_type=KineticsType.MICHAELIS_MENTEN,
                       bindings=None):
        """
        Create a Circuit instance from a stored configuration.
        """
        config = self.get_circuit_config(name)

        # Merge default parameters with custom parameters
        merged_parameters = config["default_parameters"].copy()
        if parameters:
            merged_parameters.update(parameters)

        # Use provided bindings or get from config
        circuit_bindings = bindings if bindings is not None else config.get("bindings", [])

        # Create and return a Circuit instance
        return Circuit(
            name=name,
            plasmids=config["plasmids"],
            parameters=merged_parameters,
            use_pulses=use_pulses,
            pulse_config=pulse_config,
            pulse_indices=pulse_indices,
            parameters_file=self.parameters_file,
            kinetics_type=kinetics_type,
            bindings=circuit_bindings
        )

    def list_circuits(self):
        """
        List all available circuits in the JSON file.

        Returns:
        --------
        list of circuit names
        """
        circuits = self.load_circuits()
        return list(circuits["circuits"].keys())

    def load_parameters(self, parameters_file=None):
        """
        Load model parameters from a CSV file.

        Parameters:
        -----------
        parameters_file : str or Path, optional
            Path to the parameters CSV file (overrides the default)

        Returns:
        --------
        dict
            Dictionary of parameter names and values
        """
        file_path = parameters_file or self.parameters_file
        parameters_file = Path(file_path)

        # Load parameters from CSV
        parameters_df = pd.read_csv(parameters_file)
        return dict(zip(parameters_df['Parameter'], parameters_df['Value']))
