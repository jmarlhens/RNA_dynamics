import json
from pathlib import Path
import pandas as pd
from circuits.circuit_generation.circuit import Circuit
from circuits.modules.base_modules import KineticsType


class CircuitManager:
    """
    A class to manage circuit configurations stored in a JSON file.
    Handles loading, saving, and creating Circuit instances.
    """

    def __init__(
        self, json_file="../../data/circuits/circuits.json", parameters_file=None
    ):
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
            with open(self.json_file, "r") as f:
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
        with open(self.json_file, "w") as f:
            json.dump(circuits, f, indent=2)

    def add_circuit(self, name, plasmids, default_parameters=None, bindings=None):
        """
        Add a new circuit configuration to the JSON file.

        Parameters:
        -----------
        name : str
            Name of the circuit
        plasmids : list of tuples
            List of plasmid tuples in the format [(plasmid_name, tx_control, tl_control, cds), ...]
            or legacy format [(tx_control, tl_control, cds), ...] which will be auto-named
        default_parameters : dict, optional
            Default parameters for the circuit
        bindings : list, optional
            List of tuples specifying sequestration reactions between species
        """
        circuits = self.load_circuits()

        # Convert plasmids to JSON-serializable format
        json_plasmids = []

        for i, plasmid in enumerate(plasmids):
            # Check if plasmid is in new format (with name) or legacy format
            if len(plasmid) == 4:  # New format: (name, tx_control, tl_control, cds)
                plasmid_name, tx_control, tl_control, cds = plasmid
            else:  # Legacy format: (tx_control, tl_control, cds)
                tx_control, tl_control, cds = plasmid
                plasmid_name = f"plasmid_{i}"  # Auto-generate a name

            json_tx_control = list(tx_control) if tx_control else None
            json_tl_control = list(tl_control) if tl_control else None
            json_cds = [[gene[0], gene[1]] for gene in cds]

            # Store plasmid with name
            json_plasmids.append(
                {
                    "name": plasmid_name,
                    "tx_control": json_tx_control,
                    "tl_control": json_tl_control,
                    "cds": json_cds,
                }
            )

        # Add circuit to the dictionary
        circuits["circuits"][name] = {
            "plasmids": json_plasmids,
            "default_parameters": default_parameters or {},
            "bindings": bindings or [],
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

        # Convert JSON plasmids back to Python tuples with names
        plasmids = []
        for json_plasmid in circuit["plasmids"]:
            # Check if plasmid is in new format (with name field) or legacy format
            if isinstance(json_plasmid, dict):
                # New format with explicit name field
                plasmid_name = json_plasmid["name"]
                json_tx_control = json_plasmid["tx_control"]
                json_tl_control = json_plasmid["tl_control"]
                json_cds = json_plasmid["cds"]
            else:
                # Legacy format: [tx_control, tl_control, cds]
                plasmid_name = f"plasmid_{len(plasmids)}"  # Auto-generate a name
                json_tx_control, json_tl_control, json_cds = json_plasmid

            tx_control = tuple(json_tx_control) if json_tx_control else None
            tl_control = tuple(json_tl_control) if json_tl_control else None
            cds = [(gene[0], gene[1]) for gene in json_cds]

            # Store plasmid with name in the new format
            plasmids.append((plasmid_name, tx_control, tl_control, cds))

        # Get default parameters and bindings
        default_parameters = circuit.get("default_parameters", {})
        bindings = circuit.get("bindings", [])

        return {
            "name": name,
            "plasmids": plasmids,
            "default_parameters": default_parameters,
            "bindings": bindings,
        }

    def create_circuit(
        self,
        name,
        parameters=None,
        use_pulses=False,
        pulse_config=None,
        pulse_plasmids=None,
        kinetics_type=KineticsType.MICHAELIS_MENTEN,
        bindings=None,
    ):
        """
        Create a Circuit instance from a stored configuration.

        Parameters:
        -----------
        name : str
            Name of the circuit
        parameters : dict, optional
            Custom parameters to override defaults
        use_pulses : bool, optional
            Whether to use pulse configuration
        pulse_config : dict, optional
            Configuration for the pulse (start, end, etc.)
        pulse_plasmids : list, optional
            List of plasmid names to pulse (replaces pulse_indices)
        kinetics_type : KineticsType, optional
            Type of kinetics to use
        bindings : list, optional
            List of binding interactions

        Returns:
        --------
        Circuit
            A configured Circuit instance
        """
        config = self.get_circuit_config(name)

        # Merge default parameters with custom parameters
        merged_parameters = config["default_parameters"].copy()
        if parameters:
            merged_parameters.update(parameters)

        # Use provided bindings or get from config
        circuit_bindings = (
            bindings if bindings is not None else config.get("bindings", [])
        )

        # Handle backward compatibility for pulse_indices
        pulse_indices = None
        if pulse_plasmids is not None:
            # Convert plasmid names to indices
            plasmid_name_to_index = {
                plasmid[0]: i for i, plasmid in enumerate(config["plasmids"])
            }
            pulse_indices = [
                plasmid_name_to_index[name]
                for name in pulse_plasmids
                if name in plasmid_name_to_index
            ]

        # Create and return a Circuit instance
        return Circuit(
            name=name,
            plasmids=config["plasmids"],
            parameters=merged_parameters,
            use_pulses=use_pulses,
            pulse_config=pulse_config,
            pulse_indices=pulse_indices,  # For backward compatibility
            pulse_plasmids=pulse_plasmids,  # New parameter
            parameters_file=self.parameters_file,
            kinetics_type=kinetics_type,
            bindings=circuit_bindings,
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
        return dict(zip(parameters_df["Parameter"], parameters_df["Value"]))
