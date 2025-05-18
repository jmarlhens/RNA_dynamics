# Synthetic Circuit Simulation System: Comprehensive Guide

This documentation provides a detailed explanation of the synthetic circuit simulation system, which enables the modeling, simulation, and analysis of various genetic circuits.

## Table of Contents

- [System Architecture](#system-architecture)
- [Circuit Configuration](#circuit-configuration)
- [Getting Started](#getting-started)
- [Working with Circuits](#working-with-circuits)
- [Running Simulations](#running-simulations)
- [Pulse Configuration](#pulse-configuration)
- [Visualization and Analysis](#visualization-and-analysis)
- [Named Plasmids Feature](#named-plasmids-feature)
- [Extending the System](#extending-the-system)
- [Example Workflows](#example-workflows)
- [Troubleshooting](#troubleshooting)

## System Architecture

The system consists of two main classes:

1. **CircuitManager**: Manages circuit configurations stored in a JSON file, handling loading, saving, and creating circuit instances.
2. **Circuit**: Represents a specific circuit model, handling setup and simulation.

Additionally, we have helper modules:

- **register_circuits.py**: Registers predefined circuit templates
- **usage_examples.py**: Demonstrates usage with various examples

The system utilizes PySB underneath for model generation and simulation.

## Circuit Configuration

### Plasmid Structure

Circuits are defined by plasmids, which are represented as tuples with the following structure:

```python
(plasmid_name, transcriptional_control, translational_control, cds)
```

The `plasmid_name` field is a unique identifier for the plasmid within the circuit.

### JSON Storage

Circuit configurations are stored in a JSON file with the following structure:

```json
{
  "circuits": {
    "circuit_name": {
      "plasmids": [...],
      "default_parameters": {...}
    },
    ...
  }
}
```

The JSON storage contains the circuit structure and default parameters, while simulation configurations (like pulse behavior) are specified at runtime.

## Getting Started

### Installation

The system requires Python 3.6+ and depends on:

- PySB
- NumPy
- Pandas
- Matplotlib

### Initializing the Circuit Database

```python
from circuit_manager import CircuitManager
from register_circuits import register_all_circuits

# Create a manager with a path to parameters file
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")

# Register all predefined circuits
register_all_circuits(manager)

# List available circuits
available_circuits = manager.list_circuits()
print(f"Available circuits: {available_circuits}")
```

## Working with Circuits

### Creating a Circuit Instance

```python
# Basic circuit creation
gfp_circuit = manager.create_circuit("gfp")

# Circuit with custom parameters
star_circuit = manager.create_circuit(
    "star", 
    parameters={
        "k_Sense6_GFP_concentration": 1.5,
        "k_Star6_concentration": 2.0
    }
)

# Circuit with pulse configuration using named plasmids
pulsed_circuit = manager.create_circuit(
    "gfp",
    parameters={"k_prot_deg": 0.1, "k_rna_deg": 0.1},
    use_pulses=True,
    pulse_config={
        'use_pulse': True,
        'pulse_start': 4,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    },
    pulse_plasmids=["gfp_plasmid"]  # Pulse the plasmid by name
)
```

### Registering a New Circuit

```python
# Define plasmids for a new circuit with names
custom_plasmids = [
    ("gfp_plasmid", ("STAR1", "SENSE1"), None, [(True, "GFP")]),
    ("gene2_plasmid", None, None, [(False, "Gene2")])
]

# Add to database with default parameters
manager.add_circuit(
    name="my_custom_circuit",
    plasmids=custom_plasmids,
    default_parameters={"k_Gene1_concentration": 1.5}
)
```

## Running Simulations

### Basic Simulation

```python
# Create a circuit instance
circuit = manager.create_circuit("star")

# Run a simulation with default time span
result = circuit.simulate()

# Run simulation with custom time span and visualization
import numpy as np
t_span = np.linspace(0, 50, 5001)  # 50 time units, 5001 steps
result = circuit.simulate(t_span=t_span, plot=True)
```

### Advanced Simulation Options

```python
# Simulate with rules and ODEs printing
result = circuit.simulate(
    plot=True,              # Show plot
    print_rules=True,       # Print PySB rules
    print_odes=True         # Print ODEs
)

# Generate LaTeX equations
latex_eqs = circuit.print_latex_equations()
```

## Pulse Configuration

The system allows pulsed inputs for plasmids, which is useful for modeling input signals or perturbations.

### Creating a Pulsed Circuit

```python
# Define pulse configuration
pulse_config = {
    'use_pulse': True,           # Enable pulse
    'pulse_start': 5,            # Start time
    'pulse_end': 15,             # End time
    'pulse_concentration': 5.0,  # High concentration
    'base_concentration': 0.0    # Low/base concentration
}

# Create circuit with pulse on a specific plasmid by name
toehold_circuit = manager.create_circuit(
    "toehold_trigger",
    parameters={"k_Toehold3_GFP_concentration": 5},
    use_pulses=True,
    pulse_config=pulse_config,
    pulse_plasmids=["trigger3_plasmid"]  # Pulse the plasmid by name
)

# Simulate
result = toehold_circuit.simulate(plot=True)
```

### How Pulsing Works

1. When a circuit is configured with `use_pulses=True`:
    
    - The system automatically removes concentration parameters for pulsed plasmids
    - A time-dependent expression is created to represent the concentration
    - The pulse follows a square wave pattern defined by the pulse configuration
2. The `pulse_plasmids` parameter indicates which plasmids should be pulsed by name. For example:
    
    - `["gfp_plasmid"]`: Pulse the GFP plasmid
    - `["star6_plasmid", "trigger3_plasmid"]`: Pulse both Star6 and Trigger3 plasmids

## Visualization and Analysis

### Default Visualization

When `plot=True` is passed to the `simulate()` method, the system generates a plot showing:

For constant circuits:

- Protein concentrations over time
- RNA concentrations over time

For pulsed circuits:

- Protein concentrations over time
- RNA concentrations over time
- Pulse profile over time

### Custom Visualization

You can create custom visualizations using the simulation results:

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a circuit and simulate
circuit = manager.create_circuit("star")
t_span = np.linspace(0, 30, 3001)
result = circuit.simulate(t_span=t_span, plot=False)

# Create custom visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_span, result.observables['obs_Protein_GFP'], label='GFP')
ax.set_xlabel('Time')
ax.set_ylabel('Concentration')
ax.set_title('GFP Expression')
ax.legend()
ax.grid(True)
plt.show()
```

### Parameter Sampling and Visualization

For parameter sampling and visualization with pulses, use the `ParameterSamplingManager`:

```python
from circuits.circuit_generation.parameter_sampling_and_simulation import ParameterSamplingManager

# Create a manager
sampling_manager = ParameterSamplingManager(circuit_manager)

# Define parameter values to sweep
param_df = {"k_prot_deg": [0.05, 0.1, 0.2, 0.3]}

# Define pulse configuration
pulse_configuration = {
    'use_pulse': True,
    'pulse_start': 5,
    'pulse_end': 15,
    'pulse_concentration': 5.0,
    'base_concentration': 0.0
}

# Run parameter sweep with visualization
sampling_manager.plot_parameter_sweep_with_pulse(
    circuit_name="toehold_trigger",
    param_df=param_df,
    k_prot_deg=0.1,
    pulse_configuration=pulse_configuration,  # Note the parameter name
    pulse_plasmids=["trigger3_plasmid"],  # Pulse by name
    show_protein=True,
    show_rna=True,
    show_pulse=True
)
```

### Comparing Multiple Simulations

```python
# Compare different parameter values
fig, ax = plt.subplots(figsize=(10, 6))

for conc in [0.5, 1.0, 2.0, 5.0]:
    circuit = manager.create_circuit(
        "star", 
        parameters={"k_Star6_concentration": conc}
    )
    result = circuit.simulate(t_span=t_span, plot=False)
    ax.plot(t_span, result.observables['obs_Protein_GFP'], 
            label=f'Star6 conc = {conc}')

ax.set_xlabel('Time')
ax.set_ylabel('GFP Concentration')
ax.set_title('Effect of Star6 Concentration')
ax.legend()
plt.show()
```

## Named Plasmids Feature

The named plasmids feature replaces the previous index-based approach for referencing plasmids in pulse configurations.

### Overview

Previously, the system used numerical indices to identify which plasmids to pulse in a circuit. This approach was error-prone and not user-friendly, especially when working with complex circuits containing multiple plasmids.

The new approach introduces explicit names for plasmids, making it easier to:
- Reference specific plasmids in pulse configurations
- Understand circuit designs
- Maintain and modify circuits

### Changes Made

1. **Extended Plasmid Structure**: Plasmids now include a name field
   - Old format: `(tx_control, tl_control, cds)`
   - New format: `(name, tx_control, tl_control, cds)`

2. **Updated JSON Storage**: Plasmids in JSON files now include a name field
   - Old format: `[tx_control, tl_control, cds]`
   - New format: `{"name": "plasmid_name", "tx_control": [...], "tl_control": [...], "cds": [...]}`

3. **New Pulse Configuration**: Circuit creation now accepts `pulse_plasmids` (a list of plasmid names)
   - Old approach: `pulse_indices=[0, 1]`
   - New approach: `pulse_plasmids=["star6_plasmid", "trigger3_plasmid"]`

4. **Parameter Sampling**: The `plot_parameter_sweep_with_pulse` function now uses `pulse_configuration` instead of `pulse_config` for consistency

5. **Backward Compatibility**: The system maintains compatibility with existing code
   - `pulse_indices` is still supported
   - Legacy plasmid formats are automatically upgraded with generated names

### Creating Circuits with Named Plasmids

```python
from circuit_manager import CircuitManager
from registering_all_circuits_in_json import register_all_circuits

# Initialize manager
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Create a circuit with pulsed plasmids (by name)
toehold_circuit = manager.create_circuit(
    "toehold_trigger",
    use_pulses=True,
    pulse_config={
        'use_pulse': True,
        'pulse_start': 5,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    },
    pulse_plasmids=["trigger3_plasmid"]  # Reference by name instead of index
)

# Simulate
result, t_span = toehold_circuit.simulate(plot=True)
```

### Parameter Sweep with Named Plasmids

```python
import numpy as np
from circuits.circuit_generation.parameter_sampling_and_simulation import ParameterSamplingManager

# Setup
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)
sampling_manager = ParameterSamplingManager(manager)

# Define parameter values to sweep
param_df = {
    'k_prot_deg': np.linspace(0.05, 0.5, 5)  # Vary protein degradation rate
}

# Define pulse configuration
pulse_configuration = {
    'use_pulse': True,
    'pulse_start': 5,
    'pulse_end': 15,
    'pulse_concentration': 5.0,
    'base_concentration': 0.0
}

# Run parameter sweep with visualization
sampling_manager.plot_parameter_sweep_with_pulse(
    circuit_name="toehold_trigger",
    param_df=param_df,
    k_prot_deg=0.1,
    pulse_configuration=pulse_configuration,  # Note the parameter name
    pulse_plasmids=["trigger3_plasmid"],  # Pulse by name
    show_protein=True,
    show_rna=True,
    show_pulse=True
)
```

## Extending the System

### Adding New Circuit Types

To add a new circuit type, create a registration function in `register_circuits.py`:

```python
def register_my_circuit(manager):
    """Register my new circuit"""
    plasmids = [
        # Define plasmid structure with names
        ("geneA_plasmid", None, None, [(True, "GeneA")]),
        ("geneB_plasmid", None, None, [(False, "GeneB")])
    ]
    
    manager.add_circuit(
        name="my_circuit",
        plasmids=plasmids,
        default_parameters={
            "k_GeneA_concentration": 1.0,
            "k_GeneB_concentration": 1.5
        }
    )
```

Then add your function to `register_all_circuits()`.

### Customizing the Circuit Class

If you need to extend the Circuit class with custom functionality:

```python
# Inherit from Circuit class
class MyCustomCircuit(Circuit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_property = kwargs.get('custom_property')
    
    def custom_method(self):
        # Custom functionality
        pass
```

## Example Workflows

### Basic Workflow

```python
# Initialize
from circuit_manager import CircuitManager
from registering_all_circuits_in_json import register_all_circuits

manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Create and simulate a circuit
circuit = manager.create_circuit("gfp")
result = circuit.simulate(plot=True)
```

### Advanced Workflow: Pulse Response Analysis

```python
# Setup
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Define different pulse durations
pulse_durations = [5, 10, 15, 20]
peak_responses = []

for duration in pulse_durations:
    pulse_config = {
        'use_pulse': True,
        'pulse_start': 5,
        'pulse_end': 5 + duration,  # Variable duration
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    }
    
    circuit = manager.create_circuit(
        "toehold_trigger",
        use_pulses=True,
        pulse_config=pulse_config,
        pulse_plasmids=["trigger3_plasmid"]  # Pulse by name
    )
    
    result = circuit.simulate(t_span=np.linspace(0, 50, 5001), plot=False)
    peak_gfp = np.max(result.observables['obs_Protein_GFP'])
    peak_responses.append(peak_gfp)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(pulse_durations, peak_responses, 'o-', linewidth=2)
plt.xlabel('Pulse Duration')
plt.ylabel('Peak GFP Response')
plt.title('Effect of Pulse Duration on GFP Expression')
plt.grid(True)
plt.show()
```

## Troubleshooting

### Common Issues

1. **File not found errors**:
    
    - Make sure the parameters file path is correct
    - Try using an absolute path if relative paths don't work

2. **Parameter errors**:
    
    - When using pulses, the system automatically removes concentration parameters for pulsed plasmids
    - Make sure not to provide concentration parameters for pulsed plasmids in your parameters dictionary

3. **Simulation errors**:
    
    - Check if all required parameters are defined
    - Ensure plasmid structure matches the expected format
    - Verify that plasmid names exist if using `pulse_plasmids`

4. **Visualization issues**:
    
    - Make sure matplotlib is properly installed
    - Check if the simulation result contains the expected observables

5. **Parameter sampling issues**:
    
    - Ensure the parameter name `pulse_configuration` is used correctly in `plot_parameter_sweep_with_pulse`
    - Verify plasmid names match those in the circuit definition

### Debugging Tips

1. **Print model rules and ODEs**:
    
    ```python
    circuit.simulate(print_rules=True, print_odes=True)
    ```
    
2. **Examine the circuit model**:
    
    ```python
    # Print model parameters
    for param in circuit.model.parameters:
        print(param)
    
    # Print observables
    for obs in circuit.model.observables:
        print(obs)
    ```
    
3. **Debug pulse removal**:
    
    ```python
    params_to_remove = circuit._get_parameters_to_remove()
    print(f"Parameters that will be removed: {params_to_remove}")
    ```
    
4. **View available plasmid names**:
    
    ```python
    circuit = manager.create_circuit("toehold_trigger")
    plasmid_names = [plasmid[0] for plasmid in circuit.plasmids]
    print(f"Available plasmid names: {plasmid_names}")
    ```
    
5. **Save circuit configurations to examine**:
    
    ```python
    import json
    
    circuits = manager.load_circuits()
    with open('debug_circuits.json', 'w') as f:
        json.dump(circuits, f, indent=2)
    ```

### Backward Compatibility

For backward compatibility, the system still supports the `pulse_indices` parameter. When both `pulse_indices` and `pulse_plasmids` are provided, both sets of plasmids will be pulsed.