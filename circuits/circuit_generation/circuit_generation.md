
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
(transcriptional_control, translational_control, cds)
```


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

The JSON storage only contains the circuit structure and default parameters, while simulation configurations (like pulse behavior) are specified at runtime.

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

# Circuit with pulse configuration
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
    pulse_indices=[0]  # Pulse the first plasmid
)
```

### Registering a New Circuit

```python
# Define plasmids for a new circuit
custom_plasmids = [
    (("STAR1", "SENSE1"), None, [(True, "GFP")]),
    (None, None, [(False, "Gene2")])
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

# Create circuit with pulse on the second plasmid
toehold_circuit = manager.create_circuit(
    "toehold",
    parameters={"k_Toehold3_GFP_concentration": 5},
    use_pulses=True,
    pulse_config=pulse_config,
    pulse_indices=[1]  # Pulse the second plasmid
)

# Simulate
result = toehold_circuit.simulate(plot=True)
```

### How Pulsing Works

1. When a circuit is configured with `use_pulses=True`:
    
    - The system automatically removes concentration parameters for pulsed plasmids
    - A time-dependent expression is created to represent the concentration
    - The pulse follows a square wave pattern defined by the pulse configuration
2. The `pulse_indices` parameter indicates which plasmids should be pulsed. For example:
    
    - `[0]`: Pulse the first plasmid
    - `[1,2]`: Pulse the second and third plasmids

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

## Extending the System

### Adding New Circuit Types

To add a new circuit type, create a registration function in `register_circuits.py`:

```python
def register_my_circuit(manager):
    """Register my new circuit"""
    plasmids = [
        # Define plasmid structure
        (None, None, [(True, "GeneA")]),
        (None, None, [(False, "GeneB")])
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
from register_circuits import register_all_circuits

manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Create and simulate a circuit
circuit = manager.create_circuit("gfp")
result = circuit.simulate(plot=True)
```

### Advanced Workflow: Parameter Sweep

```python
import numpy as np
import matplotlib.pyplot as plt

# Setup
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Define parameter values to sweep
star_concentrations = np.linspace(0.1, 5.0, 10)
max_gfp_values = []

# Perform parameter sweep
for conc in star_concentrations:
    circuit = manager.create_circuit(
        "star", 
        parameters={"k_Star6_concentration": conc}
    )
    result = circuit.simulate(plot=False)
    max_gfp = np.max(result.observables['obs_Protein_GFP'])
    max_gfp_values.append(max_gfp)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(star_concentrations, max_gfp_values, 'o-')
plt.xlabel('Star6 Concentration')
plt.ylabel('Max GFP Expression')
plt.title('Parameter Sweep: Effect of Star6 on GFP Expression')
plt.grid(True)
plt.show()
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
        "toehold",
        use_pulses=True,
        pulse_config=pulse_config,
        pulse_indices=[1]
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
4. **Visualization issues**:
    
    - Make sure matplotlib is properly installed
    - Check if the simulation result contains the expected observables

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
    
4. **Save circuit configurations to examine**:
    
    ```python
    import json
    
    circuits = manager.load_circuits()
    with open('debug_circuits.json', 'w') as f:
        json.dump(circuits, f, indent=2)
    ```


# Named Plasmids Feature Documentation

This document explains the new named plasmids feature for the circuit simulation system, which replaces the previous index-based approach for referencing plasmids in pulse configurations.

## Overview

Previously, the system used numerical indices to identify which plasmids to pulse in a circuit. This approach was error-prone and not user-friendly, especially when working with complex circuits containing multiple plasmids.

The new approach introduces explicit names for plasmids, making it easier to:
- Reference specific plasmids in pulse configurations
- Understand circuit designs
- Maintain and modify circuits

## Changes Made

1. **Extended Plasmid Structure**: Plasmids now include a name field
   - Old format: `(tx_control, tl_control, cds)`
   - New format: `(name, tx_control, tl_control, cds)`

2. **Updated JSON Storage**: Plasmids in JSON files now include a name field
   - Old format: `[tx_control, tl_control, cds]`
   - New format: `{"name": "plasmid_name", "tx_control": [...], "tl_control": [...], "cds": [...]}`

3. **New Pulse Configuration**: Circuit creation now accepts `pulse_plasmids` (a list of plasmid names)
   - Old approach: `pulse_indices=[0, 1]`
   - New approach: `pulse_plasmids=["star6_plasmid", "trigger3_plasmid"]`

4. **Backward Compatibility**: The system maintains compatibility with existing code
   - `pulse_indices` is still supported
   - Legacy plasmid formats are automatically upgraded with generated names

## Using Named Plasmids

### Creating Circuits with Named Plasmids

```python
from circuit_manager import CircuitManager
from registering_all_circuits_in_json import register_all_circuits

# Initialize manager
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Create a circuit with pulsed plasmids (by name)
toehold_circuit = manager.create_circuit(
    "toehold",
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

### Registering New Circuits with Named Plasmids

```python
# Define plasmids with names
custom_plasmids = [
    ("gfp_expression", None, None, [(True, "GFP")]),
    ("star6_expression", None, None, [(False, "Star6")])
]

# Add to database
manager.add_circuit(
    name="my_custom_circuit",
    plasmids=custom_plasmids,
    default_parameters={"k_GFP_concentration": 1.5}
)
```

### Accessing Plasmids by Name

```python
# Get a specific circuit
circuit = manager.create_circuit("star")

# Access a plasmid by name
gfp_plasmid = circuit.get_plasmid_by_name("sense_gfp_plasmid")
if gfp_plasmid:
    print(f"Found plasmid: {gfp_plasmid}")
```

## Migrating Existing Circuits

A migration script is provided to update existing JSON circuit files. This script will add names to all plasmids in the file.

### How to Use the Migration Script

```bash
python migrate_circuits.py path/to/circuits.json --output path/to/updated_circuits.json
```

To overwrite the existing file:

```bash
python migrate_circuits.py path/to/circuits.json
```

The migration script:
1. Loads the existing circuits JSON file
2. Adds descriptive names to each plasmid based on its contents
3. Saves the updated JSON file

## Example Workflows

### Basic Workflow with Named Plasmids

```python
# Initialize
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Create and simulate a circuit with named plasmid pulses
circuit = manager.create_circuit(
    "star_antistar_1", 
    use_pulses=True,
    pulse_config={
        'use_pulse': True,
        'pulse_start': 5,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    },
    pulse_plasmids=["star1_plasmid"]  # Pulse by name
)

result, t_span = circuit.simulate(plot=True)
```

### Parameter Sweep with Named Plasmids

```python
import numpy as np

# Setup
manager = CircuitManager(parameters_file="data/model_parameters_priors.csv")
register_all_circuits(manager)

# Define parameter values to sweep
param_values = {
    'k_prot_deg': np.linspace(0.05, 0.5, 5)  # Vary protein degradation rate
}

# Create circuit with named pulse plasmid
circuit = manager.create_circuit(
    "toehold",
    use_pulses=True,
    pulse_config={
        'use_pulse': True,
        'pulse_start': 5,
        'pulse_end': 15,
        'pulse_concentration': 5.0,
        'base_concentration': 0.0
    },
    pulse_plasmids=["trigger3_plasmid"]  # Use name instead of index
)

# Run parameter sweep
results, t_span = circuit.simulate(
    t_span=np.linspace(0, 50, 5001),
    param_values=param_values
)
```

## Troubleshooting

### Common Issues

1. **Plasmid not found error**:
   - Check that the plasmid name exactly matches what's in the circuit
   - Use `circuit.plasmid_name_to_index` to see available plasmid names

2. **Parameter errors with pulsed plasmids**:
   - Ensure the correct plasmid names are specified in `pulse_plasmids`
   - Remember that concentration parameters for pulsed plasmids are automatically removed

3. **Migration issues**:
   - If the migration script encounters errors, check the JSON file structure
   - Make sure the JSON has a `circuits` key at the top level
   - Make sure each circuit has a `plasmids` list with the expected format

### Backward Compatibility

For backward compatibility, the system still supports the `pulse_indices` parameter. When both `pulse_indices` and `pulse_plasmids` are provided, both sets of plasmids will be pulsed.

## Future Considerations

1. **Fully Deprecate Index-Based API**:
   - In a future version, the `pulse_indices` parameter may be deprecated
   - All code should transition to using named plasmids

2. **Enhanced Plasmid Class**:
   - A proper Plasmid class could be introduced for more flexibility
   - This would allow for additional plasmid metadata and methods

3. **Circuit Visualization**:
   - Named plasmids make it easier to create visual representations of circuits
   - A future enhancement could include circuit diagrams with labeled plasmids