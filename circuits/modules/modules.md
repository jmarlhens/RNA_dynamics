
## Overview

This system models synthetic gene circuits in E. coli using PySB (Python Systems Biology) to create rule-based models for RNA-based regulation. The architecture supports two primary regulation mechanisms:

1. **Transcriptional Regulation (STAR/Sense)** - Controls RNA synthesis
2. **Translational Regulation (Trigger/Toehold)** - Controls protein synthesis

The system enables both constant and pulsed expression patterns and supports different kinetic modeling approaches (Michaelis-Menten and Mass Action).

## System Architecture

The system is organized into several key modules that work together:

```
┌───────────────┐
│  build_model  │ ← Circuit configuration (JSON)
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────┐
│      Regulation Mechanisms            │
├─────────────────┬─────────────────────┤
│     STAR        │       Toehold       │
│  (Transcription)│    (Translation)    │
└────────┬────────┴──────────┬──────────┘
         │                   │
         ▼                   ▼
┌────────────────────────────────────────┐
│           Base Modules                 │
├─────────────────┬──────────────────────┤
│  Transcription  │     Translation      │
└────────┬────────┴──────────┬───────────┘
         │                   │
         ▼                   ▼
┌────────────────────────────────────────┐
│           Molecules                    │
├─────────────────┬──────────────────────┤
│       RNA       │      Protein         │
└─────────────────┴──────────────────────┘
```

## Core Modules

### 1. Molecules (`molecules.py`)

Defines the basic molecular components used in the system.

#### `MyMonomer` Class

- Base class for biological entities
- Provides name generation and instance retrieval functionalities
- Implements a prefix naming system

#### `RNA` Class

- Represents RNA molecules in the circuit
- Properties:
    - Binding sites: `b`, `sense`, `toehold`, `state`
    - States: `full`, `partial`, `init`
- Automatically adds degradation rules

#### `Protein` Class

- Represents protein molecules in the circuit
- Properties:
    - State: `mature` or `immature`
- Automatically adds maturation and degradation rules

### 2. Reaction Complex (`reactioncomplex.py`)

Provides a base class for reaction mechanisms.

#### `ReactionComplex` Class

- Represents reactions between substrate and product
- Stores model and parameter references
- Used as a base class for more specific reactions

### 3. Base Modules (`base_modules.py`)

Implements core transcription and translation processes with different kinetic models.

#### Enumerations

- `KineticsType`: `MICHAELIS_MENTEN` or `MASS_ACTION`
- `TranscriptionType`: `CONSTANT` or `PULSED`

#### Transcription Classes

- `Transcription`: Basic constant rate transcription using Michaelis-Menten kinetics
- `PulsedTranscription`: Time-dependent transcription rates
- `MassActionTranscription`: Explicit modeling of polymerase binding

#### Translation Classes

- `Translation`: Basic translation using Michaelis-Menten kinetics
- `MassActionTranslation`: Explicit modeling of ribosome binding

#### Factory Classes

- `TranscriptionFactory`: Creates appropriate transcription instance based on type and kinetics
- `TranslationFactory`: Creates appropriate translation instance based on kinetics

### 4. STAR Regulation (`star.py`)

Implements the STAR (Small Transcription Activating RNA) mechanism for transcriptional regulation.

#### `STAR` Class

- Models RNA-RNA interactions that regulate transcription
- Supports two kinetic modeling approaches:
    - `_setup_michaelis_menten`: Simplified model with effective rates
    - `_setup_mass_action`: Detailed model with explicit binding steps
- Key Parameters:
    - `k_star_bind`: STAR-sense RNA binding rate
    - `k_star_unbind`: STAR-sense RNA unbinding rate
    - `k_star_act`: Activation rate without regulator
    - `k_star_act_reg`: Activation rate with regulator
    - `k_star_stop`: Termination rate without regulator
    - `k_star_stop_reg`: Termination rate with regulator

### 5. Toehold Regulation (`toehold.py`)

Implements the Toehold Switch mechanism for translational regulation.

#### `Toehold` Class

- Models RNA-RNA interactions that regulate translation
- Supports two kinetic modeling approaches:
    - `_setup_michaelis_menten`: Simplified model with effective rates
    - `_setup_mass_action`: Detailed model with explicit ribosome binding steps
- Key Parameters:
    - `k_tl_unbound`: Translation rate when toehold is unbound
    - `k_tl_bound`: Translation rate when toehold is bound to trigger
    - `k_trigger_binding`: Trigger-toehold binding rate
    - `k_trigger_unbinding`: Trigger-toehold unbinding rate

### 6. Model Building (`build_model.py`)

Orchestrates the construction of complete circuit models.

#### `setup_model` Function

- Main entry point for model creation
- Processes plasmid configurations
- Adds parameters and observables
- Supports pulsed expression

#### `process_plasmid` Function

- Processes individual plasmid configurations
- Sets up transcription, Csy4 cleavage, and translation
- Connects regulation mechanisms

#### `generate_observables` Function

- Creates observables for model components

## Kinetic Models

The system supports two approaches to modeling kinetics:

### Michaelis-Menten Kinetics

- Simplified representation
- Uses effective rates
- More computationally efficient
- Example: `v = k_cat * [E]_total * [S] / (K_m + [S])`

### Mass Action Kinetics

- More detailed, mechanistic representation
- Explicitly models binding and catalytic steps
- More parameters required
- Example: `E + S ⟷ ES → E + P`

## Regulation Mechanisms

### STAR (Small Transcription Activating RNA) Mechanism

STAR regulates at the transcriptional level by controlling RNA synthesis:

1. Target gene has a sense region that forms a terminator hairpin
2. Terminator prevents RNA polymerase from completing transcription
3. STAR RNA binds to sense region, disrupting terminator formation
4. RNA polymerase completes transcription, producing functional RNA

```
Without STAR:
DNA → RNA (partial) → Rapid degradation

With STAR:
STAR RNA + DNA → Complete transcription → Functional RNA
```

### Toehold Switch Mechanism

Toehold regulates at the translational level by controlling protein synthesis:

1. Target RNA forms a secondary structure that blocks ribosome binding site (RBS)
2. Trigger RNA binds to toehold region on the target RNA
3. This binding causes structural changes that expose the RBS
4. Ribosomes can now bind and translate the RNA into protein

```
Without Trigger:
RNA (RBS blocked) → Low/no translation

With Trigger:
Trigger RNA + RNA → RNA (RBS exposed) → Efficient translation → Protein
```

## Circuit Configuration

Circuits are configured using a JSON-like structure:

```python
{
  "circuit_name": {
    "plasmids": [
      [
        transcriptional_control,  # [type, regulator] or null
        translational_control,    # [type, regulator] or null
        [
          [translate, gene_name], # Boolean, string
          ...
        ]
      ],
      ...
    ],
    "default_parameters": {
      "param_name": value,
      ...
    },
    "bindings": [
      [species1, species2],
      ...
    ]
  }
}
```

### Example: Toehold-Trigger Circuit

```python
"toehold_trigger": {  
  "plasmids": [  
    [  
      null,  
      [  
        "Toehold3",  
        "Trigger3"  
      ],  
      [  
        [  
          true,  
          "GFP"  
        ]  
      ]  
    ],  
    [  
      null,  
      null,  
      [  
        [  
          false,  
          "Trigger3"  
        ]  
      ]  
    ]  
  ],  
  "default_parameters": {  
    "k_Toehold3_GFP_concentration": 1,  
    "k_Trigger3_concentration": 1  
  },  
  "bindings": []  
}
```

This circuit:

1. First plasmid: Produces RNA with Toehold3 regulation that can be translated to GFP when Trigger3 is present
2. Second plasmid: Produces Trigger3 RNA (not translated)

## Common Parameters

|Parameter|Description|
|---|---|
|k_tx|Transcription rate constant|
|k_tl|Translation rate constant|
|k_mat|Protein maturation rate|
|k_rna_deg|RNA degradation rate|
|k_prot_deg|Protein degradation rate|
|k_X_concentration|Concentration of gene X|
|k_trigger_binding|Binding rate for toehold-trigger|
|k_trigger_unbinding|Unbinding rate for toehold-trigger|
|k_star_bind|Binding rate for STAR-sense|
|k_star_unbind|Unbinding rate for STAR-sense|

## Additional Resources

- [PySB Documentation](https://pysb.org/)
