# Current Code Structure: RNA-based Feedforward Loops Modeling Framework

## Overview

The current codebase provides a framework for fitting parameters of RNA-based feedforward loop models to experimental data. It uses a Bayesian approach with Markov Chain Monte Carlo (MCMC) sampling to estimate posterior distributions of model parameters.

## File Structure

```
.
├── likelihood_functions/
│   ├── base.py              # Core fitting and MCMC classes
│   ├── config.py            # Circuit configuration classes
│   ├── likelihood.py        # Likelihood calculation functions
│   ├── utils.py             # Utility functions for data preparation
│   ├── visualization.py     # Plotting and visualization functions
│   └── mcmc_analysis.py     # Analysis of MCMC results
├── utils/
│   ├── import_and_visualise_data.py  # Data loading utilities
│   ├── GFP_calibration.py            # Fluorescence calibration
│   └── calibration_gfp/              # Calibration data
├── circuits/
│   └── circuit_generation/
│       └── circuit_manager.py        # Circuit creation and management
├── data/
│   ├── prior/
│   │   └── model_parameters_priors.csv  # Prior distributions
│   └── circuits/
│       ├── circuit_configs.py           # Circuit configuration data
│       └── circuits.json                # Circuit definitions
└── multi_circuit_parameter_estimation.py  # Main script for parameter estimation
```

## Core Classes and Functions

### CircuitFitter (`base.py`)

The central class for parameter estimation, handling simulation and likelihood calculation.

```python
class CircuitFitter:
    def __init__(self, configs, parameters_to_fit, model_parameters_priors, calibration_data):
        # Initialize with circuit configurations, parameters to fit, priors, and calibration data
        
    def _cache_experimental_data(self):
        # Pre-calculate and cache experimental data means and variances
        
    def _validate_configs(self):
        # Validate circuit configurations
        
    def _setup_simulators(self):
        # Setup simulators for each circuit
        
    def _setup_priors(self):
        # Setup prior parameters in log space
        
    def log_to_linear_params(self, log_params, param_names):
        # Convert parameters from log space to linear space
        
    def linear_to_log_params(self, linear_params):
        # Convert parameters from linear space to log space
        
    def generate_test_parameters(self, n_sets=20):
        # Generate test parameters based on Gaussian priors
        
    def simulate_parameters(self, log_params):
        # Run simulations for given parameter sets across all conditions
        
    def calculate_likelihood_from_simulation(self, simulation_data):
        # Calculate likelihood using cached experimental data
        
    def calculate_log_likelihood(self, log_params):
        # Calculate log likelihood for parameters
        
    def calculate_log_prior(self, log_params):
        # Calculate log prior probability
        
    def calculate_log_posterior(self, log_params):
        # Calculate log posterior (prior + likelihood)
```

### MCMCAdapter (`base.py`)

Adapts CircuitFitter for use with MCMC sampling.

```python
class MCMCAdapter:
    def __init__(self, circuit_fitter):
        # Initialize with circuit fitter
        
    def get_initial_parameters(self):
        # Get initial parameters from prior means
        
    def get_log_likelihood_function(self):
        # Return function for likelihood calculation
        
    def get_log_prior_function(self):
        # Return function for prior calculation
        
    def setup_parallel_tempering(self, n_walkers=1, n_chains=10):
        # Configure parallel tempering for sampling
```

### CircuitConfig (`config.py`)

Configuration class for individual circuits.

```python
@dataclass
class CircuitConfig:
    model: Any                                # PySB model
    name: str                                 # Circuit name
    condition_params: Dict[str, Dict[str, float]]  # Parameters for each condition
    experimental_data: pd.DataFrame           # Experimental measurements
    tspan: np.ndarray                         # Time points for simulation
    max_time: Optional[float] = None          # Maximum time for analysis
    min_time: Optional[float] = None          # Minimum time for analysis
    calibration_params: Optional[Dict] = None  # Calibration parameters
    
    def __post_init__(self):
        # Process data after initialization
    
    @classmethod
    def from_data(cls, model, name, condition_params, data_path, 
                 calibration_params, max_time=None, min_time=None):
        # Create config from data file
```

### Likelihood Functions (`likelihood.py`)

Functions for calculating likelihoods.

```python
def compute_condition_likelihood(simulation_results, experimental_data, 
                                tspan, combined_params, condition):
    # Compute likelihood for a specific condition
    
def calculate_likelihoods(sim_values, exp_means, exp_vars):
    # Calculate log likelihoods from simulation values and experimental data
```

### Main Workflow (`multi_circuit_parameter_estimation.py`)

Functions implementing the main parameter estimation workflow.

```python
def setup_calibration():
    # Setup GFP calibration parameters
    
def create_circuit_configs(circuit_manager, circuit_names, min_time=30, max_time=210):
    # Create circuit configurations for multiple circuits
    
def fit_multiple_circuits(circuit_configs, parameters_to_fit, priors, 
                         calibration_params, n_samples=2000, n_walkers=10, 
                         n_chains=6, n_sets=60):
    # Fit multiple circuits simultaneously with shared parameters
    
def main():
    # Main entry point
```

## Current Parameter Handling

The framework currently handles parameter estimation as follows:

1. **Prior Definition**: Gaussian priors are defined for each parameter in log space, with means and standard deviations specified in a CSV file.
    
2. **Parameter Generation**: Test parameters are generated from independent Gaussian distributions.
    
3. **Simulation**: Each parameter set is used to simulate all circuits and conditions.
    
4. **Likelihood Calculation**: Likelihood is calculated by comparing simulation results to experimental data.
    
5. **MCMC Sampling**: Parallel tempering samples the posterior distribution.
    
6. **Parameter Sharing**: Currently, the same parameters are shared across all circuits without a hierarchical structure - there is a single Gaussian prior for each parameter.
    
7. **Noise Model**: Current noise model is not explicitly heteroscedastic.
    

This structure provides a foundation for the hierarchical Bayesian model implementation outlined in the next document.
