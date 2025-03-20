import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from utils.GFP_calibration import fit_gfp_calibration, get_brightness_correction_factor
from typing import Dict, List, Optional, Tuple, Any


def setup_calibration():
    """Set up GFP calibration parameters"""
    data = pd.read_csv('../calibration_gfp/gfp_Calibration.csv')
    calibration_results = fit_gfp_calibration(
        data,
        concentration_col='GFP Concentration (nM)',
        fluorescence_pattern='F.I. (a.u)'
    )
    correction_factor, _ = get_brightness_correction_factor('avGFP', 'sfGFP')

    return {
        'slope': calibration_results['slope'],
        'intercept': calibration_results['intercept'],
        'brightness_correction': correction_factor
    }


def get_circuit_config(circuit_type: str) -> Dict[str, Any]:
    """Get circuit-specific configuration"""
    config = {}

    if circuit_type.lower() == 'toehold':
        from obsolete.toehold import test_toehold
        config['model_function'] = test_toehold
        config['input_plasmid_idx'] = 1  # Trigger3 (to be pulsed)
        config['input_name'] = 'Trigger3'
        config['input_concentration_param'] = 'k_Trigger3_concentration'
        config['rna_observable'] = 'RNA_Toehold3_GFP'
        config['protein_observable'] = 'obs_Protein_GFP'
    elif circuit_type.lower() == 'sense':
        from obsolete.star import test_star
        config['model_function'] = test_star
        config['input_plasmid_idx'] = 1  # Star6 (to be pulsed)
        config['input_name'] = 'Star6'
        config['input_concentration_param'] = 'k_Star6_concentration'
        config['rna_observable'] = 'RNA_Sense6_GFP'
        config['protein_observable'] = 'obs_Protein_GFP'
    elif circuit_type.lower() == 'cascade':
        from obsolete.cascade import test_cascade
        config['model_function'] = test_cascade
        config['input_plasmid_idx'] = 2  # Star6 (to be pulsed)
        config['input_name'] = 'Star6'
        config['input_concentration_param'] = 'k_Star6_concentration'
        config['rna_observable'] = 'RNA_Toehold3_GFP'
        config['protein_observable'] = 'obs_Protein_GFP'
    elif circuit_type.lower() == 'cffl':
        from obsolete.cffl_type_1 import test_coherent_feed_forward_loop
        config['model_function'] = test_coherent_feed_forward_loop
        config['input_plasmid_idx'] = 1  # Star6 (to be pulsed)
        config['input_name'] = 'Star6'
        config['input_concentration_param'] = 'k_Star6_concentration'
        config['rna_observable'] = 'RNA_Sense6_Toehold3_GFP'
        config['protein_observable'] = 'obs_Protein_GFP'
    else:
        raise ValueError(f"Unsupported circuit type: {circuit_type}")

    return config


def setup_model_with_pulse(model_function, parameters, pulse_config, pulse_indices):
    """Helper function to setup model with pulse for specific plasmids"""
    from circuits.build_model import setup_model

    # First, get the proper plasmid design based on the model function name
    plasmids = []
    if 'toehold' in model_function.__name__:
        plasmids = [
            (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),
            (None, None, [(False, "Trigger3")]),
        ]
    elif 'star' in model_function.__name__:
        plasmids = [
            (None, ("Sense6", "Star6"), [(True, "GFP")]),
            (None, None, [(False, "Star6")]),
        ]
    elif 'cascade' in model_function.__name__:
        plasmids = [
            (None, ("Toehold3", "Trigger3"), [(True, "GFP")]),
            (None, ("Sense6", "Star6"), [(False, "Trigger3")]),
            (None, None, [(False, "Star6")]),
        ]
    elif 'feed_forward' in model_function.__name__ or 'cffl' in model_function.__name__:
        plasmids = [
            (None, ("Sense6", "Star6"), [(True, "Toehold3"), (True, "GFP")]),
            (None, None, [(False, "Star6")]),
            (None, ("Sense6", "Star6"), [(False, "Trigger3")]),
        ]

    # Create model with pulse setup
    return setup_model(
        plasmids,
        parameters,
        use_pulses=True,
        pulse_config=pulse_config,
        pulse_indices=pulse_indices
    )


def simulate_multiple_parameter_sets(
        circuit_config: Dict[str, Any],
        parameter_sets: List[Dict],
        time_span: np.ndarray,
        pulse_config: Dict,
) -> Dict[str, np.ndarray]:
    """
    Simulate multiple parameter sets efficiently (similar to likelihood function approach)

    Parameters:
    -----------
    circuit_config : Dict
        Circuit configuration with model_function, observables, etc.
    parameter_sets : List[Dict]
        List of parameter dictionaries to simulate
    time_span : np.ndarray
        Time points for simulation
    pulse_config : Dict
        Configuration for input pulse

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with RNA and protein trajectories for all parameter sets
    """
    from pysb.simulator import ScipyOdeSimulator
    import pandas as pd

    # Load default parameters from CSV if needed
    try:
        parameters_df = pd.read_csv('../data/model_parameters_priors.csv')
        default_params = dict(zip(parameters_df['Parameter'], parameters_df['Value']))
    except:
        default_params = {}
        print("Warning: Could not load default parameters from CSV")

    # Setup model with base parameters (we'll override with param_values during simulation)
    # First make sure all parameters are positive and in linear scale
    base_params = parameter_sets[0].copy()
    for k, v in base_params.items():
        if v <= 0:
            # If we have a non-positive value, replace with a default
            if k in default_params:
                base_params[k] = default_params[k]
            else:
                base_params[k] = 1.0  # Fallback default

    # Create the model
    model = setup_model_with_pulse(
        circuit_config['model_function'],
        base_params,
        pulse_config,
        [circuit_config['input_plasmid_idx']]
    )

    # Create simulator
    simulator = ScipyOdeSimulator(
        model,
        time_span,
        compiler='cython',
        cleanup=True
    )

    # Get the list of parameters actually in the model
    model_param_names = [p.name for p in model.parameters]

    # Prepare parameter values for all simulations as a DataFrame
    param_dicts = []
    for i, params in enumerate(parameter_sets):
        # Filter to include only parameters that exist in the model
        filtered_params = {k: v for k, v in params.items() if k in model_param_names}

        # Ensure all values are positive
        for k, v in filtered_params.items():
            if v <= 0:
                if k in default_params:
                    filtered_params[k] = default_params[k]
                else:
                    filtered_params[k] = 1.0  # Fallback default

        param_dicts.append(filtered_params)

    # Convert to format expected by ScipyOdeSimulator
    param_df = pd.DataFrame(param_dicts)

    # Run simulations for all parameter sets at once
    results = simulator.run(param_values=param_df)

    # Extract trajectories
    rna_trajectories = np.array([
        results.observables[i][circuit_config['rna_observable']]
        for i in range(len(parameter_sets))
    ])

    protein_trajectories = np.array([
        results.observables[i][circuit_config['protein_observable']]
        for i in range(len(parameter_sets))
    ])

    return {
        'rna': rna_trajectories,
        'protein': protein_trajectories
    }


def get_input_signal(time_span: np.ndarray, pulse_config: Dict) -> np.ndarray:
    """Generate input signal array based on pulse configuration"""
    signal = np.ones_like(time_span) * pulse_config['base_concentration']

    pulse_mask = (time_span >= pulse_config['pulse_start']) & (time_span <= pulse_config['pulse_end'])
    signal[pulse_mask] = pulse_config['pulse_concentration']

    return signal


def prepare_parameter_sets(
        posterior_samples: pd.DataFrame,
        n_samples: int,
        k_prot_deg: float
) -> Tuple[List[Dict], np.ndarray]:
    """
    Prepare parameter sets from posterior samples for simulation

    Parameters:
    -----------
    posterior_samples : pd.DataFrame
        DataFrame with posterior parameter samples
    n_samples : int
        Number of parameter sets to use
    k_prot_deg : float
        Protein degradation rate

    Returns:
    --------
    Tuple[List[Dict], np.ndarray]
        Prepared parameter dictionaries and corresponding likelihood values
    """
    # Sort by likelihood and select top n_samples
    if len(posterior_samples) > n_samples:
        selected_samples = posterior_samples.sort_values('likelihood', ascending=False).head(n_samples)
    else:
        selected_samples = posterior_samples

    # Extract likelihood values
    likelihoods = selected_samples['likelihood'].values if 'likelihood' in selected_samples.columns else np.zeros(
        len(selected_samples))

    # Create parameter dictionaries from samples
    parameter_sets = []
    for _, row in selected_samples.iterrows():
        # Extract parameters and convert from log scale if needed
        params = {}

        # First, identify if we have log-transformed parameters
        has_log_params = any(col.startswith('logk_') for col in row.index)

        for col in row.index:
            # Skip non-parameter columns
            if not (col.startswith('k_') or col.startswith('logk_')) or col == 'likelihood':
                continue

            if has_log_params and col.startswith('logk_'):
                # Handle log-transformed parameters
                param_name = col[3:]  # Remove 'log' prefix
                params[param_name] = 10 ** row[col]  # Convert from log10 to linear
            elif not has_log_params and col.startswith('k_'):
                # If we don't have log params, assume the parameters are already in linear scale
                params[col] = row[col]

        # Ensure protein degradation rate is set
        params['k_prot_deg'] = k_prot_deg
        params['k_rna_deg'] = 0.6  # Set a default RNA degradation rate if not specified

        parameter_sets.append(params)

    return parameter_sets, likelihoods


def plot_step_response(
        circuit_type: str,
        posterior_samples: pd.DataFrame,
        output_path: Optional[str] = None,
        n_samples: int = 100,
        time_span: np.ndarray = None,
        k_prot_deg: float = 0.6,
        pulse_config: Dict = None,
        ll_quartile: int = 30,
        figsize: tuple = (10, 12)
):
    """
    Plot step response for a circuit using posterior samples.

    Parameters:
    -----------
    circuit_type : str
        Circuit type to analyze ('toehold', 'sense', 'cascade', or 'cffl')
    posterior_samples : pd.DataFrame
        DataFrame containing posterior parameter samples with 'likelihood' column
    output_path : str, optional
        Path to save the resulting plot, if None plot will be displayed
    n_samples : int
        Number of parameter samples to use from the posterior
    time_span : np.ndarray, optional
        Time span for simulation (defaults to 0-30 with 3001 steps)
    k_prot_deg : float
        Protein degradation rate to use for simulation
    pulse_config : Dict, optional
        Configuration for input pulse (defaults provided if None)
    ll_quartile : int
        Percentile cutoff for likelihood coloring (higher keeps more diversity)
    figsize : tuple
        Figure size (width, height) in inches

    Returns:
    --------
    plt.Figure
        The generated figure
    """
    # Default time span if not provided
    if time_span is None:
        time_span = np.linspace(0, 30, 3001)

    # Default pulse configuration if not provided
    if pulse_config is None:
        pulse_config = {
            'use_pulse': True,
            'pulse_start': 4,
            'pulse_end': 15,
            'pulse_concentration': 5.0,
            'base_concentration': 0.0
        }

    # Get circuit configuration
    circuit_config = get_circuit_config(circuit_type)

    # Prepare parameter sets from posterior samples
    parameter_sets, likelihoods = prepare_parameter_sets(
        posterior_samples,
        n_samples,
        k_prot_deg
    )

    # Normalize likelihoods for coloring
    norm_likelihoods = (likelihoods - np.min(likelihoods)) / (np.max(likelihoods) - np.min(likelihoods))

    # Simulate all parameter sets
    trajectories = simulate_multiple_parameter_sets(
        circuit_config,
        parameter_sets,
        time_span,
        pulse_config
    )

    # Calculate statistics for trajectories
    rna_median = np.median(trajectories['rna'], axis=0)
    rna_low = np.percentile(trajectories['rna'], 2.5, axis=0)
    rna_high = np.percentile(trajectories['rna'], 97.5, axis=0)

    protein_median = np.median(trajectories['protein'], axis=0)
    protein_low = np.percentile(trajectories['protein'], 2.5, axis=0)
    protein_high = np.percentile(trajectories['protein'], 97.5, axis=0)

    # Create figure with three rows
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Generate input signal based on pulse config
    input_signal = get_input_signal(time_span, pulse_config)

    # Plot input pulse
    axes[0].plot(time_span, input_signal, 'r-', linewidth=2)
    axes[0].axvspan(pulse_config['pulse_start'], pulse_config['pulse_end'],
                    color='lightgray', alpha=0.3, label='Pulse Duration')
    axes[0].set_ylabel(f'{circuit_config["input_name"]} (nM)')
    axes[0].set_title(f'{circuit_type.capitalize()} Circuit Step Response')
    axes[0].grid(True, alpha=0.3)

    # Create colormap for likelihood values
    cmap = plt.cm.viridis

    # Plot RNA trajectories
    for i, rna_traj in enumerate(trajectories['rna']):
        color = cmap(norm_likelihoods[i])
        axes[1].plot(time_span, rna_traj, color=color, alpha=0.4, linewidth=1)

    # Add RNA confidence interval and median
    axes[1].fill_between(time_span, rna_low, rna_high, color='gray', alpha=0.2)
    axes[1].plot(time_span, rna_median, 'k-', linewidth=2)
    axes[1].set_ylabel(f'{circuit_config["rna_observable"]} (nM)')
    axes[1].grid(True, alpha=0.3)

    # Plot Protein trajectories
    for i, protein_traj in enumerate(trajectories['protein']):
        color = cmap(norm_likelihoods[i])
        axes[2].plot(time_span, protein_traj, color=color, alpha=0.4, linewidth=1)

    # Add Protein confidence interval and median
    axes[2].fill_between(time_span, protein_low, protein_high, color='gray', alpha=0.2)
    axes[2].plot(time_span, protein_median, 'k-', linewidth=2)
    axes[2].set_ylabel('GFP Protein (nM)')
    axes[2].set_xlabel('Time (h)')
    axes[2].grid(True, alpha=0.3)

    # Add colorbar for likelihood
    norm = mcolors.Normalize(vmin=np.min(likelihoods), vmax=np.max(likelihoods))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, pad=0.01)
    cbar.set_label('Log Likelihood')

    # Adjust layout
    plt.tight_layout()

    # Save or show figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()

    return fig


def compare_circuit_step_responses(
        posterior_samples_dict: Dict[str, pd.DataFrame],
        output_path: Optional[str] = None,
        n_samples: int = 50,
        time_span: np.ndarray = None,
        k_prot_deg: float = 0.6,
        pulse_config: Dict = None,
        figsize: tuple = (12, 12)
):
    """
    Compare step responses of multiple circuits using posterior samples.

    Parameters:
    -----------
    posterior_samples_dict : Dict[str, pd.DataFrame]
        Dictionary mapping circuit types to DataFrames containing posterior samples
    output_path : str, optional
        Path to save the resulting plot, if None plot will be displayed
    n_samples : int
        Number of parameter samples to use per circuit
    time_span : np.ndarray, optional
        Time span for simulation (defaults to 0-30 with 3001 steps)
    k_prot_deg : float
        Protein degradation rate to use for simulation
    pulse_config : Dict, optional
        Configuration for input pulse (defaults provided if None)
    figsize : tuple
        Figure size (width, height) in inches

    Returns:
    --------
    plt.Figure
        The generated figure
    """
    # Default time span if not provided
    if time_span is None:
        time_span = np.linspace(0, 30, 3001)

    # Default pulse configuration if not provided
    if pulse_config is None:
        pulse_config = {
            'use_pulse': True,
            'pulse_start': 4,
            'pulse_end': 15,
            'pulse_concentration': 5.0,
            'base_concentration': 0.0
        }

    # Create figure with three rows
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Generate input signal based on pulse config
    input_signal = get_input_signal(time_span, pulse_config)

    # Plot input pulse
    axes[0].plot(time_span, input_signal, 'r-', linewidth=2)
    axes[0].axvspan(pulse_config['pulse_start'], pulse_config['pulse_end'],
                    color='lightgray', alpha=0.3, label='Pulse Duration')
    axes[0].set_ylabel('Input Signal (nM)')
    axes[0].set_title('Comparison of Circuit Step Responses')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Define colors for each circuit
    circuit_colors = {
        'toehold': 'blue',
        'sense': 'green',
        'cascade': 'purple',
        'cffl': 'orange'
    }

    # Store results for RNA and protein plots
    rna_results = {}
    protein_results = {}

    # For each circuit type
    for circuit_type, samples in posterior_samples_dict.items():
        print(f"Processing {circuit_type}...")

        # Get circuit configuration
        try:
            circuit_config = get_circuit_config(circuit_type)
        except ValueError:
            print(f"Skipping unsupported circuit type: {circuit_type}")
            continue

        # Prepare parameter sets from posterior samples
        parameter_sets, _ = prepare_parameter_sets(
            samples,
            n_samples,
            k_prot_deg
        )

        # Simulate all parameter sets
        trajectories = simulate_multiple_parameter_sets(
            circuit_config,
            parameter_sets,
            time_span,
            pulse_config
        )

        # Calculate statistics for trajectories
        rna_median = np.median(trajectories['rna'], axis=0)
        rna_low = np.percentile(trajectories['rna'], 2.5, axis=0)
        rna_high = np.percentile(trajectories['rna'], 97.5, axis=0)

        protein_median = np.median(trajectories['protein'], axis=0)
        protein_low = np.percentile(trajectories['protein'], 2.5, axis=0)
        protein_high = np.percentile(trajectories['protein'], 97.5, axis=0)

        # Store results
        rna_results[circuit_type] = {
            'median': rna_median,
            'low': rna_low,
            'high': rna_high
        }

        protein_results[circuit_type] = {
            'median': protein_median,
            'low': protein_low,
            'high': protein_high
        }

    # Plot RNA results for all circuits
    for circuit_type, results in rna_results.items():
        color = circuit_colors.get(circuit_type, 'gray')
        axes[1].plot(time_span, results['median'], color=color, linewidth=2, label=circuit_type.capitalize())
        axes[1].fill_between(time_span, results['low'], results['high'], color=color, alpha=0.2)

    axes[1].set_ylabel('RNA Output (nM)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot Protein results for all circuits
    for circuit_type, results in protein_results.items():
        color = circuit_colors.get(circuit_type, 'gray')
        axes[2].plot(time_span, results['median'], color=color, linewidth=2, label=circuit_type.capitalize())
        axes[2].fill_between(time_span, results['low'], results['high'], color=color, alpha=0.2)

    axes[2].set_ylabel('GFP Protein (nM)')
    axes[2].set_xlabel('Time (h)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Adjust layout
    plt.tight_layout()

    # Save or show figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")
    else:
        plt.show()

    return fig


def batch_process_step_responses(
        circuit_types: List[str],
        posterior_samples_paths: Dict[str, str],
        output_dir: str,
        n_samples: int = 50,
        k_prot_deg: float = 0.6,
        pulse_config: Dict = None
):
    """
    Batch process step responses for multiple circuits and save plots.

    Parameters:
    -----------
    circuit_types : List[str]
        List of circuit types to process ('toehold', 'sense', 'cascade', 'cffl')
    posterior_samples_paths : Dict[str, str]
        Mapping of circuit types to file paths of posterior samples
    output_dir : str
        Directory to save output plots
    n_samples : int
        Number of parameter samples to use per circuit
    k_prot_deg : float
        Protein degradation rate to use for simulation
    pulse_config : Dict, optional
        Configuration for input pulse (defaults provided if None)
    """
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process individual circuit step responses
    for circuit_type in circuit_types:
        if circuit_type not in posterior_samples_paths:
            print(f"Skipping {circuit_type} - no posterior samples path provided")
            continue

        # Load posterior samples
        samples_path = posterior_samples_paths[circuit_type]
        samples = pd.read_csv(samples_path)

        # Generate output path
        output_path = os.path.join(output_dir, f"{circuit_type}_step_response.png")

        # Plot step response
        plot_step_response(
            circuit_type,
            samples,
            output_path=output_path,
            n_samples=n_samples,
            k_prot_deg=k_prot_deg,
            pulse_config=pulse_config
        )

    # Load all posterior samples for comparison
    posterior_samples_dict = {}
    for circuit_type, path in posterior_samples_paths.items():
        posterior_samples_dict[circuit_type] = pd.read_csv(path)

    # Compare circuit step responses
    comparison_path = os.path.join(output_dir, "circuit_comparison.png")
    compare_circuit_step_responses(
        posterior_samples_dict,
        output_path=comparison_path,
        n_samples=n_samples,
        k_prot_deg=k_prot_deg,
        pulse_config=pulse_config
    )


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import os

    # Example 1: Plot step response for a single circuit
    toehold_samples = pd.read_csv("../data/fit_data/individual_circuits/results_toehold_trigger_20250224_133718.csv")

    plot_step_response(
        'toehold',
        toehold_samples,
        output_path="toehold_step_response.png",
        n_samples=50,
        k_prot_deg=0.4,
    )

    # Example 2: Compare multiple circuits
    circuit_samples = {
        'toehold': pd.read_csv("../data/fit_data/individual_circuits/results_toehold.csv"),
        'sense': pd.read_csv("../data/fit_data/individual_circuits/results_sense.csv"),
        'cascade': pd.read_csv("../data/fit_data/individual_circuits/results_cascade.csv"),
        'cffl': pd.read_csv("../data/fit_data/individual_circuits/results_cffl.csv")
    }

    compare_circuit_step_responses(
        circuit_samples,
        output_path="circuit_comparison.png",
        n_samples=30,
        k_prot_deg=0.4
    )

    # Example 3: Batch process all circuits
    batch_process_step_responses(
        ['toehold', 'sense', 'cascade', 'cffl'],
        {
            'toehold': "../fit_data/individual_circuits/results_toehold.csv",
            'sense': "../fit_data/individual_circuits/results_sense.csv",
            'cascade': "../fit_data/individual_circuits/results_cascade.csv",
            'cffl': "../fit_data/individual_circuits/results_cffl.csv"
        },
        output_dir="step_response_plots",
        n_samples=40,
        k_prot_deg=0.5,
        pulse_config={
            'use_pulse': True,
            'pulse_start': 4,
            'pulse_end': 15,
            'pulse_concentration': 5.0,
            'base_concentration': 0.0
        }
    )
