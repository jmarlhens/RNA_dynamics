import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.import_and_visualise_data import load_and_process_csv
from data.circuits.circuit_configs import DATA_FILES, get_data_file


def calculate_mean_std_by_timepoint(data):
    """
    Calculate mean and standard deviation for each condition and time point across replicates.

    Parameters:
    - data (pd.DataFrame): DataFrame with columns 'time', 'condition', 'replicate', 'fluorescence'

    Returns:
    - pd.DataFrame: DataFrame with columns 'time', 'condition', 'mean_fluorescence', 'std_fluorescence'
    """
    # Group by time and condition, then calculate mean and std across replicates
    summary_stats = (
        data.groupby(["time", "condition"])["fluorescence"]
        .agg(
            [
                ("mean_fluorescence", "mean"),
                ("std_fluorescence", "std"),
                ("n_replicates", "count"),
            ]
        )
        .reset_index()
    )

    # Remove rows where std is NaN (happens when only 1 replicate)
    summary_stats = summary_stats.dropna(subset=["std_fluorescence"])

    return summary_stats


def load_multiple_datasets(data_paths):
    """
    Load multiple experimental datasets and combine them.

    Parameters:
    - data_paths (dict): Dictionary with circuit names as keys and file paths as values

    Returns:
    - pd.DataFrame: Combined dataset with additional 'circuit' column
    """
    all_data = []

    for circuit_name, file_path in data_paths.items():
        try:
            data, _ = load_and_process_csv(file_path)
            data["circuit"] = circuit_name
            all_data.append(data)
            print(f"Loaded {circuit_name}: {len(data)} data points")
        except Exception as e:
            print(f"Error loading {circuit_name} from {file_path}: {e}")

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        return pd.DataFrame()


def load_datasets_from_config(circuit_names=None):
    """
    Load datasets using the configuration from circuit_configs.py

    Parameters:
    - circuit_names (list, optional): List of specific circuit names to load.
                                                                     If None, loads all available circuits.

    Returns:
    - pd.DataFrame: Combined dataset with additional 'circuit' column
    """
    if circuit_names is None:
        # Load all circuits from config
        data_paths = DATA_FILES
    else:
        # Load only specified circuits
        data_paths = {
            name: get_data_file(name)
            for name in circuit_names
            if get_data_file(name) is not None
        }

        # Check for invalid circuit names
        invalid_names = [name for name in circuit_names if get_data_file(name) is None]
        if invalid_names:
            print(f"Warning: Unknown circuit names: {invalid_names}")
            print(f"Available circuits: {list(DATA_FILES.keys())}")

    return load_multiple_datasets(data_paths)


def plot_heteroscedasticity_by_circuit(summary_data):
    """
    Plot mean vs std colored by circuit.
    """
    plt.figure(figsize=(10, 6))

    # Create scatter plot colored by circuit
    sns.scatterplot(
        data=summary_data,
        x="mean_fluorescence",
        y="std_fluorescence",
        hue="circuit",
        alpha=0.7,
        s=50,
    )

    plt.xlabel("Mean Fluorescence")
    plt.ylabel("Standard Deviation")
    plt.title("Heteroscedasticity Analysis: Mean vs Standard Deviation by Circuit")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_heteroscedasticity_by_condition(summary_data):
    """
    Plot mean vs std colored by condition.
    """
    plt.figure(figsize=(12, 6))

    # Create scatter plot colored by condition
    sns.scatterplot(
        data=summary_data,
        x="mean_fluorescence",
        y="std_fluorescence",
        hue="condition",
        style="circuit",
        alpha=0.7,
        s=50,
    )

    plt.xlabel("Mean Fluorescence")
    plt.ylabel("Standard Deviation")
    plt.title("Heteroscedasticity Analysis: Mean vs Standard Deviation by Condition")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_heteroscedasticity_pattern(summary_data):
    """
    Analyze the relationship between mean and standard deviation.
    """
    # Calculate correlation
    correlation = np.corrcoef(
        summary_data["mean_fluorescence"], summary_data["std_fluorescence"]
    )[0, 1]

    # Fit linear relationship in log-log space to check for power law
    log_mean = np.log10(
        summary_data["mean_fluorescence"] + 1e-6
    )  # Add small constant to avoid log(0)
    log_std = np.log10(summary_data["std_fluorescence"] + 1e-6)

    # Remove infinite values
    valid_mask = np.isfinite(log_mean) & np.isfinite(log_std)
    if np.sum(valid_mask) > 1:
        log_correlation = np.corrcoef(log_mean[valid_mask], log_std[valid_mask])[0, 1]
        slope = np.polyfit(log_mean[valid_mask], log_std[valid_mask], 1)[0]
    else:
        log_correlation = np.nan
        slope = np.nan

    print(f"Linear correlation (mean vs std): {correlation:.3f}")
    print(f"Log-log correlation: {log_correlation:.3f}")
    print(f"Power law exponent (slope in log-log): {slope:.3f}")

    if slope > 0.8:
        print("Strong evidence for heteroscedasticity (std ∝ mean)")
    elif slope > 0.4:
        print("Moderate heteroscedasticity")
    else:
        print("Weak heteroscedasticity")


if __name__ == "__main__":
    print("Available circuits in config:")
    for circuit_name in DATA_FILES.keys():
        print(f"  - {circuit_name}")

    # Option 1: Load all circuits from config
    print("\nLoading all datasets from config...")
    combined_data = load_datasets_from_config()

    # Option 2: Load specific circuits only (uncomment to use)
    # specific_circuits = ['trigger_antitrigger', 'sense_star_6', 'cascade']
    # print(f"\nLoading specific datasets: {specific_circuits}")
    # combined_data = load_datasets_from_config(specific_circuits)

    if combined_data.empty:
        print("No data loaded. Check file paths in config.")
        exit()

    print(f"Total data points loaded: {len(combined_data)}")
    print(f"Circuits: {combined_data['circuit'].unique()}")
    print("Conditions per circuit:")
    for circuit in combined_data["circuit"].unique():
        conditions = combined_data[combined_data["circuit"] == circuit][
            "condition"
        ].unique()
        print(f"  {circuit}: {len(conditions)} conditions")

    # Calculate mean and std for each time point and condition
    print("\nCalculating summary statistics...")
    summary_data = calculate_mean_std_by_timepoint(combined_data)

    # Add circuit information to summary data
    condition_to_circuit = combined_data[["condition", "circuit"]].drop_duplicates()
    summary_data = summary_data.merge(condition_to_circuit, on="condition", how="left")

    print(f"Summary data points: {len(summary_data)}")

    # Analyze heteroscedasticity pattern
    print("\nAnalyzing heteroscedasticity pattern...")
    analyze_heteroscedasticity_pattern(summary_data)

    # Create visualizations
    print("\nCreating visualizations...")

    # Plot by circuit
    plot_heteroscedasticity_by_circuit(summary_data)

    # Plot by condition
    plot_heteroscedasticity_by_condition(summary_data)

    # Optional: Show data summary
    print("\nData Summary:")
    print(summary_data.describe())
