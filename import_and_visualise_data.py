import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns


def convert_time_string(time_str, time_unit_output='min'):
    """
    Convert time strings to minutes or hours.

    Parameters:
    - time_str (str): The time string to convert.
    - time_unit_output (str): The unit for the output time, either 'min' or 'hour'.

    Returns:
    - float: The converted time.
    """
    time_unit_output = time_unit_output.lower()
    if time_unit_output not in ['min', 'hour']:
        raise ValueError('time_unit_output must be "min" or "hour"')

    time_pattern = re.compile(r'(?:(?P<hours>\d+)\s*h\s*)?(?:(?P<minutes>\d+)\s*min)?')
    match = time_pattern.match(time_str.strip())
    if match:
        hours = int(match.group('hours')) if match.group('hours') else 0
        minutes = int(match.group('minutes')) if match.group('minutes') else 0
        total_minutes = hours * 60 + minutes
        return total_minutes / 60 if time_unit_output == 'hour' else total_minutes
    else:
        raise ValueError(f"Time string '{time_str}' is not in the expected format")


def load_and_process_csv(file_path, time_unit_output='min', time_col_name='time'):
    """
    Load and process the CSV file to convert it into a long format DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.
    - time_unit_output (str): The unit for the output time, either 'min' or 'hour'. Default is 'min'.
    - time_col_name (str): The name of the column containing the time strings. Default is 'time'.

    Returns:
    - pd.DataFrame: The processed DataFrame in long format with columns: 'time', 'condition', 'replicate', 'fluorescence'.
    """
    # Read the CSV file without a header
    data = pd.read_csv(file_path, header=None)

    # Extract the first row to use as column names
    data.columns = data.iloc[0]  # Use the first row as column names
    data = data[1:]  # Remove the first row (which contained the headers)

    # Convert time strings to numeric values
    data[time_col_name] = data[time_col_name].apply(lambda x: convert_time_string(x, time_unit_output))
    time_col = data[time_col_name]  # Extract the time column
    data = data.drop(columns=[time_col_name])  # Remove the time column from the main DataFrame

    # Convert all values to float
    data = data.astype(float)

    # Create a new DataFrame with columns: time, condition, replicate, value
    melted_data = []
    for condition in data.columns.unique():  # For each unique condition
        sub_df = data[condition]
        for i, replicate in enumerate(sub_df.columns):  # Enumerate columns
            # Create a new DataFrame with time, condition, replicate, value
            new_df = pd.DataFrame({
                'time': time_col,
                'condition': condition.split('_')[0],
                'fluorescence': sub_df.iloc[:, i],
                'replicate': i + 1
            })
            melted_data.append(new_df)

    # Concatenate all the new DataFrames
    new_data = pd.concat(melted_data, ignore_index=True)

    return new_data


def plot_replicates(data, title):
    """
    Plot the data columns with unique colors for each condition.

    Parameters:
    - data (pd.DataFrame): DataFrame containing 'time', 'condition', 'replicate', and 'fluorescence' columns.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='time', y='fluorescence', hue='condition', units='replicate', estimator=None, lw=1)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Fluorescence')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Load and process the CSV file
file_path = './data/AND-Gate_Se6To3.csv'
new_data = load_and_process_csv(file_path)
title = 'AND-Gate'

# Plot the processed data
plot_replicates(new_data, title)

