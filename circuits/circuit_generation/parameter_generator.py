import pandas as pd
import os
from pathlib import Path


def generate_mass_action_parameters(
    input_csv, output_csv=None, rnap_0=100, ribosome_0=1000
):
    """
    Load existing parameter priors and generate new mass action parameters.

    Args:
        input_csv: Path to existing parameters CSV file
        output_csv: Path to save updated parameters (if None, will add '_with_mass_action' to original name)
        rnap_0: Initial concentration of RNA polymerase
        ribosome_0: Initial concentration of ribosomes

    Returns:
        DataFrame with all parameters
    """
    # Set default output path if not provided
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = (
            input_path.parent / f"{input_path.stem}_with_mass_action{input_path.suffix}"
        )

    # Load existing parameters
    print(f"Loading parameters from {input_csv}")
    df = pd.read_csv(input_csv)

    # Convert parameters to a dictionary for easier access
    param_dict = dict(zip(df["Parameter"], df["Value"]))

    # Create a list to hold new parameters
    new_params = []

    # Calculate mass action parameters for transcription
    k_tx = param_dict["k_tx"]
    K_tx = param_dict["K_tx"]
    k_tx_bind = 10 ** (-2)  # diffusive limit for binding
    k_tx_cat = k_tx / rnap_0  # Scale catalytic rate by RNAP_0
    # Calculate unbinding rate to maintain the same Km
    # From Michaelis-Menten theory: Km = (k_unbind + k_cat) / k_bind
    k_tx_unbind = max(0.001, k_tx_bind * K_tx - k_tx_cat)  # Ensure positive value

    # Calculate mass action parameters for translation
    k_tl = param_dict["k_tl"]
    K_tl = param_dict["K_tl"]
    k_tl_bind = 10 ** (-2)  # diffusive limit for binding
    k_tl_cat = k_tl / ribosome_0  # Scale catalytic rate by Ribosome_0
    # Calculate unbinding rate to maintain the same Km
    k_tl_unbind = max(0.001, k_tl_bind * K_tl - k_tl_cat)  # Ensure positive value

    # STAR-specific parameters
    k_star_act = param_dict["k_star_act"]
    k_star_act_reg = param_dict["k_star_act_reg"]

    # Enhanced binding and catalysis rates for STAR activation
    star_factor = k_star_act_reg / max(
        k_star_act, 0.01
    )  # Ratio of activated to non-activated

    # Toehold-specific parameters
    k_tl_unbound = param_dict["k_tl_unbound_toehold"]
    k_tl_bound = param_dict["k_tl_bound_toehold"]

    # Ratio of bound to unbound translation for toehold
    toehold_factor = k_tl_bound / max(k_tl_unbound, 0.01)

    # Define new parameters
    new_params = [
        {
            "Parameter": "k_tx_bind",
            "Value": k_tx_bind,
            "Description": "RNA polymerase binding rate to DNA",
        },
        {
            "Parameter": "k_tx_unbind",
            "Value": k_tx_unbind,
            "Description": "RNA polymerase unbinding rate from DNA",
        },
        {
            "Parameter": "k_tx_cat",
            "Value": k_tx_cat,
            "Description": f"Catalytic rate of RNA production (k_tx/{rnap_0})",
        },
        {
            "Parameter": "k_tl_bind",
            "Value": k_tl_bind,
            "Description": "Ribosome binding rate to RNA",
        },
        {
            "Parameter": "k_tl_unbind",
            "Value": k_tl_unbind,
            "Description": "Ribosome unbinding rate from RNA",
        },
        {
            "Parameter": "k_tl_cat",
            "Value": k_tl_cat,
            "Description": f"Catalytic rate of protein production (k_tl/{ribosome_0})",
        },
        {
            "Parameter": "RNAP_0",
            "Value": rnap_0,
            "Description": "Initial concentration of RNA polymerase",
        },
        {
            "Parameter": "Ribosome_0",
            "Value": ribosome_0,
            "Description": "Initial concentration of ribosomes",
        },
        # STAR-specific mass action parameters
        {
            "Parameter": "k_tx_bind_star",
            "Value": k_tx_bind * star_factor,
            "Description": "Enhanced polymerase binding rate with STAR activation",
        },
        {
            "Parameter": "k_tx_cat_star",
            "Value": k_tx_cat * star_factor,
            "Description": "Enhanced catalytic rate with STAR activation",
        },
        # Toehold-specific mass action parameters
        {
            "Parameter": "k_tl_bind_unbound",
            "Value": k_tl_bind * 0.1,
            "Description": "Ribosome binding rate to unbound toehold RNA",
        },
        {
            "Parameter": "k_tl_cat_unbound",
            "Value": k_tl_cat * 0.1,
            "Description": "Translation rate from unbound toehold RNA",
        },
        {
            "Parameter": "k_tl_bind_bound",
            "Value": k_tl_bind * toehold_factor,
            "Description": "Ribosome binding rate to bound toehold RNA",
        },
        {
            "Parameter": "k_tl_cat_bound",
            "Value": k_tl_cat * toehold_factor,
            "Description": "Translation rate from bound toehold RNA",
        },
    ]

    # Check if parameters already exist in the original dataframe
    existing_params = set(df["Parameter"])
    filtered_new_params = [
        p for p in new_params if p["Parameter"] not in existing_params
    ]

    # Create dataframe for new parameters
    if filtered_new_params:
        new_params_df = pd.DataFrame(filtered_new_params)

        # Ensure the new dataframe has the same columns as the original
        for col in df.columns:
            if col not in new_params_df.columns:
                new_params_df[col] = None

        # Combine original and new parameters
        combined_df = pd.concat([df, new_params_df[df.columns]], ignore_index=True)

        # Save the updated parameters
        combined_df.to_csv(output_csv, index=False)
        print(
            f"Added {len(filtered_new_params)} mass action parameters to {output_csv}"
        )
        print(f"New parameters: {[p['Parameter'] for p in filtered_new_params]}")
    else:
        combined_df = df
        print("All mass action parameters already exist in the file.")

    return combined_df


def print_parameter_mappings(df, rnap_0=100, ribosome_0=1000):
    """
    Print a summary of how original parameters map to mass action parameters.

    Args:
        df: DataFrame with all parameters
        rnap_0: Initial concentration of RNA polymerase
        ribosome_0: Initial concentration of ribosomes
    """
    param_dict = dict(zip(df["Parameter"], df["Value"]))

    print("\n=== Parameter Mappings ===")
    print("Michaelis-Menten to Mass Action Relationships:\n")

    # Enzyme concentrations
    print("Enzyme Concentrations:")
    print(f"  RNAP_0: {rnap_0}")
    print(f"  Ribosome_0: {ribosome_0}")

    # Transcription
    print("\nTranscription:")
    print(f"  k_tx (MM max rate): {param_dict.get('k_tx', 'N/A')}")
    print(f"  K_tx (MM constant): {param_dict.get('K_tx', 'N/A')}")
    print(f"  k_tx_bind (MA binding): {param_dict.get('k_tx_bind', 'N/A')}")
    print(f"  k_tx_unbind (MA unbinding): {param_dict.get('k_tx_unbind', 'N/A')}")
    print(
        f"  k_tx_cat (MA catalytic): {param_dict.get('k_tx_cat', 'N/A')} (= k_tx / RNAP_0)"
    )

    # Translation
    print("\nTranslation:")
    print(f"  k_tl (MM max rate): {param_dict.get('k_tl', 'N/A')}")
    print(f"  K_tl (MM constant): {param_dict.get('K_tl', 'N/A')}")
    print(f"  k_tl_bind (MA binding): {param_dict.get('k_tl_bind', 'N/A')}")
    print(f"  k_tl_unbind (MA unbinding): {param_dict.get('k_tl_unbind', 'N/A')}")
    print(
        f"  k_tl_cat (MA catalytic): {param_dict.get('k_tl_cat', 'N/A')} (= k_tl / Ribosome_0)"
    )

    # STAR system
    print("\nSTAR system:")
    print(f"  k_star_act (original): {param_dict.get('k_star_act', 'N/A')}")
    print(f"  k_star_act_reg (original): {param_dict.get('k_star_act_reg', 'N/A')}")
    print(
        f"  k_tx_bind_star (MA enhanced binding): {param_dict.get('k_tx_bind_star', 'N/A')}"
    )
    print(
        f"  k_tx_cat_star (MA enhanced catalytic): {param_dict.get('k_tx_cat_star', 'N/A')}"
    )

    # Toehold system
    print("\nToehold system:")
    print(
        f"  k_tl_unbound_toehold (original): {param_dict.get('k_tl_unbound_toehold', 'N/A')}"
    )
    print(
        f"  k_tl_bound_toehold (original): {param_dict.get('k_tl_bound_toehold', 'N/A')}"
    )
    print(
        f"  k_tl_bind_unbound (MA unbound binding): {param_dict.get('k_tl_bind_unbound', 'N/A')}"
    )
    print(
        f"  k_tl_cat_unbound (MA unbound catalytic): {param_dict.get('k_tl_cat_unbound', 'N/A')}"
    )
    print(
        f"  k_tl_bind_bound (MA bound binding): {param_dict.get('k_tl_bind_bound', 'N/A')}"
    )
    print(
        f"  k_tl_cat_bound (MA bound catalytic): {param_dict.get('k_tl_cat_bound', 'N/A')}"
    )


# Example usage
if __name__ == "__main__":
    # Path to your parameters CSV file
    # Replace with your actual parameters file path
    parameters_file = "../../data/prior/model_parameters_priors.csv"

    # Define initial concentrations for RNAP and Ribosome
    rnap_0 = 1000
    ribosome_0 = 1000

    if os.path.exists(parameters_file):
        # Generate the parameters
        updated_params = generate_mass_action_parameters(
            parameters_file, rnap_0=rnap_0, ribosome_0=ribosome_0
        )

        # Print parameter mappings
        print_parameter_mappings(updated_params, rnap_0=rnap_0, ribosome_0=ribosome_0)
    else:
        print(f"Error: File not found: {parameters_file}")
        print("Please specify the correct path to your parameters file.")
