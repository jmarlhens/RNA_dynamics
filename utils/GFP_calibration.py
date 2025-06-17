import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import io
import requests


@dataclass
class FluorescentProteinProperties:
    """Class to store relevant fluorescent protein properties"""

    name: str
    brightness: float
    quantum_yield: float
    ext_coeff: float


class FPbaseAPI:
    """Class to interact with the FPbase API"""

    BASE_URL = "https://www.fpbase.org/api/proteins/"

    @staticmethod
    def get_protein_data() -> Optional[pd.DataFrame]:
        """Fetch protein data from FPbase API"""
        try:
            response = requests.get(FPbaseAPI.BASE_URL)
            if response.status_code == 200:
                # Parse CSV data
                return pd.read_csv(io.StringIO(response.content.decode("utf-8")))
            else:
                print(f"Error fetching protein data: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error accessing FPbase API: {e}")
            return None

    @staticmethod
    def get_protein(protein_slug: str) -> Optional[FluorescentProteinProperties]:
        """
        Get properties for a specific protein

        Parameters
        ----------
        protein_slug : str
            Protein identifier

        Returns
        -------
        Optional[FluorescentProteinProperties]
            Protein properties if found
        """
        df_protein_data = FPbaseAPI.get_protein_data()
        if df_protein_data is None:
            return FPbaseAPI._get_hardcoded_properties(protein_slug)

        # Filter for the specific protein
        protein_data = df_protein_data[df_protein_data["slug"] == protein_slug]

        if len(protein_data) == 0:
            return FPbaseAPI._get_hardcoded_properties(protein_slug)

        row_protein_data = protein_data.iloc[0]

        # Extract properties from the first state (assuming it's the main state)
        try:
            return FluorescentProteinProperties(
                name=row_protein_data["name"],
                brightness=float(row_protein_data["states.0.brightness"]),
                quantum_yield=float(row_protein_data["states.0.qy"]),
                ext_coeff=float(row_protein_data["states.0.ext_coeff"]),
            )
        except (KeyError, ValueError):
            return FPbaseAPI._get_hardcoded_properties(protein_slug)

    @staticmethod
    def _get_hardcoded_properties(slug: str) -> Optional[FluorescentProteinProperties]:
        """Fallback to hardcoded values if API fails"""
        hardcoded_values = {
            "avGFP": {
                "name": "Wild-type A. victoria GFP",
                "brightness": 18.0,
                "quantum_yield": 0.30,
                "ext_coeff": 60000,
            },
            "sfGFP": {
                "name": "Superfolder GFP",
                "brightness": 54.0,
                "quantum_yield": 0.65,
                "ext_coeff": 83300,
            },
        }

        if slug in hardcoded_values:
            data = hardcoded_values[slug]
            return FluorescentProteinProperties(**data)
        return None


def fit_gfp_calibration(
    data: pd.DataFrame,
    concentration_col: str = "GFP Concentration (nM)",
    fluorescence_pattern: str = "F.I. (a.u)",
) -> Dict:
    """
    Fit a linear regression to GFP calibration data with replicates.
    """
    # Get concentrations
    concentrations = data[concentration_col]

    # Get all fluorescence replicate columns
    fluorescence_cols = data.filter(like=fluorescence_pattern).columns

    # Calculate mean fluorescence for each concentration
    mean_fluorescence = data[fluorescence_cols].mean(axis=1)

    # Calculate standard error for each concentration
    std_error = data[fluorescence_cols].std(axis=1) / np.sqrt(len(fluorescence_cols))

    # Perform linear regression
    result = stats.linregress(concentrations, mean_fluorescence)

    # Calculate 95% confidence intervals
    confidence_level = 0.95
    degrees_freedom = len(concentrations) - 2
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)

    slope_ci = (
        result.slope - t_value * result.stderr,
        result.slope + t_value * result.stderr,
    )

    intercept_ci = (
        result.intercept - t_value * result.intercept_stderr,
        result.intercept + t_value * result.intercept_stderr,
    )

    return {
        "slope": result.slope,
        "intercept": result.intercept,
        "r_squared": result.rvalue**2,
        "std_error": result.stderr,
        "p_value": result.pvalue,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "mean_fluorescence": mean_fluorescence,
        "fluorescence_stderr": std_error,
    }


def get_brightness_correction_factor(
    calibration_protein_slug: str = "avGFP", target_protein_slug: str = "sfGFP"
) -> Tuple[float, Dict]:
    """
    Calculate brightness correction factor between two fluorescent proteins
    """
    api_fp_base_api = FPbaseAPI()

    # Fetch protein data
    cal_protein = api_fp_base_api.get_protein(calibration_protein_slug)
    target_protein = api_fp_base_api.get_protein(target_protein_slug)

    if not cal_protein or not target_protein:
        raise ValueError(
            f"Could not get data for proteins: {calibration_protein_slug}, {target_protein_slug}"
        )

    # Calculate correction factor
    brightness_correction_factor = target_protein.brightness / cal_protein.brightness

    # Prepare protein info dictionary
    protein_information = {
        "calibration": {
            "name": cal_protein.name,
            "brightness": cal_protein.brightness,
            "quantum_yield": cal_protein.quantum_yield,
            "ext_coeff": cal_protein.ext_coeff,
        },
        "target": {
            "name": target_protein.name,
            "brightness": target_protein.brightness,
            "quantum_yield": target_protein.quantum_yield,
            "ext_coeff": target_protein.ext_coeff,
        },
    }

    return brightness_correction_factor, protein_information


def convert_nm_to_au(
    concentration: np.ndarray,
    slope: float,
    intercept: float,
    brightness_correction: float = 1.0,
) -> np.ndarray:
    """Convert GFP concentration from nM to arbitrary units"""
    corrected_slope = slope * brightness_correction
    return concentration * corrected_slope + intercept


def convert_au_to_nm(
    fluorescence: np.ndarray,
    slope: float,
    intercept: float,
    brightness_correction: float = 1.0,
) -> np.ndarray:
    """Convert GFP fluorescence from arbitrary units to nM"""
    corrected_slope = slope * brightness_correction
    return (fluorescence - intercept) / corrected_slope


if __name__ == "__main__":
    import requests

    # First, let's check what proteins are available in the API
    api = FPbaseAPI()
    df = api.get_protein_data()
    if df is not None:
        # Find GFP variants
        gfp_variants = df[df["name"].str.contains("GFP", case=False, na=False)]
        print("Available GFP variants:")
        for _, row in gfp_variants.iterrows():
            print(f"Name: {row['name']}, Slug: {row['slug']}")

    # Now let's get our correction factor
    correction_factor, protein_info = get_brightness_correction_factor("avGFP", "sfGFP")

    print("\nProtein properties:")
    print("\nCalibration protein (avGFP):")
    for key, value in protein_info["calibration"].items():
        print(f"{key}: {value}")
    print("\nTarget protein (sfGFP):")
    for key, value in protein_info["target"].items():
        print(f"{key}: {value}")
    print(f"\nBrightness correction factor: {correction_factor:.2f}")
