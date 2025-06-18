import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional
from utils.process_experimental_data import (
    process_background_fluorescence,
    apply_time_constraints,
)
from utils.GFP_calibration import convert_au_to_nm


@dataclass
class CircuitConfig:
    """Configuration for a specific circuit with multiple conditions"""

    model: Any
    name: str
    condition_params: Dict[str, Dict[str, float]]
    experimental_data: pd.DataFrame
    tspan: np.ndarray
    max_time: Optional[float] = None
    min_time: Optional[float] = None
    calibration_params: Optional[Dict] = None

    def __post_init__(self):
        """Process the data after initialization to apply time limits, negative control subtraction,
        and unit conversion from AU to nM"""
        # First handle time constraints and shifting
        self.experimental_data, self.tspan, self.max_time = apply_time_constraints(
            self.experimental_data, self.tspan, self.min_time, self.max_time
        )

        # Process negative controls (data still in AU)
        self.experimental_data = process_background_fluorescence(self.experimental_data)

        # Convert fluorescence from AU to nM if calibration params are provided
        if self.calibration_params is not None:
            self.experimental_data["fluorescence"] = convert_au_to_nm(
                self.experimental_data["fluorescence"],
                self.calibration_params["slope"],
                self.calibration_params["intercept"],
                self.calibration_params["brightness_correction"],
            )

    @classmethod
    def from_data(
        cls,
        model,
        name,
        condition_params,
        data_path,
        calibration_params,
        max_time=None,
        min_time=None,
    ):
        """Alternative constructor that loads and processes data from file"""
        df = pd.read_csv(data_path)
        tspan = np.sort(df["time"].unique())

        return cls(
            model=model,
            name=name,
            condition_params=condition_params,
            experimental_data=df,
            tspan=tspan,
            max_time=max_time,
            min_time=min_time,
            calibration_params=calibration_params,
        )
