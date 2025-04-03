import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .utils import process_negative_controls
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
        if self.min_time is not None or self.max_time is not None:
            # Create a copy of the data to avoid modifying the original
            self.experimental_data = self.experimental_data.copy()

            # Apply min time filter if specified
            if self.min_time is not None:
                # Filter out data points before min_time
                self.experimental_data = self.experimental_data[
                    self.experimental_data['time'] >= self.min_time
                    ]

                # Shift time values by subtracting min_time
                self.experimental_data['time'] = self.experimental_data['time'] - self.min_time

                # Adjust max_time if it's set
                if self.max_time is not None:
                    self.max_time = max(0, self.max_time - self.min_time)

                # Update tspan to match shifted times
                original_tspan = self.tspan.copy()
                shifted_tspan = original_tspan[original_tspan >= self.min_time] - self.min_time
                self.tspan = shifted_tspan

            # Apply max time filter if specified (after min_time adjustments)
            if self.max_time is not None:
                # Truncate experimental data
                self.experimental_data = self.experimental_data[
                    self.experimental_data['time'] <= self.max_time
                    ]

                # Truncate simulation timespan
                self.tspan = self.tspan[self.tspan <= self.max_time]

        # Process negative controls (data still in AU)
        self.experimental_data = process_negative_controls(self.experimental_data)

        # Convert fluorescence from AU to nM if calibration params are provided
        if self.calibration_params is not None:
            self.experimental_data['fluorescence'] = convert_au_to_nm(
                self.experimental_data['fluorescence'],
                self.calibration_params['slope'],
                self.calibration_params['intercept'],
                self.calibration_params['brightness_correction']
            )

    @classmethod
    def from_data(cls, model, name, condition_params, data_path, calibration_params,
                  max_time=None, min_time=None):
        """Alternative constructor that loads and processes data from file"""
        df = pd.read_csv(data_path)
        tspan = np.sort(df['time'].unique())

        return cls(
            model=model,
            name=name,
            condition_params=condition_params,
            experimental_data=df,
            tspan=tspan,
            max_time=max_time,
            min_time=min_time,
            calibration_params=calibration_params
        )
