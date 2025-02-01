import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .utils import process_negative_controls
from utils.GFP_calibration import convert_AU_to_nM


@dataclass
class CircuitConfig:
    """Configuration for a specific circuit with multiple conditions"""
    model: Any
    name: str
    condition_params: Dict[str, Dict[str, float]]
    experimental_data: pd.DataFrame
    tspan: np.ndarray
    max_time: Optional[float] = None
    calibration_params: Optional[Dict] = None  # Add calibration params

    def __post_init__(self):
        """Process the data after initialization to apply time limits, negative control subtraction,
        and unit conversion from AU to nM"""
        if self.max_time is not None:
            # Truncate experimental data
            self.experimental_data = self.experimental_data[
                self.experimental_data['time'] <= self.max_time
                ].copy()

            # Truncate simulation timespan
            self.tspan = self.tspan[self.tspan <= self.max_time]

        # Process negative controls (data still in AU)
        self.experimental_data = process_negative_controls(self.experimental_data)

        # Convert fluorescence from AU to nM if calibration params are provided
        if self.calibration_params is not None:
            self.experimental_data['fluorescence'] = convert_AU_to_nM(
                self.experimental_data['fluorescence'],
                self.calibration_params['slope'],
                self.calibration_params['intercept'],
                self.calibration_params['brightness_correction']
            )

    @classmethod
    def from_data(cls, model, name, condition_params, data_path, calibration_params, max_time=None):
        """Alternative constructor that loads and processes data from file"""
        df = pd.read_csv(data_path)
        if max_time is not None:
            df = df[df['time'] <= max_time].copy()

        tspan = np.sort(df['time'].unique())

        return cls(
            model=model,
            name=name,
            condition_params=condition_params,
            experimental_data=df,
            tspan=tspan,
            max_time=max_time,
            calibration_params=calibration_params
        )