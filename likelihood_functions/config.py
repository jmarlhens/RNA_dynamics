import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .utils import process_negative_controls, load_and_process_csv


@dataclass
class CircuitConfig:
    """Configuration for a specific circuit with multiple conditions"""
    model: Any
    name: str
    condition_params: Dict[str, Dict[str, float]]
    experimental_data: pd.DataFrame
    tspan: np.ndarray
    max_time: Optional[float] = None

    def __post_init__(self):
        """Process the data after initialization to apply time limits and negative control subtraction"""
        if self.max_time is not None:
            # Truncate experimental data
            self.experimental_data = self.experimental_data[
                self.experimental_data['time'] <= self.max_time
                ].copy()

            # Truncate simulation timespan
            self.tspan = self.tspan[self.tspan <= self.max_time]

        # Process negative controls
        self.experimental_data = process_negative_controls(self.experimental_data)

    @classmethod
    def from_data(cls, model, name, condition_params, data_path, max_time=None):
        """Alternative constructor that loads and processes data from file"""
        data, tspan = load_and_process_csv(data_path, max_time)

        return cls(
            model=model,
            name=name,
            condition_params=condition_params,
            experimental_data=data,
            tspan=tspan,
            max_time=max_time
        )