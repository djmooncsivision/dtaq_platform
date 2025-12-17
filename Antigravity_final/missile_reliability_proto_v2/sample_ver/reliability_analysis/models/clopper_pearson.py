from typing import Dict, Any, Optional
import pandas as pd
from scipy.stats import beta
from .base import ReliabilityModel

class ClopperPearsonModel(ReliabilityModel):
    def __init__(self, config: Any):
        super().__init__(config)
        self.results = {}

    def fit(self, data: pd.DataFrame, indices: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Performs Clopper-Pearson analysis.
        Expects 'data' to have 'num_tested' and 'num_failures' columns, 
        or these can be passed as kwargs if data is None.
        """
        confidence_level = kwargs.get('confidence_level', 0.90)
        
        if data is not None and not data.empty:
            # Aggregate if multiple rows provided, or treat as single sample set
            num_samples = data['num_tested'].sum()
            num_failures = data['num_failures'].sum()
        else:
            num_samples = kwargs.get('num_samples', 0)
            num_failures = kwargs.get('num_failures', 0)

        self.results = self._calculate_one_sided_lower_bound(num_samples, num_failures, confidence_level)

    def _calculate_one_sided_lower_bound(self, num_samples: int, num_failures: int, confidence_level: float) -> Dict[str, Any]:
        if num_samples == 0:
            return {
                "technique": "Clopper-Pearson (One-sided)",
                "num_samples": 0,
                "num_failures": 0,
                "confidence_level": confidence_level,
                "reliability_point_estimate": 0.0,
                "lower_confidence_bound": 0.0,
            }

        num_successes = num_samples - num_failures
        alpha = 1 - confidence_level
        reliability_point_estimate = num_successes / num_samples

        if num_successes == num_samples:
            lower_bound = alpha**(1 / num_samples)
        else:
            lower_bound = beta.ppf(alpha, num_successes, num_failures + 1)

        return {
            "technique": "Clopper-Pearson (One-sided)",
            "num_samples": num_samples,
            "num_failures": num_failures,
            "confidence_level": confidence_level,
            "reliability_point_estimate": reliability_point_estimate,
            "lower_confidence_bound": lower_bound,
        }

    def get_results(self) -> Dict[str, Any]:
        return self.results
