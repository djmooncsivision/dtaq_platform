from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd

class ReliabilityModel(ABC):
    """Abstract base class for all reliability analysis models."""

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def fit(self, data: pd.DataFrame, indices: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Fits the model to the data.
        Args:
            data: Observed data DataFrame.
            indices: Dictionary of indices for hierarchical models (optional).
            **kwargs: Additional model-specific parameters.
        """
        pass

    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """
        Returns the analysis results.
        Returns:
            Dictionary containing results (e.g., reliability estimates, confidence bounds).
        """
        pass
