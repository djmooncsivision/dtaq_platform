import os
import platform
from typing import Dict, Any

class Config:
    # Base Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data') # Assuming data is in the parent directory or we will move it
    CACHE_DIR = os.path.join(BASE_DIR, '..', 'cache')
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
    
    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data Files
    STOCKPILE_DATA_FILE = "stockpile_composition_data.csv"
    TEST_CASES = {
        "test_1_0_fail": "observed_reliability_data_0_failures.csv",
        "test_2_1_fail": "observed_reliability_data_1_failure.csv"
    }

    # MCMC Configuration
    MCMC_CONFIG = {
        'draws': 1000,
        'tune': 500,
        'chains': 1,
        'random_seed': 42,
        'progressbar': True
    }

    # Model Priors
    MODEL_PRIORS = {
        "MU_GLOBAL_LOGIT_MU": 2.197,   # ~90% reliability (logit(0.9) ≈ 2.197)
        "MU_GLOBAL_LOGIT_SIGMA": 0.5,
        # "VARIANCE_DEGRADATION_RATE_SIGMA": 0.05  # Moved to scenario params
    }

    # Scenarios
    SCENARIOS = {
        "Optimistic": {
            "inter_year_sigma": 0.01, 
            "intra_lot_sigma": 0.02, 
            "degradation_effect_on_variance": True, # Enable degradation
            "degradation_rate_sigma": 0.01,         # Small degradation rate
            "color": "dodgerblue"
        },
        "Pessimistic": {
            "inter_year_sigma": 0.2, 
            "intra_lot_sigma": 0.1, 
            "degradation_effect_on_variance": True,
            "degradation_rate_sigma": 0.05,         # Larger degradation rate
            "color": "firebrick"
        }
    }

    # Visualization
    DEFAULT_FONT = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic'
    TARGET_RELIABILITY = 0.98
    SCENARIO_COLORS = {k: v['color'] for k, v in SCENARIOS.items()}
    SCENARIO_COLORS.update({
        "Binomial Model": "forestgreen",       # Distinct Green
        "Hypergeometric Model": "mediumorchid" # Distinct Purple
    })
