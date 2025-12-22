import os
import pymc as pm
import arviz as az
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable
from .base import ReliabilityModel

class HierarchicalBayesianModel(ReliabilityModel):
    def __init__(self, config: Any):
        super().__init__(config)
        self.trace = None
        self.model_name = ""

    def _build_hierarchical_structure(self, model_params: Dict[str, Any], indices: Dict[str, Any]):
        """Defines the common hierarchical structure."""
        # Global mean reliability (logit scale)
        mu_global_logit = pm.Normal('mu_global_logit', 
                                    mu=self.config.MODEL_PRIORS['MU_GLOBAL_LOGIT_MU'], 
                                    sigma=self.config.MODEL_PRIORS['MU_GLOBAL_LOGIT_SIGMA'])
        
        # Variances
        sigma_year = pm.HalfNormal('sigma_year', sigma=model_params["inter_year_sigma"])
        sigma_lot_base = pm.HalfNormal('sigma_lot_base', sigma=model_params["intra_lot_sigma"])

        if model_params["degradation_effect_on_variance"]:
            variance_degradation_rate = pm.HalfNormal('variance_degradation_rate', 
                                                      sigma=model_params.get('degradation_rate_sigma', 0.05))
            age_of_lot = 2025 - indices["year_of_lot"]
            sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base + age_of_lot * variance_degradation_rate)
        else:
            sigma_lot_effective = pm.Deterministic('sigma_lot_effective', sigma_lot_base)

        # Hierarchical effects
        theta_year = pm.Normal('theta_year', mu=mu_global_logit, sigma=sigma_year, shape=len(indices["all_years"]))
        theta_lot = pm.Normal('theta_lot', mu=theta_year[indices["year_idx_of_lot"]], sigma=sigma_lot_effective, shape=len(indices["all_lots"]))
        
        return theta_lot

    def fit(self, data: pd.DataFrame, indices: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Fits the Bayesian model.
        Args:
            data: Aggregated observed data.
            indices: Index mappings.
            kwargs:
                model_type: 'binomial' or 'hypergeometric'
                scenario_params: Dictionary of scenario parameters (sigmas, etc.)
                scenario_name: Name of the scenario (for caching)
                case_name: Name of the test case (for caching)
        """
        model_type = kwargs.get('model_type', 'hypergeometric')
        scenario_params = kwargs.get('scenario_params')
        scenario_name = kwargs.get('scenario_name', 'default')
        case_name = kwargs.get('case_name', 'default')
        
        self.model_name = f"{model_type}_{case_name}_{scenario_name}"
        
        # Check cache
        cache_filename = f"{self.model_name.replace(' ', '_')}.nc"
        cache_path = os.path.join(self.config.CACHE_DIR, cache_filename)
        
        if os.path.exists(cache_path):
            print(f"Loading cached model: {cache_path}")
            self.trace = az.from_netcdf(cache_path)
            return

        print(f"Running {model_type} model for {scenario_name}...")
        
        with pm.Model() as model:
            theta_lot = self._build_hierarchical_structure(scenario_params, indices)
            
            if model_type == 'binomial':
                reliability_lot = pm.Deterministic('reliability_lot', pm.invlogit(theta_lot))
                pm.Binomial('y_obs', 
                            n=data['num_tested'].values, 
                            p=reliability_lot[indices["observed_lot_idx"]], 
                            observed=data['num_success'].values)
            
            elif model_type == 'hypergeometric':
                p_lot = pm.invlogit(theta_lot)
                lot_quantities = indices["lot_quantities"]
                k_lot = pm.Binomial('k_lot', n=lot_quantities, p=p_lot, shape=len(indices["all_lots"]))

                pm.HyperGeometric('y_obs', 
                                  N=lot_quantities[indices["observed_lot_idx"]], 
                                  k=k_lot[indices["observed_lot_idx"]], 
                                  n=data['num_tested'].values, 
                                  observed=data['num_success'].values)

                pm.Deterministic('reliability_lot', k_lot / lot_quantities)

            # Sampling
            mcmc_config = self.config.MCMC_CONFIG
            self.trace = pm.sample(
                draws=mcmc_config['draws'],
                tune=mcmc_config['tune'],
                chains=mcmc_config['chains'],
                random_seed=mcmc_config['random_seed'],
                progressbar=mcmc_config['progressbar'],
                return_inferencedata=True
            )
            
            # Save to cache
            self.trace.to_netcdf(cache_path)
            print(f"Model saved to cache: {cache_path}")

    def get_results(self) -> az.InferenceData:
        return self.trace
