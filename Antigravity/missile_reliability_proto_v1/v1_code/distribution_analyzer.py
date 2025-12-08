import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ks_2samp
from scipy.spatial.distance import jensenshannon

class DistributionAnalyzer:
    def __init__(self, qim_df, asrp_df):
        self.qim_df = qim_df
        self.asrp_df = asrp_df
        self.sampled_qim_df = None

    def perform_sampling(self, target_col='1'):
        """
        Samples QIM data based on ASRP distribution using KDE.
        This ensures we compare 'apples to apples' in terms of distribution shape.
        """
        qim_data = self.qim_df[target_col].values
        asrp_data = self.asrp_df[target_col].values
        
        # KDE for ASRP distribution
        try:
            kde_asrp = gaussian_kde(asrp_data)
            
            # Calculate density for QIM data points
            qim_density = kde_asrp(qim_data)
            
            # Normalize probabilities
            if qim_density.sum() == 0:
                 probabilities = np.ones(len(qim_data)) / len(qim_data)
            else:
                probabilities = qim_density / qim_density.sum()
            
            # Sample QIM data
            target_count = min(len(self.asrp_df), len(self.qim_df))
            
            sampled_indices = np.random.choice(
                self.qim_df.index,
                size=target_count,
                replace=False,
                p=probabilities
            )
            
            self.sampled_qim_df = self.qim_df.loc[sampled_indices]
            return self.sampled_qim_df
            
        except Exception as e:
            print(f"Sampling failed: {e}. Returning original QIM.")
            self.sampled_qim_df = self.qim_df
            return self.qim_df

    def compare_distributions(self, col):
        """Calculates KS Statistic and JSD between QIM and ASRP."""
        qim_data = self.qim_df[col].dropna().values
        asrp_data = self.asrp_df[col].dropna().values
        
        if len(qim_data) == 0 or len(asrp_data) == 0:
            return {
                "KS_Statistic": 0,
                "P_Value": 1,
                "JSD": 0
            }
        
        # KS Test
        try:
            ks_stat, p_value = ks_2samp(qim_data, asrp_data)
        except:
            ks_stat, p_value = 0, 1
        
        # JSD (requires binning for discrete probability distribution)
        # Simplified approach: histogram binning
        try:
            range_min = min(qim_data.min(), asrp_data.min())
            range_max = max(qim_data.max(), asrp_data.max())
            
            if range_min == range_max:
                jsd = 0
            else:
                bins = np.linspace(range_min, range_max, 20)
                
                qim_hist, _ = np.histogram(qim_data, bins=bins, density=True)
                asrp_hist, _ = np.histogram(asrp_data, bins=bins, density=True)
                
                # Add small epsilon to avoid log(0)
                qim_hist += 1e-10
                asrp_hist += 1e-10
                
                jsd = jensenshannon(qim_hist, asrp_hist)
        except Exception as e:
            print(f"JSD calculation failed for {col}: {e}")
            jsd = 0
        
        return {
            "KS_Statistic": ks_stat,
            "P_Value": p_value,
            "JSD": jsd
        }
