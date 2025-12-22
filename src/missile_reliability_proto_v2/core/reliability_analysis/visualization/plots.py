import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import arviz as az
from typing import Dict, Any, List
from ..config import Config

class Visualizer:
    def __init__(self):
        self.config = Config()
        plt.rcParams['font.family'] = self.config.DEFAULT_FONT
        plt.rcParams['axes.unicode_minus'] = False

    def save_plot_and_csv(self, fig: plt.Figure, df: pd.DataFrame, base_filename: str, output_dir: str):
        """Saves plot and corresponding CSV data."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Plot
        plot_path = os.path.join(output_dir, f"{base_filename}.png")
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {plot_path}")

        # Save CSV
        csv_path = os.path.join(output_dir, f"{base_filename}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV saved: {csv_path}")
        
        plt.close(fig)

    def plot_stockpile_reliability_comparison(self, traces: Dict[str, az.InferenceData], indices: Dict[str, Any], case_name: str, output_dir: str):
        """Plots weighted average reliability distribution for the entire stockpile."""
        fig, ax = plt.subplots(figsize=(12, 8))
        weights = indices["lot_quantities"]
        all_draws_df = []

        for name, trace in traces.items():
            posterior_samples = trace.posterior['reliability_lot'].values
            # Weighted average across lots for each draw
            weighted_mean_per_draw = np.average(posterior_samples, axis=-1, weights=weights).flatten()
            
            color = self.config.SCENARIO_COLORS.get(name, 'gray')
            sns.kdeplot(weighted_mean_per_draw, label=f'Scenario: {name}', fill=True, 
                        alpha=0.6, color=color, ax=ax)
            
            hdi_bounds = az.hdi(weighted_mean_per_draw, hdi_prob=0.9)
            ax.axvspan(hdi_bounds[0], hdi_bounds[1], color=color, alpha=0.1)
            
            all_draws_df.append(pd.DataFrame({
                'scenario': name,
                'weighted_mean_reliability': weighted_mean_per_draw
            }))

        ax.set_title(f'[{case_name}] Stockpile Weighted Average Reliability Distribution', fontsize=20, pad=20)
        # ax.axvline(x=self.config.TARGET_RELIABILITY, color='grey', linestyle='--', linewidth=1.5, label=f'Target ({self.config.TARGET_RELIABILITY})')
        ax.set_xlabel('Reliability', fontsize=15)
        ax.set_ylabel('Density', fontsize=15)
        ax.legend(title='Scenario', fontsize=13)
        # Auto-scale x-axis is default in matplotlib/seaborn unless set_xlim is called.
        fig.tight_layout()

        self.save_plot_and_csv(fig, pd.concat(all_draws_df), "stockpile_reliability_comparison", output_dir)

    def plot_forest(self, traces: Dict[str, az.InferenceData], indices: Dict[str, Any], case_name: str, output_dir: str):
        """Plots forest plot for LOT reliability."""
        fig, ax = plt.subplots(figsize=(12, 16))
        
        y_positions = np.arange(len(indices["all_lots"]))
        all_hdi_df = []

        for i, (name, trace) in enumerate(traces.items()):
            posterior_data = trace.posterior['reliability_lot']
            hdi_data = az.hdi(posterior_data, hdi_prob=0.9)['reliability_lot'].values * 100
            mean_data = posterior_data.mean(dim=['chain', 'draw']).values * 100
            
            y_offset = -0.15 + i * 0.3
            color = self.config.SCENARIO_COLORS.get(name, 'gray')
            
            ax.errorbar(x=mean_data, y=y_positions + y_offset, 
                        xerr=[mean_data - hdi_data[:, 0], hdi_data[:, 1] - mean_data],
                        fmt='none', elinewidth=1.5, capsize=4, 
                        color=color, alpha=0.7)
            
            ax.plot(mean_data, y_positions + y_offset, 'o', markersize=6, 
                    color=color, label=name)

            hdi_df = pd.DataFrame({
                'lot': indices["all_lots"],
                'mean': mean_data,
                'hdi_5%': hdi_data[:, 0],
                'hdi_95%': hdi_data[:, 1],
                'source': name
            })
            all_hdi_df.append(hdi_df)

        ax.set_yticks(ticks=y_positions, labels=indices["all_lots"], fontsize=10)
        ax.set_title(f'[{case_name}] LOT Reliability Estimation (90% HDI)', fontsize=20, pad=20)
        ax.set_xlabel('Reliability (%)', fontsize=15)
        ax.set_ylabel('Production LOT', fontsize=15)
        # ax.set_xlim(50, 100) # Removed fixed limit for auto-scaling
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        ax.legend(title='Scenario', fontsize=13)
        fig.tight_layout()

        self.save_plot_and_csv(fig, pd.concat(all_hdi_df), "lot_reliability_forest_plot", output_dir)

    def plot_posterior_distributions(self, traces: Dict[str, az.InferenceData], case_name: str, output_dir: str):
        """Plots posterior distributions of key parameters."""
        params_to_plot = {
            'mu_global_logit': 'Global Mean Reliability (Logit Scale)',
            'sigma_year': 'Inter-Year Sigma',
            'sigma_lot_base': 'Intra-Lot Base Sigma',
            'variance_degradation_rate': 'Variance Degradation Rate'
        }
        
        for param, title in params_to_plot.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            all_samples_df = []
            has_plot = False

            for name, trace in traces.items():
                if param in trace.posterior:
                    has_plot = True
                    samples = trace.posterior[param].values.flatten()
                    color = self.config.SCENARIO_COLORS.get(name, 'gray')
                    sns.kdeplot(samples, label=name, color=color, 
                                fill=True, alpha=0.6, ax=ax)
                    all_samples_df.append(pd.DataFrame({'scenario': name, 'parameter': param, 'value': samples}))

            if not has_plot:
                plt.close(fig)
                continue

            ax.set_title(f'[{case_name}] Posterior Distribution: {title}', fontsize=20, pad=20)
            ax.set_xlabel('Parameter Value', fontsize=15)
            ax.set_ylabel('Density', fontsize=15)
            ax.legend(title='Scenario', fontsize=13)
            fig.tight_layout()

            self.save_plot_and_csv(fig, pd.concat(all_samples_df), f"posterior_{param}", output_dir)
