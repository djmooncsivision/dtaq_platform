import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd

# Set font for Korean display
try:
    if os.name == 'nt':
        plt.rc('font', family='Malgun Gothic')
    else:
        plt.rc('font', family='AppleGothic')
    plt.rc('axes', unicode_minus=False)
except:
    pass

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_distribution_comparison(self, qim_data, asrp_data, col_name):
        """Plots overlapping KDE distributions."""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(qim_data, label='QIM (Initial)', fill=True, alpha=0.3)
        sns.kdeplot(asrp_data, label='ASRP (Aged)', fill=True, alpha=0.3)
        plt.title(f'Distribution Comparison: Item {col_name}')
        plt.xlabel('Measurement Value')
        plt.legend()
        
        filename = f'dist_comp_{col_name}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename

    def plot_data_overview(self, df, target_col):
        """Plots Box Plot with Strip Plot for clear data overview."""
        plt.figure(figsize=(10, 6))
        
        # Create a temporary grouping column
        plot_df = df.copy()
        plot_df['Group'] = plot_df['운용월'].apply(lambda x: 'QIM (0 Month)' if x == 0 else 'ASRP (>0 Month)')
        
        # Box Plot
        sns.boxplot(data=plot_df, x='Group', y=target_col, palette="Set2", width=0.5, showfliers=False)
        
        # Strip Plot (Raw Data Points)
        sns.stripplot(data=plot_df, x='Group', y=target_col, color='black', alpha=0.5, jitter=True)
        
        # Custom Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.5, label='Box: IQR (25-75%)'),
            Line2D([0], [0], marker='o', color='w', label='Points: Raw Data',
                   markerfacecolor='black', markersize=6, alpha=0.5)
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.title(f'Data Overview (Box + Scatter): Item {target_col}')
        plt.grid(True, axis='y', alpha=0.3)
        
        filename = f'data_overview_{target_col}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename

    def plot_multi_model_prediction(self, df, target_col, future_months, predictions):
        """Plots predictions from multiple models."""
        plt.figure(figsize=(14, 8))
        
        # Historical Data
        sns.scatterplot(data=df, x='운용월', y=target_col, color='gray', alpha=0.5, label='Historical Data')
        
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, (model_name, (y_pred, lower, upper)) in enumerate(predictions.items()):
            color = colors[i % len(colors)]
            plt.plot(future_months, y_pred, color=color, label=f'{model_name} Trend', linewidth=2)
            plt.fill_between(future_months, lower, upper, color=color, alpha=0.1, label=f'{model_name} 90% CI')
        
        plt.title(f'Multi-Model Trend Prediction: Item {target_col}')
        plt.xlabel('Operation Month')
        plt.ylabel('Measurement Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f'multi_trend_{target_col}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename

    def plot_matched_pairs(self, df, matched_pairs, target_col):
        """Visualizes the matching logic by connecting pairs."""
        plt.figure(figsize=(12, 6))
        
        # Plot all points
        sns.scatterplot(data=df, x='운용월', y=target_col, hue='운용월', palette='viridis', legend=False, alpha=0.6)
        
        # Draw lines for a subset of pairs (to avoid clutter)
        sample_pairs = matched_pairs.sample(min(20, len(matched_pairs)))
        
        for _, row in sample_pairs.iterrows():
            qim_val = df.loc[row['qim_idx'], target_col]
            asrp_val = df.loc[row['asrp_idx'], target_col]
            
            plt.plot([0, row['asrp_month']], [qim_val, asrp_val], 'k-', alpha=0.3, linewidth=1)
            
        plt.title(f'Matched Pairs Visualization (Sample): Item {target_col}')
        plt.xlabel('Operation Month')
        plt.ylabel('Measurement Value')
        
        filename = f'matched_pairs_{target_col}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename

    def plot_screening_summary(self, screening_df, top_n=10):
        """Plots a bar chart of the Top N items with the highest absolute slope."""
        plt.figure(figsize=(12, 6))
        
        top_df = screening_df.head(top_n)
        
        sns.barplot(data=top_df, x='Item', y='Abs_Slope', palette='Reds_r')
        
        plt.title(f'Top {top_n} High-Risk Items (by Rate of Change)')
        plt.xlabel('Measurement Item')
        plt.ylabel('Absolute Slope (Magnitude of Change)')
        plt.grid(True, axis='y', alpha=0.3)
        
        filename = 'screening_summary.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename
