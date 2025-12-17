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

    def plot_distribution_comparison(self, qim_data, asrp_data, overhaul_data, col_name, limits=None):
        """Plots overlapping KDE distributions for QIM, ASRP, and Overhaul."""
        plt.figure(figsize=(10, 6))
        
        # Plot KDEs
        if not qim_data.empty:
            sns.kdeplot(qim_data, label='QIM (Initial)', fill=True, alpha=0.3, color='blue')
        if not asrp_data.empty:
            sns.kdeplot(asrp_data, label='ASRP (Aged)', fill=True, alpha=0.3, color='orange')
        if not overhaul_data.empty:
            sns.kdeplot(overhaul_data, label='Overhaul (Maintenance)', fill=True, alpha=0.3, color='green')
            
        # Plot Limits
        if limits is not None:
            item_limit = limits[limits['Item'] == col_name]
            if not item_limit.empty:
                usl = item_limit['USL'].values[0]
                lsl = item_limit['LSL'].values[0]
                if not np.isnan(usl):
                    plt.axvline(x=usl, color='red', linestyle='--', linewidth=2, label='Upper Limit (USL)')
                if not np.isnan(lsl):
                    plt.axvline(x=lsl, color='red', linestyle='--', linewidth=2, label='Lower Limit (LSL)')

        plt.title(f'Distribution Comparison (3 Groups): Item {col_name}')
        plt.xlabel('Measurement Value')
        plt.legend()
        
        filename = f'dist_comp_{col_name}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename

    def plot_data_overview(self, df, target_col, limits=None):
        """Plots Box Plot with Strip Plot for QIM, ASRP, Overhaul."""
        plt.figure(figsize=(10, 6))
        
        # Use 'Dataset' column for grouping if available, else derive
        if 'Dataset' in df.columns:
            plot_df = df.copy()
            # Ensure order: QIM -> ASRP -> Overhaul
            order = ['QIM', 'ASRP', 'Overhaul']
        else:
            plot_df = df.copy()
            plot_df['Dataset'] = plot_df['운용월'].apply(lambda x: 'QIM' if x == 0 else 'ASRP')
            order = ['QIM', 'ASRP']
        
        # Box Plot
        sns.boxplot(data=plot_df, x='Dataset', y=target_col, order=order, palette="Set2", width=0.5, showfliers=False)
        
        # Strip Plot
        sns.stripplot(data=plot_df, x='Dataset', y=target_col, order=order, color='black', alpha=0.5, jitter=True)
        
        # Plot Limits
        if limits is not None:
            item_limit = limits[limits['Item'] == target_col]
            if not item_limit.empty:
                usl = item_limit['USL'].values[0]
                lsl = item_limit['LSL'].values[0]
                if not np.isnan(usl):
                    plt.axhline(y=usl, color='red', linestyle='--', linewidth=2, label='Upper Limit (USL)')
                if not np.isnan(lsl):
                    plt.axhline(y=lsl, color='red', linestyle='--', linewidth=2, label='Lower Limit (LSL)')

        # Custom Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.5, label='Box: IQR (25-75%)'),
            Line2D([0], [0], marker='o', color='w', label='Points: Raw Data',
                   markerfacecolor='black', markersize=6, alpha=0.5)
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.title(f'Data Overview (3 Groups): Item {target_col}')
        plt.grid(True, axis='y', alpha=0.3)
        
        filename = f'data_overview_{target_col}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename

    def plot_multi_model_prediction(self, df, target_col, future_months, predictions, limits=None):
        """Plots trend predictions from multiple models (Linear, Poly, Bayesian, GPR, SVR, NN)."""
        plt.figure(figsize=(12, 6))
        
        # Plot Raw Data with Colors by Dataset
        # QIM: Blue, ASRP: Red, Overhaul: Green
        dataset_colors = {'QIM': 'blue', 'ASRP': 'red', 'Overhaul': 'green'}
        sns.scatterplot(data=df, x='운용월', y=target_col, hue='Dataset', palette=dataset_colors, alpha=0.6, s=50)
        
        # Plot Predictions
        colors = {
            'Linear': 'gray',
            'Polynomial': 'blue',
            'Bayesian': 'purple',
            'GaussianProcess': 'green',
            'SVR': 'orange',
            'NeuralNetwork': 'red',
            'Linear_Matching': 'cyan', # From matching
            'RF_Matching': 'magenta'   # From matching
        }
        
        for model_name, (y_pred, lower, upper) in predictions.items():
            color = colors.get(model_name, 'gray')
            # Plot line
            plt.plot(future_months, y_pred, label=f'{model_name} Trend', color=color, linewidth=2)
            # Plot CI (lighter shade)
            plt.fill_between(future_months, lower, upper, color=color, alpha=0.1)
            
        # Plot Limits
        if limits is not None:
            item_limit = limits[limits['Item'] == target_col]
            if not item_limit.empty:
                usl = item_limit['USL'].values[0]
                lsl = item_limit['LSL'].values[0]
                if not np.isnan(usl):
                    plt.axhline(y=usl, color='red', linestyle='--', linewidth=2, label='USL')
                if not np.isnan(lsl):
                    plt.axhline(y=lsl, color='red', linestyle='--', linewidth=2, label='LSL')

        plt.title(f'Multi-Model Trend Prediction: Item {target_col}')
        plt.xlabel('Operation Months')
        plt.ylabel('Measurement Value')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Move legend outside
        plt.grid(True, alpha=0.3)
        plt.tight_layout() # Adjust layout for outside legend
        
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
        """Plots a bar chart of the Top N items with the highest Normalized Slope."""
        plt.figure(figsize=(12, 6))
        
        top_df = screening_df.head(top_n)
        
        # Check if Norm_Slope exists (backward compatibility)
        if 'Norm_Slope' in top_df.columns and top_df['Norm_Slope'].sum() > 0:
            y_col = 'Norm_Slope'
            y_label = 'Normalized Slope (% of Spec Range / Month)'
            title = f'Top {top_n} High-Risk Items (Normalized by Spec Range)'
        else:
            y_col = 'Abs_Slope'
            y_label = 'Absolute Slope (Magnitude of Change)'
            title = f'Top {top_n} High-Risk Items (by Absolute Rate of Change)'
        
        sns.barplot(data=top_df, x='Item', y=y_col, palette='Reds_r')
        
        plt.title(title)
        plt.xlabel('Measurement Item')
        plt.ylabel(y_label)
        plt.grid(True, axis='y', alpha=0.3)
        
        filename = 'screening_summary.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename

    def plot_model_performance(self, rmse_metrics, target_col):
        """Plots a bar chart comparing RMSE of different models."""
        plt.figure(figsize=(10, 6))
        
        models = list(rmse_metrics.keys())
        rmse_values = list(rmse_metrics.values())
        
        # Sort by RMSE (lower is better)
        sorted_indices = np.argsort(rmse_values)
        models = [models[i] for i in sorted_indices]
        rmse_values = [rmse_values[i] for i in sorted_indices]
        
        sns.barplot(x=rmse_values, y=models, palette='viridis')
        
        plt.title(f'Model Performance Comparison (RMSE): Item {target_col}')
        plt.xlabel('RMSE (Lower is Better)')
        plt.ylabel('Model')
        plt.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(rmse_values):
            plt.text(v, i, f' {v:.4f}', va='center')
            
        filename = f'model_performance_{target_col}.png'
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        return filename
