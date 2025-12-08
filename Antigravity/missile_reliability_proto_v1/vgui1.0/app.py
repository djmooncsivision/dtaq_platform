import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add v1_code to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v1_code')))

from data_loader import DataLoader
from distribution_analyzer import DistributionAnalyzer
from trend_predictor import TrendPredictor
from visualizer import Visualizer

# Page Config
st.set_page_config(page_title="Reliability Analysis Tool vgui1.0", layout="wide")

st.title("🚀 Missile Reliability Analysis Tool (vgui1.0)")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")

# Data Directory Selection
default_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'v1_code', 'scenario_data'))
data_dir = st.sidebar.text_input("Data Directory", value=default_data_dir)

if not os.path.exists(data_dir):
    st.sidebar.error(f"Directory not found: {data_dir}")
    st.stop()

# Load Data (Cached)
@st.cache_data
def load_data(path):
    loader = DataLoader(path)
    loader.load_data()
    return loader

try:
    loader = load_data(data_dir)
    st.sidebar.success(f"Data Loaded: {len(loader.df)} rows")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

# Item Selection
items = [col for col in loader.df.columns if col.isdigit()] # Assuming items are '1', '2', ...
items.sort(key=int)
selected_item = st.sidebar.selectbox("Select Item to Analyze", items)

# Model Selection
st.sidebar.subheader("Trend Models")
model_options = ['Linear', 'Polynomial', 'Bayesian', 'GaussianProcess', 'SVR', 'NeuralNetwork']
selected_models = []
for model in model_options:
    if st.sidebar.checkbox(model, value=True):
        selected_models.append(model)

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Trend Prediction", "📋 Screening Summary"])

# Initialize Analyzers
qim_df, asrp_df = loader.split_data() # Note: split_data in v1 might need update for Overhaul if not handled
# Let's manually split for visualization to be safe and explicit
df = loader.df
qim_data = df[df['Dataset'] == 'QIM']
asrp_data = df[df['Dataset'] == 'ASRP']
overhaul_data = df[df['Dataset'] == 'Overhaul']

dist_analyzer = DistributionAnalyzer(qim_data, asrp_data)
trend_predictor = TrendPredictor(df)

# --- Tab 1: Data Overview ---
with tab1:
    st.header(f"Data Distribution: Item {selected_item}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("KDE Distribution Plot")
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=qim_data, x=selected_item, fill=True, label='QIM', color='blue', ax=ax_dist)
        sns.kdeplot(data=asrp_data, x=selected_item, fill=True, label='ASRP', color='red', ax=ax_dist)
        if not overhaul_data.empty:
            sns.kdeplot(data=overhaul_data, x=selected_item, fill=True, label='Overhaul', color='green', ax=ax_dist)
        ax_dist.set_title(f"Distribution Comparison (Item {selected_item})")
        ax_dist.legend()
        st.pyplot(fig_dist)
        
    with col2:
        st.subheader("Box Plot Summary")
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x='Dataset', y=selected_item, palette={'QIM': 'blue', 'ASRP': 'red', 'Overhaul': 'green'}, ax=ax_box)
        ax_box.set_title(f"Box Plot (Item {selected_item})")
        st.pyplot(fig_box)
        
    # Statistics
    st.subheader("Basic Statistics")
    stats_df = df.groupby('Dataset')[selected_item].describe()[['count', 'mean', 'std', 'min', 'max']]
    st.dataframe(stats_df)

# --- Tab 2: Trend Prediction ---
with tab2:
    st.header(f"Trend Prediction: Item {selected_item}")
    
    if st.button("Run Prediction"):
        with st.spinner("Training models..."):
            # Fit Models
            trend_predictor.fit_population_models(selected_item)
            
            # Predict
            future_months = np.linspace(0, 240, 100)
            predictions = trend_predictor.predict_population(selected_item, future_months)
            
            # Filter by selected models
            filtered_predictions = {k: v for k, v in predictions.items() if k in selected_models}
            
            # Plot
            st.subheader("Multi-Model Trend Graph")
            fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
            
            # Raw Data
            dataset_colors = {'QIM': 'blue', 'ASRP': 'red', 'Overhaul': 'green'}
            sns.scatterplot(data=df, x='운용월', y=selected_item, hue='Dataset', palette=dataset_colors, alpha=0.6, s=50, ax=ax_trend)
            
            # Predictions
            model_colors = {
                'Linear': 'gray', 'Polynomial': 'blue', 'Bayesian': 'purple',
                'GaussianProcess': 'green', 'SVR': 'orange', 'NeuralNetwork': 'red'
            }
            
            for name, (y_pred, lower, upper) in filtered_predictions.items():
                color = model_colors.get(name, 'black')
                ax_trend.plot(future_months, y_pred, label=name, color=color)
                ax_trend.fill_between(future_months, lower, upper, color=color, alpha=0.1)
                
            # Limits
            limits = loader.limits_df
            if limits is not None:
                item_limit = limits[limits['Item'] == selected_item]
                if not item_limit.empty:
                    usl = item_limit['USL'].values[0]
                    lsl = item_limit['LSL'].values[0]
                    if not np.isnan(usl):
                        ax_trend.axhline(y=usl, color='red', linestyle='--', label='USL')
                    if not np.isnan(lsl):
                        ax_trend.axhline(y=lsl, color='red', linestyle='--', label='LSL')
            
            ax_trend.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax_trend.grid(True, alpha=0.3)
            st.pyplot(fig_trend)
            
            # RMSE Comparison
            st.subheader("Model Performance (RMSE)")
            metrics = trend_predictor.model_metrics.get(selected_item, {})
            if metrics:
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Model', 'RMSE'])
                metrics_df = metrics_df.sort_values('RMSE')
                
                fig_rmse, ax_rmse = plt.subplots(figsize=(10, 5))
                sns.barplot(data=metrics_df, x='RMSE', y='Model', palette='viridis', ax=ax_rmse)
                for i, v in enumerate(metrics_df['RMSE']):
                    ax_rmse.text(v, i, f" {v:.4f}", va='center')
                st.pyplot(fig_rmse)
            else:
                st.warning("No metrics available.")

# --- Tab 3: Screening Summary ---
with tab3:
    st.header("Full Screening Summary")
    
    if st.button("Run Full Screening"):
        with st.spinner("Calculating trends for all items..."):
            screening_df = trend_predictor.calculate_all_trends(limits_df=loader.limits_df)
            st.dataframe(screening_df.style.highlight_max(axis=0, subset=['Norm_Slope', 'Var_Ratio'], color='pink'))
            
            # Highlight selected item
            st.info(f"Currently Selected Item: {selected_item}")
            selected_row = screening_df[screening_df['Item'] == selected_item]
            st.dataframe(selected_row)
