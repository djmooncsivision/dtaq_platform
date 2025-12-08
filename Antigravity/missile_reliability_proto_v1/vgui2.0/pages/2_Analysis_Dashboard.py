import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add v1_code to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'v1_code')))

from distribution_analyzer import DistributionAnalyzer
from trend_predictor import TrendPredictor
from visualizer import Visualizer

st.set_page_config(page_title="분석 대시보드", page_icon="📊", layout="wide")

st.title("📊 분석 대시보드 (Analysis Dashboard)")

# Check if data is loaded
if 'data_loader' not in st.session_state or st.session_state['data_loader'] is None:
    st.warning("⚠️ 데이터가 로드되지 않았습니다. '데이터 설정' 페이지로 이동하여 데이터를 로드하거나 생성해주세요.")
    st.stop()

loader = st.session_state['data_loader']
df = loader.df

# Sidebar Configuration
st.sidebar.header("분석 설정 (Analysis Settings)")

# Item Selection
items = [col for col in df.columns if col.isdigit()]
items.sort(key=int)
if items:
    selected_item = st.sidebar.selectbox("분석 항목 선택 (Select Item)", items)
else:
    st.error("데이터에 숫자형 항목 컬럼이 없습니다.")
    st.stop()

# Model Selection
st.sidebar.subheader("추세 예측 모델 (Trend Models)")
model_options = ['Linear', 'Polynomial', 'Bayesian', 'GaussianProcess', 'SVR', 'NeuralNetwork']
selected_models = []
for model in model_options:
    if st.sidebar.checkbox(model, value=True):
        selected_models.append(model)

# Initialize Analyzers
qim_df, asrp_df = loader.split_data()
overhaul_df = df[df['Dataset'] == 'Overhaul']

dist_analyzer = DistributionAnalyzer(qim_df, asrp_df)
trend_predictor = TrendPredictor(df)

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["📊 데이터 개요 (Data Overview)", "📈 추세 예측 (Trend Prediction)", "📋 전체 스크리닝 (Screening Summary)"])

# --- Tab 1: Data Overview ---
with tab1:
    st.header(f"데이터 분포: Item {selected_item}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("KDE 분포 그래프")
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        
        if not qim_df.empty:
            sns.kdeplot(data=qim_df, x=selected_item, fill=True, label='QIM (초기)', color='blue', ax=ax_dist)
        if not asrp_df.empty:
            sns.kdeplot(data=asrp_df, x=selected_item, fill=True, label='ASRP (저장)', color='red', ax=ax_dist)
        if not overhaul_df.empty:
            sns.kdeplot(data=overhaul_df, x=selected_item, fill=True, label='Overhaul (창정비)', color='green', ax=ax_dist)
            
        ax_dist.set_title(f"분포 비교 (Item {selected_item})")
        ax_dist.legend()
        st.pyplot(fig_dist)
        
    with col2:
        st.subheader("Box Plot 요약")
        fig_box, ax_box = plt.subplots(figsize=(8, 5))
        dataset_colors = {'QIM': 'blue', 'ASRP': 'red', 'Overhaul': 'green'}
        # Filter colors to only present datasets
        present_datasets = df['Dataset'].unique()
        palette = {k: v for k, v in dataset_colors.items() if k in present_datasets}
        
        sns.boxplot(data=df, x='Dataset', y=selected_item, palette=palette, ax=ax_box)
        ax_box.set_title(f"Box Plot (Item {selected_item})")
        st.pyplot(fig_box)
        
    # Statistics
    st.subheader("기초 통계량 (Basic Statistics)")
    stats_df = df.groupby('Dataset')[selected_item].describe()[['count', 'mean', 'std', 'min', 'max']]
    st.dataframe(stats_df)

# --- Tab 2: Trend Prediction ---
with tab2:
    st.header(f"추세 예측: Item {selected_item}")
    
    if st.button("예측 실행 (Run Prediction)"):
        with st.spinner("모델 학습 중..."):
            # Fit Models
            trend_predictor.fit_population_models(selected_item)
            
            # Predict
            future_months = np.linspace(0, 240, 100)
            predictions = trend_predictor.predict_population(selected_item, future_months)
            
            # Filter by selected models
            filtered_predictions = {k: v for k, v in predictions.items() if k in selected_models}
            
            # Plot
            st.subheader("다중 모델 추세 그래프")
            fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
            
            # Raw Data
            dataset_colors = {'QIM': 'blue', 'ASRP': 'red', 'Overhaul': 'green'}
            present_datasets = df['Dataset'].unique()
            palette = {k: v for k, v in dataset_colors.items() if k in present_datasets}
            
            sns.scatterplot(data=df, x='운용월', y=selected_item, hue='Dataset', palette=palette, alpha=0.6, s=50, ax=ax_trend)
            
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
            if hasattr(loader, 'limits_df') and loader.limits_df is not None:
                item_limit = loader.limits_df[loader.limits_df['Item'] == selected_item]
                if not item_limit.empty:
                    usl = item_limit['USL'].values[0]
                    lsl = item_limit['LSL'].values[0]
                    if not np.isnan(usl):
                        ax_trend.axhline(y=usl, color='red', linestyle='--', label='USL (상한)')
                    if not np.isnan(lsl):
                        ax_trend.axhline(y=lsl, color='red', linestyle='--', label='LSL (하한)')
            
            ax_trend.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax_trend.set_xlabel("운용월 (Months)")
            ax_trend.set_ylabel(f"측정값 (Item {selected_item})")
            ax_trend.grid(True, alpha=0.3)
            st.pyplot(fig_trend)
            
            # RMSE Comparison
            st.subheader("모델 성능 비교 (RMSE)")
            metrics = trend_predictor.model_metrics.get(selected_item, {})
            if metrics:
                metrics_df = pd.DataFrame(list(metrics.items()), columns=['Model', 'RMSE'])
                metrics_df = metrics_df.sort_values('RMSE')
                
                fig_rmse, ax_rmse = plt.subplots(figsize=(10, 5))
                sns.barplot(data=metrics_df, x='RMSE', y='Model', palette='viridis', ax=ax_rmse)
                for i, v in enumerate(metrics_df['RMSE']):
                    ax_rmse.text(v, i, f" {v:.4f}", va='center')
                ax_rmse.set_xlabel("RMSE (낮을수록 좋음)")
                st.pyplot(fig_rmse)
            else:
                st.warning("사용 가능한 성능 지표가 없습니다.")

# --- Tab 3: Screening Summary ---
with tab3:
    st.header("전체 항목 스크리닝 요약")
    
    screening_df = None
    
    if st.button("전체 스크리닝 실행"):
        with st.spinner("모든 항목에 대해 추세 분석 중..."):
            limits = getattr(loader, 'limits_df', None)
            screening_df = trend_predictor.calculate_all_trends(limits_df=limits)
            st.session_state['screening_df'] = screening_df # Cache for report
            
            st.dataframe(screening_df.style.highlight_max(axis=0, subset=['Norm_Slope', 'Var_Ratio'], color='pink'))
            
            # Highlight selected item
            st.info(f"현재 선택된 항목: {selected_item}")
            selected_row = screening_df[screening_df['Item'] == selected_item]
            st.dataframe(selected_row)
            
    # Check if cached
    if 'screening_df' in st.session_state:
        screening_df = st.session_state['screening_df']

# --- Report Generation Section ---
st.markdown("---")
st.header("📄 보고서 생성 (Report Generation)")

from report_generator import ReportGenerator
import io

if st.button("보고서 생성 준비"):
    if 'screening_df' not in st.session_state:
        st.warning("먼저 '전체 스크리닝 실행'을 완료해주세요.")
    else:
        with st.spinner("보고서 생성 중..."):
            # 1. Re-generate figures for report (to ensure they are captured)
            # We need to recreate them because Streamlit displays them but doesn't easily give us the object back unless we stored it.
            # For simplicity, we'll quickly recreate the plots here using the same logic.
            
            figures = {}
            
            # Dist Plot
            fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
            if not qim_df.empty: sns.kdeplot(data=qim_df, x=selected_item, fill=True, label='QIM', color='blue', ax=ax_dist)
            if not asrp_df.empty: sns.kdeplot(data=asrp_df, x=selected_item, fill=True, label='ASRP', color='red', ax=ax_dist)
            if not overhaul_df.empty: sns.kdeplot(data=overhaul_df, x=selected_item, fill=True, label='Overhaul', color='green', ax=ax_dist)
            ax_dist.set_title(f"Distribution Comparison (Item {selected_item})")
            ax_dist.legend()
            buf_dist = io.BytesIO()
            fig_dist.savefig(buf_dist, format='png')
            buf_dist.seek(0)
            figures['distribution_plot'] = buf_dist
            plt.close(fig_dist)
            
            # Box Plot
            fig_box, ax_box = plt.subplots(figsize=(8, 5))
            dataset_colors = {'QIM': 'blue', 'ASRP': 'red', 'Overhaul': 'green'}
            present_datasets = df['Dataset'].unique()
            palette = {k: v for k, v in dataset_colors.items() if k in present_datasets}
            sns.boxplot(data=df, x='Dataset', y=selected_item, palette=palette, ax=ax_box)
            ax_box.set_title(f"Box Plot (Item {selected_item})")
            buf_box = io.BytesIO()
            fig_box.savefig(buf_box, format='png')
            buf_box.seek(0)
            figures['box_plot'] = buf_box
            plt.close(fig_box)
            
            # Trend Plot
            # Need to re-run prediction logic if not cached, but let's assume user ran it or we run it now.
            trend_predictor.fit_population_models(selected_item)
            future_months = np.linspace(0, 240, 100)
            predictions = trend_predictor.predict_population(selected_item, future_months)
            filtered_predictions = {k: v for k, v in predictions.items() if k in selected_models}
            
            fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='운용월', y=selected_item, hue='Dataset', palette=palette, alpha=0.6, s=50, ax=ax_trend)
            model_colors = {'Linear': 'gray', 'Polynomial': 'blue', 'Bayesian': 'purple', 'GaussianProcess': 'green', 'SVR': 'orange', 'NeuralNetwork': 'red'}
            for name, (y_pred, lower, upper) in filtered_predictions.items():
                color = model_colors.get(name, 'black')
                ax_trend.plot(future_months, y_pred, label=name, color=color)
                ax_trend.fill_between(future_months, lower, upper, color=color, alpha=0.1)
            ax_trend.legend()
            buf_trend = io.BytesIO()
            fig_trend.savefig(buf_trend, format='png')
            buf_trend.seek(0)
            figures['trend_plot'] = buf_trend
            plt.close(fig_trend)
            
            # Metrics
            metrics = trend_predictor.model_metrics.get(selected_item, {})
            
            # Generate Report
            generator = ReportGenerator()
            docx_file = generator.generate_report(
                df, selected_item, stats_df, st.session_state['screening_df'], figures, metrics
            )
            
            st.download_button(
                label="📥 보고서 다운로드 (.docx)",
                data=docx_file,
                file_name=f"Reliability_Report_Item_{selected_item}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            st.success("보고서가 준비되었습니다! 위 버튼을 눌러 다운로드하세요.")
