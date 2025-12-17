import os
import numpy as np
import pandas as pd
from data_loader import DataLoader
from distribution_analyzer import DistributionAnalyzer
from trend_predictor import TrendPredictor
from visualizer import Visualizer

class ReliabilityAnalysisTool:
    def __init__(self, data_path, output_dir="analysis_results"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.loader = DataLoader(data_path)
        self.visualizer = Visualizer(output_dir)
        
    def run(self):
        print("1. Loading Data...")
        self.loader.load_data()
        qim_df, asrp_df = self.loader.split_data()
        print(f"   - QIM: {len(qim_df)}, ASRP: {len(asrp_df)}")
        
        # Initialize Analyzers
        dist_analyzer = DistributionAnalyzer(qim_df, asrp_df)
        trend_predictor = TrendPredictor(self.loader.df)
        
        # --- Phase 1: Full Screening ---
        print("2. Performing Full Screening (All Items)...")
        screening_df = trend_predictor.calculate_all_trends()
        
        # Visualize Screening Summary
        screening_img = self.visualizer.plot_screening_summary(screening_df, top_n=10)
        print("   - Screening Complete. Top 5 Risks identified.")
        
        # Select Top 5 Items
        top_items = screening_df.head(5)['Item'].values
        
        # --- Phase 2: Detailed Analysis Loop ---
        item_reports = []
        
        for target_col in top_items:
            print(f"\n3. Analyzing Top Risk Item: {target_col}...")
            
            # Distribution Analysis
            dist_analyzer.perform_sampling(target_col)
            stats = dist_analyzer.compare_distributions(target_col)
            dist_img = self.visualizer.plot_distribution_comparison(
                qim_df[target_col], asrp_df[target_col], target_col
            )
            overview_img = self.visualizer.plot_data_overview(self.loader.df, target_col)
            
            # Trend Prediction
            future_months = np.linspace(0, 240, 100)
            pop_results = trend_predictor.predict_population(target_col, future_months)
            match_results = trend_predictor.predict_matching(target_col, future_months)
            all_predictions = {**pop_results, **match_results}
            
            trend_img = self.visualizer.plot_multi_model_prediction(
                self.loader.df, target_col, future_months, all_predictions
            )
            pairs_img = self.visualizer.plot_matched_pairs(
                self.loader.df, trend_predictor.matched_pairs, target_col
            )
            
            # Generate Section Report
            item_report = self.generate_item_section(target_col, stats, dist_img, overview_img, trend_img, pairs_img)
            item_reports.append(item_report)
            
        print("\n4. Generating Final Aggregated Report...")
        self.generate_final_report(screening_df, screening_img, item_reports)
        print("Done.")

    def generate_item_section(self, col, stats, dist_img, overview_img, trend_img, pairs_img):
        return f"""
## [Item {col}] 상세 분석

### 1. 데이터 분포 (Distribution)
*   **KS 통계량**: {stats['KS_Statistic']:.4f}
*   **JSD (유사도)**: {stats['JSD']:.4f}

| 분포 비교 (KDE) | 데이터 개요 (Box) |
| :---: | :---: |
| ![]({dist_img}) | ![]({overview_img}) |

### 2. 추세 예측 (Trend Prediction)
*   **예측 범위**: 20년 (240개월)
*   **신뢰 구간**: 90%

| 다중 모델 예측 | 개별 매칭 경로 |
| :---: | :---: |
| ![]({trend_img}) | ![]({pairs_img}) |

---
"""

    def generate_final_report(self, screening_df, screening_img, item_reports):
        # Convert screening table to markdown
        table_md = screening_df.head(10).to_markdown(index=False)
        
        # Automated Interpretation Logic
        top_risk = screening_df.iloc[0]
        top_3_items = screening_df.head(3)['Item'].tolist()
        
        interpretation = f"""
### 3.1 종합 해석 (Comprehensive Interpretation)
전체 27개 항목에 대한 분석 결과, 다음과 같은 경향성이 식별되었습니다.

1.  **주요 노후화 항목**: **{', '.join(top_3_items)}** 항목들이 시간 경과에 따라 가장 뚜렷한 변화를 보이고 있습니다.
    *   특히 **Item {top_risk['Item']}**은(는) 월평균 **{top_risk['Slope']:.4f}**의 변화율을 보이며, 가장 급격한 상태 변화가 관찰됩니다.
2.  **데이터 일관성**: 상위 위험 항목들의 R-squared 값이 전반적으로 높게 나타난다면, 이는 노후화가 무작위가 아닌 **예측 가능한 패턴**으로 진행됨을 의미합니다.
3.  **관리 제언**: 상기 Top-5 항목들은 향후 고장 발생의 선행 지표가 될 가능성이 높으므로, 예방 정비 시 **중점 점검 대상**으로 관리할 것을 권장합니다.
"""

        report = f"""# 신뢰도 분석 종합 리포트

## 1. 전수 스크리닝 (Full Screening)
전체 27개 측정 항목에 대해 변화율(Slope)과 변동성(Variance)을 분석하여 위험 순위를 도출했습니다.

### 1.1 위험 순위 Top 10 (Risk Ranking)
{table_md}

### 1.2 위험도 시각화
![Screening Summary]({screening_img})

## 2. Top-5 항목 정밀 분석
위험도가 가장 높은 상위 5개 항목에 대한 심층 분석 결과입니다.

{''.join(item_reports)}

## 3. 결론 및 제언
{interpretation}
"""
        with open(os.path.join(self.output_dir, 'final_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)

if __name__ == "__main__":
    tool = ReliabilityAnalysisTool("c:/Antigravity/missile_reliability_proto_v1/sample_reliability_data.csv", output_dir="v0_reproduction_output")
    tool.run()
