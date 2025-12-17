import os
import argparse
from datetime import datetime
from reliability_analysis.config import Config
from reliability_analysis.data.loader import DataLoader
from reliability_analysis.models.bayesian import HierarchicalBayesianModel
from reliability_analysis.models.clopper_pearson import ClopperPearsonModel
from reliability_analysis.visualization.plots import Visualizer

def main():
    # 1. Setup
    config = Config()
    loader = DataLoader()
    visualizer = Visualizer()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 2. Iterate over all test cases
    for case_name, data_filename in config.TEST_CASES.items():
        print(f"\n{'='*30}")
        print(f"Starting Analysis: {case_name}")
        print(f"{'='*30}\n")

        clean_case_name = case_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_dir = os.path.join(config.OUTPUT_DIR, f"{clean_case_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        # 3. Load Data
        try:
            data, indices = loader.prepare_data_for_analysis(data_filename)
            print("Data loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        # 4. Run Scenarios (Hypergeometric)
        scenario_traces = {}
        for scenario_name, params in config.SCENARIOS.items():
            print(f"Running Scenario: {scenario_name}")
            model = HierarchicalBayesianModel(config)
            model.fit(
                data=data, 
                indices=indices, 
                model_type='hypergeometric',
                scenario_params=params,
                scenario_name=scenario_name,
                case_name=case_name
            )
            scenario_traces[scenario_name] = model.get_results()

        # 5. Visualize Scenario Comparison
        print("\nGenerating scenario comparison plots...")
        visualizer.plot_stockpile_reliability_comparison(scenario_traces, indices, case_name, output_dir)
        visualizer.plot_forest(scenario_traces, indices, case_name, output_dir)
        visualizer.plot_posterior_distributions(scenario_traces, case_name, output_dir)

        # 6. Model Comparison (Binomial vs Hypergeometric for Pessimistic)
        print("\nRunning Model Comparison (Binomial vs Hypergeometric)...")
        pessimistic_name = "Pessimistic"
        if pessimistic_name in config.SCENARIOS:
            pessimistic_params = config.SCENARIOS[pessimistic_name]
            
            # Run Binomial Model
            binomial_model = HierarchicalBayesianModel(config)
            binomial_model.fit(
                data=data,
                indices=indices,
                model_type='binomial',
                scenario_params=pessimistic_params,
                scenario_name=pessimistic_name,
                case_name=case_name
            )
            
            traces_to_compare = {
                "Binomial Model": binomial_model.get_results(),
                "Hypergeometric Model": scenario_traces[pessimistic_name]
            }
            
            visualizer.plot_forest(
                traces_to_compare, 
                indices, 
                f"{case_name} - Model Comparison (Pessimistic)", 
                output_dir
            )

        print(f"\nAnalysis for {case_name} complete. Check '{output_dir}' for results.")

if __name__ == "__main__":
    main()
