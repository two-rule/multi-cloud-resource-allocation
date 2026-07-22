from src.multi_cloud_allocation.data_loader import CloudDataLoader
from src.multi_cloud_allocation.portfolio_engine import PortfolioEngine
from src.multi_cloud_allocation.optimizer import CloudOptimizer, SCENARIO_WEIGHTS
from src.multi_cloud_allocation.paths import DATA_DIR, OUTPUT_DIR
from src.multi_cloud_allocation.visualizer import CloudVisualizer
import pandas as pd
import numpy as np

def run_full_thesis_analysis(show_plots=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- DATA PREPARATION ---
    print("\n" + "="*50)
    print("STEP 1: DATA INTEGRATION AND PREPROCESSING")
    print("="*50)
    loader = CloudDataLoader(DATA_DIR / 'cloud_pricing.csv')
    df = loader.merge_all_data()
    
    print("\n--- Sample Dataset (First 5 Rows) ---")
    print(df[['provider', 'region', 'hourly_cost', 'sla_risk', 'carbon_intensity']].head())

    # --- MATHEMATICAL ENGINE AND OPTIMIZATION SETUP ---
    engine = PortfolioEngine(df)
    engine.save_covariance_matrix(OUTPUT_DIR / 'covariance_matrix.csv')

    solver = CloudOptimizer(engine)
    
    # Scenarios (Alpha: Cost, Beta: Risk, Gamma: Carbon)
    scenarios = SCENARIO_WEIGHTS

    # --- SCENARIO ANALYSIS AND DECISION TABLE ---
    print("\n" + "="*50)
    print("STEP 2: SCENARIO ANALYSIS AND PORTFOLIO ALLOCATIONS")
    print("="*50)
    
    print("Dynamic Optimization Coefficients and Their Meanings:")
    print(f"{'Coefficient':<12} | {'Description'}")
    print("-" * 50)
    print(f"{'alpha (α)':<12} | Cost Minimization Priority")
    print(f"{'beta  (β)':<12} | Risk and SLA Reliability Priority")
    print(f"{'gamma (γ)':<12} | Carbon Footprint Minimization Priority")
    print("-" * 50)

    # Prepare the comparative decision table
    decision_table = df[['provider', 'region']].copy()
    decision_table['Asset'] = decision_table['provider'] + "_" + decision_table['region']
    
    score_summary = []
    scenario_allocations = {}

    for name, weights in scenarios.items():
        a, b, g = weights
        print(f">>> Running {name} Scenario (α={a}, β={b}, γ={g})...")
        
        # Run the solver
        optimal_w = solver.run_optimization(alpha=a, beta=b, gamma=g)
        scenario_allocations[name] = optimal_w
        
        # Calculate the metrics
        c, r, e = engine.calculate_total_metrics(optimal_w)
        score_summary.append({
            "Scenario": name,
            "Cost_Score": round(c, 4),
            "Risk_Score": round(r, 4),
            "Carbon_Score": round(e, 4)
        })
        
        # Add the weights to the table (percentage format)
        decision_table[name] = [f"%{round(w*100, 2)}" if w > 0.001 else "%0.0" for w in optimal_w]

    # Print the results
    print("\nTABLE 5.3: SCENARIO-BASED OPTIMAL WEIGHT ALLOCATIONS")
    print(decision_table.drop(columns=['provider', 'region']).to_string(index=False))

    print("\nTABLE 5.2: SCENARIO-BASED NORMALIZED PERFORMANCE SCORES")
    summary_df = pd.DataFrame(score_summary)
    print(summary_df.to_string(index=False))

    decision_table.to_csv(OUTPUT_DIR / 'scenario_allocation_results.csv', index=False)
    summary_df.to_csv(OUTPUT_DIR / 'scenario_performance_scores.csv', index=False)

    # --- VISUALIZATION (EFFICIENT FRONTIER) ---
    print("\n" + "="*50)
    print("STEP 3: EFFICIENT FRONTIER VISUALIZATION")
    print("="*50)
    visualizer = CloudVisualizer(engine, solver)
    visualizer.plot_scenario_allocations(
        scenario_allocations,
        output_path=OUTPUT_DIR / 'scenario_workload_allocation.png',
        show=show_plots
    )
    # 3,000 iterations provide a clear view of the solution space
    visualizer.plot_efficient_frontier(
        iterations=3000,
        seed=42,
        output_path=OUTPUT_DIR / 'efficient_frontier.png',
        show=show_plots
    )

if __name__ == "__main__":
    run_full_thesis_analysis()
