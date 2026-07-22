import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimizer import SCENARIO_WEIGHTS
from .paths import OUTPUT_DIR

class CloudVisualizer:
    def __init__(self, engine, solver):
        self.engine = engine
        self.solver = solver

    def _find_pareto_frontier(self, results_df):
        values = results_df[['Cost', 'Risk', 'Carbon']].values
        is_pareto = np.ones(values.shape[0], dtype=bool)

        for i, current in enumerate(values):
            if is_pareto[i]:
                dominated_by_others = (
                    np.all(values <= current, axis=1) &
                    np.any(values < current, axis=1)
                )

                if np.any(dominated_by_others):
                    is_pareto[i] = False

        return results_df[is_pareto].sort_values(by='Cost')

    def _simulate_portfolios(self, iterations, seed):
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("Iterations must be a positive integer.")

        rng = np.random.default_rng(seed)
        results = []
        print(f"\nSimulating {iterations} random portfolios...")

        for _ in range(iterations):
            # Generate random weights and normalize them (sum=1)
            w = rng.random(self.engine.n)
            w /= np.sum(w)

            cost, risk, carbon = self.engine.calculate_total_metrics(w)
            results.append([cost, risk, carbon])

        return pd.DataFrame(results, columns=['Cost', 'Risk', 'Carbon'])

    def plot_scenario_allocations(
        self,
        scenario_allocations,
        output_path=OUTPUT_DIR / 'scenario_workload_allocation.png',
        show=True
    ):
        scenario_names = list(SCENARIO_WEIGHTS)
        missing_scenarios = [
            name for name in scenario_names if name not in scenario_allocations
        ]
        if missing_scenarios:
            raise ValueError(
                "Scenario allocations are missing: " + ", ".join(missing_scenarios)
            )

        allocation_matrix = np.asarray(
            [scenario_allocations[name] for name in scenario_names],
            dtype=float
        )
        expected_shape = (len(scenario_names), self.engine.n)
        if allocation_matrix.shape != expected_shape:
            raise ValueError(
                f"Scenario allocations must have shape {expected_shape}."
            )
        if not np.all(np.isfinite(allocation_matrix)):
            raise ValueError("Scenario allocations must contain only finite values.")
        if np.any(allocation_matrix < -1e-9) or np.any(allocation_matrix > 1 + 1e-9):
            raise ValueError("Scenario allocations must be between 0 and 1.")
        if not np.allclose(allocation_matrix.sum(axis=1), 1.0, rtol=0, atol=1e-8):
            raise ValueError("Each scenario allocation must sum to 1.")

        asset_names = (self.engine.df['provider'] + "_" + self.engine.df['region']).tolist()
        plt.figure(figsize=(12, 7))
        left = np.zeros(len(scenario_names))

        for asset_index, asset_name in enumerate(asset_names):
            percentages = allocation_matrix[:, asset_index] * 100
            plt.barh(
                scenario_names,
                percentages,
                left=left,
                label=asset_name
            )
            left += percentages

        plt.title('Scenario-Based Workload Allocation Percentages', fontsize=14)
        plt.xlabel('Workload Allocation (%)', fontsize=12)
        plt.xlim(0, 100)
        plt.gca().invert_yaxis()
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.14),
            ncol=3
        )
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show and 'agg' not in plt.get_backend().lower():
            plt.show()
        plt.close()
        print(f"Scenario allocation plot saved as '{output_path}'.")

    def plot_efficient_frontier(
        self,
        iterations=2000,
        seed=42,
        output_path=OUTPUT_DIR / 'efficient_frontier.png',
        show=True
    ):
        results_df = self._simulate_portfolios(iterations, seed)
        pareto_df = self._find_pareto_frontier(results_df)

        # Plot
        plt.figure(figsize=(12, 8))
        
        # 1. All feasible solutions (cloud of portfolios)
        # The color scale represents carbon intensity
        scatter = plt.scatter(
            results_df['Cost'],
            results_df['Risk'],
            c=results_df['Carbon'],
            cmap='viridis_r',
            alpha=0.4,
            s=10,
            label='Feasible Portfolios'
        )

        plt.scatter(
            pareto_df['Cost'],
            pareto_df['Risk'],
            color='black',
            s=12,
            label='Approximate Pareto-Optimal Portfolios',
            zorder=4
        )
        
        plt.colorbar(scatter, label='Normalized Carbon Intensity')

        # 2. Add custom scenario points
        scenario_styles = {
            "Balanced": ('red', '*'),
            "Cost-Oriented": ('blue', 'D'),
            "Risk-Oriented": ('orange', 'X'),
            "Green-Oriented": ('green', 's')
        }

        for name, (color, marker) in scenario_styles.items():
            a, b, g = SCENARIO_WEIGHTS[name]
            best_w = self.solver.run_optimization(alpha=a, beta=b, gamma=g)
            c, r, e = self.engine.calculate_total_metrics(best_w)
            plt.scatter(c, r, color=color, marker=marker, s=200, 
                        edgecolor='black', label=f'Optimal: {name}', zorder=5)

        plt.title('Efficient Frontier and Scenario-Based Optima', fontsize=14)
        plt.xlabel('Normalized Operational Cost', fontsize=12)
        plt.ylabel('Normalized Portfolio Risk (SLA Violation)', fontsize=12)
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=3
        )
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save the file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if show and 'agg' not in plt.get_backend().lower():
            plt.show()
        plt.close()
        print(f"Efficient frontier plot saved as '{output_path}'.")
