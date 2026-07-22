import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from src.multi_cloud_allocation.data_loader import CloudDataLoader
from src.multi_cloud_allocation.optimizer import CloudOptimizer, SCENARIO_WEIGHTS
from src.multi_cloud_allocation.paths import DATA_DIR, OUTPUT_DIR
from src.multi_cloud_allocation.portfolio_engine import PortfolioEngine


SCENARIOS = {
    "Cost-Oriented": (0.80, 0.10, 0.10),
    "Risk-Oriented": (0.10, 0.80, 0.10),
    "Green-Oriented": (0.10, 0.10, 0.80),
    "Balanced": (0.33, 0.33, 0.34),
}

EXPECTED_SCORES = {
    "Cost-Oriented": (0.0000, 0.4520, 0.6250),
    "Risk-Oriented": (0.2845, 0.2590, 0.2624),
    "Green-Oriented": (0.3463, 0.3594, 0.0000),
    "Balanced": (0.3135, 0.3284, 0.0589),
}

EXPECTED_ALLOCATIONS = {
    "Cost-Oriented": [
        "%64.86", "%0.0", "%0.0", "%35.14", "%0.0",
        "%0.0", "%0.0", "%0.0", "%0.0",
    ],
    "Risk-Oriented": [
        "%19.22", "%33.15", "%4.46", "%10.04", "%19.47",
        "%1.52", "%2.97", "%9.06", "%0.12",
    ],
    "Green-Oriented": [
        "%0.0", "%74.24", "%0.0", "%0.0", "%25.76",
        "%0.0", "%0.0", "%0.0", "%0.0",
    ],
    "Balanced": [
        "%6.12", "%67.35", "%0.0", "%3.31", "%23.22",
        "%0.0", "%0.0", "%0.0", "%0.0",
    ],
}

EXPECTED_ASSETS = [
    "AWS_US-EAST",
    "AWS_EU-WEST",
    "AWS_ASIA-PACIFIC",
    "AZURE_US-EAST",
    "AZURE_EU-WEST",
    "AZURE_ASIA-PACIFIC",
    "GCP_US-EAST",
    "GCP_EU-WEST",
    "GCP_ASIA-PACIFIC",
]


class ThesisRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = CloudDataLoader(DATA_DIR / "cloud_pricing.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.data = loader.merge_all_data()
        cls.engine = PortfolioEngine(cls.data)
        cls.solver = CloudOptimizer(cls.engine)

    def test_table_5_2_performance_scores(self):
        for name, coefficients in SCENARIOS.items():
            with self.subTest(scenario=name):
                weights = self.solver.run_optimization(*coefficients)
                scores = self.engine.calculate_total_metrics(weights)
                self.assertEqual(
                    tuple(round(float(score), 4) for score in scores),
                    EXPECTED_SCORES[name],
                )

    def test_table_5_3_workload_allocations(self):
        for name, coefficients in SCENARIOS.items():
            with self.subTest(scenario=name):
                weights = self.solver.run_optimization(*coefficients)
                allocations = [
                    f"%{round(weight * 100, 2)}" if weight > 0.001 else "%0.0"
                    for weight in weights
                ]
                self.assertEqual(allocations, EXPECTED_ALLOCATIONS[name])
                self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=8)
                self.assertTrue(np.all(weights >= -1e-9))
                self.assertTrue(np.all(weights <= 1.0 + 1e-9))

    def test_covariance_matrix_matches_archived_behavior(self):
        risks = self.data["sla_risk"].to_numpy()
        expected = np.zeros((len(self.data), len(self.data)))

        for i in range(len(self.data)):
            for j in range(len(self.data)):
                if i == j:
                    correlation = 1.0
                elif self.data.iloc[i]["region"] == self.data.iloc[j]["region"]:
                    correlation = 0.4
                else:
                    correlation = 0.0
                expected[i, j] = correlation * risks[i] * risks[j]

        covariance = self.engine.get_covariance_matrix()
        np.testing.assert_allclose(covariance, expected, rtol=0, atol=1e-18)
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-18)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-12)

    def test_checked_in_covariance_artifact(self):
        covariance = pd.read_csv(OUTPUT_DIR / "covariance_matrix.csv", index_col=0)
        self.assertEqual(covariance.index.tolist(), EXPECTED_ASSETS)
        self.assertEqual(covariance.columns.tolist(), EXPECTED_ASSETS)
        np.testing.assert_allclose(
            covariance.to_numpy(),
            self.engine.get_covariance_matrix(),
            rtol=0,
            atol=1e-18,
        )

    def test_checked_in_scenario_artifacts(self):
        allocations = pd.read_csv(
            OUTPUT_DIR / "scenario_allocation_results.csv",
            dtype=str,
        )
        self.assertEqual(allocations["Asset"].tolist(), EXPECTED_ASSETS)
        for scenario, expected in EXPECTED_ALLOCATIONS.items():
            self.assertEqual(allocations[scenario].tolist(), expected)

        scores = pd.read_csv(OUTPUT_DIR / "scenario_performance_scores.csv")
        for scenario, expected in EXPECTED_SCORES.items():
            actual = scores.loc[
                scores["Scenario"] == scenario,
                ["Cost_Score", "Risk_Score", "Carbon_Score"],
            ].iloc[0]
            self.assertEqual(tuple(float(value) for value in actual), expected)

    def test_objective_is_weighted_sum_of_metrics(self):
        weights = np.full(self.engine.n, 1.0 / self.engine.n)
        coefficients = SCENARIOS["Balanced"]
        metrics = self.engine.calculate_total_metrics(weights)
        expected = sum(
            coefficient * metric
            for coefficient, metric in zip(coefficients, metrics)
        )
        actual = self.engine.objective_function(weights, *coefficients)
        self.assertAlmostEqual(actual, expected, places=15)

    def test_scenario_weights_are_canonical(self):
        self.assertEqual(SCENARIO_WEIGHTS, SCENARIOS)

        from main import SCENARIO_WEIGHTS as main_scenarios
        from src.multi_cloud_allocation.visualizer import (
            SCENARIO_WEIGHTS as visualizer_scenarios,
        )

        self.assertIs(main_scenarios, SCENARIO_WEIGHTS)
        self.assertIs(visualizer_scenarios, SCENARIO_WEIGHTS)


if __name__ == "__main__":
    unittest.main()
