import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.multi_cloud_allocation.data_loader import CloudDataLoader
from src.multi_cloud_allocation.optimizer import CloudOptimizer, SCENARIO_WEIGHTS
from src.multi_cloud_allocation.paths import DATA_DIR, OUTPUT_DIR
from src.multi_cloud_allocation.portfolio_engine import PortfolioEngine
from src.multi_cloud_allocation.visualizer import CloudVisualizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = CloudDataLoader(DATA_DIR / "cloud_pricing.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            data = loader.merge_all_data()
        cls.engine = PortfolioEngine(data)
        cls.solver = CloudOptimizer(cls.engine)
        cls.visualizer = CloudVisualizer(cls.engine, cls.solver)

    def test_seeded_simulation_is_deterministic(self):
        with contextlib.redirect_stdout(io.StringIO()):
            first = self.visualizer._simulate_portfolios(25, seed=42)
            second = self.visualizer._simulate_portfolios(25, seed=42)
        pd.testing.assert_frame_equal(first, second)

        rng = np.random.default_rng(42)
        weights = rng.random(self.engine.n)
        weights /= np.sum(weights)
        expected_first_row = self.engine.calculate_total_metrics(weights)
        np.testing.assert_allclose(
            first.iloc[0].to_numpy(),
            expected_first_row,
            rtol=0,
            atol=1e-15,
        )

    def test_invalid_iteration_count_is_rejected(self):
        for iterations in (0, -1, 1.5):
            with self.subTest(iterations=iterations):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    self.visualizer._simulate_portfolios(iterations, seed=42)

    def test_both_figures_are_generated_from_canonical_scenarios(self):
        allocations = {
            name: self.solver.run_optimization(*coefficients)
            for name, coefficients in SCENARIO_WEIGHTS.items()
        }

        with tempfile.TemporaryDirectory(prefix="figure-generation-") as temp_name:
            allocation_path = Path(temp_name) / "scenario_workload_allocation.png"
            frontier_path = Path(temp_name) / "efficient_frontier.png"

            with contextlib.redirect_stdout(io.StringIO()):
                self.visualizer.plot_scenario_allocations(
                    allocations,
                    output_path=allocation_path,
                    show=False,
                )
                with patch.object(
                    self.solver,
                    "run_optimization",
                    wraps=self.solver.run_optimization,
                ) as run_optimization:
                    self.visualizer.plot_efficient_frontier(
                        iterations=25,
                        seed=42,
                        output_path=frontier_path,
                        show=False,
                    )

            run_optimization.assert_any_call(alpha=0.33, beta=0.33, gamma=0.34)
            self.assertGreater(allocation_path.stat().st_size, 0)
            self.assertGreater(frontier_path.stat().st_size, 0)

    def test_missing_scenario_allocation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            self.visualizer.plot_scenario_allocations({}, show=False)


class EntrypointIntegrationTests(unittest.TestCase):
    def test_entrypoint_runs_from_parent_directory(self):
        with tempfile.TemporaryDirectory(prefix="entrypoint-integration-") as temp_name:
            temp_root = Path(temp_name)
            project_copy = temp_root / "project"
            project_copy.mkdir()

            shutil.copy2(PROJECT_ROOT / "main.py", project_copy / "main.py")
            shutil.copytree(
                PROJECT_ROOT / "src",
                project_copy / "src",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            shutil.copytree(PROJECT_ROOT / "data", project_copy / "data")

            environment = os.environ.copy()
            environment.update(
                MPLBACKEND="Agg",
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONIOENCODING="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, str(project_copy / "main.py")],
                cwd=temp_root,
                env=environment,
                text=True,
                capture_output=True,
            )
            if completed.returncode:
                self.fail(
                    f"Entrypoint failed:\n{completed.stdout}\n{completed.stderr}"
                )

            for name in (
                "covariance_matrix.csv",
                "scenario_allocation_results.csv",
                "scenario_performance_scores.csv",
            ):
                self.assertEqual(
                    (project_copy / "outputs" / name).read_bytes(),
                    (OUTPUT_DIR / name).read_bytes(),
                )
                self.assertFalse((project_copy / name).exists())

            for name in (
                "scenario_workload_allocation.png",
                "efficient_frontier.png",
            ):
                self.assertGreater(
                    (project_copy / "outputs" / name).stat().st_size,
                    0,
                )
                self.assertFalse((project_copy / name).exists())


if __name__ == "__main__":
    unittest.main()
