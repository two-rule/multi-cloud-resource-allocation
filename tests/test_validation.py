import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.multi_cloud_allocation.data_loader import CloudDataLoader
from src.multi_cloud_allocation.optimizer import CloudOptimizer
from src.multi_cloud_allocation.paths import DATA_DIR
from src.multi_cloud_allocation.portfolio_engine import PortfolioEngine


class DataValidationTests(unittest.TestCase):
    def _load_csv(self, contents):
        temporary_directory = tempfile.TemporaryDirectory(prefix="loader-validation-")
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "pricing.csv"
        path.write_text(contents, encoding="utf-8")
        return CloudDataLoader(path)

    def test_missing_file_is_rejected(self):
        loader = CloudDataLoader(DATA_DIR / "missing-pricing.csv")
        with self.assertRaisesRegex(FileNotFoundError, "was not found"):
            loader.load_pricing_data()

    def test_empty_data_is_rejected(self):
        loader = self._load_csv("provider,region,instance_type,hourly_cost\n")
        with self.assertRaisesRegex(ValueError, "at least one row"):
            loader.load_pricing_data()

        loader = self._load_csv("")
        with self.assertRaisesRegex(ValueError, "header and at least one row"):
            loader.load_pricing_data()

    def test_missing_columns_are_rejected(self):
        loader = self._load_csv("provider,region,hourly_cost\nAWS,US-East,0.096\n")
        with self.assertRaisesRegex(ValueError, "instance_type"):
            loader.load_pricing_data()

    def test_blank_identifier_is_rejected(self):
        loader = self._load_csv(
            "provider,region,instance_type,hourly_cost\n"
            ",US-East,m5.large,0.096\n"
        )
        with self.assertRaisesRegex(ValueError, "provider"):
            loader.load_pricing_data()

    def test_invalid_hourly_cost_is_rejected(self):
        for value in ("invalid", "NaN", "inf", "0", "-1"):
            with self.subTest(value=value):
                loader = self._load_csv(
                    "provider,region,instance_type,hourly_cost\n"
                    f"AWS,US-East,m5.large,{value}\n"
                )
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    loader.load_pricing_data()

    def test_duplicate_normalized_asset_is_rejected(self):
        loader = self._load_csv(
            "provider,region,instance_type,hourly_cost\n"
            "AWS,US-East,m5.large,0.096\n"
            " aws , us-east ,m5.xlarge,0.192\n"
        )
        with self.assertRaisesRegex(ValueError, "duplicate provider-region"):
            loader.load_pricing_data()

    def test_missing_carbon_mapping_is_rejected(self):
        loader = self._load_csv(
            "provider,region,instance_type,hourly_cost\n"
            "AWS,UNKNOWN,m5.large,0.096\n"
        )
        with self.assertRaisesRegex(ValueError, "Carbon intensity mappings"):
            loader.merge_all_data()

    def test_missing_risk_mapping_is_rejected(self):
        loader = self._load_csv(
            "provider,region,instance_type,hourly_cost\n"
            "UNKNOWN,US-East,m5.large,0.096\n"
        )
        with self.assertRaisesRegex(ValueError, "SLA risk mappings"):
            loader.merge_all_data()

    def test_getters_require_loaded_data(self):
        loader = CloudDataLoader()
        with self.assertRaisesRegex(RuntimeError, "has not been loaded"):
            loader.get_provider_list()
        with self.assertRaisesRegex(RuntimeError, "has not been loaded"):
            loader.get_region_list()


class ModelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = CloudDataLoader(DATA_DIR / "cloud_pricing.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.data = loader.merge_all_data()
        cls.engine = PortfolioEngine(cls.data)

    def test_invalid_correlation_is_rejected(self):
        for value in (-0.1, 1.1, np.nan, np.inf, "invalid"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "correlation"):
                    PortfolioEngine(self.data, same_region_correlation=value)

    def test_invalid_portfolio_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            PortfolioEngine(pd.DataFrame({"provider": ["AWS"]}))

        zero_risk = self.data.copy()
        zero_risk["sla_risk"] = 0.0
        with self.assertRaisesRegex(ValueError, "finite positive"):
            PortfolioEngine(zero_risk)

    def test_invalid_weight_vector_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 9"):
            self.engine.calculate_total_metrics(np.ones(8))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.engine.calculate_total_metrics(np.full(9, np.nan))

    def test_invalid_objective_coefficients_are_rejected(self):
        solver = CloudOptimizer(self.engine)
        invalid_values = (
            (-0.1, 0.5, 0.6),
            (0.4, 0.4, 0.4),
            (np.nan, 0.5, 0.5),
            ("invalid", 0.5, 0.5),
        )
        for coefficients in invalid_values:
            with self.subTest(coefficients=coefficients):
                with self.assertRaises(ValueError):
                    solver.run_optimization(*coefficients)

    def test_optimizer_failure_is_reported(self):
        solver = CloudOptimizer(self.engine)
        failed_result = SimpleNamespace(success=False, message="test failure")
        with patch(
            "src.multi_cloud_allocation.optimizer.minimize",
            return_value=failed_result,
        ):
            with self.assertRaisesRegex(ValueError, "test failure"):
                solver.run_optimization()

    def test_malformed_optimizer_result_is_rejected(self):
        solver = CloudOptimizer(self.engine)
        malformed_result = SimpleNamespace(
            success=True,
            x=np.zeros(self.engine.n),
            fun=0.0,
        )
        with patch(
            "src.multi_cloud_allocation.optimizer.minimize",
            return_value=malformed_result,
        ):
            with self.assertRaisesRegex(ValueError, "do not sum to 1"):
                solver.run_optimization()

        missing_objective = SimpleNamespace(
            success=True,
            x=np.full(self.engine.n, 1.0 / self.engine.n),
        )
        with patch(
            "src.multi_cloud_allocation.optimizer.minimize",
            return_value=missing_objective,
        ):
            with self.assertRaisesRegex(ValueError, "invalid objective value"):
                solver.run_optimization()


if __name__ == "__main__":
    unittest.main()
