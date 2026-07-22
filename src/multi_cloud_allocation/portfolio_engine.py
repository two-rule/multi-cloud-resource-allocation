import numpy as np
import pandas as pd

from .paths import OUTPUT_DIR

class PortfolioEngine:
    def __init__(self, data_df, same_region_correlation=0.4):
        self._validate_data(data_df)
        try:
            correlation = float(same_region_correlation)
        except (TypeError, ValueError) as exc:
            raise ValueError("Same-region correlation must be a finite number.") from exc
        if not np.isfinite(correlation) or not 0 <= correlation <= 1:
            raise ValueError("Same-region correlation must be between 0 and 1.")

        self.df = data_df
        self.n = len(data_df)
        self.same_region_correlation = correlation
        self.cov_matrix = self._construct_covariance_matrix()
        self._validate_covariance_matrix()
        self.normalized_df = self._normalize_metrics()

    def _validate_data(self, data_df):
        if not isinstance(data_df, pd.DataFrame):
            raise TypeError("Portfolio data must be a pandas DataFrame.")

        required_columns = {
            'provider', 'region', 'hourly_cost', 'sla_risk', 'carbon_intensity'
        }
        missing_columns = sorted(required_columns - set(data_df.columns))
        if missing_columns:
            raise ValueError(
                "Portfolio data is missing required columns: "
                + ", ".join(missing_columns)
            )
        if data_df.empty:
            raise ValueError("Portfolio data must contain at least one asset.")
        if data_df[['provider', 'region']].isna().any().any():
            raise ValueError("Portfolio provider and region values must not be missing.")
        if data_df.duplicated(['provider', 'region']).any():
            raise ValueError("Portfolio assets must have unique provider-region pairs.")

        for column in ('hourly_cost', 'sla_risk', 'carbon_intensity'):
            try:
                values = data_df[column].to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Portfolio column '{column}' must be numeric."
                ) from exc
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(
                    f"Portfolio column '{column}' must contain finite positive values."
                )

    def _construct_covariance_matrix(self):
        """Construct the archived 0.4 same-region, zero cross-region covariance."""
        risks = self.df['sla_risk'].values
        corr_matrix = np.eye(self.n)

        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.df.iloc[i]['region'] == self.df.iloc[j]['region']:
                    corr_matrix[i, j] = self.same_region_correlation

        diag_risks = np.diag(risks)
        return diag_risks @ corr_matrix @ diag_risks

    def _validate_covariance_matrix(self):
        if not np.all(np.isfinite(self.cov_matrix)):
            raise ValueError("Covariance matrix must contain only finite values.")
        if not np.allclose(self.cov_matrix, self.cov_matrix.T):
            raise ValueError("Covariance matrix must be symmetric.")
        if np.linalg.eigvalsh(self.cov_matrix).min() < -1e-12:
            raise ValueError("Covariance matrix must be positive semidefinite.")

    def _validate_weights(self, weights):
        try:
            weights = np.asarray(weights, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Portfolio weights must be numeric.") from exc
        if weights.ndim != 1 or len(weights) != self.n:
            raise ValueError(f"Portfolio weights must contain exactly {self.n} values.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("Portfolio weights must contain only finite values.")
        return weights

    def get_covariance_matrix(self):
        return self.cov_matrix

    def save_covariance_matrix(self, filename=OUTPUT_DIR / 'covariance_matrix.csv'):
        asset_names = self.df['provider'] + "_" + self.df['region']

        cov_df = pd.DataFrame(
            self.cov_matrix,
            index=asset_names,
            columns=asset_names
        )

        cov_df.to_csv(filename)
        return cov_df

    def calculate_portfolio_risk(self, weights):
        weights = self._validate_weights(weights)
        variance = weights.T @ self.cov_matrix @ weights
        return np.sqrt(variance)
    
    def _normalize_metrics(self):
        norm_df = self.df.copy()

        for col in ['hourly_cost', 'sla_risk', 'carbon_intensity']:
            min_val = norm_df[col].min()
            max_val = norm_df[col].max()

            if max_val == min_val:
                norm_df[f'norm_{col}'] = 0.0
            else:
                norm_df[f'norm_{col}'] = (norm_df[col] - min_val) / (max_val - min_val)

        return norm_df
    
    def get_normalized_data(self):
        return self.normalized_df
    
    def calculate_total_metrics(self, weights):
        weights = self._validate_weights(weights)
        
        # 1. Total normalized cost
        total_cost = np.sum(weights * self.normalized_df['norm_hourly_cost'])
        
        # 2. Portfolio risk (MPT variance-covariance structure)
        # Risk is the standard deviation calculated from the covariance matrix.
        raw_risk = self.calculate_portfolio_risk(weights)
        total_risk = raw_risk / self.df['sla_risk'].max()
        
        # 3. Total normalized carbon
        total_carbon = np.sum(weights * self.normalized_df['norm_carbon_intensity'])
        
        return total_cost, total_risk, total_carbon
    
    def objective_function(self, weights, alpha=0.33, beta=0.33, gamma=0.34):
        cost, risk, carbon = self.calculate_total_metrics(weights)
        
        # Weighted aggregate score for Pareto optimality
        score = (alpha * cost) + (beta * risk) + (gamma * carbon)
        return score
