from pathlib import Path

import pandas as pd
import numpy as np

from .paths import DATA_DIR

REQUIRED_PRICING_COLUMNS = ('provider', 'region', 'instance_type', 'hourly_cost')

class CloudDataLoader:
    def __init__(self, pricing_file=None):
        self.pricing_file = (
            DATA_DIR / 'cloud_pricing.csv'
            if pricing_file is None
            else Path(pricing_file)
        )
        self.raw_data = None
        
        self.carbon_mapping = {
            'US-EAST': 450,
            'EU-WEST': 200,
            'ASIA-PACIFIC': 600
        }
        self.carbon_api_mock = self.carbon_mapping
        
        self.risk_mapping = {
            'AWS':   {'US-EAST': 0.0010, 'EU-WEST': 0.0008, 'ASIA-PACIFIC': 0.0015},
            'AZURE': {'US-EAST': 0.0012, 'EU-WEST': 0.0009, 'ASIA-PACIFIC': 0.0018},
            'GCP':   {'US-EAST': 0.0015, 'EU-WEST': 0.0010, 'ASIA-PACIFIC': 0.0020}
        }
        
    def load_pricing_data(self):
        if not self.pricing_file.is_file():
            raise FileNotFoundError(f"{self.pricing_file} was not found.")
        try:
            self.raw_data = pd.read_csv(self.pricing_file)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                "Pricing data must contain a header and at least one row."
            ) from exc
        self._validate_pricing_data()
        self.preprocess_names()
        self._validate_unique_assets()
        return self.raw_data

    def _validate_pricing_data(self):
        missing_columns = [
            column for column in REQUIRED_PRICING_COLUMNS
            if column not in self.raw_data.columns
        ]
        if missing_columns:
            raise ValueError(
                "Pricing data is missing required columns: "
                + ", ".join(missing_columns)
            )
        if self.raw_data.empty:
            raise ValueError("Pricing data must contain at least one row.")

        for column in ('provider', 'region', 'instance_type'):
            invalid = ~self.raw_data[column].map(
                lambda value: isinstance(value, str) and bool(value.strip())
            )
            if invalid.any():
                rows = (self.raw_data.index[invalid] + 2).tolist()
                raise ValueError(
                    f"Column '{column}' contains missing or blank values "
                    f"at CSV rows: {rows}"
                )

        numeric_cost = pd.to_numeric(self.raw_data['hourly_cost'], errors='coerce')
        invalid_cost = numeric_cost.isna() | ~np.isfinite(numeric_cost) | (numeric_cost <= 0)
        if invalid_cost.any():
            rows = (self.raw_data.index[invalid_cost] + 2).tolist()
            raise ValueError(
                "Column 'hourly_cost' must contain finite positive numbers "
                f"at CSV rows: {rows}"
            )
        self.raw_data['hourly_cost'] = numeric_cost

    def preprocess_names(self):
        if self.raw_data is not None:
            self.raw_data['provider'] = self.raw_data['provider'].str.strip().str.upper()
            self.raw_data['region'] = self.raw_data['region'].str.strip().str.upper()

    def _validate_unique_assets(self):
        duplicates = self.raw_data.duplicated(['provider', 'region'], keep=False)
        if duplicates.any():
            assets = sorted(
                set(
                    self.raw_data.loc[duplicates, 'provider']
                    + "_"
                    + self.raw_data.loc[duplicates, 'region']
                )
            )
            raise ValueError(
                "Pricing data contains duplicate provider-region assets: "
                + ", ".join(assets)
            )

    def _validate_mappings(self):
        missing_carbon = sorted(
            set(self.raw_data['region']) - set(self.carbon_mapping)
        )
        if missing_carbon:
            raise ValueError(
                "Carbon intensity mappings are missing for regions: "
                + ", ".join(missing_carbon)
            )

        missing_risk = sorted(
            {
                f"{row.provider}_{row.region}"
                for row in self.raw_data.itertuples()
                if row.provider not in self.risk_mapping
                or row.region not in self.risk_mapping[row.provider]
            }
        )
        if missing_risk:
            raise ValueError(
                "SLA risk mappings are missing for assets: "
                + ", ".join(missing_risk)
            )

    def fetch_carbon_intensity(self, region):
        if region not in self.carbon_api_mock:
            raise ValueError(f"Carbon intensity mapping is missing for region: {region}")
        return self.carbon_api_mock[region]

    def merge_all_data(self):
        if self.raw_data is None:
            self.load_pricing_data()

        self._validate_mappings()
        
        self.raw_data['carbon_intensity'] = self.raw_data['region'].apply(
            self.fetch_carbon_intensity
        )
        
        # Add scenario-based SLA risk indicators
        self.raw_data['sla_risk'] = self.raw_data.apply(
            lambda x: self.risk_mapping[x['provider']][x['region']], axis=1
        )
        
        print("Success: Multi-objective dataset (cost, risk, and carbon) created.")
        return self.raw_data

    def get_provider_list(self):
        """Returns the unique providers in the system [cite: 167]."""
        if self.raw_data is None:
            raise RuntimeError("Pricing data has not been loaded.")
        return self.raw_data['provider'].unique().tolist()

    def get_region_list(self):
        """Returns the unique regions in the system [cite: 168]."""
        if self.raw_data is None:
            raise RuntimeError("Pricing data has not been loaded.")
        return self.raw_data['region'].unique().tolist()

# Test block
if __name__ == "__main__":
    loader = CloudDataLoader()
    # Load and merge all data
    df = loader.merge_all_data()
    
    print("\n--- Integrated Dataset (Cost & Carbon) ---")
    print(df[['provider', 'region', 'hourly_cost', 'carbon_intensity']].head())
