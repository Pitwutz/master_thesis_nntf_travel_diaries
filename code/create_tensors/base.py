from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import geopandas as gpd
import numpy as np
import logging
import json


@dataclass
class TensorConfig:
    """Configuration for tensor processing"""
    data_path: Path
    odt_file: Path
    zips_file: Path
    processed_path: Path
    cities: List[str]


class BaseTensorProcessor:
    """Base class for tensor processing"""

    def __init__(self, config: TensorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def format_postal_code(self, x: Any) -> Optional[str]:
        """Format postal code to 4-digit string, handling various input types."""
        if pd.isna(x):
            return None
        try:
            return f"{int(float(x)):04d}"
        except (ValueError, TypeError):
            return None

    def get_city_postal_codes(self, netherlands_gdf: gpd.GeoDataFrame, city: str) -> Set[str]:
        """Get postal codes for a city"""
        city_gdf = netherlands_gdf[netherlands_gdf['gem_name'] == city]
        postal_codes = set(city_gdf['pc4_code'].apply(self.format_postal_code))
        postal_codes.discard(None)
        return postal_codes

    def create_index_mappings(self, city_trips: pd.DataFrame) -> Dict:
        """Create index mappings for all dimensions"""
        raise NotImplementedError("Subclasses must implement this method")

    def save_sparse_tensor(self, grouped_df: pd.DataFrame, dim_mappings: Dict,
                           output_path: Path, tensor_type: str = "ODT") -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Convert grouped data to sparse tensor format and save."""
        # Extract coordinates and values
        coords = np.array([
            grouped_df[dim].map(dim_mappings[dim]).values
            for dim in dim_mappings
        ])
        values = grouped_df['trip_count'].values
        shape = [len(mapping) for mapping in dim_mappings.values()]

        # Verify uniqueness of combinations
        unique_coords = np.unique(coords, axis=1)
        if len(unique_coords[0]) != len(coords[0]):
            self.logger.warning(
                "Duplicate coordinates found! This should not happen.")
            # Keep only unique combinations
            _, unique_idx = np.unique(coords.T, axis=0, return_index=True)
            coords = coords[:, unique_idx]
            values = values[unique_idx]

        # Save metadata
        metadata = {
            'shape': shape,
            'nnz': len(values),
            'density': len(values) / np.prod(shape),
            'dimension_names': list(dim_mappings.keys()),
            'dimension_mappings': {k: {str(v): int(i) for v, i in mapping.items()}
                                   for k, mapping in dim_mappings.items()}
        }

        # Save coordinates, values, and metadata
        np.savez(output_path,
                 coords=coords,
                 data=values,
                 metadata=json.dumps(metadata))

        self.logger.info(f"Sparse tensor shape: {shape}")
        self.logger.info(f"Non-zero elements: {metadata['nnz']}")
        self.logger.info(
            f"Memory usage: {(coords.nbytes + values.nbytes) / 1024:.2f} KB")
        self.logger.info(f"Density: {metadata['density']:.6f}")

        return coords, values, metadata

    def process_city_data(self, trips_df: pd.DataFrame, city_postal_codes: Set[str],
                          city_name: str, output_path: Path) -> Dict:
        """Process data for a specific city"""
        raise NotImplementedError("Subclasses must implement this method")

    def run(self) -> None:
        """Main execution flow"""
        self.logger.info("Loading geographical data...")
        netherlands_gdf = gpd.read_file(self.config.zips_file)
        self.logger.info(
            f"Loaded geographical data with shape: {netherlands_gdf.shape}")

        self.logger.info("\nLoading ODT data...")
        odt_df = pd.read_csv(self.config.odt_file)
        self.logger.info(f"Loaded ODT data with shape: {odt_df.shape}")

        self.logger.info("\nFormatting postal codes...")
        odt_df['pc4_departure'] = odt_df['pc4_departure'].apply(
            self.format_postal_code)
        odt_df['pc4_arrival'] = odt_df['pc4_arrival'].apply(
            self.format_postal_code)
        odt_df = odt_df.dropna(subset=['pc4_departure', 'pc4_arrival'])
        self.logger.info(
            f"Shape after removing invalid postal codes: {odt_df.shape}")

        for city in self.config.cities:
            self.logger.info(f"\nProcessing {city}...")
            city_path = self.config.processed_path / city.lower()
            city_path.mkdir(parents=True, exist_ok=True)

            postal_codes = self.get_city_postal_codes(netherlands_gdf, city)
            self.logger.info(
                f"Found {len(postal_codes)} postal codes in {city}")

            stats = self.process_city_data(
                odt_df, postal_codes, city, city_path)
            self.logger.info(f"\n{city} Statistics:")
            for key, value in stats.items():
                self.logger.info(f"{key}: {value}")
