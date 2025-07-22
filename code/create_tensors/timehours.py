from datetime import time
from pathlib import Path
from typing import Dict, Set, Optional, Tuple
import pandas as pd
import numpy as np
from base import BaseTensorProcessor, TensorConfig
import json
import logging


class TimeHoursProcessor(BaseTensorProcessor):
    """Processor for 24-hour based tensor creation with weekday/weekend separation"""

    def __init__(self, config: TensorConfig, tensor_name: Optional[str] = None):
        """Initialize the processor with config and optional custom tensor name."""
        super().__init__(config)
        self.tensor_name = tensor_name
        # Get the folder name from the input CSV file
        self.folder_name = Path(config.odt_file).stem

    def get_tensor_name(self, city_name: str, is_weekend: bool) -> str:
        """Get the tensor name based on city and time period."""
        period = "weekend" if is_weekend else "weekday"
        if self.tensor_name:
            return f"{self.tensor_name}_{period}"
        return f"odt_processed_{city_name.lower()}_hourly_{period}"

    def assign_hour(self, dt: pd.Timestamp) -> int:
        """Assign an hour index (0-23) based on the time of day."""
        return dt.hour

    def create_index_mappings(self, city_trips: pd.DataFrame) -> Dict:
        """Create index mappings for the tensor dimensions."""
        # Get unique postal codes from both origins and destinations
        unique_origins = set(city_trips['pc4_departure'].unique())
        unique_destinations = set(city_trips['pc4_arrival'].unique())
        # Use union of both sets to ensure symmetry
        all_postal_codes = sorted(unique_origins.union(unique_destinations))

        unique_hours = list(range(24))  # 24 hours

        mappings = {
            'origins': {str(pc): int(idx) for idx, pc in enumerate(all_postal_codes)},
            'destinations': {str(pc): int(idx) for idx, pc in enumerate(all_postal_codes)},
            'hour': {int(h): int(h) for h in unique_hours}
        }

        # Create reverse mappings
        mappings['idx_to_origins'] = {
            int(idx): str(pc) for pc, idx in mappings['origins'].items()}
        mappings['idx_to_destinations'] = {
            int(idx): str(pc) for pc, idx in mappings['destinations'].items()}

        # Add sizes
        mappings['sizes'] = {
            'origins': int(len(all_postal_codes)),
            'destinations': int(len(all_postal_codes)),
            'hour': int(len(unique_hours))
        }

        return mappings

    def calculate_tensor_statistics(self, tensor: np.ndarray, trip_counts: pd.Series) -> Dict:
        """Calculate statistics for the tensor."""
        return {
            'total_trips': int(trip_counts.sum()),
            'mean_trips': float(trip_counts.mean()),
            'median_trips': float(trip_counts.median()),
            'density': float(np.count_nonzero(tensor) / tensor.size),
            'max_trips_per_hour': int(trip_counts.max()),
            'min_trips_per_hour': int(trip_counts.min()),
            'std_trips': float(trip_counts.std())
        }

    def process_city_data(self, trips_df: pd.DataFrame, city_postal_codes: Set[str],
                          city_name: str, output_path: Path) -> Dict:
        """Process data for a specific city with hourly analysis and weekday/weekend separation."""
        self.logger.info(f"\nProcessing data for {city_name}...")

        # Filter trips for the city
        city_trips = trips_df[
            (trips_df['pc4_departure'].isin(city_postal_codes)) &
            (trips_df['pc4_arrival'].isin(city_postal_codes))
        ]

        if len(city_trips) == 0:
            raise ValueError(f"No trips found for {city_name}!")

        # Convert timestamps and add hour
        city_trips['timestamp_departure'] = pd.to_datetime(
            city_trips['timestamp_departure'])
        city_trips['hour'] = city_trips['timestamp_departure'].apply(
            self.assign_hour)
        city_trips['is_weekend'] = city_trips['timestamp_departure'].dt.dayofweek >= 5

        # Create tensor directories using the input CSV filename
        tensor_path = output_path / self.folder_name
        tensor_path.mkdir(exist_ok=True)
        csv_path = tensor_path / "csv"
        csv_path.mkdir(exist_ok=True)

        stats = {}

        # Process weekday and weekend separately
        for is_weekend in [False, True]:
            period = "weekend" if is_weekend else "weekday"
            self.logger.info(f"\nProcessing {period} data...")

            # Filter trips for the period
            period_trips = city_trips[city_trips['is_weekend'] == is_weekend]

            if len(period_trips) == 0:
                self.logger.warning(
                    f"No {period} trips found for {city_name}!")
                continue

            # Create index mappings
            index_mappings = self.create_index_mappings(period_trips)

            # Save index mappings
            mappings_path = tensor_path / \
                f"index_mappings_{city_name.lower()}_{period}_hourly.json"
            with open(mappings_path, 'w') as f:
                json.dump(index_mappings, f, indent=2)
            self.logger.info(f"Saved index mappings to {mappings_path}")

            # Create ODT aggregation
            odt_grouped = period_trips.groupby(
                ['pc4_departure', 'pc4_arrival', 'hour']
            ).size().reset_index(name='trip_count')

            # Create tensor
            odt_tensor = np.zeros((
                index_mappings['sizes']['origins'],
                index_mappings['sizes']['destinations'],
                index_mappings['sizes']['hour']
            ))

            for _, row in odt_grouped.iterrows():
                i = index_mappings['origins'][row['pc4_departure']]
                j = index_mappings['destinations'][row['pc4_arrival']]
                k = row['hour']
                odt_tensor[i, j, k] = row['trip_count']

            # Calculate statistics
            tensor_stats = self.calculate_tensor_statistics(
                odt_tensor, odt_grouped['trip_count'])
            stats[period] = tensor_stats

            # Save ODT data
            odt_output_path = csv_path / \
                f"odt_processed_{city_name.lower()}_hourly_{period}.csv"
            odt_grouped.to_csv(odt_output_path, index=False)
            self.logger.info(f"Saved ODT data to {odt_output_path}")

            # Save tensor
            tensor_name = self.get_tensor_name(city_name, is_weekend)
            np.savez(tensor_path / f"{tensor_name}.npz",
                     tensor=odt_tensor,
                     origins=list(index_mappings['origins'].keys()),
                     destinations=list(index_mappings['destinations'].keys()),
                     hours=list(range(24)))

            # Save metadata
            metadata = {
                'tensor_name': tensor_name,
                'city': city_name,
                'period': period,
                'source_file': str(self.config.odt_file),
                'statistics': tensor_stats,
                'dimensions': {
                    'origins': index_mappings['sizes']['origins'],
                    'destinations': index_mappings['sizes']['destinations'],
                    'hours': 24
                }
            }

            metadata_path = tensor_path / f"{tensor_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"Saved metadata to {metadata_path}")

        return stats
