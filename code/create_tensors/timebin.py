from datetime import time, datetime, timedelta
from pathlib import Path
from typing import Dict, Set, Optional
import pandas as pd
import numpy as np
from base import BaseTensorProcessor, TensorConfig
import json


class TimebinProcessor(BaseTensorProcessor):
    """Processor for timebin-based tensor creation"""

    TIME_BINS = [
        (time(0, 0), time(6, 30)),   # Early Morning
        (time(6, 30), time(9, 0)),   # Morning Peak
        (time(9, 0), time(16, 0)),   # Day
        (time(16, 0), time(18, 30)),  # Evening Peak
        (time(18, 30), time(23, 59, 59)),  # Night
    ]

    def __init__(self, config: TensorConfig, tensor_name: Optional[str] = None):
        """Initialize the processor with config and optional custom tensor name."""
        super().__init__(config)
        self.tensor_name = tensor_name
        self.bin_durations_hours: list[float] = self._calculate_bin_durations()

    def _calculate_bin_durations(self) -> list[float]:
        """Calculate the duration of each time bin in hours."""
        durations = []
        for i, (start_time, end_time) in enumerate(self.TIME_BINS):
            start_dt = datetime.combine(datetime.min.date(), start_time)
            # For the last bin, assume it extends to midnight of the next day for duration calculation
            if i == len(self.TIME_BINS) - 1:
                end_dt = datetime.combine(
                    datetime.min.date() + timedelta(days=1), time(0, 0, 0))
            else:
                end_dt = datetime.combine(datetime.min.date(), end_time)

            duration_seconds = (end_dt - start_dt).total_seconds()
            durations.append(duration_seconds / 3600.0)  # Convert to hours
        return durations

    def get_tensor_name(self, city_name: str) -> str:
        """Get the tensor name, using custom name if provided."""
        if self.tensor_name:
            return self.tensor_name
        return f"odt_processed_{city_name.lower()}_timebin"

    def assign_time_bin(self, dt: pd.Timestamp) -> int:
        """Assign a time bin index (0-4) based on the time of day."""
        t = dt.time()
        for i, (start, end) in enumerate(self.TIME_BINS):
            if start <= t < end or (i == 4 and t >= start):
                return i
        return 0  # fallback (should not happen)

    def create_index_mappings(self, city_trips: pd.DataFrame) -> Dict:
        """Create index mappings for the tensor dimensions."""
        # Get unique postal codes from both origins and destinations
        unique_origins = set(city_trips['pc4_departure'].unique())
        unique_destinations = set(city_trips['pc4_arrival'].unique())
        # Use union of both sets to ensure symmetry
        all_postal_codes = sorted(unique_origins.union(unique_destinations))

        unique_timebins = list(range(35))  # 7 weekdays × 5 bins
        unique_modes = sorted(city_trips['mode_of_transport'].unique(
        )) if 'mode_of_transport' in city_trips else []

        mappings = {
            'origins': {str(pc): int(idx) for idx, pc in enumerate(all_postal_codes)},
            'destinations': {str(pc): int(idx) for idx, pc in enumerate(all_postal_codes)},
            'timebin': {int(tb): int(tb) for tb in unique_timebins},
            'modes': {str(mode): int(idx) for idx, mode in enumerate(unique_modes)}
        }

        # Create reverse mappings
        mappings['idx_to_origins'] = {
            int(idx): str(pc) for pc, idx in mappings['origins'].items()}
        mappings['idx_to_destinations'] = {
            int(idx): str(pc) for pc, idx in mappings['destinations'].items()}
        mappings['idx_to_modes'] = {
            int(idx): str(mode) for mode, idx in mappings['modes'].items()}

        # Add sizes
        mappings['sizes'] = {
            'origins': int(len(all_postal_codes)),
            'destinations': int(len(all_postal_codes)),
            'timebin': int(len(unique_timebins)),
            'modes': int(len(unique_modes))
        }

        return mappings

    def process_city_data(self, trips_df: pd.DataFrame, city_postal_codes: Set[str],
                          city_name: str, output_path: Path) -> Dict:
        """Process data for a specific city with timebin-based analysis."""
        self.logger.info(f"\nProcessing data for {city_name}...")

        # Log total number of postal codes in geographical data
        total_postal_codes = len(city_postal_codes)
        self.logger.info(
            f"Total postal codes in {city_name}: {total_postal_codes}")

        # Filter trips for the city
        city_trips = trips_df[
            (trips_df['pc4_departure'].isin(city_postal_codes)) &
            (trips_df['pc4_arrival'].isin(city_postal_codes))
        ]

        if len(city_trips) == 0:
            raise ValueError(f"No trips found for {city_name}!")

        # Create index mappings
        index_mappings = self.create_index_mappings(city_trips)

        # Log postal code usage statistics
        used_origins = set(city_trips['pc4_departure'].unique())
        used_destinations = set(city_trips['pc4_arrival'].unique())
        all_used_codes = used_origins.union(used_destinations)
        unused_codes = city_postal_codes - all_used_codes

        self.logger.info(f"Postal codes used as origins: {len(used_origins)}")
        self.logger.info(
            f"Postal codes used as destinations: {len(used_destinations)}")
        self.logger.info(
            f"Total unique postal codes used: {len(all_used_codes)}")
        self.logger.info(f"Unused postal codes: {len(unused_codes)}")
        if unused_codes:
            self.logger.info(
                f"Unused postal codes: {sorted(list(unused_codes))}")

        # Save mappings
        mappings_path = output_path / \
            f"index_mappings_{city_name.lower()}_timebin.json"
        with open(mappings_path, 'w') as f:
            json.dump(index_mappings, f, indent=2)
        self.logger.info(f"Saved index mappings to {mappings_path}")

        # Convert timestamps and add timebin
        city_trips['timestamp_departure'] = pd.to_datetime(
            city_trips['timestamp_departure'])
        city_trips['weekday'] = city_trips['timestamp_departure'].dt.dayofweek
        city_trips['time_bin'] = city_trips['timestamp_departure'].apply(
            self.assign_time_bin)
        city_trips['timebin'] = city_trips['weekday'] * \
            5 + city_trips['time_bin']

        # Create ODT aggregation
        self.logger.info("\nCreating ODT aggregation...")
        odt_grouped = city_trips.groupby(
            ['pc4_departure', 'pc4_arrival', 'timebin']
        ).size().reset_index(name='trip_count')

        # Normalize trip counts by time bin duration (trips/hour)
        # First, map the combined 'timebin' (weekday*5 + time_bin_original_index) back to the original time bin index (0-4)
        odt_grouped['time_bin_original_idx'] = odt_grouped['timebin'] % 5
        odt_grouped['bin_duration_hours'] = odt_grouped['time_bin_original_idx'].apply(
            lambda x: self.bin_durations_hours[x]
        )
        odt_grouped['trip_count'] = odt_grouped['trip_count'] / \
            odt_grouped['bin_duration_hours']
        # Drop helper columns
        odt_grouped.drop(
            columns=['time_bin_original_idx', 'bin_duration_hours'], inplace=True)

        # Create and save ODT tensor
        odt_tensor = np.zeros((
            index_mappings['sizes']['origins'],
            index_mappings['sizes']['destinations'],
            index_mappings['sizes']['timebin']
        ))

        for _, row in odt_grouped.iterrows():
            i = index_mappings['origins'][row['pc4_departure']]
            j = index_mappings['destinations'][row['pc4_arrival']]
            k = row['timebin']
            odt_tensor[i, j, k] = row['trip_count']

        # Create tensor directories
        tensor_path = output_path / "tensors"
        tensor_path.mkdir(exist_ok=True)
        csv_path = tensor_path / "csv"
        csv_path.mkdir(exist_ok=True)

        # Save ODT data
        odt_output_path = csv_path / \
            f"odt_processed_{city_name.lower()}_timebin.csv"
        odt_grouped.to_csv(odt_output_path, index=False)
        self.logger.info(f"Saved ODT data to {odt_output_path}")

        # Process ODTM if mode_of_transport exists
        if 'mode_of_transport' in city_trips:
            self.logger.info("\nCreating ODTM aggregation...")
            odtm_grouped = city_trips.groupby(
                ['pc4_departure', 'pc4_arrival', 'timebin', 'mode_of_transport']
            ).size().reset_index(name='trip_count')

            # Normalize trip counts by time bin duration (trips/hour)
            odtm_grouped['time_bin_original_idx'] = odtm_grouped['timebin'] % 5
            odtm_grouped['bin_duration_hours'] = odtm_grouped['time_bin_original_idx'].apply(
                lambda x: self.bin_durations_hours[x]
            )
            odtm_grouped['trip_count'] = odtm_grouped['trip_count'] / \
                odtm_grouped['bin_duration_hours']
            # Drop helper columns
            odtm_grouped.drop(
                columns=['time_bin_original_idx', 'bin_duration_hours'], inplace=True)

            odtm_tensor = np.zeros((
                index_mappings['sizes']['origins'],
                index_mappings['sizes']['destinations'],
                index_mappings['sizes']['timebin'],
                index_mappings['sizes']['modes']
            ))

            for _, row in odtm_grouped.iterrows():
                i = index_mappings['origins'][row['pc4_departure']]
                j = index_mappings['destinations'][row['pc4_arrival']]
                k = row['timebin']
                m = index_mappings['modes'][str(row['mode_of_transport'])]
                odtm_tensor[i, j, k, m] = row['trip_count']

            odtm_output_path = csv_path / \
                f"odtm_processed_{city_name.lower()}_timebin.csv"
            odtm_grouped.to_csv(odtm_output_path, index=False)
            self.logger.info(f"Saved ODTM data to {odtm_output_path}")

        # Save tensors
        tensor_name = self.get_tensor_name(city_name)
        np.savez(tensor_path / f"{tensor_name}.npz",
                 tensor=odt_tensor,
                 origins=list(index_mappings['origins'].keys()),
                 destinations=list(index_mappings['destinations'].keys()),
                 timebins=list(range(35)))

        if 'mode_of_transport' in city_trips:
            np.savez(tensor_path / f"{tensor_name}_mode.npz",
                     tensor=odtm_tensor,
                     origins=list(index_mappings['origins'].keys()),
                     destinations=list(index_mappings['destinations'].keys()),
                     timebins=list(range(35)),
                     modes=list(index_mappings['modes'].keys()))

        self.logger.info(f"Saved tensor data to {tensor_path}")

        # Calculate statistics from the grouped data and tensor
        non_zero = np.nonzero(odt_tensor)
        values = odt_tensor[non_zero]
        unique_combinations = set(zip(*non_zero))
        nnz = len(unique_combinations)
        density = nnz / np.prod(odt_tensor.shape)

        # Log tensor statistics
        self.logger.info("\nTensor Statistics:")
        self.logger.info(f"Shape: {odt_tensor.shape}")
        self.logger.info(f"Non-zero elements: {nnz}")
        self.logger.info(f"Density: {density:.6f}")
        self.logger.info(f"Maximum value: {np.max(values):.2f}")
        self.logger.info(f"Mean value (non-zero): {np.mean(values):.2f}")

        # Calculate value distribution
        unique_values, counts = np.unique(values, return_counts=True)
        self.logger.info("\nValue distribution:")
        for val, count in zip(unique_values, counts):
            self.logger.info(f"  {val:.2f}: {count} occurrences")

        # Return statistics
        return {
            'total_trips': len(city_trips),
            'unique_origins': index_mappings['sizes']['origins'],
            'unique_destinations': index_mappings['sizes']['destinations'],
            'unique_timebins': index_mappings['sizes']['timebin'],
            'odt_shape': odt_tensor.shape,
            'odtm_shape': odtm_tensor.shape if 'mode_of_transport' in city_trips else None,
            'max_odt_count': float(np.max(odt_tensor)),
            'max_odtm_count': float(np.max(odtm_tensor)) if 'mode_of_transport' in city_trips else None,
            'non_zero_elements': nnz,
            'density': float(density),
            'value_distribution': {float(val): int(count) for val, count in zip(unique_values, counts)}
        }
