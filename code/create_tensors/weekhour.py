from pathlib import Path
from typing import Dict, Set, Optional
import pandas as pd
import numpy as np
from base import BaseTensorProcessor, TensorConfig
import json


class WeekhourProcessor(BaseTensorProcessor):
    """Processor for weekhour-based tensor creation"""

    def __init__(self, config: TensorConfig, tensor_name: Optional[str] = None):
        """Initialize the processor with config and optional custom tensor name."""
        super().__init__(config)
        self.tensor_name = tensor_name

    def get_tensor_name(self, city_name: str) -> str:
        """Get the tensor name, using custom name if provided."""
        if self.tensor_name:
            return self.tensor_name
        return f"odt_processed_{city_name.lower()}_weekhour"

    def create_index_mappings(self, city_trips: pd.DataFrame) -> Dict:
        """Create index mappings for all dimensions (origins, destinations, weekhours, modes)."""
        # Get unique postal codes from both origins and destinations
        unique_origins = set(city_trips['pc4_departure'].unique())
        unique_destinations = set(city_trips['pc4_arrival'].unique())
        # Use union of both sets to ensure symmetry
        all_postal_codes = sorted(unique_origins.union(unique_destinations))

        unique_weekhours = list(range(168))  # 24 hours * 7 days
        unique_modes = sorted(city_trips['mode_of_transport'].unique(
        )) if 'mode_of_transport' in city_trips else []

        mappings = {
            'origins': {str(pc): int(idx) for idx, pc in enumerate(all_postal_codes)},
            'destinations': {str(pc): int(idx) for idx, pc in enumerate(all_postal_codes)},
            'weekhours': {int(hour): int(hour) for hour in unique_weekhours},
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
            'weekhours': int(len(unique_weekhours)),
            'modes': int(len(unique_modes))
        }

        return mappings

    def process_city_data(self, trips_df: pd.DataFrame, city_postal_codes: Set[str],
                          city_name: str, output_path: Path) -> Dict:
        """Process data for a specific city with weekhour-based analysis."""
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
            f"index_mappings_{city_name.lower()}_weekhour.json"
        with open(mappings_path, 'w') as f:
            json.dump(index_mappings, f, indent=2)
        self.logger.info(f"Saved index mappings to {mappings_path}")

        # Convert timestamps and add weekhour
        city_trips['timestamp_departure'] = pd.to_datetime(
            city_trips['timestamp_departure'])
        city_trips['weekhour'] = city_trips['timestamp_departure'].dt.dayofweek * \
            24 + city_trips['timestamp_departure'].dt.hour

        # Create ODT aggregation
        self.logger.info("\nCreating ODT aggregation...")
        odt_grouped = city_trips.groupby(
            ['pc4_departure', 'pc4_arrival', 'weekhour']
        ).size().reset_index(name='trip_count')

        # Create and save ODT tensor
        odt_tensor = np.zeros((
            index_mappings['sizes']['origins'],
            index_mappings['sizes']['destinations'],
            index_mappings['sizes']['weekhours']
        ))

        for _, row in odt_grouped.iterrows():
            i = index_mappings['origins'][row['pc4_departure']]
            j = index_mappings['destinations'][row['pc4_arrival']]
            k = row['weekhour']
            odt_tensor[i, j, k] = row['trip_count']

        # Create tensor directories
        tensor_path = output_path / "tensors"
        tensor_path.mkdir(exist_ok=True)
        csv_path = tensor_path / "csv"
        csv_path.mkdir(exist_ok=True)

        # Save ODT data
        odt_output_path = csv_path / \
            f"odt_processed_{city_name.lower()}_weekhour.csv"
        odt_grouped.to_csv(odt_output_path, index=False)
        self.logger.info(f"Saved ODT data to {odt_output_path}")

        # Process ODTM if mode_of_transport exists
        if 'mode_of_transport' in city_trips:
            self.logger.info("\nCreating ODTM aggregation...")
            odtm_grouped = city_trips.groupby(
                ['pc4_departure', 'pc4_arrival', 'weekhour', 'mode_of_transport']
            ).size().reset_index(name='trip_count')

            odtm_tensor = np.zeros((
                index_mappings['sizes']['origins'],
                index_mappings['sizes']['destinations'],
                index_mappings['sizes']['weekhours'],
                index_mappings['sizes']['modes']
            ))

            for _, row in odtm_grouped.iterrows():
                i = index_mappings['origins'][row['pc4_departure']]
                j = index_mappings['destinations'][row['pc4_arrival']]
                k = row['weekhour']
                m = index_mappings['modes'][str(row['mode_of_transport'])]
                odtm_tensor[i, j, k, m] = row['trip_count']

            odtm_output_path = csv_path / \
                f"odtm_processed_{city_name.lower()}_weekhour.csv"
            odtm_grouped.to_csv(odtm_output_path, index=False)
            self.logger.info(f"Saved ODTM data to {odtm_output_path}")

        # Save tensors
        tensor_name = self.get_tensor_name(city_name)
        np.savez(tensor_path / f"{tensor_name}.npz",
                 tensor=odt_tensor,
                 origins=list(index_mappings['origins'].keys()),
                 destinations=list(index_mappings['destinations'].keys()),
                 weekhours=list(range(168)))

        if 'mode_of_transport' in city_trips:
            np.savez(tensor_path / f"{tensor_name}_mode.npz",
                     tensor=odtm_tensor,
                     origins=list(index_mappings['origins'].keys()),
                     destinations=list(index_mappings['destinations'].keys()),
                     weekhours=list(range(168)),
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
        self.logger.info(f"Maximum value: {np.max(values)}")
        self.logger.info(f"Mean value (non-zero): {np.mean(values):.2f}")

        # Calculate value distribution
        unique_values, counts = np.unique(values, return_counts=True)
        self.logger.info("\nValue distribution:")
        for val, count in zip(unique_values, counts):
            self.logger.info(f"  {int(val)}: {count} occurrences")

        # Return statistics
        return {
            'total_trips': len(city_trips),
            'unique_origins': index_mappings['sizes']['origins'],
            'unique_destinations': index_mappings['sizes']['destinations'],
            'unique_weekhours': index_mappings['sizes']['weekhours'],
            'odt_shape': odt_tensor.shape,
            'odtm_shape': odtm_tensor.shape if 'mode_of_transport' in city_trips else None,
            'max_odt_count': int(np.max(odt_tensor)),
            'max_odtm_count': int(np.max(odtm_tensor)) if 'mode_of_transport' in city_trips else None,
            'non_zero_elements': nnz,
            'density': float(density),
            'value_distribution': {int(val): int(count) for val, count in zip(unique_values, counts)}
        }
