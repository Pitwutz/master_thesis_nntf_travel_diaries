import sys
import os

# Add the project root to sys.path so config.py can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import PROJECT_ROOT
from pathlib import Path
from typing import List
import logging
from base import TensorConfig
from timebin import TimebinProcessor
from timehours import TimeHoursProcessor
from weekhour import WeekhourProcessor
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_tensor_creation(
    odt_file: Path,
    zips_file: Path,
    output_dir: Path,
    cities: List[str]
) -> None:
    """Run all tensor creation processors for specified cities."""

    # Create base config
    config = TensorConfig(
        data_path=Path(__file__).resolve().parents[2] / "data",
        odt_file=odt_file,
        zips_file=zips_file,
        processed_path=output_dir,
        cities=cities
    )

    # Initialize processors
    processors = [
        TimebinProcessor(config),
        TimeHoursProcessor(config),
        WeekhourProcessor(config)
    ]

    # Run each processor
    for processor in processors:
        logger.info(f"\nRunning {processor.__class__.__name__}...")
        processor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run tensor creation for specified cities.")
    parser.add_argument("--cities", type=str, nargs='+',
                        default=["Rotterdam", "Utrecht"],
                        help="List of city names to process (default: Rotterdam Utrecht)")
    parser.add_argument("--input-file", type=str,
                        default=str(
                            PROJECT_ROOT) + "/data/processed/odt_no_same_od_no_rare_od_fixed_thresh.csv",
                        help="Path to input CSV file")
    args = parser.parse_args()

    # Define paths
    base_path = Path(
        str(PROJECT_ROOT) + "")
    odt_file = Path(args.input_file)
    zips_file = base_path / "data/raw/location/working_zips.geojson"

    # Process each city
    for city in args.cities:
        # Get the input file name without extension and use it for the output directory
        input_file_name = Path(args.input_file).stem
        output_dir = base_path / \
            f"data/processed/{city.lower()}/{input_file_name}_normalizedPeaks"
        logger.info(f"\nProcessing city: {city}")

        # Run tensor creation for the current city
        run_tensor_creation(
            odt_file=odt_file,
            zips_file=zips_file,
            output_dir=output_dir,
            cities=[city]
        )
