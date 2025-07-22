from pathlib import Path
import logging
import argparse
from timehours import TimeHoursProcessor
from base import TensorConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Process ODT data into hourly tensors with weekday/weekend separation')
    parser.add_argument('input_file', type=str,
                        help='Path to the input ODT CSV file')
    args = parser.parse_args()

    # Define paths
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data"
    odt_file = Path(args.input_file)
    zips_file = data_path / "raw" / "location" / "working_zips.geojson"
    processed_path = data_path / "processed"

    # Create config
    config = TensorConfig(
        data_path=data_path,
        odt_file=odt_file,
        zips_file=zips_file,
        processed_path=processed_path,
        cities=["Utrecht", "Rotterdam"]  # Only process these two cities
    )

    # Create and run processor
    processor = TimeHoursProcessor(config)
    processor.run()


if __name__ == "__main__":
    main()
