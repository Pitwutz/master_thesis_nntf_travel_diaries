#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from weekhour import WeekhourProcessor
from base import TensorConfig


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Process ODT tensors for cities using weekhours.")
    parser.add_argument('--city', type=str,
                        help='City name to process (e.g., Utrecht)')
    parser.add_argument('--tensor-path', type=str,
                        help='Custom tensor output directory')
    parser.add_argument('--tensor-name', type=str,
                        help='Custom name for the output tensor file (without extension)')
    parser.add_argument('--input-csv', type=str,
                        help='Path to input CSV file')
    args = parser.parse_args()

    # Define paths relative to project root
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data"
    config = TensorConfig(
        data_path=data_path,
        odt_file=args.input_csv if args.input_csv else data_path /
        "processed" / "odt_full.csv",
        zips_file=data_path / "raw" / "location" / "working_zips.geojson",
        processed_path=data_path / "processed",
        cities=[args.city] if args.city else [
            "Utrecht", "'s-Gravenhage", "Rotterdam"]
    )

    # Create and run processor
    processor = WeekhourProcessor(config, tensor_name=args.tensor_name)
    processor.run()


if __name__ == "__main__":
    main()
