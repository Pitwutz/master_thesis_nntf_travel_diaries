#!/usr/bin/env python3
"""
Single Tensor Processing Pipeline

This script orchestrates the complete process of creating, decomposing, and visualizing a single tensor.
It handles the entire pipeline from raw data to final visualizations for one tensor at a time.

Key Features:
- Tensor Creation: Creates a single tensor from input data (timebin or weekhour)
- Decomposition: Runs CP decomposition with specified parameters (rank, optimizer)
- Visualization: Generates detailed visualizations of the decomposition results

Usage:
    python run_single_tensor_pipeline.py --city CITY --tensor-type TYPE --input-csv PATH [options]

Example:
    python run_single_tensor_pipeline.py --city Utrecht --tensor-type timebin --input-csv data/raw/utrecht.csv

Note:
    This script is designed to process one tensor at a time. For comparing multiple tensor types,
    use compare_tensor_types.py instead.
"""

import argparse
import logging
from pathlib import Path
import sys
import subprocess
from typing import Optional, List, Union
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define default paths
DEFAULT_GEOJSON_PATH = "data/raw/location/working_zips.geojson"
DEFAULT_PROCESSED_PATH = "data/processed"
DEFAULT_DECOMP_PATH = "data/results/decompositions"
DEFAULT_VIS_PATH = "data/results/visualizations"


def run_tensor_creation(city: str, tensor_type: str, input_csv: str, tensor_name: Optional[str] = None) -> Path:
    """Run tensor creation script and return the path to the created tensor.

    Args:
        city: Name of the city to process
        tensor_type: Type of tensor to create ('timebin' or 'weekhour')
        input_csv: Path to input CSV file
        tensor_name: Optional custom name for the tensor

    Returns:
        Path to the created tensor file
    """
    logger.info(f"Creating {tensor_type} tensor for {city}...")

    # Determine which script to run
    script_name = f"run_{tensor_type}.py"
    script_path = Path(__file__).parent / "create_tensors" / script_name

    # Build command
    cmd = [sys.executable, str(script_path),
           "--city", city,
           "--input-csv", input_csv]

    if tensor_name:
        cmd.extend(["--tensor-name", tensor_name])

    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Tensor creation completed successfully")

        # Return path to created tensor
        tensor_path = Path(DEFAULT_PROCESSED_PATH) / city.lower() / "tensors"
        if tensor_name:
            tensor_file = tensor_path / f"{tensor_name}.npz"
        else:
            tensor_file = tensor_path / \
                f"odt_processed_{city.lower()}_{tensor_type}.npz"

        return tensor_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating tensor: {e}")
        raise


def run_decomposition(tensor_path: Path, ranks: Union[int, List[int]], optimizer: str = "MU") -> Path:
    """Run CP decomposition on the tensor.

    Args:
        tensor_path: Path to the tensor file
        ranks: Single rank or list of ranks to use
        optimizer: Optimization method ('ALS' or 'HALS')

    Returns:
        Path to the decomposition results directory
    """
    logger.info(f"Running CP decomposition with {optimizer}...")

    # Convert ranks to string format
    if isinstance(ranks, list):
        ranks_str = f"{min(ranks)}-{max(ranks)}"
    else:
        ranks_str = str(ranks)

    # Build command
    cmd = [sys.executable, str(Path(__file__).parent / "decompose_cp.py"),
           str(tensor_path),
           ranks_str,
           "--optimizer", optimizer]

    # Run command
    try:
        subprocess.run(cmd, check=True)
        logger.info("Decomposition completed successfully")

        # Find the most recent decomposition directory
        decomp_dir = Path(DEFAULT_DECOMP_PATH)
        run_dirs = sorted([d for d in decomp_dir.glob("run_*")],
                          key=lambda x: x.stat().st_mtime, reverse=True)
        if not run_dirs:
            raise FileNotFoundError("No decomposition results found")

        return run_dirs[0]
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running decomposition: {e}")
        raise


def run_visualization(decomp_dir: Path, city: str, tensor_type: str) -> None:
    """Run visualization script on decomposition results.

    Args:
        decomp_dir: Path to decomposition results directory
        city: Name of the city
        tensor_type: Type of tensor ('timebin' or 'weekhour')
    """
    logger.info("Running visualization...")

    # Find the most recent decomposition file
    rank_dirs = sorted([d for d in decomp_dir.glob("rank_*")],
                       key=lambda x: int(x.name.split("_")[1]))
    if not rank_dirs:
        raise FileNotFoundError("No rank directories found")

    # Process each rank
    for rank_dir in rank_dirs:
        decomp_file = rank_dir / \
            f"odt_processed_{city.lower()}_{tensor_type}_factors.npz"
        if not decomp_file.exists():
            continue

        # Build command
        cmd = [sys.executable, str(Path(__file__).parent / "visualize_decomposition.py"),
               str(decomp_file),
               "--geojson", DEFAULT_GEOJSON_PATH,
               str(Path(DEFAULT_PROCESSED_PATH) / city.lower() /
                   f"index_mappings_{city.lower()}_{tensor_type}.json"),
               str(Path(DEFAULT_VIS_PATH) / f"odt_{city.lower()}_{tensor_type}")]

        # Run command
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Visualization completed for rank {rank_dir.name}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error running visualization for rank {rank_dir.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Run complete tensor creation, decomposition, and visualization pipeline")
    parser.add_argument("--city", type=str, required=True,
                        help="City name to process (e.g., Utrecht)")
    parser.add_argument("--tensor-type", type=str, choices=["timebin", "weekhour"], required=True,
                        help="Type of tensor to create")
    parser.add_argument("--input-csv", type=str, required=True,
                        help="Path to input CSV file")
    parser.add_argument("--tensor-name", type=str,
                        help="Custom name for the tensor (without extension)")
    parser.add_argument("--ranks", type=str, default="5",
                        help="Rank or range of ranks to use (e.g., '5' or '1-10')")
    parser.add_argument("--optimizer", type=str, choices=["MU", "HALS"], default="MU",
                        help="Optimization method for decomposition")
    parser.add_argument("--skip-tensor-creation", action="store_true",
                        help="Skip tensor creation and use existing tensor")
    parser.add_argument("--tensor-path", type=str,
                        help="Path to existing tensor file (required if --skip-tensor-creation is used)")

    args = parser.parse_args()

    try:
        # Step 1: Create tensor (if not skipped)
        if args.skip_tensor_creation:
            if not args.tensor_path:
                raise ValueError(
                    "--tensor-path is required when --skip-tensor-creation is used")
            tensor_path = Path(args.tensor_path)
        else:
            tensor_path = run_tensor_creation(
                args.city,
                args.tensor_type,
                args.input_csv,
                args.tensor_name
            )

        # Step 2: Run decomposition
        decomp_dir = run_decomposition(
            tensor_path,
            args.ranks,
            args.optimizer
        )

        # Step 3: Run visualization
        run_visualization(
            decomp_dir,
            args.city,
            args.tensor_type
        )

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
