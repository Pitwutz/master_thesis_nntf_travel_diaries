import subprocess
import os
from pathlib import Path
from typing import List, Tuple
import logging
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define base paths
BASE_PATH = Path(__file__).resolve().parents[2]
OUTPUT_BASE = BASE_PATH / "data" / "results" / "decompositions" / "tucker"


def run_decomposition(tensor_path: str, rank_o: str, rank_d: str, rank_t: str, output_dir: str, optimizer: str = 'MU', args=None) -> None:
    """Run Tucker decomposition for a single tensor with specified rank combinations.

    Args:
        tensor_path: Path to the tensor file
        rank_o: Origin mode ranks (e.g., "2-6" or "2,3,4")
        rank_d: Destination mode ranks
        rank_t: Time mode ranks
        output_dir: Directory to save results
        optimizer: 'MU' or 'HALS'
        args: Command line arguments
    """

    logger.info(f"\nRunning Tucker decomposition for {tensor_path}")
    cmd = [
        "python3", "code/decompose/decompose_tucker.py",
        tensor_path,
        '--rank-o', str(rank_o),
        '--rank-d', str(rank_d),
        '--rank-t', str(rank_t),
        "--optimizer", optimizer,
        "--max-iter", "1000",
        "--tol", "1e-8",
        "--init-methods", "random,svd,nndsvd",
        "--output-dir", output_dir,
        "--force-cpu",
        "--algorithm", "fista"
    ]

    # Add symmetric-ranks flag if specified
    if args and args.symmetric_ranks:
        cmd.append("--symmetric-ranks")

    # Run command
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully completed decomposition for {tensor_path}")
        logger.debug(f"STDOUT: {result.stdout}")
        logger.debug(f"STDERR: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(
            f"Error running decomposition for {tensor_path}: {str(e)}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        print("\n===== STDOUT =====\n", e.stdout)
        print("\n===== STDERR =====\n", e.stderr)
        raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Tucker decomposition on a tensor')
    parser.add_argument('--tensor', type=str, required=True,
                        help='Path to the tensor file')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save results')
    parser.add_argument('--rank-o', type=str, default="3-6",
                        help='Origin mode ranks (e.g., "2-6" or "2,3,4")')
    parser.add_argument('--rank-d', type=str, default="3-6",
                        help='Destination mode ranks')
    parser.add_argument('--rank-t', type=str, default="2-5",
                        help='Time mode ranks')
    parser.add_argument('--optimizer', type=str, nargs='+', choices=['MU', 'HALS'], default=['MU'],
                        help='Optimizers to use: "MU" (Multiplicative Updates) and/or "HALS"')
    parser.add_argument('--symmetric-ranks', action='store_true',
                        help='If set, only use rank combinations where origin rank equals destination rank')
    return parser.parse_args()


def main():
    """Run Tucker decomposition for the specified tensor and rank combinations."""
    # Parse command line arguments
    args = parse_arguments()

    if args.output_dir is None:
        tensor_filename_stem = args.tensor.split('/')[-1].split('.')[0]
        processed_removed_stem = tensor_filename_stem.replace('_processed', '')
        args.output_dir = OUTPUT_BASE / processed_removed_stem

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    # Run decomposition for each optimizer
    for optimizer in args.optimizer:
        optimizer_base_output_dir = output_dir / optimizer
        optimizer_base_output_dir.mkdir(parents=True, exist_ok=True)

        # Create a single timestamped run directory for this optimizer run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        orchestrator_run_dir = optimizer_base_output_dir / f"run_{timestamp}"
        orchestrator_run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running decomposition with optimizer: {optimizer}")
        logger.info(f"Output for this run will be in: {orchestrator_run_dir}")

        run_decomposition(
            args.tensor,
            args.rank_o,
            args.rank_d,
            args.rank_t,
            str(orchestrator_run_dir),  # Pass the new timestamped dir
            optimizer,
            args  # Pass args to run_decomposition
        )


if __name__ == "__main__":
    main()
