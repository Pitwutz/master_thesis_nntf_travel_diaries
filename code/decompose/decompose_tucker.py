import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import sys
import scipy.sparse
# Install required packages if running on Colab
try:
    import google.colab
    print("Running on Google Colab. Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "--quiet",
                          "tensorly", "torch", "scipy", "numpy", "matplotlib"])
    print("Required packages installed successfully!")
except ImportError:
    print("Running in local environment - packages should be installed in local environment")

import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import non_negative_tucker, non_negative_tucker_hals
import matplotlib.pyplot as plt
import platform
from scipy import sparse
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure TensorLy to use PyTorch backend
tl.set_backend('pytorch')

# Check if running on Google Colab
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    base_path = Path('/content/drive/MyDrive/')
    logger.info("Running on Google Colab")
    IS_COLAB = True

    # Create a sync function to ensure files are written to drive
    def sync_drive():
        """Force synchronization of Google Drive to prevent data loss"""
        try:
            from IPython.display import display, Javascript
            display(Javascript('google.colab.files.sync()'))
            logger.info("Synced results to Google Drive")
        except Exception as e:
            logger.warning(f"Could not sync to drive: {e}")
except ImportError:
    # Dynamically determine the project root (assumes this file is in code/decompose/)
    base_path = Path(__file__).resolve().parents[2]
    logger.info("Running on local environment")
    IS_COLAB = False

    # Dummy sync function for local environment
    def sync_drive():
        pass

# Define paths
DATA_PATH = base_path / "data"
TENSOR_PATH = DATA_PATH / "processed" / "tensors"
DECOMP_PATH = DATA_PATH / "results" / "decompositions" / "tucker"
DECOMP_PATH.mkdir(parents=True, exist_ok=True)

# Set random state for reproducibility
RANDOM_STATE = 42
logger.info(f"Using random state: {RANDOM_STATE}")

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize device (will be updated in main based on args)
device = None
device_name = None


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu"), "cpu"
    # Check if MPS is available and we're on Apple Silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS doesn't handle sparse tensors well, so for processing that involves sparse tensors
        # we'll use CPU first, then move to MPS after converting to dense
        return torch.device("mps"), "mps"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"


# Initialize device with default settings
device, device_name = get_device()
logger.info(f"Using {device_name} device for computation")

# Check GPU memory if available
if device.type == "cuda":
    logger.info(
        f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logger.info(
        f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
elif device.type == "mps":
    # Unfortunately, MPS doesn't have direct memory query functions like CUDA
    logger.info(
        "Using Apple MPS (Metal) for acceleration - memory stats not available")


def load_sparse_tensor(tensor_path):
    """Load tensor data from NPZ file.

    The NPZ file should contain:
    - tensor: The dense tensor data
    - origins: List of origin postal codes
    - destinations: List of destination postal codes
    - timebins/weekhours/hours: List of time indices

    Returns:
        tensor: Sparse tensor for decomposition in PyTorch COO format
        origins: List of origin postal codes
        destinations: List of destination postal codes
        time_indices: List of time indices (timebins or weekhours)
    """
    try:
        # Load the NPZ file
        npz_data = np.load(tensor_path, allow_pickle=True)

        # Get the tensor data
        dense_tensor = npz_data['tensor']

        # Get non-zero indices and values
        non_zero = np.nonzero(dense_tensor)
        values = dense_tensor[non_zero]

        # Stack indices for PyTorch sparse tensor
        indices = torch.LongTensor(np.vstack(non_zero))
        values = torch.FloatTensor(values)
        shape = dense_tensor.shape

        # Create sparse tensor
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, size=torch.Size(shape)
        )

        # Get the dimension labels
        origins = npz_data['origins'].tolist()
        destinations = npz_data['destinations'].tolist()

        # Check for time indices in various possible keys
        if 'timebins' in npz_data:
            time_indices = npz_data['timebins'].tolist()
        elif 'weekhours' in npz_data:
            time_indices = npz_data['weekhours'].tolist()
        elif 'hours' in npz_data:
            time_indices = npz_data['hours'].tolist()
        else:
            raise KeyError(
                "Neither 'timebins', 'weekhours', nor 'hours' found in the NPZ file")

        # Calculate tensor density
        density = len(values) / np.prod(shape)
        logger.info(
            f"Tensor density: {density:.6f} ({len(values)} non-zero elements)")
        logger.info(f"Tensor shape: {shape}")

        return sparse_tensor, origins, destinations, time_indices

    except Exception as e:
        logger.error(f"Error loading tensor: {str(e)}")
        raise


def save_decomposition(tucker_tensor, ranks, tensor_name, metrics, run_dir):
    """Save decomposition results and metrics with improved error handling.

    Args:
        tucker_tensor: The Tucker tensor (core, factors) to save
        ranks: Tuple of ranks (O, D, T) for each mode
        tensor_name: The name of the tensor
        metrics: The metrics dictionary
        run_dir: The run directory to save results in
    """
    try:
        # Create rank-specific subdirectory with ODT format
        rank_dir = run_dir / f"rank_{ranks[0]}_{ranks[1]}_{ranks[2]}"
        rank_dir.mkdir(exist_ok=True)
        logger.info(f"Creating rank directory: {rank_dir}")

        # Log the full path structure
        logger.info(f"Full path structure:")
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Rank directory: {rank_dir}")

        # Unpack the Tucker tensor
        core, factors = tucker_tensor

        # Convert factors to CPU and numpy for saving
        core_np = tl.to_numpy(core)
        factors_np = [tl.to_numpy(f) for f in factors]

        # Save factors
        np_file = rank_dir / f"{tensor_name}_factors.npz"
        np.savez(np_file, core=core_np, factors=factors_np)

        # Convert metrics to JSON-serializable format
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    serializable_metrics[key] = value.item()
                else:
                    serializable_metrics[key] = value.tolist()
            elif isinstance(value, list) and any(isinstance(x, torch.Tensor) for x in value):
                # Handle lists of tensors (e.g., errors_history)
                serializable_metrics[key] = [x.item() if isinstance(
                    x, torch.Tensor) else x for x in value]
            else:
                serializable_metrics[key] = value

        # Add rank information to metrics
        serializable_metrics['ranks'] = {
            'origin': ranks[0],
            'destination': ranks[1],
            'time': ranks[2]
        }

        # Save metrics
        metrics_file = rank_dir / f"{tensor_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
            f.flush()  # Ensure file is written to disk
            os.fsync(f.fileno())  # Force OS to write to physical storage

    except Exception as e:
        logger.error(f"Error saving decomposition results: {str(e)}")
        raise


def run_tucker_decomposition(input_tensor, ranks, tensor_name, output_path, optimizer='MU', max_iter=1000, tol=1e-8, init_methods='random,svd', normalize_factors=True, exact=False, algorithm='fista'):
    """Run non-negative Tucker decomposition for multiple ranks and save results.

    Args:
        input_tensor: The input tensor to decompose
        ranks: Tuple of ranks (O, D, T) for each mode
        tensor_name: Name of the tensor
        output_path: Path to save results
        optimizer: Optimization method ('MU' or 'HALS')
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        init_methods: Comma-separated list of initialization methods
        normalize_factors: Whether to normalize factors
        exact: Whether to use exact decomposition
        algorithm: Algorithm to use for HALS optimization

    Returns:
        Dictionary containing results for each rank combination
    """
    # The output_path is now the orchestrator_run_dir.
    # No new timestamped run_dir is created here.
    # Rank-specific directories will be created directly under this output_path.
    base_run_dir = Path(output_path)
    # base_run_dir.mkdir(parents=True, exist_ok=True) # Orchestrator ensures this exists

    # The run_info.json specific to this rank and attempt will be saved in the rank_X_Y_Z subdir
    # The overall run_info for all ranks is handled by the main() function.
    rank_specific_dir = base_run_dir / f"rank_{ranks[0]}_{ranks[1]}_{ranks[2]}"
    # Ensure rank-specific dir exists
    rank_specific_dir.mkdir(parents=True, exist_ok=True)

    # Save run info for this specific rank attempt
    rank_attempt_info = {
        "tensor_name": tensor_name,
        "ranks": ranks,
        "ranks_mode": {
            "origin": ranks[0],
            "destination": ranks[1],
            "time": ranks[2]
        },
        "optimizer": optimizer,
        "max_iter": max_iter,
        "tol": tol,
        "init_methods": init_methods,
        "normalize_factors": normalize_factors,
        "exact": exact,
        "algorithm": algorithm,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running"
    }
    with open(rank_specific_dir / "rank_attempt_details.json", 'w') as f:
        json.dump(rank_attempt_info, f, indent=4)

    # Use global device and device_name
    global device, device_name
    logger.info(f"Using device: {device_name}")

    # Convert tensor to torch tensor if it's not already
    if not isinstance(input_tensor, torch.Tensor):
        tensor_for_decomp = torch.tensor(input_tensor, device=device)
    else:
        tensor_for_decomp = input_tensor.to(device)

    # Convert to dense if sparse (fix for TensorLy not supporting sparse)
    if tensor_for_decomp.is_sparse:
        logger.info(
            "Converting sparse tensor to dense format for decomposition...")
        tensor_for_decomp = tensor_for_decomp.to_dense()
        logger.info(
            f"Dense tensor created with shape {tensor_for_decomp.shape}")

    # Calculate tensor norm for error normalization
    tensor_norm = torch.linalg.vector_norm(tensor_for_decomp, ord=2)

    # Parse initialization methods
    init_methods_list = init_methods.split(',')

    # Store results
    results_dict = {}

    logger.info(f"\nRunning decomposition with ranks {ranks}")
    best_error = float('inf')
    best_tucker_tensor = None
    best_init = None
    best_errors_history = None
    computation_time = None
    tucker_tensor = None  # Initialize tucker_tensor
    errors_history = None  # Initialize errors_history

    # Try each initialization method
    for init_method in init_methods_list:
        logger.info(f"Trying initialization method: {init_method}")
        try:
            start_time = time.time()
            if optimizer == 'MU':
                tucker_tensor, errors_history = non_negative_tucker(
                    tensor_for_decomp,
                    rank=ranks,
                    n_iter_max=max_iter,
                    tol=tol,
                    init=init_method,
                    return_errors=True,
                    normalize_factors=normalize_factors
                )
            elif optimizer == 'HALS':
                tucker_tensor, errors_history = non_negative_tucker_hals(
                    tensor_for_decomp,
                    rank=ranks,
                    n_iter_max=max_iter,
                    tol=tol,
                    init=init_method,
                    return_errors=True,
                    exact=exact,
                    algorithm=algorithm
                )
            end_time = time.time()
            computation_time = end_time - start_time

            # Calculate reconstruction error
            core, factors = tucker_tensor
            reconstruction = tl.tucker_to_tensor((core, factors))
            error = torch.linalg.vector_norm(tensor_for_decomp -
                                             reconstruction, ord=2) / tensor_norm

            if error < best_error:
                best_error = error
                best_tucker_tensor = tucker_tensor
                best_init = init_method
                best_errors_history = errors_history

        except Exception as e:
            logger.error(f"Error with initialization {init_method}: {str(e)}")
            continue

    # Calculate metrics for best result
    if best_tucker_tensor is not None:
        logger.info("Computing metrics for best result")
        core, factors = best_tucker_tensor
        reconstruction = tl.tucker_to_tensor((core, factors))

        # Calculate various metrics
        error = torch.linalg.vector_norm(
            tensor_for_decomp - reconstruction, ord=2) / tensor_norm
        rmse = torch.sqrt(torch.mean(
            (tensor_for_decomp - reconstruction) ** 2))
        mean_value = torch.mean(tensor_for_decomp[tensor_for_decomp > 0])
        relative_rmse = rmse / mean_value
        explained_variance = 1 - error ** 2

        # Check if factors are non-negative
        all_non_negative = all(torch.all(f >= 0) for f in factors)

        # Calculate sparsity for each factor
        factor_sparsity = [
            f"{torch.sum(f == 0).item() / f.numel() * 100:.2f}%" for f in factors]

        # Store metrics
        metrics = {
            'normalized_frobenius_error': float(error),
            'masked_rmse': float(rmse),
            'mean_nonzero_value': float(mean_value),
            'relative_rmse': float(relative_rmse),
            'explained_variance': float(explained_variance),
            'computation_time': float(computation_time) if computation_time is not None else 0.0,
            'all_factors_non_negative': bool(all_non_negative),
            'factor_sparsity': factor_sparsity,
            'device': device_name,
            'best_initialization': best_init,
            'errors_history': best_errors_history if best_errors_history is not None else []
        }

        # Log metrics
        logger.info(f"Ranks {ranks} {optimizer} decomposition completed:")
        logger.info(f"Normalized Frobenius error: {error:.6f}")
        logger.info(f"Masked RMSE: {rmse:.6f}")
        logger.info(f"Mean non-zero value: {mean_value:.6f}")
        logger.info(f"Relative RMSE (RMSE/mean): {relative_rmse:.6f}")
        logger.info(f"Explained variance: {explained_variance:.6f}")
        logger.info(
            f"Computation time: {metrics['computation_time']:.2f} seconds")
        logger.info(f"All factors non-negative: {all_non_negative}")
        logger.info(f"Factor sparsity: {factor_sparsity}")
        logger.info(f"Using device: {device_name}")
        logger.info(f"Best initialization: {best_init}")

        # Save decomposition results
        save_decomposition(best_tucker_tensor, ranks,
                           tensor_name, metrics, base_run_dir)  # Pass base_run_dir

        # Store results with optimizer-specific key
        result_key = f"{ranks}_{optimizer.lower()}"
        results_dict[result_key] = {
            'tucker_tensor': best_tucker_tensor,
            'metrics': metrics
        }

    # Update rank attempt info with completion status
    final_rank_attempt_info = rank_attempt_info.copy()
    final_rank_attempt_info["status"] = "completed"
    final_rank_attempt_info["completed_at"] = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S")
    with open(rank_specific_dir / "rank_attempt_details.json", 'w') as f:
        json.dump(final_rank_attempt_info, f, indent=4)

    logger.info("\nDecomposition complete.")
    return results_dict


def get_system_info():
    """Gather system information for reproducibility."""
    system_info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'cpu_count': os.cpu_count(),
        'date_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tensorly_version': getattr(tl, '__version__', 'unknown'),
        'numpy_version': np.__version__,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    return system_info


def create_performance_plots(metrics_dict, output_dir, tensor_name):
    """Create plots showing the evolution of performance metrics across ranks.

    Args:
        metrics_dict: Dictionary containing metrics for each rank
        output_dir: Directory to save the plots
        tensor_name: Name of the tensor being decomposed
    """
    # Extract ranks and metrics
    ranks = []
    frobenius_errors = []
    rmses = []
    explained_variances = []

    # Sort by rank to ensure proper ordering
    for rank_key in sorted(metrics_dict.keys(), key=lambda x: int(x.split('_')[0])):
        rank = int(rank_key.split('_')[0])
        metrics = metrics_dict[rank_key]['metrics']

        ranks.append(rank)
        frobenius_errors.append(metrics['normalized_frobenius_error'])
        rmses.append(metrics['masked_rmse'])
        explained_variances.append(metrics['explained_variance'])

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot each metric
    plt.plot(ranks, frobenius_errors, 'b-o', label='Frobenius Norm Error')
    plt.plot(ranks, rmses, 'r-o', label='RMSE')
    plt.plot(ranks, explained_variances, 'g-o', label='Explained Variance')

    # Add labels and title
    plt.xlabel('Rank')
    plt.ylabel('Metric Value')
    plt.title(f'Performance Metrics vs Rank for {tensor_name}')
    plt.grid(True)
    plt.legend()

    # Set y-axis to start from 0
    plt.ylim(bottom=0)

    # Save the plot
    plot_path = output_dir / 'performance_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved performance metrics plot to {plot_path}")

    # Create a second plot with RMSE and Explained Variance on separate y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot RMSE on primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Rank')
    ax1.set_ylabel('RMSE', color=color)
    ax1.plot(ranks, rmses, 'o-', color=color, label='RMSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bottom=0)  # Set RMSE axis to start from 0

    # Create secondary y-axis for Explained Variance
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Explained Variance', color=color)
    ax2.plot(ranks, explained_variances, 'o-',
             color=color, label='Explained Variance')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bottom=0)  # Set Explained Variance axis to start from 0

    # Add title and grid
    plt.title(f'RMSE and Explained Variance vs Rank for {tensor_name}')
    ax1.grid(True)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Save the plot
    plot_path = output_dir / 'rmse_variance_metrics.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved RMSE and Explained Variance plot to {plot_path}")


def parse_rank_range(rank_str):
    """Parse rank range string into list of ranks.

    Args:
        rank_str: String specifying ranks (e.g., "1-10" or "5,10,15,20")

    Returns:
        List of ranks
    """
    if '-' in rank_str:
        # Handle range format (e.g., "1-10")
        start, end = map(int, rank_str.split('-'))
        return list(range(start, end + 1))
    else:
        # Handle list format (e.g., "5,10,15,20")
        return [int(x.strip()) for x in rank_str.split(',')]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run Tucker decomposition on a tensor')
    parser.add_argument('tensor', type=str,
                        help='Path to the tensor file')
    parser.add_argument('--rank-o', type=str, required=True,
                        help='Origin mode rank(s) (e.g., "4-12")')
    parser.add_argument('--rank-d', type=str, required=True,
                        help='Destination mode rank(s) (e.g., "4-12")')
    parser.add_argument('--rank-t', type=str, required=True,
                        help='Time mode rank(s) (e.g., "1-5")')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=1e-8,
                        help='Tolerance for convergence')
    parser.add_argument('--optimizer', type=str, choices=['MU', 'HALS'],
                        default='HALS', help='Optimization method')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--init-methods', type=str, default='random,svd',
                        help='Comma-separated list of initialization methods to try')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results (default: data/results/decompositions/Tucker)')
    parser.add_argument('--algorithm', type=str, default='fista',
                        help='Algorithm to use for optimization')
    parser.add_argument('--symmetric-ranks', action='store_true',
                        help='If set, only use rank combinations where origin rank equals destination rank')
    return parser.parse_args()


def create_run_summary(tensor_file, tensor_shape, ranks_o, ranks_d, ranks_t, optimizer, device_name, init_methods, max_iter, tol, all_results):
    """Create a comprehensive run summary similar to CP decomposition.

    Args:
        tensor_file: Path to the tensor file
        tensor_shape: Shape of the tensor
        ranks_o: List of origin ranks
        ranks_d: List of destination ranks
        ranks_t: List of time ranks
        optimizer: Optimization method used
        device_name: Device used for computation
        init_methods: List of initialization methods
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
        all_results: Dictionary containing results for all rank combinations

    Returns:
        dict: Run summary containing all relevant information
    """
    # Get system information
    system_info = get_system_info()

    # Calculate tensor statistics
    tensor_density = len(tensor_shape) / np.prod(tensor_shape)

    # Create tensor info
    tensor_info = {
        "file": str(tensor_file),
        "shape": tensor_shape,
        "non_zero_elements": len(tensor_shape),
        "density": tensor_density,
        "min_value": 1.0,  # Assuming this is the case based on the data
        "using_sparse": True
    }

    # Create decomposition parameters
    decomp_params = {
        "ranks": {
            "origin": ranks_o,
            "destination": ranks_d,
            "time": ranks_t
        },
        "max_iterations": max_iter,
        "tolerance": tol,
        "optimizer": optimizer,
        "device": device_name,
        "init_methods": init_methods
    }

    # Process results
    results = {}
    for rank_key, rank_results in all_results.items():
        result_key = f"{rank_key}_{optimizer.lower()}"
        if result_key in rank_results:
            metrics = rank_results[result_key]['metrics']
            results[rank_key] = {
                "normalized_frobenius_error": float(metrics['normalized_frobenius_error']),
                "explained_variance": float(metrics['explained_variance']),
                "computation_time": float(metrics['computation_time']),
                "best_initialization": metrics['best_initialization']
            }

    # Create the complete summary
    summary = {
        "system_info": system_info,
        "tensor_info": tensor_info,
        "decomposition_params": decomp_params,
        "results": results
    }

    return summary


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set device based on arguments
    global device, device_name
    device, device_name = get_device(args.force_cpu)
    logger.info(f"Using {device_name} device for computation")

    # Process the tensor file path
    tensor_file = Path(args.tensor)
    if not tensor_file.is_absolute():
        tensor_file = Path.cwd() / tensor_file

    # Extract tensor name
    tensor_name = tensor_file.stem
    if tensor_name.startswith('odt_processed_'):
        tensor_name = tensor_name.replace('odt_processed_', 'odt_')
    elif tensor_name.startswith('odtm_processed_'):
        tensor_name = tensor_name.replace('odtm_processed_', 'odtm_')

    # Set output directory (this is now the orchestrator_run_dir)
    # The orchestrator script (run_all_decompositions_tucker.py) now creates
    # the main timestamped run directory.
    # This script (decompose_tucker.py) will use the provided args.output_dir directly.
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = DECOMP_PATH / tensor_name + '_' + args.optimizer
    # Ensure it exists, though orchestrator should create it
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using run directory provided by orchestrator: {run_dir}")

    # Parse mode-specific ranks
    ranks_o = parse_rank_range(args.rank_o)
    ranks_d = parse_rank_range(args.rank_d)
    ranks_t = parse_rank_range(args.rank_t)
    logger.info(f"Origin ranks: {ranks_o}")
    logger.info(f"Destination ranks: {ranks_d}")
    logger.info(f"Time ranks: {ranks_t}")

    # Log configuration
    logger.info("Running with configuration:")
    logger.info(f"Tensor file: {tensor_file}")
    logger.info(f"Max iterations: {args.max_iter}")
    logger.info(f"Tolerance: {args.tol}")
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Random state: {RANDOM_STATE}")
    logger.info("Using sparse tensor representation")
    logger.info(f"Optimization method: {args.optimizer}")
    logger.info(f"Initialization methods: {args.init_methods}")

    # Load and preprocess tensor
    tensor, origins, destinations, timebins = load_sparse_tensor(tensor_file)

    # Generate all combinations of ranks
    from itertools import product
    if args.symmetric_ranks:
        # Only use combinations where origin rank equals destination rank
        rank_combinations = [(r_o, r_o, r_t)
                             for r_o in ranks_o for r_t in ranks_t]
    else:
        # Use all combinations
        rank_combinations = list(product(ranks_o, ranks_d, ranks_t))
    logger.info(f"Total rank combinations: {len(rank_combinations)}")

    # No longer creating tensor_dir or an additional run_dir here.
    # The provided run_dir (args.output_dir) is the main run directory.

    logger.info(
        "Expected directory structure within the provided run directory:")
    logger.info(f"Base run directory: {run_dir}")
    logger.info(
        f"Rank subdirectories will be created as: {run_dir}/rank_O_D_T/")

    # Save initial run info with all combinations directly in the orchestrator-provided run_dir
    initial_info = {
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tensor_name": tensor_name,
        "all_rank_combinations": [list(rc) for rc in rank_combinations],
        "tensor_shape": list(tensor.shape),
        "optimizer": args.optimizer,
        "device": device_name,
        "status": "started"
    }
    with open(run_dir / "run_info.json", 'w') as f:
        json.dump(initial_info, f, indent=4)

    # Save system information directly in the orchestrator-provided run_dir
    system_info = get_system_info()
    with open(run_dir / "system_info.json", 'w') as f:
        json.dump(system_info, f, indent=4)

    # Run decomposition for each combination
    all_results = {}
    best_combination = None
    best_error = float('inf')
    best_metrics = None

    for rank_tuple in rank_combinations:
        logger.info(
            f"\nRunning Tucker decomposition for ranks {rank_tuple}...")
        results = run_tucker_decomposition(
            tensor,
            rank_tuple,  # Pass as tuple (O, D, T)
            tensor_name,
            run_dir,  # use the single run_dir for all
            optimizer=args.optimizer,
            max_iter=args.max_iter,
            tol=args.tol,
            init_methods=args.init_methods
        )
        all_results[str(rank_tuple)] = results

        # Check if this combination is better than the current best
        result_key = f"{rank_tuple}_{args.optimizer.lower()}"
        if result_key in results:
            current_error = results[result_key]['metrics']['normalized_frobenius_error']
            if current_error < best_error:
                best_error = current_error
                best_combination = rank_tuple
                best_metrics = results[result_key]['metrics']

    # Create and save run summary
    run_summary = create_run_summary(
        tensor_file=tensor_file,
        tensor_shape=list(tensor.shape),
        ranks_o=ranks_o,
        ranks_d=ranks_d,
        ranks_t=ranks_t,
        optimizer=args.optimizer,
        device_name=device_name,
        init_methods=args.init_methods,
        max_iter=args.max_iter,
        tol=args.tol,
        all_results=all_results
    )

    # Save run summary directly in the orchestrator-provided run_dir
    with open(run_dir / "run_summary.json", 'w') as f:
        json.dump(run_summary, f, indent=4)

    logger.info(f"Saved run summary to {run_dir / 'run_summary.json'}")

    # Create summary of best combination
    if best_combination is not None:
        # Convert metrics to JSON-serializable format
        def convert_metrics(metrics):
            converted = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    converted[key] = value.cpu().numpy().tolist()
                elif isinstance(value, list) and any(isinstance(x, torch.Tensor) for x in value):
                    converted[key] = [x.cpu().numpy().tolist() if isinstance(
                        x, torch.Tensor) else x for x in value]
                else:
                    converted[key] = value
            return converted

        summary = {
            "best_combination": {
                "ranks": list(best_combination),
                "ranks_mode": {
                    "origin": best_combination[0],
                    "destination": best_combination[1],
                    "time": best_combination[2]
                },
                "metrics": convert_metrics(best_metrics)
            },
            "all_combinations": {
                str(rc): {
                    "ranks": list(rc),
                    "metrics": convert_metrics(all_results[str(rc)][f"{rc}_{args.optimizer.lower()}"]['metrics'])
                } for rc in rank_combinations
            }
        }

        # Save summary directly in the orchestrator-provided run_dir
        with open(run_dir / "best_combination_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)

        logger.info("\nBest rank combination summary:")
        logger.info(f"Ranks: {best_combination}")
        logger.info(
            f"Normalized Frobenius error: {best_metrics['normalized_frobenius_error']:.6f}")
        logger.info(
            f"Explained variance: {best_metrics['explained_variance']:.6f}")
        logger.info(
            f"Computation time: {best_metrics['computation_time']:.2f} seconds")
        logger.info(
            f"Best initialization: {best_metrics['best_initialization']}")

    # Update run info with completion status in the orchestrator-provided run_dir
    final_info = initial_info.copy()
    final_info["status"] = "completed"
    final_info["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if best_combination is not None:
        final_info["best_combination"] = list(best_combination)
        final_info["best_error"] = float(best_error)
    with open(run_dir / "run_info.json", 'w') as f:
        json.dump(final_info, f, indent=4)

    logger.info("\nAll decompositions complete.")
    return all_results


if __name__ == "__main__":
    main()
