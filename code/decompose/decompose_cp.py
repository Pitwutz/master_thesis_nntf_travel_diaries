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
from tensorly.decomposition import non_negative_parafac, non_negative_parafac_hals
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
DECOMP_PATH = DATA_PATH / "results" / "decompositions"
DECOMP_PATH.mkdir(parents=True, exist_ok=True)

# Set random state for reproducibility
RANDOM_STATE = 42
logger.info(f"Using random state: {RANDOM_STATE}")

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
    - timebins/weekhours: List of time indices

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

        # Check for either timebins or weekhours
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


def save_decomposition(cp_tensor, rank, tensor_name, metrics, run_dir):
    """Save decomposition results and metrics with improved error handling.

    Args:
        cp_tensor: The CP tensor (weights, factors) to save
        rank: The rank of the decomposition
        tensor_name: The name of the tensor
        metrics: The metrics dictionary
        run_dir: The run directory to save results in
    """
    try:
        # Create rank-specific subdirectory
        rank_dir = run_dir / f"rank_{rank}"
        rank_dir.mkdir(exist_ok=True)

        # Unpack the CP tensor
        weights, factors = cp_tensor

        # Convert factors to CPU and numpy for saving
        weights_np = tl.to_numpy(weights)
        factors_np = [tl.to_numpy(f) for f in factors]

        # Save factors
        np_file = rank_dir / f"{tensor_name}_factors.npz"
        np.savez(np_file, weights=weights_np, factors=factors_np)

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

        # Save metrics
        metrics_file = rank_dir / f"{tensor_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
            f.flush()  # Ensure file is written to disk
            os.fsync(f.fileno())  # Force OS to write to physical storage

        # Create convergence plot if errors_history exists
        if 'errors_history' in serializable_metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(serializable_metrics['errors_history'])
            plt.yscale('log')
            plt.title(f'Convergence for Rank {rank}')
            plt.xlabel('Iteration')
            plt.ylabel('Reconstruction Error (log scale)')
            plt.grid(True)
            plot_path = rank_dir / f"{tensor_name}_convergence.png"
            plt.savefig(plot_path)
            plt.close()

        logger.info(
            f"Saved decomposition results for rank {rank} to {rank_dir}")

        # Force sync to Google Drive if in Colab to prevent data loss on crash
        if IS_COLAB:
            sync_drive()

    except Exception as e:
        logger.error(f"Error saving decomposition results: {e}")
        # Try to save at least the metrics to not lose all progress
        try:
            emergency_file = run_dir / \
                f"emergency_{tensor_name}_rank{rank}_metrics.json"
            with open(emergency_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            logger.info(f"Saved emergency metrics to {emergency_file}")
            if IS_COLAB:
                sync_drive()
        except Exception as e:
            logger.error(f"Could not save emergency metrics file: {e}")


def try_initialization(tensor_for_decomp, rank, init_method, optimizer, random_state, max_iter, tol, device, l1_reg=0.0, l2_reg=0.0):
    """Try CP decomposition with a specific initialization method.

    Args:
        tensor_for_decomp: Input tensor
        rank: Rank of decomposition
        init_method: Initialization method ('random' or 'svd')
        optimizer: Optimization method ('MU' or 'HALS')
        random_state: Random state for reproducibility
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        device: Device to use for computation
        l1_reg: L1 regularization parameter
        l2_reg: L2 regularization parameter

    Returns:
        tuple: (cp_tensor, error, computation_time, errors_history)
    """
    try:
        start_time = time.time()
        if optimizer == 'MU':
            # First run without regularization to get initial factors
            cp_tensor = non_negative_parafac(
                tensor_for_decomp,
                rank=rank,
                n_iter_max=max_iter,
                tol=tol,
                init=init_method,
                random_state=random_state,
                return_errors=True,
                normalize_factors=True  # Enable normalization
            )

            # Apply regularization manually if specified
            if l1_reg > 0 or l2_reg > 0:
                weights, factors = cp_tensor
                # Apply regularization to each factor
                for i in range(len(factors)):
                    if l1_reg > 0:
                        # L1 regularization (soft thresholding)
                        factors[i] = torch.nn.functional.relu(
                            factors[i] - l1_reg)
                    if l2_reg > 0:
                        # L2 regularization (shrinkage)
                        factors[i] = factors[i] / (1 + l2_reg)
                cp_tensor = (weights, factors)
        else:  # HALS
            cp_tensor = non_negative_parafac_hals(
                tensor_for_decomp,
                rank=rank,
                n_iter_max=max_iter,
                tol=tol,
                init=init_method,
                random_state=random_state,
                return_errors=True,
                normalize_factors=True  # Enable normalization
            )

        computation_time = time.time() - start_time

        # Unpack the result if it includes error history
        if isinstance(cp_tensor, tuple) and len(cp_tensor) == 2:
            cp_tensor, errors_history = cp_tensor
        else:
            errors_history = None

        # Calculate reconstruction and error
        reconstruction = tl.cp_to_tensor(cp_tensor)
        error = torch.linalg.vector_norm(tensor_for_decomp - reconstruction, ord=2) / \
            torch.linalg.vector_norm(tensor_for_decomp, ord=2)

        return cp_tensor, error, computation_time, errors_history

    except Exception as e:
        logger.error(f"Error during {init_method} initialization: {str(e)}")
        return None, float('inf'), None, None


def compute_c1_unique_zones(factors: list, top_n: int, origins: list, destinations: list, use_percentage: bool = True, percentage: float = 0.2) -> int:
    """
    Compute c1: Number of unique postal codes in the top entries across all components and both spatial factors.
    Args:
        factors: List of factor matrices (origin, destination, ...)
        top_n: Number of top indices to select per component per factor (used if use_percentage is False)
        origins: List of origin postal codes
        destinations: List of destination postal codes
        use_percentage: Whether to use percentage-based selection (True) or fixed top_n (False)
        percentage: Percentage of accumulated weights to consider (default: 0.2 for 20%)
    Returns:
        Number of unique postal codes in the top entries
    """
    unique_postal_codes: set = set()
    # Only consider the first two factors: origin and destination
    for mode, (factor, code_list) in enumerate(zip(factors[:2], [origins, destinations])):
        for component in range(factor.shape[1]):
            # Get values for this component
            values = factor[:, component]
            if use_percentage:
                # Sort values in descending order
                sorted_indices = np.argsort(values)[::-1]
                sorted_values = values[sorted_indices]
                # Calculate cumulative sum
                cumsum = np.cumsum(sorted_values)
                # Find index where cumulative sum reaches percentage of total
                total_sum = cumsum[-1]
                threshold = total_sum * percentage
                # Get indices up to threshold
                top_indices = sorted_indices[cumsum <= threshold]
            else:
                # Use fixed top_n approach
                top_indices = np.argsort(values)[-top_n:][::-1]

            for idx in top_indices:
                unique_postal_codes.add(code_list[idx])
    return len(unique_postal_codes)


def compute_c2_unique_od_pairs(factors: list, top_n: int, origins: list, destinations: list, use_percentage: bool = True, percentage: float = 0.2) -> int:
    """
    Compute c2: Number of unique OD pairs among the top entries per component.
    Args:
        factors: List of factor matrices (origin, destination, ...)
        top_n: Number of top indices to select per component per factor (used if use_percentage is False)
        origins: List of origin postal codes
        destinations: List of destination postal codes
        use_percentage: Whether to use percentage-based selection (True) or fixed top_n (False)
        percentage: Percentage of accumulated weights to consider (default: 0.2 for 20%)
    Returns:
        Number of unique OD pairs in the top entries
    """
    unique_od_pairs: set = set()
    origin_factor = factors[0]
    dest_factor = factors[1]

    for component in range(origin_factor.shape[1]):
        if use_percentage:
            # Get top origins based on percentage
            origin_values = origin_factor[:, component]
            sorted_origin_indices = np.argsort(origin_values)[::-1]
            sorted_origin_values = origin_values[sorted_origin_indices]
            origin_cumsum = np.cumsum(sorted_origin_values)
            origin_threshold = origin_cumsum[-1] * percentage
            top_origins = sorted_origin_indices[origin_cumsum <=
                                                origin_threshold]

            # Get top destinations based on percentage
            dest_values = dest_factor[:, component]
            sorted_dest_indices = np.argsort(dest_values)[::-1]
            sorted_dest_values = dest_values[sorted_dest_indices]
            dest_cumsum = np.cumsum(sorted_dest_values)
            dest_threshold = dest_cumsum[-1] * percentage
            top_dests = sorted_dest_indices[dest_cumsum <= dest_threshold]
        else:
            # Use fixed top_n approach
            top_origins = np.argsort(
                origin_factor[:, component])[-top_n:][::-1]
            top_dests = np.argsort(dest_factor[:, component])[-top_n:][::-1]

        for o in top_origins:
            for d in top_dests:
                unique_od_pairs.add((origins[o], destinations[d]))
    return len(unique_od_pairs)


def compute_c1_unique_zone_indices(factors: list, top_n: int, use_percentage: bool = True, percentage: float = 0.2) -> set:
    """
    Return the set of unique indices (not mapped postal codes) used for c1 calculation.
    Args:
        factors: List of factor matrices (origin, destination, ...)
        top_n: Number of top indices to select per component per factor (used if use_percentage is False)
        use_percentage: Whether to use percentage-based selection (True) or fixed top_n (False)
        percentage: Percentage of accumulated weights to consider (default: 0.2 for 20%)
    Returns:
        Set of unique indices
    """
    unique_indices: set = set()
    for factor in factors[:2]:
        for component in range(factor.shape[1]):
            values = factor[:, component]
            if use_percentage:
                sorted_indices = np.argsort(values)[::-1]
                sorted_values = values[sorted_indices]
                cumsum = np.cumsum(sorted_values)
                total_sum = cumsum[-1]
                threshold = total_sum * percentage
                top_indices = sorted_indices[cumsum <= threshold]
            else:
                top_indices = np.argsort(values)[-top_n:][::-1]
            for idx in top_indices:
                unique_indices.add(int(idx))
    return unique_indices


def compute_c2_unique_od_pair_indices(factors: list, top_n: int, use_percentage: bool = True, percentage: float = 0.2) -> set:
    """
    Return the set of unique OD index pairs (not mapped postal codes) used for c2 calculation.
    Args:
        factors: List of factor matrices (origin, destination, ...)
        top_n: Number of top indices to select per component per factor (used if use_percentage is False)
        use_percentage: Whether to use percentage-based selection (True) or fixed top_n (False)
        percentage: Percentage of accumulated weights to consider (default: 0.2 for 20%)
    Returns:
        Set of unique (origin_idx, dest_idx) pairs
    """
    unique_pairs: set = set()
    origin_factor = factors[0]
    dest_factor = factors[1]
    for component in range(origin_factor.shape[1]):
        if use_percentage:
            origin_values = origin_factor[:, component]
            sorted_origin_indices = np.argsort(origin_values)[::-1]
            sorted_origin_values = origin_values[sorted_origin_indices]
            origin_cumsum = np.cumsum(sorted_origin_values)
            origin_threshold = origin_cumsum[-1] * percentage
            top_origins = sorted_origin_indices[origin_cumsum <=
                                                origin_threshold]

            dest_values = dest_factor[:, component]
            sorted_dest_indices = np.argsort(dest_values)[::-1]
            sorted_dest_values = dest_values[sorted_dest_indices]
            dest_cumsum = np.cumsum(sorted_dest_values)
            dest_threshold = dest_cumsum[-1] * percentage
            top_dests = sorted_dest_indices[dest_cumsum <= dest_threshold]
        else:
            top_origins = np.argsort(
                origin_factor[:, component])[-top_n:][::-1]
            top_dests = np.argsort(dest_factor[:, component])[-top_n:][::-1]
        for o in top_origins:
            for d in top_dests:
                unique_pairs.add((int(o), int(d)))
    return unique_pairs


def run_cp_decomposition(input_tensor, ranks, tensor_name, output_path, optimizer='MU', max_iter=1000, tol=1e-8, init_methods='random,svd', l1_reg=0.0, l2_reg=0.0, top_n=2, use_percentage=True, percentage=0.2, origins=None, destinations=None, random_state=None):
    """Run CP decomposition for multiple ranks and save results.

    Args:
        input_tensor: The input tensor to decompose (already in sparse format)
        ranks: List of ranks to try
        tensor_name: Name of the tensor for saving results
        output_path: Path to save results
        optimizer: Optimization method to use ('MU' or 'HALS')
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        init_methods: Comma-separated list of initialization methods to try
        l1_reg: L1 regularization parameter
        l2_reg: L2 regularization parameter
        top_n: Number of top indices to use for spatial/OD metrics when not using percentage
        use_percentage: Whether to use percentage-based selection for c1 and c2 metrics
        percentage: Percentage of accumulated weights to consider for c1 and c2 metrics
        origins: List of origin postal codes
        destinations: List of destination postal codes
        random_state: Random state for reproducibility (defaults to RANDOM_STATE if None)

    Returns:
        results_dict: Dictionary containing results for each rank
        run_dir: Path to the directory containing results
    """
    global device, device_name

    # Use provided random_state or default to global RANDOM_STATE
    if random_state is None:
        random_state = RANDOM_STATE
        logger.info(f"Using default random state: {random_state}")
    else:
        logger.info(f"Using provided random state: {random_state}")

    # Extract tensor type, city and time resolution from tensor name
    # Example: odt_processed_utrecht_weekhour -> odt_utrecht_weekhour
    tensor_parts = tensor_name.split('_')
    if len(tensor_parts) >= 4:  # Ensure we have enough parts
        tensor_type = tensor_parts[0]  # Get 'odt' or 'odtm'
        # Join type, city and time resolution
        city_time = '_'.join([tensor_type] + tensor_parts[2:])
    else:
        city_time = tensor_name  # Fallback to full name if format is unexpected

    # Create tensor-specific directory
    tensor_dir = output_path / city_time
    tensor_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created tensor directory: {tensor_dir}")

    # Create a directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = tensor_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")

    # Save initial run info
    initial_info = {
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tensor_name": tensor_name,
        "ranks": ranks,
        "tensor_shape": list(input_tensor.shape),
        "optimizer": optimizer,
        "device": device_name,
        "init_methods": init_methods,
        "random_state": random_state,
        "status": "started"
    }
    with open(run_dir / "run_info.json", 'w') as f:
        json.dump(initial_info, f, indent=4)

    # Save system information
    system_info = get_system_info()
    with open(run_dir / "system_info.json", 'w') as f:
        json.dump(system_info, f, indent=4)

    # Dictionary to store results
    results_dict = {}

    # Convert sparse tensor to dense for decomposition
    logger.info("Converting sparse tensor to dense format for decomposition...")
    tensor_for_decomp = input_tensor.to_dense()
    logger.info(f"Dense tensor created with shape {tensor_for_decomp.shape}")

    # Move tensor to target device and ensure it's contiguous
    try:
        tensor_for_decomp = tensor_for_decomp.to(device).contiguous()
        logger.info(f"Successfully moved tensor to {device_name} device")
    except Exception as e:
        logger.error(f"Error moving tensor to device: {str(e)}")
        raise

    # Calculate tensor statistics for evaluation metrics
    tensor_norm = torch.linalg.vector_norm(tensor_for_decomp, ord=2)
    logger.info(f"Tensor Frobenius norm: {tensor_norm}")

    # Process each rank
    for rank in ranks:
        logger.info(f"\nRunning CP decompositions for rank {rank}...")

        # Initialize variables for best result
        best_error = float('inf')
        best_cp_tensor = None
        best_init = None
        best_errors_history = None

        # Try different initialization methods
        init_methods_list = init_methods.split(',')
        for init_method in init_methods_list:
            logger.info(
                f"Attempting {optimizer} decomposition with {init_method} initialization...")

            cp_tensor, error, computation_time, errors_history = try_initialization(
                tensor_for_decomp,
                rank,
                init_method,
                optimizer,
                random_state,  # Use the provided or default random_state
                max_iter,
                tol,
                device,
                l1_reg,
                l2_reg
            )

            if cp_tensor is not None and error < best_error:
                best_error = error
                best_cp_tensor = cp_tensor
                best_init = init_method
                best_errors_history = errors_history
                logger.info(
                    f"Found better solution with {init_method} initialization (error: {error:.6f})")

        # Calculate metrics for best result
        if best_cp_tensor is not None:
            weights, factors = best_cp_tensor
            reconstruction = tl.cp_to_tensor((weights, factors))

            # Calculate various metrics
            error = torch.linalg.vector_norm(tensor_for_decomp -
                                             reconstruction, ord=2) / tensor_norm
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
                'computation_time': float(computation_time),
                'all_factors_non_negative': bool(all_non_negative),
                'factor_sparsity': factor_sparsity,
                'device': device_name,
                'best_initialization': best_init,
                'errors_history': best_errors_history if best_errors_history is not None else [],
                'l1_regularization': float(l1_reg),
                'l2_regularization': float(l2_reg),
                'random_state': random_state  # Add random state to metrics
            }

            # Compute c1 and c2 metrics
            if origins is not None and destinations is not None:
                c1 = compute_c1_unique_zones(
                    [tl.to_numpy(f) for f in factors], top_n, origins, destinations, use_percentage, percentage)
                c2 = compute_c2_unique_od_pairs(
                    [tl.to_numpy(f) for f in factors], top_n, origins, destinations, use_percentage, percentage)
                metrics['c1_unique_zones'] = c1
                metrics['c2_unique_od_pairs'] = c2
                metrics['c1_c2_approach'] = 'percentage' if use_percentage else 'top_n'
                if use_percentage:
                    metrics['c1_c2_percentage'] = percentage
                else:
                    metrics['c1_c2_top_n'] = top_n

            # Log metrics
            logger.info(f"Rank {rank} {optimizer} decomposition completed:")
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
            logger.info(f"Random state: {random_state}")

            # Save decomposition results
            save_decomposition(best_cp_tensor, rank,
                               tensor_name, metrics, run_dir)

            # Store results with optimizer-specific key
            result_key = f"{rank}_{optimizer.lower()}"
            results_dict[result_key] = {
                'cp_tensor': best_cp_tensor,
                'metrics': metrics
            }

    # Update run info with completion status
    final_info = initial_info.copy()
    final_info["status"] = "completed"
    final_info["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(run_dir / "run_info.json", 'w') as f:
        json.dump(final_info, f, indent=4)

    return results_dict, run_dir


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
        description='Run CP decomposition on a tensor')
    parser.add_argument('tensor', type=str,
                        help='Path to the tensor file')
    parser.add_argument('ranks', type=str,
                        help='Rank or range of ranks to use (e.g., "5" or "1-10")')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum number of iterations')
    parser.add_argument('--tol', type=float, default=1e-8,
                        help='Tolerance for convergence')
    parser.add_argument('--optimizer', type=str, choices=['MU', 'HALS'],
                        default='MU', help='Optimization method (MU: Multiplicative Update, HALS: Hierarchical ALS)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--init-methods', type=str, default='random,svd',
                        help='Comma-separated list of initialization methods to try')
    parser.add_argument('--l1-reg', type=float, default=0.0,
                        help='L1 regularization parameter (default: 0.0)')
    parser.add_argument('--l2-reg', type=float, default=0.0,
                        help='L2 regularization parameter (default: 0.0)')
    parser.add_argument('--top-n', type=int, default=2,
                        help='Number of top indices to use for spatial/OD metrics when not using percentage (default: 2)')
    parser.add_argument('--percentage', type=float, default=0.2,
                        help='Percentage of accumulated weights to consider for c1 and c2 metrics (default: 0.2)')
    # Mutually exclusive group for top-n vs percentage
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use-percentage', action='store_true',
                       help='Use percentage-based selection for c1 and c2 metrics')
    group.add_argument('--use-top-n', action='store_true',
                       help='Use top-n selection for c1 and c2 metrics')
    parser.add_argument('--include-indices', action='store_true', default=False,
                        help='Include c1_unique_zone_indices and c2_unique_od_pair_indices in the run summary (default: False)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Custom output directory for results (default: data/results/decompositions)')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set device based on arguments
    global device, device_name
    device, device_name = get_device(args.force_cpu)
    logger.info(f"Using {device_name} device for computation")

    # Determine which approach to use
    if hasattr(args, 'use_percentage') and args.use_percentage:
        use_percentage = True
    elif hasattr(args, 'use_top_n') and args.use_top_n:
        use_percentage = False
    else:
        use_percentage = True  # Default to percentage if neither is specified

    # Process the tensor file path
    tensor_file = Path(args.tensor)
    if not tensor_file.is_absolute():
        tensor_file = Path.cwd() / tensor_file

    # Extract tensor name
    tensor_name = tensor_file.stem
    # Remove 'processed_' if present, but keep the tensor type (odt/odtm)
    if tensor_name.startswith('odt_processed_'):
        tensor_name = tensor_name.replace('odt_processed_', 'odt_')
    elif tensor_name.startswith('odtm_processed_'):
        tensor_name = tensor_name.replace('odtm_processed_', 'odtm_')

    # Use custom output directory if specified, otherwise use default
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir
    else:
        output_dir = DECOMP_PATH

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    # Parse ranks
    ranks = parse_rank_range(args.ranks)
    logger.info(f"Using ranks: {ranks}")

    # Log configuration
    logger.info("Running with configuration:")
    logger.info(f"Tensor file: {tensor_file}")
    logger.info(f"Ranks: {ranks}")
    logger.info(f"Max iterations: {args.max_iter}")
    logger.info(f"Tolerance: {args.tol}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random state: {RANDOM_STATE}")
    logger.info("Using sparse tensor representation")
    logger.info(f"Optimization method: {args.optimizer}")
    logger.info(f"Initialization methods: {args.init_methods}")
    logger.info(f"L1 regularization: {args.l1_reg}")
    logger.info(f"L2 regularization: {args.l2_reg}")
    logger.info(
        f"C1/C2 approach: {'percentage' if use_percentage else 'top_n'}")
    if use_percentage:
        logger.info(f"C1/C2 percentage: {args.percentage}")
    else:
        logger.info(f"C1/C2 top_n: {args.top_n}")

    # Load and preprocess tensor
    tensor, origins, destinations, timebins = load_sparse_tensor(tensor_file)

    # Run decomposition
    results, run_dir = run_cp_decomposition(
        tensor,
        ranks,
        tensor_file.stem,
        output_dir,
        optimizer=args.optimizer,
        max_iter=args.max_iter,
        tol=args.tol,
        init_methods=args.init_methods,
        l1_reg=args.l1_reg,
        l2_reg=args.l2_reg,
        top_n=args.top_n,
        use_percentage=use_percentage,
        percentage=args.percentage,
        origins=origins,
        destinations=destinations
    )

    # Print summary
    logger.info("\nDecomposition summary:")
    for rank in ranks:
        for method in [args.optimizer.lower()]:
            result_key = f"{rank}_{method}"
            if result_key in results:
                metrics = results[result_key]['metrics']
                logger.info(f"\nRank {rank} - {method.upper()}:")
                logger.info(
                    f"Reconstruction error: {metrics['normalized_frobenius_error']:.6f}")
                logger.info(
                    f"Explained variance: {metrics['explained_variance']:.6f}")
                if metrics['computation_time'] is not None:
                    logger.info(
                        f"Computation time: {metrics['computation_time']:.2f} seconds")
                else:
                    logger.info("Computation time: Not available")

    # Create performance plots
    create_performance_plots(results, run_dir, tensor_name)

    # Additional summary statistics for the run
    summary = {
        'system_info': get_system_info(),
        'tensor_info': {
            'file': str(tensor_file),
            'shape': list(tensor.shape),
            'non_zero_elements': len(tensor._values()) if tensor.is_sparse else int(torch.count_nonzero(tensor).item()),
            'density': float(len(tensor._values()) / np.prod(tensor.shape) if tensor.is_sparse else torch.count_nonzero(tensor).item() / np.prod(tensor.shape)),
            'min_value': float(tensor._values().min().item() if tensor.is_sparse else tensor.min().item()),
            'using_sparse': tensor.is_sparse,
            'total_trips': int(tensor._values().sum().item() if tensor.is_sparse else tensor.sum().item()),
            'mean_trips': float(tensor._values().mean().item() if tensor.is_sparse else tensor.mean().item()),
            'median_trips': float(torch.median(tensor._values()).item() if tensor.is_sparse else torch.median(tensor).item()),
            'std_trips': float(tensor._values().std().item() if tensor.is_sparse else tensor.std().item())
        },
        'decomposition_params': {
            'ranks': ranks,
            'max_iterations': args.max_iter,
            'tolerance': args.tol,
            'optimizer': args.optimizer,
            'device': device_name,
            'init_methods': args.init_methods,
            'l1_regularization': args.l1_reg,
            'l2_regularization': args.l2_reg,
            'top_n': args.top_n,
            'use_percentage': use_percentage,
            'percentage': args.percentage
        },
        'results': {
            rank: {
                'normalized_frobenius_error': results[f"{rank}_{args.optimizer.lower()}"]['metrics']['normalized_frobenius_error'],
                'explained_variance': results[f"{rank}_{args.optimizer.lower()}"]['metrics']['explained_variance'],
                'c1_unique_zones': results[f"{rank}_{args.optimizer.lower()}"]['metrics'].get('c1_unique_zones'),
                'c2_unique_od_pairs': results[f"{rank}_{args.optimizer.lower()}"]['metrics'].get('c2_unique_od_pairs'),
                'c1_c2_approach': results[f"{rank}_{args.optimizer.lower()}"]['metrics'].get('c1_c2_approach'),
                'c1_c2_percentage': results[f"{rank}_{args.optimizer.lower()}"]['metrics'].get('c1_c2_percentage'),
                'c1_c2_top_n': results[f"{rank}_{args.optimizer.lower()}"]['metrics'].get('c1_c2_top_n'),
                'c1_unique_zone_indices': (list(compute_c1_unique_zone_indices(
                    [tl.to_numpy(
                        f) for f in results[f"{rank}_{args.optimizer.lower()}"]['cp_tensor'][1][:2]],
                    args.top_n, use_percentage, args.percentage)) if args.include_indices else None),
                'c2_unique_od_pair_indices': ([list(pair) for pair in compute_c2_unique_od_pair_indices(
                    [tl.to_numpy(
                        f) for f in results[f"{rank}_{args.optimizer.lower()}"]['cp_tensor'][1][:2]],
                    args.top_n, use_percentage, args.percentage)] if args.include_indices else None)
            } for rank in ranks
        }
    }

    # Save summary
    with open(run_dir / "run_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Saved run summary to {run_dir / 'run_summary.json'}")

    # Sync to drive if on Colab
    if IS_COLAB:
        sync_drive()

    logger.info("\nDecomposition complete. Results saved to:")
    logger.info(f"{run_dir}")

    return results, run_dir


if __name__ == "__main__":
    main()
