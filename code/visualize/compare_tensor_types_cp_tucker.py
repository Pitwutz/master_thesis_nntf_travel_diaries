import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
import glob
import ast
import argparse
from datetime import datetime
import sys
import os

# Add the project root to sys.path so config.py can be imported
project_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import PROJECT_ROOT
"""
Tensor Type Comparison Tool

This script analyzes and visualizes the differences between various tensor types and their decompositions.
It provides comparative analysis across different tensor representations of the same data.

Key Features:
- Multiple Tensor Types: Compares timebin, weekhour, hourly_weekday, and hourly_weekend tensors
- Comparative Metrics: Analyzes various performance metrics across tensor types
- City Comparison: Provides side-by-side analysis for different cities (Utrecht and Rotterdam)
- Visualization: Generates comparative plots for easy analysis

Metrics Analyzed:
- Normalized Frobenius Error
- Explained Variance
- Unique Zones (C1)
- Distinct OD Pairs (C2)
- Unique Time Bins
- Time Bin Ratios
- Number of Core Triplets (Energy Threshold)

Note:
    This script is designed for comparative analysis. For detailed analysis of a single tensor,
    use visualize_tensor_components.py instead.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare tensor types and their decompositions')
    parser.add_argument('--tucker-path', type=str,
                        default=str(
                            PROJECT_ROOT) + "/data/results/decompositions/tucker/hourly_analysis",
                        help='Path to Tucker decomposition results')
    parser.add_argument('--decomposition-mode', type=str, default='tucker',
                        choices=['cp', 'tucker'],
                        help='Decomposition mode to analyze')
    parser.add_argument('--tensor-type', type=str, default='hourly_weekday',
                        choices=['timebin', 'weekhour',
                                 'hourly_weekday', 'hourly_weekend'],
                        help='Type of tensor to analyze')
    parser.add_argument('--optimizer', type=str, default='MU',
                        choices=['MU', 'HALS'],
                        help='Optimizer to use for Tucker decomposition')
    parser.add_argument('--top-selection', type=str, default='cumulative',
                        choices=['cumulative', 'top_k'],
                        help='Method for selecting top components')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='Threshold for cumulative selection')
    parser.add_argument('--top-k', type=int, default=2,
                        help='Number of top components to select')
    parser.add_argument('--core-energy-threshold', type=float, default=0.80,
                        help='Energy threshold for core tensor analysis')
    parser.add_argument('--export-json', action='store_true',
                        help='Export performance metrics as JSON file')
    return parser.parse_args()


# Set style
plt.style.use('seaborn-v0_8-colorblind')
sns.set_palette("colorblind")

# Enable LaTeX rendering with standard font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
})

# Define paths
BASE_PATH = Path(
    str(PROJECT_ROOT) + "")
CP_COMPARE_RESULTS_PATH = BASE_PATH / \
    "data/results/decompositions/CP/final/peak hours normalized"


# Get command line arguments
args = parse_args()
TUCKER_RESULTS_BASE_PATH = Path(args.tucker_path)

# Define tensor types and their display names
TENSOR_TYPES = {
    'timebin': 'Peak Hour',
    'weekhour': 'Week Hour',
    'hourly_weekday': 'Hourly Weekday',
    'hourly_weekend': 'Hourly Weekend'
}

# Define optimizers to compare for Tucker
OPTIMIZERS_TO_COMPARE = ['MU', 'HALS']

# Number of time bins for each tensor type
TIME_BINS_PER_TYPE = {
    'timebin': 35,
    'weekhour': 168,
    'hourly_weekday': 24,
    'hourly_weekend': 24
}

# Define metrics and their properties
METRICS = {
    'normalized_frobenius_error': {
        'title': 'Normalized Frobenius Error',
        'ylim': (0, 1),
        'invert_y': True  # Lower is better
    },
    'explained_variance': {
        'title': 'Explained Variance',
        'ylim': (0, 1),
        'invert_y': False  # Higher is better
    },
    'c1_unique_zones': {
        'title': '$c_1$: Unique Zones',
        'ylim': (0, 50),
        'invert_y': False  # Higher is better
    },
    'c2': {
        'title': '$c_2$: Distinct OD Pairs',
        'ylim': (0, 50),
        'invert_y': False  # Higher is better
    },
    'c3_distinct_time_bins': {
        'title': '$c_3$: Distinct Time Bins',
        'ylim': (0, 50),
        'invert_y': False
    },
    'c3_ratio_distinct_available_time': {
        'title': '$c_{3,\\mathrm{rel}}$: Ratio Distinct/Available Time',
        'ylim': (0, 1),
        'invert_y': False
    },
    'num_core_triplets': {
        # Title for the subplot
        'title': 'Core: Num. Triplets (Energy Thresh.)',
        'ylim': (0, 25),  # Initial guess, can be adjusted
        'invert_y': False
    }
}

# Add a list of metrics that require factor loading
METRICS_REQUIRING_FACTORS = ['c1_unique_zones',
                             'c2',
                             'c3_distinct_time_bins',
                             'c3_ratio_distinct_available_time',
                             'num_core_triplets']


def find_latest_run_file(base_path: Path, filename: str = "run_summary.json") -> Path:
    """Find the latest specified JSON file (e.g., run_summary.json or best_combination_summary.json) in the given directory."""
    run_dirs = list(base_path.glob("run_*"))
    if not run_dirs:
        return None

    latest_run = max(run_dirs, key=lambda x: x.name.split('_')[1:])
    summary_path = latest_run / filename
    return summary_path if summary_path.exists() else None


def load_decomposition_data(city: str, tensor_type: str, category: str = None,  # Category is optional, mainly for CP
                            decomposition_mode: str = "cp", optimizer: str = None) -> dict:
    """Load decomposition data from the appropriate summary file."""
    data_file_to_load = "run_summary.json"  # Default for CP

    if decomposition_mode == "cp":
        if not category:
            raise ValueError(
                "Category must be provided for CP decomposition mode.")
        base_path = CP_COMPARE_RESULTS_PATH / city / \
            category / f"odt_{city}_{tensor_type}"
    elif decomposition_mode == "tucker":
        # For Tucker, path is TUCKER_RESULTS_BASE_PATH / odt_{city}_{tensor_type} / optimizer / run_* / best_combination_summary.json
        base_path = TUCKER_RESULTS_BASE_PATH / \
            f"odt_{city}_{tensor_type}" / optimizer
        # Tucker uses this for easier 'all_combinations' access
        data_file_to_load = "best_combination_summary.json"
    else:
        raise ValueError(f"Unknown decomposition_mode: {decomposition_mode}")

    # For Tucker, we need to look in the run directory
    if decomposition_mode == "tucker":
        run_dirs = list(base_path.glob("run_*"))
        if not run_dirs:
            raise FileNotFoundError(
                f"No run directories found in {base_path} for {decomposition_mode} mode (city: {city}, tensor_type: {tensor_type}, optimizer: {optimizer})")
        latest_run = max(run_dirs, key=lambda x: x.name.split('_')[1:])
        summary_path = latest_run / data_file_to_load
    else:
        summary_path = find_latest_run_file(
            base_path, filename=data_file_to_load)

    if summary_path is None or not summary_path.exists():
        raise FileNotFoundError(
            f"No '{data_file_to_load}' found in {base_path} for {decomposition_mode} mode (city: {city}, tensor_type: {tensor_type}, category: {category}, optimizer: {optimizer if optimizer else 'N/A'})")

    with open(summary_path, 'r') as f:
        return json.load(f)


def get_time_factor_index(tensor_type: str) -> int:
    """Return the index of the time mode in the factors array for each tensor type."""
    # Time is the third mode (index 2) in ODT tensor (Origin, Destination, Time)
    if tensor_type in ['timebin', 'weekhour', 'hourly_weekday', 'hourly_weekend']:
        # Assuming factors are stored [Origin, Destination, Time] for CP
        return 2
    raise ValueError(
        f"Unknown tensor type for time factor index: {tensor_type}")


def load_cp_factor_matrix(city: str, category: str, tensor_type: str, rank: int, factor_idx: int) -> np.ndarray | None:
    """Load a specific factor matrix (origin, destination, or time) for CP decomposition 
       for a given rank from the .npz file. Returns None if not found or error.
    """
    # Path for CP factors: CP_COMPARE_RESULTS_PATH / city / category / odt_city_tensor_type / run_... / rank_X / odt_processed_city_tensor_type_factors.npz
    base_run_path = CP_COMPARE_RESULTS_PATH / \
        city / category / f"odt_{city}_{tensor_type}"

    # find_latest_run_file needs the directory containing run_* folders (i.e., base_run_path)
    # Its result is the path to the summary file WITHIN a run_* folder.
    summary_file_in_latest_run = find_latest_run_file(
        base_run_path, "run_summary.json")
    if not summary_file_in_latest_run:
        # print(f"No run_summary.json found in {base_run_path} to locate factors.")
        return None
    # This is the specific run_YYYYMMDD_HHMMSS folder
    latest_run_dir_path = summary_file_in_latest_run.parent

    factors_file_name = f"odt_processed_{city}_{tensor_type}_factors.npz"
    factors_file_path = latest_run_dir_path / \
        f"rank_{rank}" / factors_file_name

    if not factors_file_path.exists():
        # print(f"CP Factors file not found: {factors_file_path}")
        return None

    try:
        data = np.load(factors_file_path, allow_pickle=True)
        factors = data.get('factors')
        if factors is None or not isinstance(factors, (list, np.ndarray)) or not (0 <= factor_idx < len(factors)):
            # print(f"Factor index {factor_idx} out of bounds or 'factors' key missing/invalid in {factors_file_path}")
            return None
        return factors[factor_idx]
    except Exception as e:
        # print(f"Error loading CP factors from {factors_file_path}: {e}")
        return None


def count_total_unique_time_bins(time_factors: np.ndarray, top_k: int = 2) -> int:
    """Count the total number of unique time bins that are in the top_k for any component."""
    indices = set()
    for i in range(time_factors.shape[1]):
        comp = time_factors[:, i]
        top_indices = np.argpartition(comp, -top_k)[-top_k:]
        indices.update(top_indices)
    return len(indices)


def count_total_unique_indices_top_k(factor_matrix: np.ndarray, top_k: int = 2) -> int:
    """Count the total number of unique indices that are in the top_k for any component."""
    indices = set()
    if factor_matrix is None or factor_matrix.ndim != 2:
        return 0
    for i in range(factor_matrix.shape[1]):
        comp = factor_matrix[:, i]
        # Ensure get_top_indices_top_k returns a set of integers
        indices.update(get_top_indices_top_k(comp, top_k))
    return len(indices)


def count_total_unique_indices_cumulative(factor_matrix: np.ndarray, threshold: float = 0.25) -> int:
    """
    Count the total number of unique indices that together account for at least `threshold`
    of the sum of values in each component, across all components.
    """
    indices = set()
    if factor_matrix is None or factor_matrix.ndim != 2:
        return 0
    for i in range(factor_matrix.shape[1]):
        comp = factor_matrix[:, i]
        # Ensure get_top_indices_cumulative returns a set of integers
        indices.update(get_top_indices_cumulative(comp, threshold))
    return len(indices)


def get_top_indices_cumulative(values: np.ndarray, threshold: float) -> set:
    """Return indices whose cumulative sum reaches the threshold percentage of the total."""
    sorted_indices = np.argsort(values)[::-1]
    sorted_values = values[sorted_indices]
    cumsum = np.cumsum(sorted_values)
    total_sum = cumsum[-1]
    if total_sum == 0:  # Avoid division by zero if all values are zero
        return set()
    cutoff = total_sum * threshold
    # Include indices up to and including the one that meets/exceeds cutoff
    # Find first index where cumsum >= cutoff
    idx_cutoff = np.searchsorted(cumsum, cutoff, side='left')
    # Take all indices up to idx_cutoff inclusive
    top_indices_list = sorted_indices[:idx_cutoff + 1]
    return set(top_indices_list)


def get_top_indices_top_k(values: np.ndarray, top_k: int) -> set:
    """Return indices of the top_k values."""
    # Ensure top_k is not greater than the number of available values
    actual_top_k = min(top_k, len(values))
    if actual_top_k <= 0:
        return set()
    top_indices = np.argpartition(values, -actual_top_k)[-actual_top_k:]
    return set(top_indices)


def load_tucker_core_tensor(city: str, base_tensor_type: str, optimizer: str,
                            rank_tuple: tuple[int, int, int]) -> np.ndarray | None:
    """Load the core tensor G for Tucker decomposition.
       Returns None if not found or error.
    """
    optimizer_run_base_path = TUCKER_RESULTS_BASE_PATH / \
        f"odt_{city}_{base_tensor_type}" / optimizer

    summary_file_in_latest_run = find_latest_run_file(
        optimizer_run_base_path, "best_combination_summary.json")
    if not summary_file_in_latest_run:
        print(
            f"No best_combination_summary.json found in {optimizer_run_base_path} to locate Tucker core tensor run folder.")
        return None
    latest_run_dir_path = summary_file_in_latest_run.parent

    rank_o, rank_d, rank_t = rank_tuple
    rank_subdir_name = f"rank_{rank_o}_{rank_d}_{rank_t}"

    processed_tensor_name_stem = f"odt_processed_{city}_{base_tensor_type}"
    # Common filename pattern for Tucker results including the core tensor
    # Usually, the core and factors are in the same file.
    # Assuming core is in the same file as factors
    results_file_name = f"{processed_tensor_name_stem}_Tucker_factors.npz"
    results_file_path = latest_run_dir_path / rank_subdir_name / results_file_name

    if not results_file_path.exists():
        # Try alternative naming if "processed" is not in the filename stem
        raw_tensor_name_stem = f"odt_{city}_{base_tensor_type}"
        # Again, assuming core is with factors
        alt_results_file_name = f"{raw_tensor_name_stem}_factors.npz"
        alt_results_file_path = latest_run_dir_path / \
            rank_subdir_name / alt_results_file_name
        if alt_results_file_path.exists():
            results_file_path = alt_results_file_path
        else:
            # Try one more common pattern if the above are not found
            # Sometimes it's just named 'tucker_decomposition.npz' or similar generic name if saved by a different script part.
            # However, sticking to convention derived from load_tucker_factor_matrix for now.
            # The key is that `decompose_tucker.py` saves core and factors into `_Tucker_factors.npz`.
            print(
                f"Tucker results file (for core) not found: {results_file_path} (and alt: {alt_results_file_path if 'alt_results_file_path' in locals() else 'N/A'})")
            return None

    try:
        data = np.load(results_file_path, allow_pickle=True)
        core_tensor = data.get('core')  # Common key for core tensor
        if core_tensor is None:
            # Try 'core_tensor' as another common key
            core_tensor = data.get('core_tensor')

        if core_tensor is None:
            print(
                f"Key 'core' or 'core_tensor' not found in {results_file_path}")
            return None
        if not isinstance(core_tensor, np.ndarray):
            print(
                f"Core tensor loaded from {results_file_path} is not a NumPy array.")
            return None
        # Basic check for 3D core tensor
        if core_tensor.ndim != 3:
            print(
                f"Loaded core tensor from {results_file_path} is not 3-dimensional (shape: {core_tensor.shape}). Expected (R_o, R_d, R_t).")
            # Depending on strictness, one might return None or try to reshape if possible, but better to be strict here.
            return None

        # Check if core tensor dimensions match the rank_tuple
        # core_tensor.shape should be (rank_o, rank_d, rank_t)
        if core_tensor.shape != rank_tuple:
            print(
                f"Warning: Loaded core tensor shape {core_tensor.shape} from {results_file_path} does not match expected rank tuple {rank_tuple}. This might indicate a mismatch.")
            # Decide if this is a fatal error. For now, proceed with a warning.
            # Potentially return None if strict matching is required.

        return core_tensor
    except Exception as e:
        print(
            f"Error loading Tucker core tensor from {results_file_path}: {e}")
        return None


def get_top_core_pattern_indices(core_tensor_G: np.ndarray, energy_threshold: float = 0.80) -> list[tuple[int, int, int]]:
    """Get 3D indices of core tensor patterns that meet the energy threshold.

    This function identifies the top core tensor triplets (i,j,k) that collectively
    meet a specified cumulative energy threshold. The energy of each triplet is
    its squared value. The function is optimized using NumPy for performance.

    Args:
        core_tensor_G: The 3D Tucker core tensor (R_o, R_d, R_t).
        energy_threshold: The cumulative energy threshold (e.g., 0.80 for 80%).
                          Must be between 0 (exclusive) and 1 (inclusive).

    Returns:
        A list of 3D indices (idx_o, idx_d, idx_t) for the selected patterns,
        sorted by energy in descending order.
    """
    if not isinstance(core_tensor_G, np.ndarray) or core_tensor_G.ndim != 3:
        # print("Error: core_tensor_G must be a 3D NumPy array.")
        return []
    if not (0 < energy_threshold <= 1.0):
        # print("Error: energy_threshold must be between 0 (exclusive) and 1 (inclusive).")
        return []

    # Calculate squared values (energies) of core tensor elements
    energies = np.square(core_tensor_G)
    total_energy = np.sum(energies)

    if total_energy == 0:
        return []  # Avoid division by zero if core tensor is all zeros

    flat_energies = energies.flatten()

    # Get indices that would sort the flattened energies in descending order
    order = np.argsort(flat_energies)[::-1]

    # Calculate cumulative energy (normalized)
    cum_energy = np.cumsum(flat_energies[order]) / total_energy

    # Find the number of components needed to reach the energy threshold.
    # np.searchsorted finds the first index where cum_energy >= threshold.
    # We add 1 to the index to get the count of elements to include.
    k_threshold = np.searchsorted(
        cum_energy, energy_threshold, side='left') + 1

    # Get the flat indices of the top components
    top_flat_indices = order[:k_threshold]

    # Convert flat indices back to 3D indices
    top_3d_indices_tuple = np.unravel_index(
        top_flat_indices, core_tensor_G.shape)

    # Convert the tuple of arrays from unravel_index into a list of (i,j,k) tuples
    selected_indices = list(zip(
        top_3d_indices_tuple[0], top_3d_indices_tuple[1], top_3d_indices_tuple[2]
    ))

    # Ensure at least one pattern is selected if any exist, as long as threshold > 0.
    # This is implicitly handled by the k_threshold calculation, but as a safeguard,
    # if the list is empty and the tensor is not, it means the threshold was not met.
    # The logic above should still select the top element if its energy is > 0.
    # This check remains as a conceptual safeguard.
    if not selected_indices and core_tensor_G.size > 0:
        top_flat_indices = order[:1]
        top_3d_indices_tuple = np.unravel_index(
            top_flat_indices, core_tensor_G.shape)
        selected_indices = list(zip(
            top_3d_indices_tuple[0], top_3d_indices_tuple[1], top_3d_indices_tuple[2]
        ))

    return selected_indices


def load_tucker_factor_matrix(city: str, base_tensor_type: str, optimizer: str,
                              rank_tuple: tuple[int, int, int], factor_idx: int) -> np.ndarray | None:
    """Load a specific factor matrix (origin, destination, or time) for Tucker decomposition.
       Returns None if not found or error.
    """
    # Path for Tucker factors:
    # TUCKER_RESULTS_BASE_PATH / odt_city_base_tensor_type / optimizer / run_... / rank_O_D_T / odt_processed_city_base_tensor_type_Tucker_factors.npz (or similar)

    optimizer_run_base_path = TUCKER_RESULTS_BASE_PATH / \
        f"odt_{city}_{base_tensor_type}" / optimizer

    # Find the latest run directory under the optimizer path
    # We need best_combination_summary.json to find the specific run folder, but factors are inside that run folder.
    # Let's assume find_latest_run_file can point us to the relevant run folder if we give it the optimizer_run_base_path.
    # The factors are typically saved per rank combination within a single run timestamped folder.

    # The best_combination_summary.json is at the root of the run_... folder.
    # So, finding it gives us the run folder path.
    summary_file_in_latest_run = find_latest_run_file(
        optimizer_run_base_path, "best_combination_summary.json")
    if not summary_file_in_latest_run:
        print(
            f"No best_combination_summary.json found in {optimizer_run_base_path} to locate Tucker factors run folder.")
        return None
    # This is the specific run_YYYYMMDD_HHMMSS folder
    latest_run_dir_path = summary_file_in_latest_run.parent

    rank_o, rank_d, rank_t = rank_tuple
    rank_subdir_name = f"rank_{rank_o}_{rank_d}_{rank_t}"

    # Construct the expected filename for Tucker factors. This might vary based on actual saving logic in decompose_tucker.py
    # Example from decompose_tucker.py save_decomposition:
    # save_path_factors = run_dir / f"{tensor_name}_Tucker_factors.npz" (where tensor_name is like odt_utrecht_timebin)
    # And run_dir for decompose_tucker is rank_specific_dir in the orchestrator context.
    # So, the file should be directly in the rank_subdir_name.
    # The tensor_name in decompose_tucker.py might be like 'odt_utrecht_timebin' or 'odt_processed_utrecht_timebin' before processing.
    # Let's assume it becomes something like 'odt_processed_city_base_tensor_type' for consistency or check decompose_tucker.py output.
    # From an earlier log: "odt_processed_utrecht_timebin_Tucker_factors.npz" seems to be a pattern.
    # Matches CP naming for processed tensors
    processed_tensor_name_stem = f"odt_processed_{city}_{base_tensor_type}"
    factors_file_name = f"{processed_tensor_name_stem}_Tucker_factors.npz"
    factors_file_path = latest_run_dir_path / rank_subdir_name / factors_file_name

    if not factors_file_path.exists():
        # Attempt an alternative common naming from decompose_tucker.py if tensor_name isn't prefixed with "processed"
        # e.g. if tensor_name was 'odt_utrecht_timebin' in save_decomposition
        raw_tensor_name_stem = f"odt_{city}_{base_tensor_type}"
        alt_factors_file_name = f"{raw_tensor_name_stem}_factors.npz"
        alt_factors_file_path = latest_run_dir_path / \
            rank_subdir_name / alt_factors_file_name
        if alt_factors_file_path.exists():
            factors_file_path = alt_factors_file_path
        else:
            print(
                f"Tucker Factors file not found: {factors_file_path} (and alt: {alt_factors_file_path})")
            return None

    try:
        data = np.load(factors_file_path, allow_pickle=True)
        # For Tucker, factors are often saved as 'core' and 'factors' (list of factor matrices)
        # We need the factor matrices from the 'factors' list.
        tucker_factor_list = data.get('factors')
        if tucker_factor_list is None or not isinstance(tucker_factor_list, (list, np.ndarray)) or not (0 <= factor_idx < len(tucker_factor_list)):
            print(
                f"Factor index {factor_idx} out of bounds or 'factors' key missing/invalid in {factors_file_path}")
            return None
        return tucker_factor_list[factor_idx]
    except Exception as e:
        print(f"Error loading Tucker factors from {factors_file_path}: {e}")
        return None


def extract_metrics(
    data: dict,
    metric: str,
    decomposition_mode: str = "cp",
    city: str = None,
    category: str = None,
    tensor_type: str = None,
    optimizer_for_factors: str = None,
    top_selection_method: str = "cumulative",
    top_k: int = 2,
    threshold: float = 0.25,
    core_energy_threshold: float = 0.80  # New parameter for Tucker core energy
) -> tuple[list[float], list[float]]:
    """Extract metric values for all ranks. For unique_time_bins, c1, and c2, calculate from factors using the selected method."""
    processed_results = []  # List of (sortable_rank_value, metric_value)

    if not data:  # Handle empty or None data early
        return [], []

    if decomposition_mode == "cp":
        source_dict = data.get("results", {})
        if not isinstance(source_dict, dict):
            source_dict = {}  # Ensure it's a dict

        for rank_str_cp, metric_data_cp in source_dict.items():
            try:
                rank_val_cp = int(rank_str_cp)
                metric_val = np.nan  # Default to NaN

                if metric in METRICS_REQUIRING_FACTORS:
                    if metric.startswith('c3_distinct_time_bins') or metric == 'c3_ratio_distinct_available_time':
                        time_factor_idx = get_time_factor_index(tensor_type)
                        factors_matrix = load_cp_factor_matrix(
                            city, category, tensor_type, rank_val_cp, time_factor_idx)
                        if factors_matrix is not None:
                            total_unique: int
                            if top_selection_method == "cumulative":
                                total_unique = count_total_unique_indices_cumulative(
                                    factors_matrix, threshold=threshold)
                            else:  # top_k
                                total_unique = count_total_unique_indices_top_k(
                                    factors_matrix, top_k=top_k)

                            if metric == 'c3_ratio_distinct_available_time':
                                denom = TIME_BINS_PER_TYPE.get(tensor_type, 1)
                                metric_val = float(
                                    total_unique) / denom if denom > 0 else np.nan
                            else:  # c3_distinct_time_bins
                                metric_val = float(total_unique)

                    elif metric == 'c1_unique_zones':
                        origin_factors = load_cp_factor_matrix(
                            city, category, tensor_type, rank_val_cp, 0)  # Factor 0: Origin
                        dest_factors = load_cp_factor_matrix(
                            city, category, tensor_type, rank_val_cp, 1)   # Factor 1: Destination

                        if origin_factors is not None and dest_factors is not None and \
                           origin_factors.ndim == 2 and dest_factors.ndim == 2 and \
                           origin_factors.shape[1] == dest_factors.shape[1]:  # Check components align

                            all_unique_zone_indices = set()
                            # Iterate over components (columns)
                            for i in range(origin_factors.shape[1]):
                                if top_selection_method == "cumulative":
                                    all_unique_zone_indices.update(
                                        get_top_indices_cumulative(origin_factors[:, i], threshold))
                                    all_unique_zone_indices.update(
                                        get_top_indices_cumulative(dest_factors[:, i], threshold))
                                else:  # top_k
                                    all_unique_zone_indices.update(
                                        get_top_indices_top_k(origin_factors[:, i], top_k))
                                    all_unique_zone_indices.update(
                                        get_top_indices_top_k(dest_factors[:, i], top_k))
                            metric_val = float(len(all_unique_zone_indices))

                    elif metric == 'c2':  # Distinct OD Pairs
                        origin_factors = load_cp_factor_matrix(
                            city, category, tensor_type, rank_val_cp, 0)  # Origin
                        dest_factors = load_cp_factor_matrix(
                            city, category, tensor_type, rank_val_cp, 1)   # Destination

                        if origin_factors is not None and dest_factors is not None and \
                           origin_factors.ndim == 2 and dest_factors.ndim == 2 and \
                           origin_factors.shape[1] == dest_factors.shape[1]:

                            unique_od_pairs = set()
                            # Iterate over components
                            for i in range(origin_factors.shape[1]):
                                current_origin_indices: set[int]
                                current_dest_indices: set[int]
                                if top_selection_method == "cumulative":
                                    current_origin_indices = get_top_indices_cumulative(
                                        origin_factors[:, i], threshold)
                                    current_dest_indices = get_top_indices_cumulative(
                                        dest_factors[:, i], threshold)
                                else:  # top_k
                                    current_origin_indices = get_top_indices_top_k(
                                        origin_factors[:, i], top_k)
                                    current_dest_indices = get_top_indices_top_k(
                                        dest_factors[:, i], top_k)

                                for o_idx in current_origin_indices:
                                    for d_idx in current_dest_indices:
                                        if o_idx != d_idx:
                                            # Store as sorted tuple to count (A,B) and (B,A) as one if desired
                                            unique_od_pairs.add(
                                                tuple(sorted((o_idx, d_idx))))
                                            # For C2 as *distinct* OD pairs, (o_idx, d_idx) is fine.
                            metric_val = float(len(unique_od_pairs))
                else:
                    if isinstance(metric_data_cp, dict):
                        metric_val = metric_data_cp.get(metric, np.nan)
                    else:
                        metric_val = np.nan
                processed_results.append((rank_val_cp, metric_val))
            except (ValueError, TypeError, KeyError, IndexError) as e:
                processed_results.append(
                    (int(rank_str_cp) if rank_str_cp.isdigit() else float('inf'), np.nan))

    elif decomposition_mode == "tucker":
        source_dict = data.get("all_combinations", {})
        if not isinstance(source_dict, dict):
            source_dict = {}

        for rank_tuple_str, rank_data_tucker in source_dict.items():
            try:
                # AST literal_eval is good for "(o,d,t)" or "[o,d,t]" string formats
                parsed_ranks_tuple = ast.literal_eval(rank_tuple_str)
                if not (isinstance(parsed_ranks_tuple, tuple) and len(parsed_ranks_tuple) == 3 and all(isinstance(r, int) for r in parsed_ranks_tuple)):
                    # Try to parse simpler formats like "3,3,2" if ast.literal_eval fails or produces wrong type
                    try:
                        parts = [int(p.strip())
                                 for p in rank_tuple_str.strip("()[]").split(',')]
                        if len(parts) == 3:
                            parsed_ranks_tuple = tuple(parts)
                        else:
                            raise ValueError(
                                "Parsed rank is not a 3-element integer tuple after basic parse")
                    except ValueError:
                        raise ValueError(
                            f"Cannot parse rank string '{rank_tuple_str}' into a 3-element integer tuple for Tucker")

                # rank_o, rank_d, rank_t = parsed_ranks_tuple # Unpacking not needed if we use the tuple directly
                # The x-axis value will be the tuple itself for sorting and later for plotting
                x_axis_identifier = parsed_ranks_tuple
                metric_val = np.nan

                if metric in METRICS_REQUIRING_FACTORS:
                    if metric.startswith('c3_distinct_time_bins') or metric == 'c3_ratio_distinct_available_time':
                        time_factor_idx = get_time_factor_index(tensor_type)
                        factors_matrix = load_tucker_factor_matrix(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple, time_factor_idx)
                        if factors_matrix is None:
                            print(
                                f"Debug: Tucker time factors_matrix is None for {city}, {tensor_type}, {optimizer_for_factors}, {parsed_ranks_tuple}")
                        if factors_matrix is not None:
                            total_unique: int
                            if top_selection_method == "cumulative":
                                total_unique = count_total_unique_indices_cumulative(
                                    factors_matrix, threshold=threshold)
                            else:
                                total_unique = count_total_unique_indices_top_k(
                                    factors_matrix, top_k=top_k)

                            if metric == 'c3_ratio_distinct_available_time':
                                denom = TIME_BINS_PER_TYPE.get(tensor_type, 1)
                                metric_val = float(
                                    total_unique) / denom if denom > 0 else np.nan
                            else:  # c3_distinct_time_bins
                                metric_val = float(total_unique)

                    elif metric == 'c1_unique_zones':
                        origin_factors = load_tucker_factor_matrix(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple, 0)
                        dest_factors = load_tucker_factor_matrix(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple, 1)
                        core_tensor_g = load_tucker_core_tensor(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple)

                        if origin_factors is not None and dest_factors is not None and core_tensor_g is not None and \
                           origin_factors.ndim == 2 and dest_factors.ndim == 2 and \
                           origin_factors.shape[1] == core_tensor_g.shape[0] and \
                           dest_factors.shape[1] == core_tensor_g.shape[1]:  # Check factor components align with core dimensions

                            selected_core_indices = get_top_core_pattern_indices(
                                core_tensor_g, core_energy_threshold)
                            all_unique_zone_indices = set()

                            if not selected_core_indices:
                                print(
                                    f"Debug: Tucker C1 - No core patterns selected for {city}, {tensor_type}, {optimizer_for_factors}, {parsed_ranks_tuple} with energy_threshold {core_energy_threshold}")

                            for core_o_idx, core_d_idx, _ in selected_core_indices:
                                # Ensure indices are within bounds for factor matrices
                                if core_o_idx < origin_factors.shape[1] and core_d_idx < dest_factors.shape[1]:
                                    origin_component_vector = origin_factors[:, core_o_idx]
                                    dest_component_vector = dest_factors[:,
                                                                         core_d_idx]

                                    if top_selection_method == "cumulative":
                                        all_unique_zone_indices.update(
                                            get_top_indices_cumulative(origin_component_vector, threshold))
                                        all_unique_zone_indices.update(
                                            get_top_indices_cumulative(dest_component_vector, threshold))
                                    else:  # top_k
                                        all_unique_zone_indices.update(
                                            get_top_indices_top_k(origin_component_vector, top_k))
                                        all_unique_zone_indices.update(
                                            get_top_indices_top_k(dest_component_vector, top_k))
                                else:
                                    print(
                                        f"Warning: Core index out of bounds for factor matrices. Core o_idx: {core_o_idx} (max: {origin_factors.shape[1]-1}), d_idx: {core_d_idx} (max: {dest_factors.shape[1]-1})")

                            metric_val = float(len(all_unique_zone_indices))
                            # print(f"Debug: Tucker C1 - Rank: {parsed_ranks_tuple}, Opt: {optimizer_for_factors}, City: {city}, Metric Val: {metric_val}")
                        # else:
                            # print(f"Debug: Tucker C1 - Factors or Core G is None or mismatched for {city}, {tensor_type}, {optimizer_for_factors}, {parsed_ranks_tuple}")
                            # if origin_factors is None: print(" origin_factors is None")
                            # if dest_factors is None: print(" dest_factors is None")
                            # if core_tensor_g is None: print(" core_tensor_g is None")
                            # if origin_factors is not None and dest_factors is not None and core_tensor_g is not None:
                            #    print(f" origin_factors.shape: {origin_factors.shape}, dest_factors.shape: {dest_factors.shape}, core_tensor_g.shape: {core_tensor_g.shape}")

                    elif metric == 'c2':
                        origin_factors = load_tucker_factor_matrix(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple, 0)
                        dest_factors = load_tucker_factor_matrix(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple, 1)
                        core_tensor_g = load_tucker_core_tensor(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple)

                        if origin_factors is not None and dest_factors is not None and core_tensor_g is not None and \
                           origin_factors.ndim == 2 and dest_factors.ndim == 2 and \
                           origin_factors.shape[1] == core_tensor_g.shape[0] and \
                           dest_factors.shape[1] == core_tensor_g.shape[1]:

                            selected_core_indices = get_top_core_pattern_indices(
                                core_tensor_g, core_energy_threshold)
                            unique_od_pairs = set()

                            if not selected_core_indices:
                                print(
                                    f"Debug: Tucker C2 - No core patterns selected for {city}, {tensor_type}, {optimizer_for_factors}, {parsed_ranks_tuple} with energy_threshold {core_energy_threshold}")

                            for core_o_idx, core_d_idx, _ in selected_core_indices:
                                if core_o_idx < origin_factors.shape[1] and core_d_idx < dest_factors.shape[1]:
                                    origin_component_vector = origin_factors[:, core_o_idx]
                                    dest_component_vector = dest_factors[:,
                                                                         core_d_idx]

                                    current_origin_indices: set[int]
                                    current_dest_indices: set[int]

                                    if top_selection_method == "cumulative":
                                        current_origin_indices = get_top_indices_cumulative(
                                            origin_component_vector, threshold)
                                        current_dest_indices = get_top_indices_cumulative(
                                            dest_component_vector, threshold)
                                    else:  # top_k
                                        current_origin_indices = get_top_indices_top_k(
                                            origin_component_vector, top_k)
                                        current_dest_indices = get_top_indices_top_k(
                                            dest_component_vector, top_k)

                                    for o_idx in current_origin_indices:
                                        for d_idx in current_dest_indices:
                                            if o_idx != d_idx:
                                                # Store as sorted tuple to count (A,B) and (B,A) as one if desired
                                                unique_od_pairs.add(
                                                    tuple(sorted((o_idx, d_idx))))
                                                # unique_od_pairs.add((o_idx, d_idx)) # If directed pairs are needed
                                else:
                                    print(
                                        f"Warning: Core index out of bounds for factor matrices. Core o_idx: {core_o_idx} (max: {origin_factors.shape[1]-1}), d_idx: {core_d_idx} (max: {dest_factors.shape[1]-1})")

                            metric_val = float(len(unique_od_pairs))
                            # print(f"Debug: Tucker C2 - Rank: {parsed_ranks_tuple}, Opt: {optimizer_for_factors}, City: {city}, Metric Val: {metric_val}")
                        # else:
                            # print(f"Debug: Tucker C2 - Factors or Core G is None or mismatched for {city}, {tensor_type}, {optimizer_for_factors}, {parsed_ranks_tuple}")
                    elif metric == 'num_core_triplets':
                        core_tensor_g = load_tucker_core_tensor(
                            city, tensor_type, optimizer_for_factors, parsed_ranks_tuple)  # tensor_type is base_tensor_type for Tucker

                        if core_tensor_g is not None:
                            # core_energy_threshold is already a parameter of extract_metrics
                            selected_core_indices = get_top_core_pattern_indices(
                                core_tensor_g, core_energy_threshold)
                            metric_val = float(len(selected_core_indices))
                            # print(f"Debug: Tucker NumCoreTriplets - Rank: {parsed_ranks_tuple}, Opt: {optimizer_for_factors}, City: {city}, Metric Val: {metric_val}, Threshold: {core_energy_threshold}")
                        else:
                            metric_val = np.nan
                            # print(f"Debug: Tucker NumCoreTriplets - Core G is None for {city}, {tensor_type}, {optimizer_for_factors}, {parsed_ranks_tuple}")
                else:
                    actual_metrics = rank_data_tucker.get("metrics", {})
                    if isinstance(actual_metrics, dict):
                        metric_val = actual_metrics.get(metric, np.nan)
                processed_results.append((x_axis_identifier, metric_val))
            except (ValueError, SyntaxError, TypeError, KeyError, IndexError) as e:
                # Added more info and printing e
                print(
                    f"Error processing Tucker metric for rank_tuple {rank_tuple_str}, metric {metric}, city {city}, opt {optimizer_for_factors}: {e}")
                # Add a placeholder that won't sort first and can be filtered or identified
                processed_results.append(
                    ((float('inf'), float('inf'), float('inf')), np.nan))

    if not processed_results:
        return [], []

    # Sort by the x_axis_identifier (integer for CP, tuple for Tucker)
    # Ensure NaNs or problematic entries in x_axis_identifier are handled for sorting if any ((inf,inf,inf) sorts last for tuples)
    processed_results.sort(key=lambda x: x[0])

    x_values = [item[0] for item in processed_results]
    y_values = [item[1] for item in processed_results]

    # Filter out any placeholder sort keys if they were added due to error
    if decomposition_mode == "tucker":
        valid_indices = [i for i, x in enumerate(x_values) if x != (
            float('inf'), float('inf'), float('inf'))]
        x_values = [x_values[i] for i in valid_indices]
        y_values = [y_values[i] for i in valid_indices]

    return x_values, y_values


def plot_metric(ax: plt.Axes, city: str, category: str, metric: str, props: dict,
                top_selection_method: str, top_k: int, threshold: float,
                decomposition_mode: str = "cp",
                base_tensor_type_for_tucker: str = None,
                core_energy_threshold_for_tucker: float = 0.80):
    """Plot a single metric. For CP, lines are tensor_types. For Tucker, lines are optimizers."""

    # Determine selection_info based on the metric
    selection_info = ""  # Default to no selection info

    if metric == 'num_core_triplets':
        if decomposition_mode == "tucker":  # This metric is Tucker specific
            selection_info = f" (Core Energy: {core_energy_threshold_for_tucker*100:.0f}%)"
    elif metric.startswith('c') and len(metric) > 1 and metric[1].isdigit() and metric in METRICS_REQUIRING_FACTORS:
        # These use top_selection_method (cumulative or top_k)
        if top_selection_method == "top_k":
            selection_info = f" (Top {top_k})"
        else:  # cumulative
            selection_info = f" (Top {threshold*100:.0f}\\% cum.)"
    # Metrics not in METRICS_REQUIRING_FACTORS (unless they are c-metrics) or not 'num_core_triplets'
    # or not c-metrics will have selection_info = "" based on current structure.
    # This means other metrics in METRICS_REQUIRING_FACTORS but not starting with 'c' won't get this specific title suffix.

    # plot_title_base = props['title'] # Original metric title
    # Subplot title will now include selection_info directly
    current_plot_title = props['title'] + selection_info

    if decomposition_mode == "cp":
        iterable_to_loop = TENSOR_TYPES.items()
        data_loading_key_is_optimizer = False
        x_axis_label_text = 'Rank'
    elif decomposition_mode == "tucker":
        iterable_to_loop = [(opt, opt.upper())
                            for opt in OPTIMIZERS_TO_COMPARE]
        data_loading_key_is_optimizer = True
        x_axis_label_text = 'Rank Combination (O, D, T)'
    else:
        return

    lines_plotted = False
    for item_key, display_name_for_line in iterable_to_loop:
        try:
            current_optimizer = item_key if data_loading_key_is_optimizer else None
            current_tensor_type = base_tensor_type_for_tucker if decomposition_mode == "tucker" else item_key

            if current_tensor_type is None and (decomposition_mode == "tucker" or decomposition_mode == "cp"):
                # print(f"Skipping plot: current_tensor_type is None for metric {metric}, city {city}, mode {decomposition_mode}")
                continue

            data = load_decomposition_data(city=city,
                                           tensor_type=current_tensor_type,
                                           category=category,
                                           decomposition_mode=decomposition_mode,
                                           optimizer=current_optimizer)

            x_axis_values, y_axis_values = extract_metrics(
                data=data, metric=metric, decomposition_mode=decomposition_mode,
                city=city, category=category, tensor_type=current_tensor_type,
                optimizer_for_factors=current_optimizer,
                top_selection_method=top_selection_method, top_k=top_k, threshold=threshold,
                core_energy_threshold=core_energy_threshold_for_tucker
            )

            if not x_axis_values or not y_axis_values or all(np.isnan(y) for y in y_axis_values):
                # print(f"No valid data for {display_name_for_line} on metric {metric}, city {city}, tensor {current_tensor_type}")
                continue
            lines_plotted = True

            label_to_plot = display_name_for_line
            if decomposition_mode == "cp" and (metric.startswith('c3_distinct_time_bins') or metric == 'c3_ratio_distinct_available_time'):
                label_to_plot = f"{display_name_for_line} ({TIME_BINS_PER_TYPE.get(item_key, '?')})"

            print(
                f"Debug Plot: About to plot for {display_name_for_line}, Metric: {metric}, City: {city}")
            print(f"Debug Plot: X-values: {x_axis_values}")
            print(f"Debug Plot: Y-values: {y_axis_values}")

            # If x_axis_values are tuples, convert them to strings for explicit categorical plotting.
            if decomposition_mode == "tucker" and x_axis_values and isinstance(x_axis_values[0], tuple):
                # Plot data starting from x-position 1, as x-position 0 is reserved for "O\nD\nT" label
                ax.plot(range(1, len(x_axis_values) + 1), y_axis_values,
                        marker='o', label=label_to_plot)
            # CP mode or if Tucker x_values are not tuples (e.g. if all were error placeholders)
            else:
                ax.plot(x_axis_values, y_axis_values,
                        marker='o', label=label_to_plot)

        except FileNotFoundError as e:
            print(
                f"Data file not found for {display_name_for_line} (metric: {metric}, city: {city}, tensor: {current_tensor_type}, opt: {current_optimizer if current_optimizer else 'N/A'}): {e}")
            pass
        except Exception as e:
            print(
                f"Error plotting line for {display_name_for_line} (metric: {metric}, city: {city}, tensor: {current_tensor_type}, opt: {current_optimizer}): {e}")
            # import traceback; traceback.print_exc()
            pass

    ax.set_xlabel(x_axis_label_text)
    ax.set_ylabel(props['title'])
    ax.set_title(current_plot_title, fontsize=10)
    ax.set_ylim(props['ylim'])
    if props.get('invert_y', False):
        ax.invert_yaxis()
    ax.grid(True)

    if lines_plotted:
        ax.legend(bbox_to_anchor=(1.05, 0.2),
                  loc='upper left', fontsize='small')
        # Potentially rotate x-tick labels if they are too crowded (especially for Tucker tuples)
        if decomposition_mode == "tucker" and x_axis_values and isinstance(x_axis_values[0], tuple):
            # Ensure x-axis includes position 0 for the O/D/T label
            # Get current limits to preserve the upper limit based on data
            # current_xlim = ax.get_xlim()
            # Set new limits, ensuring 0 is visible and there's some padding.
            # The upper limit should be based on the number of data points + 1 (for ODT label) - 0.5 for padding.
            # Number of rank tuple labels is len(x_axis_values).
            # Total number of labels including ODT is len(x_axis_values) + 1.
            # So, max x-tick position is len(x_axis_values).
            # Adjusted to ensure all labels fit
            ax.set_xlim(0, len(x_axis_values) + 0.5)

            # Create the multiline labels for the rank tuples
            rank_tuple_labels = [
                "\n".join(str(val) for val in tup)
                for tup in x_axis_values
            ]
            # Prepend the "O\nD\T" label for the first position
            all_xticklabels = ["O\nD\nT"] + rank_tuple_labels

            # Set tick positions from 0 to len(x_axis_values)
            # (0 for "O\nD\T", 1 for first rank tuple, etc.)
            tick_positions = range(len(all_xticklabels))

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(all_xticklabels, rotation=0,
                               ha="center", fontsize='x-small')

            # Make the first tick line (at x=0, for "O\nD\T") invisible
            if tick_positions:  # Check if there are any tick positions
                first_tick = ax.xaxis.get_major_ticks()[0]
                first_tick.tick1line.set_visible(
                    False)  # Main tick line on the axis
                # Secondary tick line (usually not visible by default)
                first_tick.tick2line.set_visible(False)

            ax.margins(x=0.01)  # Keep small margins
    else:
        ax.text(0.5, 0.5, "No data available", horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, color='grey')


def create_comparison_plot(category: str,
                           top_selection_method: str = "cumulative",
                           top_k: int = 2,
                           threshold: float = 0.25,
                           decomposition_mode: str = "cp",
                           base_tensor_type_for_tucker: str = None,
                           core_energy_threshold_tucker: float = 0.80):
    """Create the comparison plot with subplots for all metrics (2 rows, N columns)."""
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(2, n_metrics, figsize=(
        5.5 * n_metrics, 10), constrained_layout=True)
    # Using constrained_layout=True for better spacing with bbox_inches='tight' later

    title_selection_info = ""
    if any(m in METRICS_REQUIRING_FACTORS for m in METRICS.keys()):
        if top_selection_method == 'top_k':
            sel_text = f'Top {top_k}'
        else:
            sel_text = f'{threshold*100:.0f}% cumulative'
        title_selection_info = f" (Selection: {sel_text}"
        # Corrected condition: Check if relevant metrics are in METRICS.keys() for Tucker mode
        if decomposition_mode == "tucker" and any(key in ['c1_unique_zones', 'c2'] for key in METRICS.keys()):
            title_selection_info += f", Core Energy: {core_energy_threshold_tucker*100:.0f}%"
        title_selection_info += ")"

    # Determine parts of title and filename based on mode
    if decomposition_mode == "cp":
        main_title_subject = "CP Decomposition Across Time Binning Strategies"
        filename_suffix_mode = "cp_comparison"
        filename_category_part = category
    elif decomposition_mode == "tucker":
        tucker_data_descr = TENSOR_TYPES.get(
            base_tensor_type_for_tucker, base_tensor_type_for_tucker)
        main_title_subject = f"Tucker Decompositions ({tucker_data_descr} Data) Across Optimizers"
        filename_suffix_mode = f"tucker_{base_tensor_type_for_tucker}_comparison"
        filename_category_part = category
    else:
        raise ValueError(f"Unknown decomposition mode: {decomposition_mode}")

    fig.suptitle(
        # f'{main_title_subject} - Category: {category}{title_selection_info}', fontsize=16,
        main_title_subject, fontsize=16,
    )

    for row, city in enumerate(['utrecht', 'rotterdam']):
        for col, (metric, props) in enumerate(METRICS.items()):
            ax = axes[row, col] if n_metrics > 1 else axes[row]
            if n_metrics == 1 and len(METRICS.items()) == 1:
                ax = axes[row]  # if only one city and one metric
            elif n_metrics == 1 and len(METRICS.items()) > 1:
                ax = axes[col]  # if only one city but multiple metrics

            plot_metric(ax, city, category, metric, props,
                        top_selection_method, top_k, threshold,
                        decomposition_mode,
                        base_tensor_type_for_tucker=base_tensor_type_for_tucker,
                        core_energy_threshold_for_tucker=core_energy_threshold_tucker)
            if col == 0:
                ax.text(-0.25, 0.5, city.capitalize(),
                        rotation=90, transform=ax.transAxes,
                        fontsize=14, va='center', ha='center')  # Adjusted position and size

    # plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle if not using constrained_layout

    # Add selection method to filename for traceability
    filename_selection_suffix = f"_{top_selection_method}"
    if top_selection_method == "top_k":
        filename_selection_suffix += str(top_k)
    else:
        filename_selection_suffix += str(int(threshold*100)) + "pct"

    # Add core energy to filename if Tucker and C1/C2 present
    if decomposition_mode == "tucker" and any(m in ['c1_unique_zones', 'c2'] for m in METRICS.keys()):
        filename_selection_suffix += f"_coreE{int(core_energy_threshold_tucker*100)}"

    # Determine output directory based on mode
    if decomposition_mode == "cp":
        output_dir = CP_COMPARE_RESULTS_PATH
    elif decomposition_mode == "tucker":
        # Save Tucker comparisons in a subfolder of TUCKER_RESULTS_BASE_PATH or a new compare folder
        output_dir = TUCKER_RESULTS_BASE_PATH / "comparison_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:  # Fallback, should not happen
        output_dir = Path(".")

    output_filename = f"{filename_suffix_mode}_{filename_category_part}{filename_selection_suffix}.png"
    output_path = output_dir / output_filename

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close(fig)  # Close the figure to free memory


def plot_core_tensor_energy(core_tensor_G: np.ndarray,
                            city: str,
                            base_tensor_type: str,
                            optimizer: str,
                            rank_tuple: tuple[int, int, int],
                            decomposition_mode: str,
                            output_dir_base: Path,
                            top_n_elements_to_plot: int = 20):
    """Plots the energy distribution of a Tucker or CP core tensor G.

    Generates:
    1. Bar plot of individual energies of the top N elements.
    2. Line plot of cumulative energy of top N elements, marking thresholds, with grid structure and ODT marker.

    Args:
        core_tensor_G: The 3D Tucker core tensor.
        city: City name.
        base_tensor_type: Base tensor type.
        optimizer: Optimizer used.
        rank_tuple: Rank tuple (Ro, Rd, Rt).
        decomposition_mode: 'tucker' or other.
        output_dir_base: Base directory to save the plots.
        top_n_elements_to_plot: Number of top elements for both plots.
    """
    # Validate input
    if not isinstance(core_tensor_G, np.ndarray) or core_tensor_G.ndim != 3:
        print("Skipping plot: core_tensor_G must be a 3D NumPy array.")
        return

    # Flatten and compute squared energies
    flat_vals = core_tensor_G.flatten()
    flat_energies = flat_vals ** 2
    total_energy = flat_energies.sum()
    if total_energy == 0:
        print("Skipping plot: total core tensor energy is zero.")
        return

    # Sort energies descending
    sorted_idx = np.argsort(flat_energies)[::-1]
    sorted_energies = flat_energies[sorted_idx]
    shape = core_tensor_G.shape

    # Prepare output directory
    rank_str = f"rank_{rank_tuple[0]}_{rank_tuple[1]}_{rank_tuple[2]}"
    plot_dir = Path(output_dir_base) / optimizer / rank_str
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Plot 1: Bar plot of top N individual energies ---
    n_bar = min(top_n_elements_to_plot, sorted_energies.size)
    top_indices = sorted_idx[:n_bar]
    energies_bar = sorted_energies[:n_bar]
    idx_3d = [np.unravel_index(i, shape) for i in top_indices]
    labels = [f"({i},{j},{k})" for i, j, k in idx_3d]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(n_bar), energies_bar)
    ax.set_xticks(range(n_bar))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize='small')
    ax.set_xlabel('Core Element (i,j,k)')
    ax.set_ylabel('Energy (g_ijk^2)')
    ax.set_title(
        f"Top {n_bar} Core Energies — {city}, {base_tensor_type}, {optimizer}, Rank={rank_tuple}, total energy={total_energy:.2f}")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize='small')

    fig.tight_layout()
    fig.savefig(
        plot_dir / f"core_energy_bar_{n_bar}_{city}_{optimizer}_{rank_str}.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: Cumulative energy of top N elements ---
    n_display = n_bar
    cum_norm = np.cumsum(sorted_energies) / total_energy
    x = np.arange(1, n_display + 1)
    y = cum_norm[:n_display]
    displayed_indices = sorted_idx[:n_display]
    x_axis_values = [np.unravel_index(i, shape) for i in displayed_indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker='o')

    # Underlying gridlines
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Draw horizontal threshold lines across the full x span
    thresholds = [0.8, 0.9, 0.95]
    for thr in thresholds:
        ax.hlines(thr, xmin=0.5, xmax=n_display + 0.5, colors='C0',
                  linestyles='--', label=f"{int(thr*100)}% energy")

    # Draw verticals and annotate only where within display, using 'at' for clarity
    for thr in thresholds:
        idx_thr = np.searchsorted(cum_norm, thr) + 1
        if idx_thr <= n_display:
            ax.vlines(idx_thr, ymin=0, ymax=1.05, colors='C0', linestyles=':')
            ax.text(idx_thr, 0.1, f'{int(thr*100)}%\nat {idx_thr}',
                    ha='center', va='top', fontsize='small', backgroundcolor='white')

    # Ticks: ODT at 0, then top N elements
    tick_positions = np.arange(0, n_display + 1)
    tick_labels = ['O\nD\nT'] + \
        ['\n'.join(str(val) for val in tup) for tup in x_axis_values]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, va='top',
                       ha='center', fontsize='small')
    ax.set_xlim(0, n_display + 0.5)

    ax.set_xlabel(
        'Core Element Index (idx_O, idx_D, idx_T) - Sorted by Decreasing Energy')
    ax.set_ylabel('Cumulative Energy (Normalized to Total Core Energy)')
    ax.set_title(
        f"Cumulative Energy of Top {n_display} Core Elements\n{city.capitalize()} - {base_tensor_type} - {optimizer} - Rank {rank_tuple}")
    ax.legend(loc='lower right', fontsize='small')
    fig.tight_layout()
    fig.savefig(
        plot_dir / f"core_energy_cumulative_top{n_display}_{city}_{base_tensor_type}_{optimizer}_{rank_str}.png", dpi=150)
    plt.close(fig)


def export_performance_metrics_json(
    category: str,
    top_selection_method: str = "cumulative",
    top_k: int = 2,
    threshold: float = 0.25,
    decomposition_mode: str = "cp",
    base_tensor_type_for_tucker: str = None,
    core_energy_threshold_tucker: float = 0.80
) -> dict:
    """
    Export performance metrics as a structured JSON dictionary.

    Structure:
    {
        "city_name": {
            "tensor_type": {
                "rank": {
                    "metric_name": value,
                    ...
                }
            }
        }
    }

    For CP: tensor_types are the different time binning strategies
    For Tucker: tensor_types are the optimizers (MU, HALS)
    """
    results = {}

    cities = ['utrecht', 'rotterdam']

    if decomposition_mode == "cp":
        tensor_types = list(TENSOR_TYPES.keys())
        data_loading_key_is_optimizer = False
    elif decomposition_mode == "tucker":
        tensor_types = OPTIMIZERS_TO_COMPARE
        data_loading_key_is_optimizer = True
    else:
        raise ValueError(f"Unknown decomposition mode: {decomposition_mode}")

    for city in cities:
        results[city] = {}

        for tensor_type in tensor_types:
            results[city][tensor_type] = {}

            try:
                current_optimizer = tensor_type if data_loading_key_is_optimizer else None
                current_tensor_type = base_tensor_type_for_tucker if decomposition_mode == "tucker" else tensor_type

                # Load decomposition data
                data = load_decomposition_data(
                    city=city,
                    tensor_type=current_tensor_type,
                    category=category,
                    decomposition_mode=decomposition_mode,
                    optimizer=current_optimizer
                )

                if not data:
                    print(f"No data found for {city}, {tensor_type}")
                    continue

                # Extract metrics for all ranks
                for metric_name in METRICS.keys():
                    try:
                        x_values, y_values = extract_metrics(
                            data=data,
                            metric=metric_name,
                            decomposition_mode=decomposition_mode,
                            city=city,
                            category=category,
                            tensor_type=current_tensor_type,
                            optimizer_for_factors=current_optimizer,
                            top_selection_method=top_selection_method,
                            top_k=top_k,
                            threshold=threshold,
                            core_energy_threshold=core_energy_threshold_tucker
                        )

                        # Store metrics for each rank
                        for i, (rank, metric_value) in enumerate(zip(x_values, y_values)):
                            # Convert rank to string for JSON serialization
                            rank_str = str(rank)

                            if rank_str not in results[city][tensor_type]:
                                results[city][tensor_type][rank_str] = {}

                            # Store metric value, handling NaN values
                            if np.isnan(metric_value):
                                results[city][tensor_type][rank_str][metric_name] = None
                            else:
                                results[city][tensor_type][rank_str][metric_name] = float(
                                    metric_value)

                    except Exception as e:
                        print(
                            f"Error extracting metric {metric_name} for {city}, {tensor_type}: {e}")
                        continue

            except Exception as e:
                print(f"Error processing {city}, {tensor_type}: {e}")
                continue

    return results


def save_performance_metrics_json(
    metrics_data: dict,
    category: str,
    top_selection_method: str = "cumulative",
    top_k: int = 2,
    threshold: float = 0.25,
    decomposition_mode: str = "cp",
    base_tensor_type_for_tucker: str = None,
    core_energy_threshold_tucker: float = 0.80
) -> Path:
    """
    Save performance metrics to a JSON file with appropriate naming.

    Returns:
        Path to the saved JSON file
    """
    # Determine output directory based on mode
    if decomposition_mode == "cp":
        output_dir = CP_COMPARE_RESULTS_PATH
    elif decomposition_mode == "tucker":
        output_dir = TUCKER_RESULTS_BASE_PATH / "comparison_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(".")

    # Create filename with selection parameters
    filename_selection_suffix = f"_{top_selection_method}"
    if top_selection_method == "top_k":
        filename_selection_suffix += str(top_k)
    else:
        filename_selection_suffix += str(int(threshold*100)) + "pct"

    # Add core energy to filename if Tucker and C1/C2 present
    if decomposition_mode == "tucker" and any(m in ['c1_unique_zones', 'c2'] for m in METRICS.keys()):
        filename_selection_suffix += f"_coreE{int(core_energy_threshold_tucker*100)}"

    # Determine mode suffix
    if decomposition_mode == "cp":
        mode_suffix = "cp"
        tensor_type_part = category
    elif decomposition_mode == "tucker":
        mode_suffix = f"tucker_{base_tensor_type_for_tucker}"
        tensor_type_part = category
    else:
        mode_suffix = "unknown"
        tensor_type_part = category

    output_filename = f"performance_metrics_{mode_suffix}_{tensor_type_part}{filename_selection_suffix}.json"
    output_path = output_dir / output_filename

    # Add metadata to the JSON
    export_data = {
        "metadata": {
            "decomposition_mode": decomposition_mode,
            "category": category,
            "top_selection_method": top_selection_method,
            "top_k": top_k,
            "threshold": threshold,
            "core_energy_threshold": core_energy_threshold_tucker if decomposition_mode == "tucker" else None,
            # Placeholder for timestamp
            "export_timestamp": str(datetime.now()),
            "metrics_included": list(METRICS.keys())
        },
        "data": metrics_data
    }

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"Saved performance metrics to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Determine category based on decomposition mode
    if args.decomposition_mode == "cp":
        # Correct category for CP
        category = "odt_no_same_od_no_rare_od_fixed_thresh_normalizedPeaks"
    else:  # tucker
        category = f"{args.tensor_type}_tucker_analysis"

    # Export performance metrics as JSON if requested
    if args.export_json:
        print("Exporting performance metrics as JSON...")
        metrics_data = export_performance_metrics_json(
            category=category,
            top_selection_method=args.top_selection,
            top_k=args.top_k,
            threshold=args.threshold,
            decomposition_mode=args.decomposition_mode,
            base_tensor_type_for_tucker=args.tensor_type if args.decomposition_mode == "tucker" else None,
            core_energy_threshold_tucker=args.core_energy_threshold
        )

        # Save the JSON file
        json_file_path = save_performance_metrics_json(
            metrics_data=metrics_data,
            category=category,
            top_selection_method=args.top_selection,
            top_k=args.top_k,
            threshold=args.threshold,
            decomposition_mode=args.decomposition_mode,
            base_tensor_type_for_tucker=args.tensor_type if args.decomposition_mode == "tucker" else None,
            core_energy_threshold_tucker=args.core_energy_threshold
        )

        print(f"Performance metrics exported to: {json_file_path}")

    # Create comparison plot with command line arguments
    print("Creating comparison plot...")
    create_comparison_plot(
        category=category,
        top_selection_method=args.top_selection,
        top_k=args.top_k,
        threshold=args.threshold,
        decomposition_mode=args.decomposition_mode,
        base_tensor_type_for_tucker=args.tensor_type if args.decomposition_mode == "tucker" else None,
        core_energy_threshold_tucker=args.core_energy_threshold
    )

    # Generate core energy plots only for Tucker decomposition
    if args.decomposition_mode == "tucker":
        print("Generating core energy plots...")
        for city in ['utrecht', 'rotterdam']:
            for optimizer in OPTIMIZERS_TO_COMPARE:
                try:
                    # Load the decomposition data to get all rank combinations
                    data = load_decomposition_data(
                        city=city,
                        tensor_type=args.tensor_type,
                        decomposition_mode="tucker",
                        optimizer=optimizer
                    )

                    if not data or "all_combinations" not in data:
                        print(
                            f"No data found for {city}, {optimizer}, {args.tensor_type}")
                        continue

                    # Create output directory for core energy plots
                    output_dir = TUCKER_RESULTS_BASE_PATH / \
                        f"{city}_{args.tensor_type}" / \
                        "core_energy_analysis_plots"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Process each rank combination
                    for rank_tuple_str in data["all_combinations"].keys():
                        try:
                            rank_tuple = ast.literal_eval(rank_tuple_str)
                            if not isinstance(rank_tuple, tuple) or len(rank_tuple) != 3:
                                continue

                            # Load core tensor for this rank combination
                            core_tensor = load_tucker_core_tensor(
                                city=city,
                                base_tensor_type=args.tensor_type,
                                optimizer=optimizer,
                                rank_tuple=rank_tuple
                            )

                            if core_tensor is not None:
                                # Generate core energy plots
                                plot_core_tensor_energy(
                                    core_tensor_G=core_tensor,
                                    city=city,
                                    base_tensor_type=args.tensor_type,
                                    optimizer=optimizer,
                                    rank_tuple=rank_tuple,
                                    decomposition_mode="tucker",
                                    output_dir_base=output_dir,
                                    # Using 18 as it seems to be a common value from your examples
                                    top_n_elements_to_plot=18
                                )
                                print(
                                    f"Generated core energy plots for {city}, {optimizer}, rank {rank_tuple}")
                            else:
                                print(
                                    f"Could not load core tensor for {city}, {optimizer}, rank {rank_tuple}")

                        except (ValueError, SyntaxError) as e:
                            print(
                                f"Error processing rank tuple {rank_tuple_str}: {e}")
                            continue

                except Exception as e:
                    print(f"Error processing {city}, {optimizer}: {e}")
                    continue

    print("All tasks completed successfully!")
