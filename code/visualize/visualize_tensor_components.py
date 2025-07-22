from matplotlib.colors import LinearSegmentedColormap
import json
import os
from datetime import datetime, timedelta
import logging
from typing import List, Tuple, Dict, Literal
import seaborn as sns
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import argparse

"""
Detailed Tensor Component Visualization

This script provides detailed visualization and analysis of individual tensor decomposition components.
It creates comprehensive visualizations of spatial and temporal patterns in the decomposed tensor.

Key Features:
- Spatial Components: Visualizes origin and destination patterns using geographical data
- Temporal Components: Analyzes time-based patterns and their variations
- Weight Analysis: Examines the importance of different components
- Statistical Analysis: Provides detailed metrics about the decomposition
- Metadata Generation: Creates comprehensive documentation of the analysis

Visualization Types:
- Spatial Maps: Geographical representation of origin/destination patterns
- Temporal Plots: Time-based pattern analysis
- Weight Distributions: Analysis of component importance
- Statistical Summaries: Detailed metrics and statistics

Time Granularities Supported:
1. Timebin (35 bins): 5 time bins per day × 7 days
   - Early Morning: 00:00-06:30
   - Morning Peak: 06:30-09:00
   - Day: 09:00-16:00
   - Evening Peak: 16:00-18:30
   - Night: 18:30-23:59:59
2. Weekhour (168 bins): 24 hours × 7 days
3. Hourly (24 bins): 24 hours of a day

Note:
    This script is designed for detailed analysis of a single tensor decomposition.
    For comparing multiple tensor types, use compare_tensor_types.py instead.
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default paths
DEFAULT_GEOJSON_PATH = "data/raw/location/working_zips.geojson"

# Define time bin configurations
TIME_BINS = {
    'timebin': {
        'bins_per_day': 5,
        'total_bins': 35,
        'bin_labels': ['Early Morning', 'Morning Peak', 'Day', 'Evening Peak', 'Night'],
        'bin_times': ['00:00-06:30', '06:30-09:00', '09:00-16:00', '16:00-18:30', '18:30-24:00']
    },
    'weekhour': {
        'bins_per_day': 24,
        'total_bins': 168,
        'bin_labels': [f'{h:02d}:00' for h in range(24)],
        'bin_times': [f'{h:02d}:00-{(h+1):02d}:00' for h in range(24)]
    },
    'hourly': {
        'bins_per_day': 24,
        'total_bins': 24,
        'bin_labels': [f'{h:02d}:00' for h in range(24)],
        'bin_times': [f'{h:02d}:00-{(h+1):02d}:00' for h in range(24)]
    }
}


class FigureStyler:
    """Class to manage consistent figure styling across all plots."""

    def __init__(self):
        # Figure dimensions
        self.spatial_figsize = (10, 8)
        self.temporal_figsize = (8, 6)  # Default for CP
        self.temporal_figsize_tucker = (14, 6)  # For Tucker decompositions

        # Frame styling
        self.frame_color = 'gray'
        self.frame_linewidth = 0.5

        # Grid styling
        self.grid_alpha = 0.3

        # Color scale
        self.vmin = 0
        self.vmax = 1

        # Temporal y-axis padding to ensure lines are visible when zooming out
        self.temporal_ymin = -0.05  # Add 5% padding below zero
        self.temporal_ymax = 1.05   # Add 5% padding above one

        # Custom colormap for spatial plots
        self.colors = [
            (1, 1, 1, 0),      # White with 0 opacity
            (0.2, 0.4, 0.8, 0.3),  # Light blue with low opacity
            (0.1, 0.2, 0.6, 0.6),  # Medium blue with medium opacity
            (0, 0, 0.4, 0.9)   # Dark blue with high opacity
        ]
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'custom', self.colors)

        # Boundary styling for spatial plots
        self.boundary_color = '#CCCCCC'
        self.boundary_width = 0.25

    def create_spatial_figure(self) -> tuple[plt.Figure, plt.Axes]:
        """Create a figure with consistent spatial plot styling."""
        fig, ax = plt.subplots(1, 1, figsize=self.spatial_figsize)
        self.apply_frame_style(ax)
        return fig, ax

    def create_temporal_figure(self, is_tucker: bool = False) -> tuple[plt.Figure, plt.Axes]:
        """Create a figure with consistent temporal plot styling."""
        figsize = self.temporal_figsize_tucker if is_tucker else self.temporal_figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        self.apply_frame_style(ax)
        return fig, ax

    def apply_frame_style(self, ax: plt.Axes) -> None:
        """Apply consistent frame styling to an axis."""
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Add subtle frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(self.frame_color)
            spine.set_linewidth(self.frame_linewidth)

    def apply_temporal_style(self, ax: plt.Axes) -> None:
        """Apply temporal-specific styling."""
        ax.grid(True, alpha=self.grid_alpha)
        ax.set_ylim(self.temporal_ymin, self.temporal_ymax)

    def get_spatial_plot_kwargs(self) -> dict:
        """Get keyword arguments for spatial plotting."""
        return {
            'cmap': self.custom_cmap,
            'vmin': self.vmin,
            'vmax': self.vmax,
            'edgecolor': self.boundary_color,
            'linewidth': self.boundary_width,
            'missing_kwds': {'color': 'white'}
        }


# Create global styler instance
styler = FigureStyler()


def determine_time_granularity(decomp_path: str) -> Literal['timebin', 'weekhour', 'hourly']:
    """Determine the time granularity from the decomposition path.

    Args:
        decomp_path: Path to the decomposition file

    Returns:
        Time granularity type ('timebin', 'weekhour', or 'hourly')
    """
    path = Path(decomp_path)
    if 'timebin' in path.parts:
        return 'timebin'
    elif 'weekhour' in path.parts:
        return 'weekhour'
    elif 'hourly' in path.parts:
        return 'hourly'
    else:
        # Try to determine from the number of time bins
        data = np.load(decomp_path, allow_pickle=True)
        time_dim = data['factors'][2].shape[0]
        if time_dim == 35:
            return 'timebin'
        elif time_dim == 168:
            return 'weekhour'
        elif time_dim == 24:
            return 'hourly'
        else:
            raise ValueError(
                f"Unknown time granularity with {time_dim} time bins")


def load_decomposition(file_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load the decomposition factors from NPZ file.

    Args:
        file_path: Path to the decomposition NPZ file

    Returns:
        Tuple of (weights, factors)
    """
    data = np.load(file_path, allow_pickle=True)

    # Check if this is a Tucker decomposition (has 'core' and 'factors')
    if 'core' in data and 'factors' in data:
        # For Tucker decomposition, we'll use the core tensor as weights
        # and the factors as is
        core = data['core']
        factors = data['factors']

        # Convert core tensor to weights (sum of absolute values for each component)
        weights = np.array([np.sum(np.abs(core[i]))
                           for i in range(core.shape[0])])

        return weights, factors
    # Check if this is a CP decomposition (has 'weights' and 'factors')
    elif 'weights' in data and 'factors' in data:
        return data['weights'], data['factors']
    else:
        raise ValueError(
            "Decomposition file must contain either 'core' and 'factors' (Tucker) or 'weights' and 'factors' (CP)")


def load_geojson(file_path: str, relevant_postal_codes: List[str]) -> gpd.GeoDataFrame:
    """Load postal code geometries from GeoJSON.

    Args:
        file_path: Path to the GeoJSON file
        relevant_postal_codes: List of postal codes used in the decomposition

    Returns:
        GeoDataFrame with postal code geometries
    """
    gdf = gpd.read_file(file_path)

    # Log the CRS information
    logger.info(f"Original CRS: {gdf.crs}")

    # Ensure CRS is set or assumed correctly
    if gdf.crs is None:
        logger.info("No CRS found, assuming EPSG:4326 (WGS84)")
        gdf = gdf.set_crs(epsg=4326)

    # Reproject to WGS84 (EPSG:4326) to ensure proper north orientation
    gdf = gdf.to_crs(epsg=4326)
    logger.info(f"Reprojected CRS: {gdf.crs}")

    # Ensure pc4_code is string type for matching
    gdf['pc4_code'] = gdf['pc4_code'].astype(str).str.zfill(4)

    # Filter to only relevant postal codes
    relevant_gdf = gdf[gdf['pc4_code'].isin(relevant_postal_codes)].copy()

    # Log information about postal codes
    logger.info(f"Total number of postal codes in GeoJSON: {len(gdf)}")
    logger.info(
        f"Number of relevant postal codes from decomposition: {len(relevant_postal_codes)}")
    logger.info(f"Number of postal codes being plotted: {len(relevant_gdf)}")

    # Check if any postal codes are missing
    missing_codes = set(relevant_postal_codes) - set(relevant_gdf['pc4_code'])
    if missing_codes:
        logger.warning(f"Missing geometries for postal codes: {missing_codes}")

    return relevant_gdf


def load_index_mappings(file_path: str) -> Dict:
    """Load index mappings from JSON file.

    Args:
        file_path: Path to the index mappings JSON file

    Returns:
        Dictionary containing index mappings
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def calculate_sparsity(factor: np.ndarray) -> float:
    """Calculate the sparsity of a factor matrix.

    Args:
        factor: Factor matrix

    Returns:
        Sparsity ratio (percentage of near-zero values)
    """
    return float(np.sum(factor <= 1e-10) / factor.size)


def plot_spatial_components(context_gdf: gpd.GeoDataFrame,
                            relevant_gdf: gpd.GeoDataFrame,
                            factor: np.ndarray,
                            idx_to_postal: Dict[str, str],
                            mode_name: str,
                            save_path: str,
                            tensor_name: str,
                            plot_individual_ranks: bool = False) -> None:
    """Plot all spatial components on a single figure with subplots.

    Args:
        context_gdf: GeoDataFrame with postal codes (not used anymore)
        relevant_gdf: GeoDataFrame with relevant postal codes
        factor: Factor matrix for spatial component
        idx_to_postal: Mapping from indices to postal codes
        mode_name: Name of the mode (Origin/Destination)
        save_path: Path to save the plot
        tensor_name: Name of the tensor being visualized
        plot_individual_ranks: If True, create individual plots for each rank
    """
    n_components = factor.shape[1]

    if plot_individual_ranks:
        # Create individual plots for each rank
        for rank in range(n_components):
            individual_save_path = save_path.replace(
                '.png', f'_rank_{rank+1}.png')
            plot_single_spatial_component(
                relevant_gdf, factor, idx_to_postal, mode_name,
                individual_save_path, tensor_name, rank
            )
        return

    n_rows = (n_components + 2) // 3  # Ceiling division for number of rows
    n_cols = min(3, n_components)  # Max 3 columns

    # Calculate sparsity
    sparsity = calculate_sparsity(factor)
    sparsity_percent = sparsity * 100

    # Create figure with extra space for colorbar
    fig = plt.figure(figsize=(15, 5*n_rows))
    gs = fig.add_gridspec(n_rows, n_cols + 1,  # Add extra column for colorbar
                          # Make colorbar column narrower
                          width_ratios=[1]*n_cols + [0.05],
                          hspace=0.3, wspace=0.4)  # Increase spacing

    # Use fixed color scale 0-1 for consistent interpretation
    vmin, vmax = 0, 1

    # Set colormap based on mode_name
    if mode_name.lower() == "origin":
        # Blue colormap for origin
        colors = [
            (1, 1, 1, 0),      # White with 0 opacity
            (0.2, 0.4, 0.8, 0.3),  # Light blue with low opacity
            (0.1, 0.2, 0.6, 0.6),  # Medium blue with medium opacity
            (0, 0, 0.4, 0.9)   # Dark blue with high opacity
        ]
    else:
        # Red colormap for destination
        colors = [
            (1, 1, 1, 0),      # White with 0 opacity
            (0.8, 0.3, 0.3, 0.3),  # Light red with low opacity
            (0.6, 0.1, 0.1, 0.6),  # Medium red with medium opacity
            (0.4, 0, 0, 0.9)   # Dark red with high opacity
        ]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    # Define boundary color (light gray) and width
    boundary_color = '#CCCCCC'
    boundary_width = 0.25

    # Create a mapping from postal codes to values for each component
    for idx in range(n_components):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        values = factor[:, idx]
        postal_to_value = {
            idx_to_postal[str(i)]: val for i, val in enumerate(values)}

        # Add values to the relevant GeoDataFrame
        temp_gdf = relevant_gdf.copy()
        temp_gdf['value'] = temp_gdf['pc4_code'].map(postal_to_value)

        # Plot the areas with values
        im = temp_gdf.plot(column='value',
                           ax=ax,
                           legend=False,  # We'll add colorbar manually
                           cmap=custom_cmap,
                           missing_kwds={'color': 'white'},
                           vmin=0,
                           vmax=1,
                           edgecolor=boundary_color,
                           linewidth=boundary_width)

        # Overlay black postal code boundaries for consistency with spatial flows
        relevant_gdf.boundary.plot(
            ax=ax, color='black', linewidth=0.5, zorder=1)

        # Plot boundaries for zero-value areas separately to ensure visibility
        zero_value_gdf = temp_gdf[temp_gdf['value'] <= 1e-10]
        if not zero_value_gdf.empty:
            zero_value_gdf.boundary.plot(
                ax=ax, color=boundary_color, linewidth=boundary_width)

        # Improve the map appearance
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a subtle frame
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)

    # Remove any empty subplots
    for idx in range(n_components, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.remove()

    # Add a single colorbar for all subplots
    cbar_ax = fig.add_subplot(gs[:, -1])  # Use full height for colorbar
    sm = plt.cm.ScalarMappable(
        cmap=custom_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Factor Value (0-1 scale)', rotation=270, labelpad=15)

    # Add overall title with sparsity information
    fig.suptitle(f'{mode_name} Patterns Across Components for {tensor_name}\n'
                 f'Sparsity: {sparsity_percent:.1f}% (values ≤ 1e-10)',
                 y=1.02, fontsize=14)

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_single_spatial_component(relevant_gdf: gpd.GeoDataFrame,
                                  factor: np.ndarray,
                                  idx_to_postal: Dict[str, str],
                                  mode_name: str,
                                  save_path: str,
                                  tensor_name: str,
                                  rank: int) -> None:
    """Plot a single spatial component for a specific rank.

    Args:
        relevant_gdf: GeoDataFrame with relevant postal codes
        factor: Factor matrix for spatial component
        idx_to_postal: Mapping from indices to postal codes
        mode_name: Name of the mode (Origin/Destination)
        save_path: Path to save the plot
        tensor_name: Name of the tensor being visualized
        rank: Rank index to plot (0-based)
    """
    # Create figure with consistent styling
    fig, ax = styler.create_spatial_figure()

    # Get values for this rank
    values = factor[:, rank]

    # Create mapping from postal codes to values
    postal_to_value = {
        idx_to_postal[str(i)]: val for i, val in enumerate(values)}

    # Add values to the relevant GeoDataFrame
    temp_gdf = relevant_gdf.copy()
    temp_gdf['value'] = temp_gdf['pc4_code'].map(postal_to_value)

    # Set colormap based on mode_name
    if mode_name.lower() == "origin":
        # Blue colormap for origin
        colors = [
            (1, 1, 1, 0),      # White with 0 opacity
            (0.2, 0.4, 0.8, 0.3),  # Light blue with low opacity
            (0.1, 0.2, 0.6, 0.6),  # Medium blue with medium opacity
            (0, 0, 0.4, 0.9)   # Dark blue with high opacity
        ]
    else:
        # Red colormap for destination
        colors = [
            (1, 1, 1, 0),      # White with 0 opacity
            (0.8, 0.3, 0.3, 0.3),  # Light red with low opacity
            (0.6, 0.1, 0.1, 0.6),  # Medium red with medium opacity
            (0.4, 0, 0, 0.9)   # Dark red with high opacity
        ]
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    # Plot the areas with values using consistent styling
    plot_kwargs = styler.get_spatial_plot_kwargs()
    # Remove cmap from plot_kwargs since we're using custom_cmap
    plot_kwargs.pop('cmap', None)
    im = temp_gdf.plot(column='value',
                       ax=ax,
                       legend=False,
                       cmap=custom_cmap,
                       **plot_kwargs)

    # Overlay black postal code boundaries for consistency with spatial flows
    relevant_gdf.boundary.plot(
        ax=ax, color=styler.boundary_color, linewidth=styler.boundary_width, zorder=1)
    relevant_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, zorder=2)

    # Plot boundaries for zero-value areas separately to ensure visibility
    zero_value_gdf = temp_gdf[temp_gdf['value'] <= 1e-10]
    if not zero_value_gdf.empty:
        zero_value_gdf.boundary.plot(
            ax=ax, color=styler.boundary_color, linewidth=styler.boundary_width)

    # Set aspect ratio for proper map proportions
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_temporal_components(factor: np.ndarray,
                             save_path: str,
                             tensor_name: str,
                             time_granularity: Literal['timebin', 'weekhour', 'hourly'],
                             plot_individual_ranks: bool = False,
                             is_tucker: bool = False,
                             show_time_labels: bool = False) -> None:
    """Plot all temporal components on a single figure with subplots.

    Args:
        factor: Factor matrix for temporal component
        save_path: Path to save the plot
        tensor_name: Name of the tensor being visualized
        time_granularity: Type of time granularity ('timebin', 'weekhour', or 'hourly')
        plot_individual_ranks: If True, create individual plots for each rank
    """
    n_components = factor.shape[1]
    time_points = factor.shape[0]
    time_config = TIME_BINS[time_granularity]

    # Calculate sparsity
    sparsity = calculate_sparsity(factor)
    sparsity_percent = sparsity * 100

    if plot_individual_ranks:
        # Create individual plots for each rank
        for rank in range(n_components):
            individual_save_path = save_path.replace(
                '.png', f'_rank_{rank+1}.png')
            plot_single_temporal_component(
                factor, individual_save_path, tensor_name,
                time_granularity, rank, is_tucker=is_tucker,
                show_time_labels=show_time_labels
            )
        return

    # Create figure with subplots stacked vertically
    fig, axes = plt.subplots(n_components, 1, figsize=(15, 3*n_components))
    if n_components == 1:
        axes = [axes]

    # Day labels for annotation
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Prepare x-axis ticks and labels
    if time_granularity == 'timebin':
        # 5 bins per day, 7 days
        bins_per_day = time_config['bins_per_day']
        # Calculate mean time for each bin
        bin_means = ["03:15", "07:45", "12:30", "17:15",
                     "21:15"]  # hardcoded for the 5 bins
        x_ticks = np.arange(time_points)
        x_ticklabels = [bin_means[i % bins_per_day]
                        for i in range(time_points)]
        day_starts = np.arange(0, time_points, bins_per_day)
    elif time_granularity == 'weekhour':
        # 24 hours per day, 7 days
        bins_per_day = 24
        x_ticks = np.arange(0, time_points, 3)  # every 3 hours
        x_ticklabels = [f"{(i%24):02d}:00" for i in x_ticks]
        day_starts = np.arange(0, time_points, bins_per_day)
    else:  # hourly
        bins_per_day = 24
        x_ticks = np.arange(0, time_points, 3)
        x_ticklabels = [f"{i:02d}:00" for i in x_ticks]
        day_starts = []

    # Plot each component
    for idx, ax in enumerate(axes):
        values = factor[:, idx]
        ax.plot(np.arange(time_points), values,
                '-', linewidth=1.5, color=f'C{idx}')

        # Add day label as annotation above the plot at the start of each day
        for d, start in enumerate(day_starts):
            ymax = 1.05  # Fixed ymax for 0-1 scale
            ax.annotate(day_labels[d], xy=(start + bins_per_day/2 - 0.5, ymax), xytext=(0, 10),
                        textcoords='offset points', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
            ax.axvline(x=start, color='gray', linestyle='--', alpha=0.3)

        # Format y-axis and x-axis based on show_time_labels parameter
        if show_time_labels:
            # Show x-axis labels for Rotterdam Tucker decompositions
            ax.tick_params(axis='x', labelsize=20)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticklabels, rotation=0, ha='center')
            ax.set_xlabel('Hour of day', fontsize=24)
        else:
            # Remove x-axis ticks and labels
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_xlabel('')   # Remove x-axis label
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Component {idx + 1}')
        ax.tick_params(axis='y', labelsize=28)
        # Use fixed 0-1 scale for consistent interpretation
        ax.set_ylim(0, 1)
        # Set specific y-axis ticks at 0.25 intervals
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        # Move y-axis to the right side
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

        # Add a subtle frame to match spatial plots
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('gray')
            spine.set_linewidth(0.5)

    plt.tight_layout()
    # Create title based on time granularity
    if time_granularity == 'timebin':
        title = f'Weekly Temporal Patterns for {tensor_name} (7 days × 5 time bins)\n'
    elif time_granularity == 'weekhour':
        title = f'Weekly Temporal Patterns for {tensor_name} (7 days × 24 hours)\n'
    else:  # hourly
        title = f'Daily Temporal Patterns for {tensor_name} (24 hours)\n'
    title += f'Sparsity: {sparsity_percent:.1f}% (values ≤ 1e-10)'
    fig.suptitle(title, y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_single_temporal_component(factor: np.ndarray,
                                   save_path: str,
                                   tensor_name: str,
                                   time_granularity: Literal['timebin', 'weekhour', 'hourly'],
                                   rank: int,
                                   is_tucker: bool = False,
                                   show_time_labels: bool = False) -> None:
    """Plot a single temporal component for a specific rank.

    Args:
        factor: Factor matrix for temporal component
        save_path: Path to save the plot
        tensor_name: Name of the tensor being visualized
        time_granularity: Type of time granularity ('timebin', 'weekhour', or 'hourly')
        rank: Rank index to plot (0-based)
    """
    time_points = factor.shape[0]
    time_config = TIME_BINS[time_granularity]

    # Create figure with consistent styling
    fig, ax = styler.create_temporal_figure(is_tucker=is_tucker)

    # Day labels for annotation
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Prepare x-axis ticks and labels
    if time_granularity == 'timebin':
        # 5 bins per day, 7 days
        bins_per_day = time_config['bins_per_day']
        # Calculate mean time for each bin
        bin_means = ["03:15", "07:45", "12:30", "17:15",
                     "21:15"]  # hardcoded for the 5 bins
        x_ticks = np.arange(time_points)
        x_ticklabels = [bin_means[i % bins_per_day]
                        for i in range(time_points)]
        day_starts = np.arange(0, time_points, bins_per_day)
    elif time_granularity == 'weekhour':
        # 24 hours per day, 7 days
        bins_per_day = 24
        x_ticks = np.arange(0, time_points, 3)  # every 3 hours
        x_ticklabels = [f"{(i%24)}" for i in x_ticks]
        day_starts = np.arange(0, time_points, bins_per_day)
    else:  # hourly
        bins_per_day = 24
        x_ticks = np.arange(0, time_points, 3)
        x_ticklabels = [f"{i}" for i in x_ticks]
        day_starts = []

    # Plot the component
    values = factor[:, rank]
    ax.plot(np.arange(time_points), values,
            '-', linewidth=2, color='blue')

    # Add day label as annotation above the plot at the start of each day
    for d, start in enumerate(day_starts):
        ymax = 1.05  # Fixed ymax for 0-1 scale
        ax.annotate(day_labels[d], xy=(start + bins_per_day/2 - 0.5, ymax), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')
        ax.axvline(x=start, color='gray', linestyle='--', alpha=0.3)

    # Apply temporal-specific styling
    styler.apply_temporal_style(ax)

    # Add x-axis timescale based on decomposition type and show_time_labels parameter
    ax.tick_params(axis='y', labelsize=28)
    if not is_tucker:
        # For CP decompositions: show x-axis labels
        ax.tick_params(axis='x', labelsize=20)
        # Set x-axis ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, rotation=0, ha='center')
        ax.set_xlabel('Hour of day', fontsize=24)
    elif show_time_labels:
        # For Rotterdam Tucker decompositions: show x-axis labels
        ax.tick_params(axis='x', labelsize=20)
        # Set x-axis ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, rotation=0, ha='center')
        ax.set_xlabel('Hour of day', fontsize=24)
    else:
        # For other Tucker decompositions: remove x-axis labels
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_xlabel('')   # Remove x-axis label
    # Set specific y-axis ticks at 0.25 intervals
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    # Move y-axis to the right side
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def create_analysis_directory(base_path: str, decomp_path: str) -> str:
    """Create a directory for analysis results.

    Args:
        base_path: Base path for analysis results
        decomp_path: Path to the decomposition file to extract tensor info

    Returns:
        Path to the created directory
    """
    # Extract tensor type, city and time resolution from decomposition path
    # Example: .../odt_utrecht_weekhour/run_20250509_155307/... -> odt_utrecht_weekhour
    decomp_parts = Path(decomp_path).parts
    for part in decomp_parts:
        if part.startswith(('odt_', 'odtm_')):
            tensor_dir = part
            break
    else:
        # Fallback to timestamp if no tensor info found
        tensor_dir = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create tensor-specific directory
    tensor_dir_path = Path(base_path) / tensor_dir
    tensor_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created tensor directory: {tensor_dir_path}")

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = tensor_dir_path / f"run_{timestamp}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created analysis directory: {analysis_dir}")

    return str(analysis_dir)


def save_metadata(output_dir: str,
                  decomp_path: str,
                  geojson_path: str,
                  index_mappings_path: str,
                  weights: np.ndarray,
                  factors: List[np.ndarray]) -> None:
    """Save metadata about the decomposition and visualization.

    Args:
        output_dir: Directory to save metadata
        decomp_path: Path to decomposition file
        geojson_path: Path to GeoJSON file
        index_mappings_path: Path to index mappings
        weights: Decomposition weights
        factors: Decomposition factors
    """
    # Calculate statistics for each mode
    origin_stats = {
        'shape': factors[0].shape,
        'min': float(np.min(factors[0])),
        'max': float(np.max(factors[0])),
        'mean': float(np.mean(factors[0])),
        'sparsity': float(calculate_sparsity(factors[0]))
    }

    dest_stats = {
        'shape': factors[1].shape,
        'min': float(np.min(factors[1])),
        'max': float(np.max(factors[1])),
        'mean': float(np.mean(factors[1])),
        'sparsity': float(calculate_sparsity(factors[1]))
    }

    temporal_stats = {
        'shape': factors[2].shape,
        'min': float(np.min(factors[2])),
        'max': float(np.max(factors[2])),
        'mean': float(np.mean(factors[2])),
        'sparsity': float(calculate_sparsity(factors[2]))
    }

    metadata = {
        'input_files': {
            'decomposition': decomp_path,
            'geojson': geojson_path,
            'index_mappings': index_mappings_path
        },
        'decomposition_info': {
            'rank': len(weights),
            'weights': weights.tolist(),
            'origin_mode': origin_stats,
            'destination_mode': dest_stats,
            'temporal_mode': temporal_stats
        },
        'visualization_info': {
            'spatial_plot': {
                'colormap': 'viridis',
                'context_alpha': 0.3,
                'context_color': 'lightgray',
                'context_edge_color': 'gray',
                'max_components_per_row': 3
            },
            'temporal_plot': {
                'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'line_style': '-o',
                'grid_alpha': 0.3
            }
        },
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'script_version': '1.0.0'
        }
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, 'visualization_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def plot_top_weights_per_rank(factors: List[np.ndarray],
                              n_top: int = 20,
                              save_path: str = None,
                              tensor_name: str = None,
                              mappings: dict = None,
                              time_granularity: str = None) -> None:
    """Plot the top weights for each rank across all modes, using postal codes for origin/destination and time labels for time mode if available."""
    n_ranks = factors[0].shape[1]  # number of patterns/ranks
    mode_names = ['Origin', 'Destination', 'Time']
    idx_to_postal = [
        mappings['idx_to_origins'] if mappings is not None else None,
        mappings['idx_to_destinations'] if mappings is not None else None,
        None  # For time mode, handled below
    ]
    # Prepare time labels if possible
    time_labels = None
    if time_granularity is not None and time_granularity in TIME_BINS:
        time_labels = TIME_BINS[time_granularity]['bin_labels']
    # Create a figure for each rank
    for rank in range(n_ranks):
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        # Plot weights for each mode
        for i, (factor_matrix, name) in enumerate(zip(factors, mode_names)):
            # Get weights for this rank (column of the factor matrix)
            rank_weights = np.abs(factor_matrix[:, rank])
            # Sort weights in descending order and get top n
            sorted_indices = np.argsort(rank_weights)[::-1][:n_top]
            sorted_weights = rank_weights[sorted_indices]
            # Compute cumulative sum and determine cutoff for 25%, 30%, and 35%
            total_weight = np.sum(rank_weights)
            cumulative = np.cumsum(rank_weights[sorted_indices])
            cutoff_25_idx = np.searchsorted(cumulative, 0.25 * total_weight)
            cutoff_30_idx = np.searchsorted(cumulative, 0.30 * total_weight)
            cutoff_35_idx = np.searchsorted(cumulative, 0.35 * total_weight)
            # Create subplot
            ax = fig.add_subplot(gs[i//3, i % 3])
            bars = ax.bar(range(1, len(sorted_weights) + 1),
                          sorted_weights, color='skyblue')
            ax.set_title(f'{name} Mode - Rank {rank+1}\nTop {n_top} Weights')
            ax.set_xlabel('Component Index')
            ax.set_ylabel('Weight')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            # Set x-axis labels
            if idx_to_postal[i] is not None:
                x_labels = [idx_to_postal[i].get(
                    str(idx), str(idx)) for idx in sorted_indices]
            elif i == 2 and time_labels is not None:
                # Time mode: use time labels if available
                x_labels = [time_labels[idx %
                                        len(time_labels)] for idx in sorted_indices]
            else:
                x_labels = [str(idx + 1) for idx in sorted_indices]
            ax.set_xticks(range(1, len(sorted_weights) + 1))
            ax.set_xticklabels(x_labels, rotation=45)
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2e}',
                        ha='center', va='bottom')
            # Draw dashed lines to separate top 25%, 30%, and 35% cumulative weight
            if cutoff_25_idx < len(sorted_weights):
                ax.axvline(x=cutoff_25_idx + 0.5, color='darkred',
                           linestyle='--', label='Top 25% Cumulative')
            if cutoff_30_idx < len(sorted_weights):
                ax.axvline(x=cutoff_30_idx + 0.5, color='darkgreen',
                           linestyle='--', label='Top 30% Cumulative')
            if cutoff_35_idx < len(sorted_weights):
                ax.axvline(x=cutoff_35_idx + 0.5, color='darkblue',
                           linestyle='--', label='Top 35% Cumulative')
            ax.legend()
        # Add a title for the entire figure
        title = f'Top {n_top} Weights for Rank {rank+1} Across All Modes'
        if tensor_name:
            title += f' for {tensor_name}'
        fig.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        if save_path:
            # Create rank-specific save path
            rank_save_path = save_path.replace('.png', f'_rank_{rank+1}.png')
            plt.savefig(rank_save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


def get_tucker_core_energies(core: np.ndarray, threshold: float = 0.8):
    """Compute the core energy for each entry (squared), sort, and select minimum set covering threshold of total energy.

    Args:
        core: The Tucker core tensor (numpy array)
        threshold: Fraction of total energy to cover (default 0.8)

    Returns:
        selected_indices: List of (i, j, k) tuples for selected entries
        selected_energies: List of energies for selected entries
        cumulative_energy: Cumulative energy values (sorted)
        total_energy: Total energy in the core tensor
    """
    # Flatten core and get squared energies
    flat_core = (core ** 2).flatten()
    total_energy = np.sum(flat_core)
    sorted_indices = np.argsort(flat_core)[::-1]  # Descending
    sorted_energies = flat_core[sorted_indices]
    cumulative = np.cumsum(sorted_energies)
    # Find minimum number of entries to reach threshold
    cutoff_idx = np.searchsorted(cumulative, threshold * total_energy) + 1
    selected_indices_flat = sorted_indices[:cutoff_idx]
    selected_energies = sorted_energies[:cutoff_idx]
    # Convert flat indices to (i, j, k)
    selected_indices = [np.unravel_index(
        idx, core.shape) for idx in selected_indices_flat]
    return selected_indices, selected_energies, cumulative, total_energy


def plot_lambda_weights(weights: np.ndarray,
                        save_path: str,
                        tensor_name: str,
                        is_tucker: bool = False,
                        core: np.ndarray = None,
                        selected_indices: list = None,
                        max_entries: int = 15) -> None:
    """Plot the lambda weights (CP) or core energies (Tucker) for the decomposition.

    Args:
        weights: 1D array of weights (CP) or core energies (Tucker)
        save_path: Path to save the plot
        tensor_name: Name of the tensor being visualized
        is_tucker: If True, plot core energies instead of weights
        core: The core tensor (for Tucker)
        selected_indices: List of selected (i, j, k) indices (for annotation)
        selected_energies: List of selected energies (for annotation)
        total_energy: Total energy in the core tensor
    """
    plt.figure(figsize=(12, 6))

    if is_tucker and core is not None:
        # For Tucker, plot the top 15 core entries with squared energies
        core_entries = []
        for i in range(core.shape[0]):
            for j in range(core.shape[1]):
                for k in range(core.shape[2]):
                    core_entries.append(((i, j, k), core[i, j, k] ** 2))

        # Sort by energy and take top 15
        core_entries.sort(key=lambda x: x[1], reverse=True)
        top_entries = core_entries[:max_entries]

        # Plot the top 15 entries
        indices = [f"({i+1},{j+1},{k+1})" for (i, j, k), _ in top_entries]
        values = [val for _, val in top_entries]

        plt.bar(range(len(values)), values, color='skyblue')
        plt.xticks(range(len(indices)), indices, rotation=45, ha='right')
        plt.title(
            f'Top {max_entries} Tucker Core Energies - {tensor_name}\n(Sorted, g_ijk^2)')
        plt.xlabel('Core Indices (i,j,k)')
        plt.ylabel('Energy (g_ijk^2)')

        # Add the 80% energy cutoff line if we have the information
        if selected_indices is not None and len(selected_indices) <= 15:
            cutoff = len(selected_indices)
            plt.axvline(x=cutoff-0.5, color='red',
                        linestyle='--', label='80% Energy Cutoff')
            plt.legend()

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

    else:
        # For CP, plot all weights with original formatting
        bars = plt.bar(range(1, len(weights) + 1), weights, color='skyblue')
        plt.title(
            f'Normalized Component Weights for {tensor_name}\n(Unit-length normalized factors with weights)')
        plt.xlabel('Component')
        plt.ylabel('Weight (absorbed normalization constants)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2e}',
                     ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_all_temporal_components_single_chart(factor: np.ndarray,
                                              save_path: str,
                                              tensor_name: str,
                                              time_granularity: Literal['timebin', 'weekhour', 'hourly'],
                                              is_tucker: bool = False) -> None:
    """Plot all temporal components in a single chart with different colors and markers, with y-axis fixed from 0 to 1, and distinct marker shapes for each pattern."""
    n_components = factor.shape[1]
    time_points = factor.shape[0]
    time_config = TIME_BINS[time_granularity]

    # Use a strong, distinct color palette (matplotlib tab10)
    pattern_colors = ['#d62728',  # red
                      '#1f77b4',  # blue
                      '#2ca02c',  # green
                      '#ff7f0e',  # orange
                      '#9467bd']  # purple
    # Use distinct marker shapes for each pattern
    # square, triangle, circle, diamond, down triangle
    pattern_markers = ['s', '^', 'o', 'D', 'v']

    plt.figure(figsize=(10, 10))
    for idx in range(n_components):
        color = pattern_colors[idx % len(pattern_colors)]
        marker = pattern_markers[idx % len(pattern_markers)]
        plt.plot(
            np.arange(time_points),
            factor[:, idx],
            label=f'T{idx+1}',
            color=color,
            marker=marker,
            linewidth=3,
            markersize=10
        )
    plt.xlabel('Time of day $t$')
    plt.ylabel('$P(t)$')
    plt.title(f'All Temporal Patterns for {tensor_name}')
    # plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def create_timeline_axis(time_granularity: Literal['timebin', 'weekhour', 'hourly'],
                         save_path: str,
                         fig_width: float = 14,
                         legend_height: float = 5) -> None:
    """Create a standalone timeline axis for temporal plots.

    Args:
        time_granularity: Type of time granularity
        save_path: Path to save the timeline
        fig_width: Width of the figure (should match temporal plots)
    """
    time_config = TIME_BINS[time_granularity]
    time_points = time_config['total_bins']

    # Create figure with minimal height for just the axis
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, legend_height))

    # Day labels for annotation
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Prepare x-axis ticks and labels
    if time_granularity == 'timebin':
        # 5 bins per day, 7 days
        bins_per_day = time_config['bins_per_day']
        # Calculate mean time for each bin
        bin_means = ["03:15", "07:45", "12:30", "17:15", "21:15"]
        x_ticks = np.arange(time_points)
        x_ticklabels = [bin_means[i % bins_per_day]
                        for i in range(time_points)]
        day_starts = np.arange(0, time_points, bins_per_day)
    elif time_granularity == 'weekhour':
        # 24 hours per day, 7 days
        bins_per_day = 24
        x_ticks = np.arange(0, time_points, 3)  # every 3 hours
        x_ticklabels = [f"{(i%24):02d}:00" for i in x_ticks]
        day_starts = np.arange(0, time_points, bins_per_day)
    else:  # hourly
        bins_per_day = 24
        x_ticks = np.arange(0, time_points, 3)
        x_ticklabels = [f"{i:02d}" for i in x_ticks]
        day_starts = []

    # Create the axis
    ax.set_xlim(0, time_points - 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, ha='center')
    ax.set_xlabel('Time', fontsize=12)

    # Add day labels
    for d, start in enumerate(day_starts):
        ax.annotate(day_labels[d], xy=(start + bins_per_day/2 - 0.5, 0.5),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.axvline(x=start, color='gray', linestyle='--', alpha=0.3)

        # Remove y-axis and spines
    ax.set_yticks([])
    for spine in ['left', 'right', 'top']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('gray')
    ax.spines['bottom'].set_linewidth(0.5)

    # Adjust layout to prevent tight layout warning
    plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def create_color_legend(mode_name: str,
                        save_path: str,
                        fig_width: float = 14,
                        legend_height: float = 5) -> None:
    """Create a standalone color legend for spatial plots.

    Args:
        mode_name: Name of the mode (Origin/Destination)
        save_path: Path to save the legend
        fig_width: Width of the figure (should match spatial plots)
        legend_height: Height of the legend (should match timeline height)
    """
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, legend_height))

    # Set colormap based on mode_name - EXACTLY the same as in spatial plots
    if mode_name.lower() == "origin":
        # Blue colormap for origin - same as in plot_single_spatial_component
        colors = [
            (1, 1, 1, 0),      # White with 0 opacity
            (0.2, 0.4, 0.8, 0.3),  # Light blue with low opacity
            (0.1, 0.2, 0.6, 0.6),  # Medium blue with medium opacity
            (0, 0, 0.4, 0.9)   # Dark blue with high opacity
        ]
        label = "Origin Factor Value"
    else:
        # Red colormap for destination - same as in plot_single_spatial_component
        colors = [
            (1, 1, 1, 0),      # White with 0 opacity
            (0.8, 0.3, 0.3, 0.3),  # Light red with low opacity
            (0.6, 0.1, 0.1, 0.6),  # Medium red with medium opacity
            (0.4, 0, 0, 0.9)   # Dark red with high opacity
        ]
        label = "Destination Factor Value"

    custom_cmap = LinearSegmentedColormap.from_list('custom', colors)

    # Create a gradient bar that shows the actual colors
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    im = ax.imshow(gradient, cmap=custom_cmap,
                   aspect='auto', extent=[0, 1, 0, 1])

    # Set up the axis to look like a colorbar
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])
    ax.set_yticks([])
    ax.set_xlabel(label, fontsize=12)
    ax.tick_params(axis='x', labelsize=10)

    # Add a subtle frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('gray')
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_spatial_flows_cp_components(relevant_gdf, origin_factors, dest_factors, idx_to_origins, idx_to_destinations, output_dir, tensor_name):
    """Plot spatial flows as arrows for each CP component (rank)."""
    import matplotlib.patches as mpatches
    n_components = origin_factors.shape[1]
    # Precompute centroids for all postal codes
    centroids = {row['pc4_code']: row['geometry'].centroid for _,
                 row in relevant_gdf.iterrows()}
    for comp in range(n_components):
        fig, ax = plt.subplots(figsize=(10, 6))
        # Plot the postal code boundaries
        relevant_gdf.boundary.plot(
            ax=ax, color='black', linewidth=0.5, zorder=1)
        # Compute OD flow matrix for this component
        origin_vec = origin_factors[:, comp]
        dest_vec = dest_factors[:, comp]
        # shape: (n_origins, n_destinations)
        flow_matrix = np.outer(origin_vec, dest_vec)
        # Top-K logic
        K = 5
        threshold = 1e-3
        top_origin_indices = np.argsort(origin_vec)[-K:][::-1]
        top_dest_indices = np.argsort(dest_vec)[-K:][::-1]
        # Find max flow among plotted flows for legend
        plotted_flows = []
        for i in top_origin_indices:
            for j in top_dest_indices:
                flow_strength = flow_matrix[i, j]
                if flow_strength < threshold or i == j:
                    continue
                origin_code = idx_to_origins[str(i)]
                dest_code = idx_to_destinations[str(j)]
                if origin_code in centroids and dest_code in centroids:
                    o = centroids[origin_code]
                    d = centroids[dest_code]
                    plotted_flows.append(flow_strength)
        max_flow = max(plotted_flows) if plotted_flows else 0

        # Collect all arrows with their properties for proper layering
        arrows_to_plot = []
        for i in top_origin_indices:
            for j in top_dest_indices:
                flow_strength = flow_matrix[i, j]
                if flow_strength < threshold or i == j:
                    continue
                origin_code = idx_to_origins[str(i)]
                dest_code = idx_to_destinations[str(j)]
                if origin_code in centroids and dest_code in centroids:
                    o = centroids[origin_code]
                    d = centroids[dest_code]
                    # Arrow width proportional to flow_strength (relative to max_flow)
                    width = 4 * (flow_strength /
                                 max_flow) if max_flow > 0 else 0.5
                    # Color gradient: light red for weak flows, dark red for strong flows
                    color_intensity = flow_strength / max_flow if max_flow > 0 else 0
                    # Use a red color gradient: light red (0.8, 0.3, 0.3) to dark red (0.4, 0, 0)
                    red = 0.8 - 0.4 * color_intensity
                    green = 0.3 - 0.3 * color_intensity
                    blue = 0.3 - 0.3 * color_intensity
                    arrow_color = (red, green, blue)

                    arrows_to_plot.append({
                        'origin': o,
                        'destination': d,
                        'flow_strength': flow_strength,
                        'width': width,
                        'color': arrow_color
                    })

        # Sort arrows by flow strength (weakest first, strongest last for proper layering)
        arrows_to_plot.sort(key=lambda x: x['flow_strength'])

        # Plot arrows in order (weakest first = background, strongest last = foreground)
        for arrow in arrows_to_plot:
            ax.annotate('', xy=(arrow['destination'].x, arrow['destination'].y),
                        xytext=(arrow['origin'].x, arrow['origin'].y),
                        arrowprops=dict(arrowstyle='->', color=arrow['color'],
                                        lw=arrow['width'], alpha=0.8,
                                        mutation_scale=20, shrinkA=0, shrinkB=0), zorder=2)
        # ax.set_title(f'Spatial Flows (Arrows) for {tensor_name} - Component {comp+1}')
        # Apply consistent frame styling
        styler.apply_frame_style(ax)
        ax.set_aspect('equal')
        plt.tight_layout()
        save_path = os.path.join(
            output_dir, f'spatial_flows_rank_{comp+1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def analyze_decomposition(decomp_path: str,
                          geojson_path: str,
                          index_mappings_path: str,
                          output_base_path: str,
                          plot_individual_ranks: bool = False,
                          weights=None,
                          factors=None,
                          mappings=None,
                          tensor_name=None,
                          time_granularity=None,
                          plot_single_temporal_chart: bool = False,
                          no_timestamp_dir: bool = False) -> None:
    """Analyze and visualize tensor decomposition."""
    # Use provided weights/factors/mappings if available, else load
    if weights is None or factors is None:
        weights, factors = load_decomposition(decomp_path)
    if mappings is None:
        mappings = load_index_mappings(index_mappings_path)
    if time_granularity is None:
        time_granularity = determine_time_granularity(decomp_path)
    if tensor_name is None:
        decomp_parts = Path(decomp_path).parts
        for part in decomp_parts:
            if part.startswith(('odt_', 'odtm_')):
                tensor_name = part
                break
        if tensor_name is None:
            tensor_name = Path(decomp_path).stem

    # Create output directory
    if no_timestamp_dir:
        output_dir = output_base_path
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = create_analysis_directory(output_base_path, decomp_path)
    logger.info(f"Saving analysis results to {output_dir}")

    # Save metadata
    save_metadata(output_dir, decomp_path, geojson_path,
                  index_mappings_path, weights, factors)

    # Plot origin components
    logger.info("Processing origin components...")
    plot_spatial_components(
        None,  # context_gdf is not used anymore
        load_geojson(geojson_path, list(mappings['origins'].keys())),
        factors[0],
        mappings['idx_to_origins'],
        'Origin',
        os.path.join(output_dir, "origin_components.png"),
        tensor_name,
        plot_individual_ranks
    )

    # Plot destination components
    logger.info("Processing destination components...")
    plot_spatial_components(
        None,  # context_gdf is not used anymore
        load_geojson(geojson_path, list(mappings['destinations'].keys())),
        factors[1],
        mappings['idx_to_destinations'],
        'Destination',
        os.path.join(output_dir, "destination_components.png"),
        tensor_name,
        plot_individual_ranks
    )

    # Plot temporal components
    logger.info("Processing temporal components...")
    # Check if this is a Tucker decomposition
    is_tucker = 'core' in np.load(decomp_path, allow_pickle=True)

    # Determine if we should show time labels (Rotterdam Tucker decompositions)
    show_time_labels = False
    if is_tucker and 'rotterdam' in decomp_path.lower():
        show_time_labels = True

    plot_temporal_components(
        factors[2],
        os.path.join(output_dir, "temporal_components.png"),
        tensor_name,
        time_granularity,
        plot_individual_ranks,
        is_tucker=is_tucker,
        show_time_labels=show_time_labels
    )

    # Create standalone timeline axis
    logger.info("Creating timeline axis...")
    create_timeline_axis(
        time_granularity,
        os.path.join(output_dir, "timeline_axis.png"),
        fig_width=3.5,
        legend_height=0.3
    )

    # Create separate color legends
    logger.info("Creating color legends...")
    create_color_legend(
        'Origin',
        os.path.join(output_dir, "origin_color_legend.png"),
        fig_width=3.5,
        legend_height=0.3
    )
    create_color_legend(
        'Destination',
        os.path.join(output_dir, "destination_color_legend.png"),
        fig_width=3.5,
        legend_height=0.3
    )

    # For Tucker: analyze core energies and select important patterns
    if 'core' in np.load(decomp_path, allow_pickle=True) and 'factors' in np.load(decomp_path, allow_pickle=True):
        selected_indices, selected_energies, cumulative, total_energy = get_tucker_core_energies(
            np.load(decomp_path, allow_pickle=True)['core'], threshold=0.8)
        # Print summary to console
        print("\nTucker Core Energy Analysis:")
        print(
            f"Total core entries: {np.load(decomp_path, allow_pickle=True)['core'].size}")
        print(f"Total energy (sum g_ijk^2): {total_energy:.4f}")
        print(
            f"Number of entries to reach 80% energy: {len(selected_indices)}")
        print("Selected (i, j, k) indices and energies:")
        for idx, energy in zip(selected_indices, selected_energies):
            print(f"  Index {idx}: Energy {energy:.4f}")
        # Save to JSON
        core_energy_info = {
            "total_core_entries": int(np.load(decomp_path, allow_pickle=True)['core'].size),
            "total_energy": float(total_energy),
            "num_entries_80pct": int(len(selected_indices)),
            "selected_indices": [list(map(int, idx)) for idx in selected_indices],
            "selected_energies": [float(e) for e in selected_energies]
        }
        with open(os.path.join(output_dir, 'tucker_core_energy_summary.json'), 'w') as f:
            json.dump(core_energy_info, f, indent=2)
        # Plot core energies (core energy distribution for Tucker)
        plot_lambda_weights(weights, os.path.join(output_dir, "core_energy_distribution.png"), tensor_name, is_tucker=True,
                            core=np.load(decomp_path, allow_pickle=True)['core'], selected_indices=selected_indices, max_entries=15)
    else:
        # CP: plot lambda weights as before
        plot_lambda_weights(weights, os.path.join(
            output_dir, "lambda_weights.png"), tensor_name)
        # Plot top weights per rank (CP only)
        logger.info("Processing top weights per rank...")
        plot_top_weights_per_rank(
            factors,
            n_top=20,
            save_path=os.path.join(output_dir, "top_weights.png"),
            tensor_name=tensor_name,
            mappings=mappings,
            time_granularity=time_granularity
        )
        # New: Plot spatial flows for CP decompositions only
        logger.info("Plotting spatial flows as arrows for CP components...")
        # Use all relevant postal codes for centroids
        all_postal_codes = set(mappings['origins'].keys()) | set(
            mappings['destinations'].keys())
        relevant_gdf = load_geojson(geojson_path, list(all_postal_codes))
        plot_spatial_flows_cp_components(
            relevant_gdf,
            factors[0],
            factors[1],
            mappings['idx_to_origins'],
            mappings['idx_to_destinations'],
            output_dir,
            tensor_name
        )

    # If requested, plot all temporal components in a single chart
    if plot_individual_ranks and plot_single_temporal_chart:
        plot_all_temporal_components_single_chart(
            factors[2],
            os.path.join(output_dir, "temporal_components_single_chart.png"),
            tensor_name,
            time_granularity,
            is_tucker=is_tucker
        )

    logger.info("Analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize tensor decomposition results')
    parser.add_argument('decomp_path', type=str,
                        help='Path to decomposition NPZ file')
    parser.add_argument('--geojson', type=str, default=DEFAULT_GEOJSON_PATH,
                        help=f'Path to GeoJSON file (default: {DEFAULT_GEOJSON_PATH})')
    parser.add_argument('index_mappings_path', type=str,
                        help='Path to index mappings JSON file')
    parser.add_argument('output_base_path', type=str,
                        help='Base path for saving analysis results')
    parser.add_argument('--plot-individual-ranks', action='store_true',
                        help='Create individual plots for each rank instead of combined plots')
    parser.add_argument('--single-temporal-chart', action='store_true',
                        help='Plot all temporal patterns in a single chart with different colors')
    parser.add_argument('--no-timestamp-dir', action='store_true',
                        help='Store all output files directly in the provided output directory (no timestamped subfolder)')

    args = parser.parse_args()

    # Convert relative paths to absolute if needed
    decomp_path = args.decomp_path
    geojson_path = args.geojson
    index_mappings_path = args.index_mappings_path
    output_base_path = args.output_base_path

    if not os.path.isabs(geojson_path):
        geojson_path = os.path.abspath(geojson_path)
    if not os.path.isabs(decomp_path):
        decomp_path = os.path.abspath(decomp_path)
    if not os.path.isabs(index_mappings_path):
        index_mappings_path = os.path.abspath(index_mappings_path)
    if not os.path.isabs(output_base_path):
        output_base_path = os.path.abspath(output_base_path)

    logger.info(f"Using GeoJSON file: {geojson_path}")

    # Load decomposition and mappings once
    weights, factors = load_decomposition(decomp_path)
    mappings = load_index_mappings(index_mappings_path)
    time_granularity = determine_time_granularity(decomp_path)
    tensor_name = None
    decomp_parts = Path(decomp_path).parts
    for part in decomp_parts:
        if part.startswith(('odt_', 'odtm_')):
            tensor_name = part
            break
    if tensor_name is None:
        tensor_name = Path(decomp_path).stem

    analyze_decomposition(
        decomp_path,
        geojson_path,
        index_mappings_path,
        output_base_path,
        args.plot_individual_ranks,
        weights=weights,
        factors=factors,
        mappings=mappings,
        tensor_name=tensor_name,
        time_granularity=time_granularity,
        plot_single_temporal_chart=args.single_temporal_chart,
        no_timestamp_dir=args.no_timestamp_dir
    )
