# Master Thesis: Non-Negative Tensor Factorization for Travel Diaries

This repository contains the implementation of a master thesis project focusing on analyzing Origin-Destination-Time (ODT) tensors using Non-Negative Tensor Factorization (NNTF) techniques. The project analyzes travel diary data from two Dutch cities to discover temporal and spatial patterns in urban mobility.

> **Thesis PDF Available:**  
> The full master thesis describing this project, methodology, and results is included in this repository as [Master_Thesis_Peter_Falterbaum.pdf](Master_Thesis_Peter_Falterbaum.pdf).  
> Readers interested in the scientific background, detailed analysis, and conclusions are encouraged to start with the thesis document.

## Project Structure

```
.
├── data/                    # Data directory
│   ├── raw/                 # Raw ODiN travel diary data
│   ├── processed/           # Processed tensor data for multiple cities
│   └── results/             # Decomposition results and outputs
├── code/                    # Main implementation code
│   ├── decompose/           # Tensor decomposition implementations
│   ├── preprocess/          # Data preprocessing utilities
│   ├── visualize/           # Visualization tools
│   ├── utils/               # Utility functions
│   ├── create_tensors/      # Tensor creation from raw data
│   ├── rank_selection/      # Rank selection algorithms
│   ├── comparison_cp_tucker/ # Comparison between CP and Tucker decompositions
│   └── setup/               # Environment setup scripts
├── src/                     # Core source code
│   └── data_utils.py        # Data utility functions
└── README.md               # This file
```

## Key Features

### Multi-City Analysis

- **Cities Covered**: Utrecht, Rotterdam
- **Data Source**: ODiN (Onderzoek Verplaatsingen in Nederland) travel diary survey
- **Temporal Resolution**: Both timebin (5 periods/day) and weekhour (24 hours/day) representations

### Tensor Decomposition Methods

- **CP (Canonical Polyadic) Decomposition**: Primary method for pattern discovery
- **Tucker Decomposition**: Alternative method for comparison
- **Algorithms**: MU (Multiplicative Update) and HALS (Hierarchical Alternating Least Squares)
- **Non-Negative Constraints**: All decompositions maintain non-negativity for interpretability

### Performance Characteristics

- **Tested Ranks**: 5, 10, 15, 20
- **Explained Variance**: 32% (rank 5) to 59% (rank 20)
- **Sparsity**: 23-55% depending on algorithm and rank
- **GPU Support**: CUDA and Apple Silicon (MPS/Metal) acceleration

## Data Overview

This project analyzes travel diary data for two Dutch cities: **Utrecht** and **Rotterdam**. The data is sourced from the ODiN (Onderzoek Onderweg in Nederland) survey, available at [https://doi.org/10.17026/SS/FNXJEU](https://doi.org/10.17026/SS/FNXJEU).

We tested three different time binning approaches:

1. **24-hour based**: Separate tensors for weekday and weekend (24 bins each).
2. **Peak-hour bins**: 5 time bins per day, aggregated over 7 days (35 bins).
3. **Week-hour aggregation**: Trips aggregated by hour of the week (168 bins).

### Tensor Dimensionalities for Utrecht

| Metric           | 24-Hour Weekday | 24-Hour Weekend | Peak Hour | Week Hour |
|------------------|-----------------|-----------------|-----------|-----------|
| Origin           | 45              | 44              | 45        | 45        |
| Destination      | 45              | 44              | 45        | 45        |
| Time             | 24              | 24              | 35        | 168       |
| Possible Comb.   | 48,600          | 46,464          | 70,875    | 340,200   |
| Density (%)      | 2.90            | 1.21            | 2.68      | 0.66      |
| Avg. Trip Count  | 1.35            | 1.15            | 0.31      | 1.13      |
| Median TC        | 1.00            | 1.00            | 0.29      | 1.00      |
| Std. Dev. TC     | 0.73            | 0.45            | 0.21      | 0.44      |

### Tensor Dimensionalities for Rotterdam

| Metric           | 24-Hour Weekday | 24-Hour Weekend | Peak Hour | Week Hour |
|------------------|-----------------|-----------------|-----------|-----------|
| Origin           | 69              | 62              | 69        | 69        |
| Destination      | 69              | 62              | 69        | 69        |
| Time             | 24              | 24              | 35        | 168       |
| Possible Comb.   | 114,264         | 92,256          | 166,635   | 799,848   |
| Density (%)      | 1.38            | 0.70            | 1.32      | 0.32      |
| Avg. Trip Count  | 1.60            | 1.36            | 0.38      | 1.35      |
| Median TC        | 1.00            | 1.00            | 0.36      | 1.00      |
| Std. Dev. TC     | 1.02            | 0.78            | 0.30      | 0.78      |

### Tensor Characteristics

- **Sparsity**: 0.56% - 4.62% depending on temporal resolution
- **Symmetric Spaces**: Origin and destination spaces are identical
- **Complete Coverage**: All postal codes in each city are represented

## Getting Started

### Prerequisites

- Python 3.11+
- Conda or pip for package management
- GPU support (optional, for acceleration)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Pitwutz/master_thesis_nntf_travel_diaries.git
   cd master_thesis_nntf_travel_diaries
   ```

2. **Create Conda Environment**:

   ```bash
   conda create -n nntf python=3.11.5
   conda activate nntf
   ```

3. **Install Dependencies**:
   ```bash
   pip install tensorly torch numpy scipy matplotlib pandas jupyter
   ```

### Basic Usage

1. **Run CP Decomposition**:

   ```bash
   cd code/decompose
   python decompose_cp.py --tensor odt_processed_utrecht_timebin.npz --ranks 5 10 15 20
   ```

2. **Run Tucker Decomposition**:

   ```bash
   python decompose_tucker.py --tensor odt_processed_utrecht_timebin.npz --ranks 5 10 15 20
   ```

3. **Run All Decompositions**:
   ```bash
   python run_all_decompositions_cp.py
   python run_all_decompositions_tucker.py
   ```

## Key Findings

- **NNTF (CP and Tucker decompositions)** enabled interpretable analysis of travel diary data for Utrecht and Rotterdam, revealing both common and city-specific mobility patterns.
- **Preprocessing pipeline** emphasized data sparsity and meaningful trip flows, using robust time binning (weekday/weekend, peak-hour, week-hour) and exclusion of intra-zonal/low-frequency OD pairs to enhance latent structure.
- **Rank selection** using explained variance (EV) and pattern entropy (PE) identified optimal CP ranks of 5–6 and Tucker dimension-specific ranks of 4–6, balancing accuracy and interpretability.
- **Key mobility patterns:**
  - Utrecht: City center evening peaks, suburban car commutes, shopping transit, and weekend leisure/shopping trips.
  - Rotterdam: Student commuting to university (morning/evening peaks), senior suburban flows, and strong weekend shopping/homebound patterns.
- **Practical value:** The extracted patterns provide actionable insights for urban and transport planning, e.g., highlighting the effectiveness of main transport routes to city centers and the need for improved suburban connectivity.
- **Both CP and Tucker** showed high correlation in extracted patterns, validating the effectiveness of NNTF for spatio-temporal mobility analysis.

## Development Workflow

### Data Processing Pipeline

1. **Raw Data**: ODiN travel diary survey data
2. **Preprocessing**: Filtering, cleaning, and aggregation
3. **Tensor Creation**: Converting to sparse tensor format
4. **Decomposition**: Applying NNTF algorithms
5. **Analysis**: Pattern discovery and interpretation

### Version Control

- Jupyter notebooks paired with markdown files using `jupytext`
- Automated version control for reproducible research
- Comprehensive documentation of methods and results

## Dependencies

### Core Libraries

- `tensorly`: Tensor operations and decompositions
- `torch`: Deep learning framework (CPU/GPU support)
- `numpy`: Numerical computing
- `scipy`: Scientific computing
- `matplotlib`: Visualization
- `pandas`: Data manipulation

### Optional Dependencies

- `jupyter`: Interactive notebooks
- `seaborn`: Enhanced visualizations
- `plotly`: Interactive plots

## Results Structure

Decomposition results are saved in timestamped directories containing:

- Decomposition factors for each rank
- Performance metrics and convergence plots
- Comparison summaries between algorithms
- System information for reproducibility

## Future Work

### Planned Improvements

1. **Performance Optimization**: Parallel processing and distributed computing
2. **Analysis Tools**: Automated pattern discovery and statistical significance tests
3. **Feature Additions**: Support for more decomposition methods and real-time monitoring

### Research Extensions

1. **Multi-modal Analysis**: Incorporating different transportation modes
2. **Spatial Clustering**: Advanced spatial pattern analysis
3. **Temporal Dynamics**: Time-varying pattern analysis

## Reproducibility & Environment

- All scripts and notebooks use a central config.py and environment variable (PROJECT_ROOT) for robust, portable path management.
- See config.py for details on setting up your environment.