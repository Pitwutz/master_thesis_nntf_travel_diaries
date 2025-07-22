# Master Thesis: Non-Negative Tensor Factorization for Travel Diaries

This repository contains the implementation of a master thesis project focusing on analyzing Origin-Destination-Time (ODT) tensors using Non-Negative Tensor Factorization (NNTF) techniques. The project analyzes travel diary data from multiple Dutch cities to discover temporal and spatial patterns in urban mobility.

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

- **Cities Covered**: Utrecht, 's-Gravenhage (The Hague), Rotterdam
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

### City Statistics

| City          | Total Trips | Origins | Destinations | Timebin Tensor | Weekhour Tensor |
| ------------- | ----------- | ------- | ------------ | -------------- | --------------- |
| Utrecht       | 5,637       | 46      | 46           | 46×46×35       | 46×46×168       |
| 's-Gravenhage | 6,371       | 62      | 62           | 62×62×35       | 62×62×168       |
| Rotterdam     | 7,223       | 76      | 76           | 76×76×35       | 76×76×168       |

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

### Algorithm Performance

- **MU vs HALS**: MU is 20-130x faster with similar quality results
- **Factor Sparsity**: HALS produces sparser factors (36-55%) beneficial for pattern discovery
- **Convergence**: Both methods achieve similar reconstruction quality

### Rank Selection

- **Higher Ranks**: Better explained variance (up to 59% at rank 20)
- **Factor Sparsity**: Increases with rank, helping identify distinct patterns
- **Trade-off**: Between reconstruction quality and interpretability

### Temporal Patterns

- **Weekday Factors**: Show 0% sparsity (all days contribute)
- **Origin/Destination Factors**: Show 36%+ sparsity
- **Time Resolution**: Both timebin and weekhour representations reveal different patterns

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

## Contributing

This is a research project for a master thesis. For questions or collaboration, please contact the author.

## License

[Add license information here]

## Acknowledgments

- ODiN survey data provided by [source]
- Tensor decomposition algorithms based on [references]
- Research conducted as part of [institution] master thesis
