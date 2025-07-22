# Tensor Decomposition Implementation

This directory contains the implementation of tensor decomposition techniques for analyzing Origin-Destination-Time (ODT) tensors. The implementation focuses on efficient processing of sparse tensors and provides both ALS (Alternating Least Squares) and HALS (Hierarchical Alternating Least Squares) decomposition methods.

## Key Files

- `decompose_cp.py`: Main implementation file for CP (Canonical Polyadic) decomposition
- `visualize_factors.py`: Visualization tools for decomposition results
- `analyze_results.py`: Analysis tools for comparing different decomposition approaches

## Implementation Features

### Sparse Tensor Handling

- Efficient loading of sparse tensors using PyTorch's sparse_coo_tensor
- Hybrid approach: loads data in sparse format, converts to dense for decomposition operations
- Memory-efficient representation for large tensors

### Decomposition Methods

1. **MU (Multiplicative Update)**
   - Fast convergence for non-negative CP
   - Similar quality results to HALS
   - Lower factor sparsity (23-53%)

2. **HALS (Hierarchical Alternating Least Squares)**
   - Produces sparser factors (36-55%)
   - Better for pattern discovery
   - More computationally intensive

### Performance Characteristics

- Tested ranks: 5, 10, 15, 20
- Explained variance increases with rank:
  - Rank 5: ~32%
  - Rank 20: ~59%
- Higher ranks show increased factor sparsity
- GPU acceleration support for both CUDA and Apple Silicon (MPS/Metal)

### Platform Support

- Local environment support
- Google Colab integration with automatic:
  - Package installation
  - Google Drive mounting
  - Result synchronization

## Usage

### Basic Usage

```python
python decompose_cp.py --tensor odt_processed_amsterdam_weekday_sparse.npz --ranks 5 10 15 20
```

### Command Line Arguments

- `--tensor`: Input tensor file name
- `--ranks`: List of ranks to try for decomposition
- `--max-iter`: Maximum number of iterations (default: 1000)
- `--tol`: Convergence tolerance (default: 1e-8)
- `--output-dir`: Custom output directory

## Results Structure

Results are saved in timestamped directories containing:

- Decomposition factors for each rank
- Performance metrics
- Convergence plots
- Comparison summaries between ALS and HALS
- System information for reproducibility

## Key Findings

1. **Algorithm Performance**

   - ALS consistently outperforms HALS in computation time
   - Both methods achieve similar reconstruction quality
   - HALS produces sparser factors, beneficial for pattern discovery

2. **Rank Selection**

   - Higher ranks (15-20) show better explained variance
   - Factor sparsity increases with rank
   - Trade-off between reconstruction quality and interpretability

3. **Factor Sparsity**
   - Weekday factors show 0% sparsity (all days contribute)
   - Origin/destination factors show 36%+ sparsity
   - Higher sparsity in higher ranks helps identify distinct patterns

## Future Improvements

1. **Performance Optimization**

   - Implement parallel processing for multiple ranks
   - Optimize memory usage for large tensors
   - Add support for distributed computing

2. **Analysis Tools**

   - Add more visualization options
   - Implement automated pattern discovery
   - Add statistical significance tests

3. **Feature Additions**
   - Support for more decomposition methods
   - Automated rank selection
   - Real-time progress monitoring

## Dependencies

- tensorly
- torch
- numpy
- scipy
- matplotlib

## Notes

- All decompositions are non-negative
- Only 'MU' and 'HALS' are valid optimizers for non-negative CP in this codebase
- Results are automatically synchronized to Google Drive when running on Colab
- Memory usage is optimized for both CPU and GPU operations
- Comprehensive error handling and logging implemented
- All scripts and notebooks use a central config.py and environment variable (PROJECT_ROOT) for robust, portable path management.
