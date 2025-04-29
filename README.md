# PixelHop and PixelHop++ Implementation

Implementation of PixelHop and PixelHop++ models for image classification on MNIST and Fashion-MNIST datasets.

## Overview

This codebase implements and compares two models:

1. **PixelHop**: Uses standard Saab transform with single energy threshold (TH2)
2. **PixelHop++**: Uses channel-wise Saab transform with two energy thresholds (TH1, TH2)

Both models follow the same architecture with three units, each consisting of neighborhood construction, transform, and optional max-pooling, followed by an XGBoost classifier.

## Requirements

```bash
pip install -r requirements.txt
```

## Codebase Structure

```
pixelhop/
  models/
    saab_transform.py         # Regular Saab transform
    cw_saab_transform.py      # Channel-wise Saab transform
    pixelhop_unit.py          # Shared PixelHop unit implementation
    pixelhop.py               # PixelHop model
    pixelhop_pp.py            # PixelHop++ model
  utils/
    data_utils.py             # Data loading utilities
main.py                       # PixelHop++ experiments
comparison.py                 # Comparison between PixelHop and PixelHop++
```

## How to Run

### Part A: PixelHop++ Experiments

Run PixelHop++ on MNIST with default parameters:
```bash
python main.py --dataset mnist
```

Explore different TH1 values on MNIST:
```bash
python main.py --dataset mnist --explore_th1
```

Run on Fashion-MNIST:
```bash
python main.py --dataset fashion_mnist
```

Options:
- `--dataset`: Choose dataset ('mnist', 'fashion_mnist', 'both')
- `--samples`: Number of training samples (default: 10000)
- `--th1`: Energy threshold for intermediate nodes (default: 0.005)
- `--th2`: Energy threshold for discarded nodes (default: 0.001)
- `--explore_th1`: Explore different TH1 values

### Part B: Comparison between PixelHop and PixelHop++

Compare both models on MNIST:
```bash
python comparison.py --dataset mnist
```

Compare on Fashion-MNIST:
```bash
python comparison.py --dataset fashion_mnist
```

Compare on both datasets:
```bash
python comparison.py --dataset both
```

Options:
- `--dataset`: Choose dataset ('mnist', 'fashion_mnist', 'both') 
- `--samples`: Number of training samples (default: 10000)
- `--th2`: Energy threshold for PixelHop / discarded nodes in PixelHop++ (default: 0.001)

## Results

Results are saved in timestamped directories:
- Experiment results: `experiment_YYYYMMDD_HHMMSS/`
- Comparison results: `comparison_mnist_YYYYMMDD_HHMMSS/` or `comparison_fashion_mnist_YYYYMMDD_HHMMSS/`

Each directory contains:
- Summary JSON files with accuracy, training time, and model size
- Visualization charts comparing performance metrics
- Confusion matrices as images

## Features

- **Memory-efficient Implementation** - Optimized for handling large datasets through batch processing
- **Multi-dataset Support** - Works with MNIST, Fashion-MNIST, and combined datasets
- **Performance Analysis** - Comprehensive metrics and visualizations for model evaluation
- **Threshold Exploration** - Tools to analyze the impact of energy thresholds on model performance
- **Data Augmentation** - Optional image augmentation to improve model generalization
- **Filter Visualization** - Tools to visualize the learned filters at each layer

## Architecture

The implementation follows the PixelHop++ architecture consisting of:

1. **Three PixelHop++ Units** - Each combining neighborhood construction and c/w Saab transforms
2. **Max-pooling Layers** - Implemented with optimized vectorized operations
3. **XGBoost Classifier** - Applied to the final hop features

## Implementation Details

- **Saab Transform** - Optimized with SVD for numerical stability
- **Neighborhood Construction** - Efficient patch extraction with stride support
- **Max-pooling** - Vectorized implementation with reshape operations
- **Batch Processing** - Memory-efficient processing of large datasets
- **Data Utilities** - Support for balanced sampling and data augmentation

## References

- Yueru Chen et al., "PixelHop++: A Small Successive-Subspace-Learning-Based (SSL-based) Model for Image Classification," https://arxiv.org/abs/2002.03141, 2020. 