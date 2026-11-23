"""
Imputation Baselines - Runner Script

Runs baseline imputation methods using hyperimpute and saves imputed datasets.
For evaluation of imputed datasets (from any source), use evaluate_imputation.py.

Directory structure for saved imputed datasets:
    imputed_root/
    └── {dataset}/
        └── {split}/
            └── {missingness_type}/
                └── {rate}/
                    └── {mask_idx}/
                        └── {method}/
                            └── imputed_{method}_{dataset}_{type}_{rate}_{mask}.csv

Example:
    artifacts/imputed/adult/test/MCAR/rate30/mask_0/missforest/imputed_missforest_adult_MCAR_30_0.csv

Usage:
    # Run imputation methods and save datasets in structured directory
    python experiment_imputation_baselines.py \
        --data-root data \
        --dataset adult \
        --mask-type MCAR \
        --rate 30 \
        --mask-idx 0 \
        --imputed-dir artifacts/imputed \
        --methods mean median mice \
        --seed 42
    
    # Evaluate existing imputed datasets
    python evaluate_imputation.py \
        --imputed-dir artifacts/imputed \
        --data-root data \
        --output-dir results \
        --aggregate \
        --seed 42
"""

import os
import re
import argparse
import gc
import json
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Try to import hyperimpute - will be available on cluster
try:
    from hyperimpute.plugins.imputers import Imputers
except ImportError:
    print("Warning: hyperimpute not available. Install with: pip install hyperimpute")
    Imputers = None


# Constants
DEFAULT_METHODS = ['mean_mode', 'median_mode', 'mice', 'missforest']
DEEP_LEARNING_METHODS = ['gain', 'miwae', 'vaec']
DEFAULT_SEED = 42
DEFAULT_DEVICE = 'cpu'

# Pattern to extract missingness details from mask path
# Expects: .../{dataset}/masks/rate{N}/{TYPE}/{split}_mask_{idx}.npy
# Example: data/adult/masks/rate30/MCAR/test_mask_0.npy
MASK_PATTERN = re.compile(
    r'(?P<dataset>[^/]+)/'
    r'masks/'
    r'rate(?P<rate>\d+)/'
    r'(?P<type>[^/]+)/'
    r'(?P<split>train|test)_mask_(?P<idx>\d+)\.npy'
)


def free_memory():
    """Free GPU memory."""
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def extract_missingness_details(mask_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract missingness details and dataset name from mask file path.
    
    Args:
        mask_path: Path to mask file
    
    Returns:
        Dictionary with keys: 'dataset', 'rate', 'type', 'idx', 'split'
        Returns None if pattern doesn't match
    """
    mask_path = os.path.normpath(mask_path)
    match = MASK_PATTERN.search(mask_path)
    
    if match:
        return {
            'dataset': match.group('dataset'),
            'rate': int(match.group('rate')),
            'type': match.group('type'),
            'idx': int(match.group('idx')),
            'split': match.group('split')
        }
    
    return None


def load_data_and_masks(
    data_root: str,
    dataset: str,
    mask_type: str,
    rate: int,
    mask_idx: int,
    split: str
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    """
    Load dataset and masks from structured directories.
    
    Args:
        data_root: Root data directory (e.g., data/)
        dataset: Dataset name (e.g., adult)
        mask_type: Missingness type (e.g., MCAR, MAR)
        rate: Missingness rate (e.g., 30)
        mask_idx: Mask index (e.g., 0)
        split: 'train' or 'test'
    
    Returns:
        df_masked, df_complete, mask, split_name
    """
    dataset_dir = os.path.join(data_root, dataset)
    mask_dir = os.path.join(data_root, dataset, "masks", f"rate{rate}", mask_type)
    
    dataset_path = os.path.join(dataset_dir, f"{split}.csv")
    mask_filename = f"{split}_mask_{mask_idx}.npy"
    mask_path = os.path.join(mask_dir, mask_filename)
    
    df_complete = pd.read_csv(dataset_path)
    mask = np.load(mask_path)
    df_masked = df_complete.mask(mask)
    
    return df_masked, df_complete, mask, split


def get_column_types(df: pd.DataFrame, info_path: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns."""
    numeric_cols = []
    categorical_cols = []
    
    if info_path and os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        if "metadata" in info and "columns" in info["metadata"]:
            for col_name, col_info in info["metadata"]["columns"].items():
                col_type = col_info.get("sdtype", None)
                if col_type in ["numerical", "integer"]:
                    numeric_cols.append(col_name)
                elif col_type in ["categorical", "boolean"]:
                    categorical_cols.append(col_name)
            
            if numeric_cols or categorical_cols:
                return numeric_cols, categorical_cols
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return numeric_cols, categorical_cols


def simple_statistical_imputation(
    df_masked: pd.DataFrame,
    numeric_strategy: str = 'mean',
    categorical_strategy: str = 'most_frequent'
) -> pd.DataFrame:
    """
    Simple statistical imputation using scikit-learn's SimpleImputer.
    
    Applies different strategies for numeric and categorical columns:
    - Numeric: mean or median
    - Categorical: most_frequent (mode)
    
    Args:
        df_masked: DataFrame with missing values
        numeric_strategy: 'mean' or 'median' for numeric columns
        categorical_strategy: 'most_frequent' for categorical columns
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df_masked.copy()
    
    # Get column types
    numeric_cols, categorical_cols = get_column_types(df_masked)
    
    # Impute numeric columns
    if numeric_cols:
        numeric_imputer = SimpleImputer(strategy=numeric_strategy)
        df_imputed[numeric_cols] = numeric_imputer.fit_transform(df_masked[numeric_cols])
    
    # Impute categorical columns
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy=categorical_strategy)
        df_imputed[categorical_cols] = categorical_imputer.fit_transform(df_masked[categorical_cols])
    
    return df_imputed


class ImputationRunner:
    """Runs imputation methods using hyperimpute."""

    def __init__(self, seed: int = DEFAULT_SEED, device: str = DEFAULT_DEVICE):
        self.seed = seed
        self.device = device
        
        if Imputers is None:
            raise ImportError("hyperimpute.plugins.imputers.Imputers not available")
    
    def run(self, df_masked: pd.DataFrame, method: str) -> pd.DataFrame:
        """Run a single imputation method."""
        # Check for simple statistical methods
        if method == 'mean_mode':
            return simple_statistical_imputation(
                df_masked,
                numeric_strategy='mean',
                categorical_strategy='most_frequent'
            )
        elif method == 'median_mode':
            return simple_statistical_imputation(
                df_masked,
                numeric_strategy='median',
                categorical_strategy='most_frequent'
            )
        
        # Use hyperimpute for other methods
        imputer = Imputers().get(name=method.lower(), random_state=self.seed)
        
        if self.device == "cuda" and hasattr(imputer, 'device'):
            imputer.device = self.device
        
        df_imputed = imputer.fit_transform(df_masked)
        
        if isinstance(df_imputed, np.ndarray):
            df_imputed = pd.DataFrame(df_imputed, columns=df_masked.columns)
        
        return df_imputed


def run_and_save_imputations(
    df_masked: pd.DataFrame,
    methods: List[str],
    imputed_root: str,
    dataset: str,
    split: str,
    mask_type: str,
    rate: int,
    mask_idx: int,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    overwrite: bool = False
) -> Dict[str, str]:
    """
    Run imputation methods and save imputed datasets in structured directory.
    
    Saves to: {imputed_root}/{dataset}/{split}/{mask_type}/rate{rate}/mask_{mask_idx}/{method}/imputed_{method}_{dataset}_{type}_{rate}_{mask}.csv
    
    Args:
        df_masked: DataFrame with missing values
        methods: List of imputation methods to run
        imputed_root: Root directory for imputed datasets
        dataset: Dataset name
        split: 'train' or 'test'
        mask_type: Missingness type (MCAR, MAR, MNAR)
        rate: Missingness rate (e.g., 30)
        mask_idx: Mask index
        seed: Random seed
        device: Device for computation
        overwrite: Whether to overwrite existing files
    
    Returns:
        Dict mapping method name to saved file path
    """
    runner = ImputationRunner(seed=seed, device=device)
    saved_files = {}
    
    for method in methods:
        # Create structured directory path
        method_dir = os.path.join(
            imputed_root,
            dataset,
            split,
            mask_type,
            f"rate{rate}",
            f"mask_{mask_idx}",
            method
        )
        os.makedirs(method_dir, exist_ok=True)
        
        # Create filename
        output_filename = f"imputed_{method}_{dataset}_{mask_type}_{rate}_{mask_idx}.csv"
        output_path = os.path.join(method_dir, output_filename)
        
        if os.path.exists(output_path) and not overwrite:
            print(f"Skipping {method}: {output_path} already exists")
            saved_files[method] = output_path
            continue
        
        print(f"\nRunning imputation method: {method}")
        print(f"{'='*50}")
        
        try:
            df_imputed = runner.run(df_masked, method)
            df_imputed.to_csv(output_path, index=False)
            saved_files[method] = output_path
            print(f"Saved imputed dataset to: {output_path}")
            
            del df_imputed
            free_memory()
        
        except Exception as e:
            print(f"Failed to run {method}: {e}")
    
    return saved_files


def parse_runner_args() -> argparse.Namespace:
    """Parse command line arguments for imputation runner."""
    parser = argparse.ArgumentParser(
        description='Run imputation baselines using hyperimpute and save imputed datasets.'
    )
    
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root directory containing datasets with masks (e.g., data/)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (subdirectory in data-root)')
    parser.add_argument('--mask-type', type=str, required=True,
                        help='Missingness type (e.g., MCAR, MAR, MNAR)')
    parser.add_argument('--rate', type=int, required=True,
                        help='Missingness rate (e.g., 30 for 30%)')
    parser.add_argument('--mask-idx', type=int, required=True,
                        help='Index of the missingness mask to use (0-9)')
    parser.add_argument('--imputed-dir', type=str, required=True,
                        help='Root directory to save imputed datasets')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing imputed datasets')
    parser.add_argument('--methods', type=str, nargs='+', default=DEFAULT_METHODS,
                        help=f'Imputation methods to run (default: {" ".join(DEFAULT_METHODS)})')
    parser.add_argument('--include-deep', action='store_true',
                        help=f'Include deep learning methods ({", ".join(DEEP_LEARNING_METHODS)})')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Data split to use (default: test)')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=['cpu', 'cuda'],
                        help=f'Device to use (default: {DEFAULT_DEVICE})')
    
    return parser.parse_args()


def main():
    args = parse_runner_args()
    
    # Build method list
    methods = args.methods.copy()
    if args.include_deep:
        methods.extend([m for m in DEEP_LEARNING_METHODS if m not in methods])
    
    # Print configuration
    print(f"Running imputation baselines with methods: {methods}")
    print(f"Data root: {args.data_root}")
    print(f"Dataset: {args.dataset}")
    print(f"Missingness: {args.mask_type}, Rate: {args.rate}%, Mask idx: {args.mask_idx}")
    print(f"Split: {args.split}")
    print(f"Imputed datasets root: {args.imputed_dir}")
    print(f"Seed: {args.seed}, Device: {args.device}")
    
    # Load data
    print("\nLoading data and masks...")
    try:
        df_masked, df_complete, mask, split_name = load_data_and_masks(
            data_root=args.data_root,
            dataset=args.dataset,
            mask_type=args.mask_type,
            rate=args.rate,
            mask_idx=args.mask_idx,
            split=args.split
        )
        print(f"Loaded {split_name} split: {df_masked.shape[0]} samples, {df_masked.shape[1]} features")
        print(f"Missingness rate: {mask.sum() / mask.size:.2%}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Expected data at: {args.data_root}/{args.dataset}/")
        print(f"Expected masks at: {args.data_root}/{args.dataset}/masks/rate{args.rate}/{args.mask_type}/")
        return
    
    # Get info.json path if available
    info_path = os.path.join(args.data_root, args.dataset, "info.json")
    if not os.path.exists(info_path):
        info_path = None
    
    # Run and save imputations
    print("\n" + "="*50)
    print("Running imputation methods...")
    print("="*50)
    
    saved_files = run_and_save_imputations(
        df_masked=df_masked,
        methods=methods,
        imputed_root=args.imputed_dir,
        dataset=args.dataset,
        split=args.split,
        mask_type=args.mask_type,
        rate=args.rate,
        mask_idx=args.mask_idx,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite
    )
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for method, path in saved_files.items():
        print(f"  {method}: {path}")
    
    print(f"\nDone! Saved {len(saved_files)} imputed datasets.")
    print("Use evaluate_imputation.py to compute metrics on these datasets.")


if __name__ == "__main__":
    main()
