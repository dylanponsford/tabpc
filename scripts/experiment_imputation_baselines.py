"""
Imputation Baselines - Runner Script

Runs baseline imputation methods using hyperimpute and saves imputed datasets.
For evaluation of imputed datasets (from any source), use evaluate_imputation.py.

Usage:
    # Run imputation methods and save datasets
    python experiment_imputation_baselines.py \
        --dataset-dir /path/to/dataset \
        --mask-dir /path/to/masks \
        --mask-idx 0 \
        --imputed-dir /path/to/save/imputed \
        --methods mean median mice \
        --seed 42
    
    # Evaluate existing imputed datasets
    python evaluate_imputation.py \
        --dataset-dir /path/to/dataset \
        --mask-dir /path/to/masks \
        --mask-idx 0 \
        --imputed-dir /path/to/imputed/datasets \
        --output-dir /path/to/metrics \
        --seed 42
"""

import os
import argparse
import gc
import json
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

# Try to import hyperimpute - will be available on cluster
try:
    from hyperimpute.plugins.imputers import Imputers
except ImportError:
    print("Warning: hyperimpute not available. Install with: pip install hyperimpute")
    Imputers = None


# Constants
DEFAULT_METHODS = ['mean', 'median', 'mice', 'missforest']
DEEP_LEARNING_METHODS = ['gain', 'miwae', 'vaec']
DEFAULT_SEED = 42
DEFAULT_DEVICE = 'cpu'


def free_memory():
    """Free GPU memory."""
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


def load_data_and_masks(
    dataset_dir: str,
    mask_dir: str,
    mask_idx: int,
    use_train_set: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str]:
    """Load dataset and masks."""
    split_name = "train" if use_train_set else "test"
    mask_filename = f"{split_name}_mask_{mask_idx}.npy"
    mask_path = os.path.join(mask_dir, mask_filename)
    dataset_path = os.path.join(dataset_dir, f"{split_name}.csv")
    
    df_complete = pd.read_csv(dataset_path)
    mask = np.load(mask_path)
    df_masked = df_complete.mask(mask)
    
    return df_masked, df_complete, mask, split_name


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


class ImputationRunner:
    """Runs imputation methods using hyperimpute."""
    def __init__(self, seed: int = DEFAULT_SEED, device: str = DEFAULT_DEVICE):
        self.seed = seed
        self.device = device
        
        if Imputers is None:
            raise ImportError("hyperimpute.plugins.imputers.Imputers not available")
    
    def run(self, df_masked: pd.DataFrame, method: str) -> pd.DataFrame:
        """Run a single imputation method."""        
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
    imputed_dir: str,
    seed: int = DEFAULT_SEED,
    device: str = DEFAULT_DEVICE,
    overwrite: bool = False,
    file_stem: str = ".csv"
) -> Dict[str, str]:
    """
    Run imputation methods and save imputed datasets to disk.
    
    Args:
        df_masked: DataFrame with missing values
        methods: List of imputation methods to run
        imputed_dir: Directory to save imputed datasets
        seed: Random seed
        device: Device for computation
        overwrite: Whether to overwrite existing files
    
    Returns:
        Dict mapping method name to saved file path
    """
    os.makedirs(imputed_dir, exist_ok=True)
    runner = ImputationRunner(seed=seed, device=device)
    saved_files = {}
    
    for method in methods:
        output_filename = f"imputed_{method}_{file_stem}"
        output_path = os.path.join(imputed_dir, output_filename)
        
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
    
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Folder containing train.csv and test.csv')
    parser.add_argument('--mask-dir', type=str, required=True,
                        help='Folder containing train_mask_*.npy and test_mask_*.npy files')
    parser.add_argument('--mask-idx', type=int, required=True,
                        help='Index of the missingness mask to use (0-9)')
    parser.add_argument('--imputed-dir', type=str, required=True,
                        help='Directory to save imputed datasets')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing imputed datasets')
    parser.add_argument('--methods', type=str, nargs='+', default=DEFAULT_METHODS,
                        help=f'Imputation methods to run (default: {" ".join(DEFAULT_METHODS)})')
    parser.add_argument('--include-deep', action='store_true',
                        help=f'Include deep learning methods ({", ".join(DEEP_LEARNING_METHODS)})')
    parser.add_argument('--use-train-set-for-conditioning', action='store_true',
                        help='Use train set instead of test set')
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
    print(f"Dataset: {args.dataset_dir}")
    print(f"Masks: {args.mask_dir}, mask index: {args.mask_idx}")
    print(f"Imputed datasets dir: {args.imputed_dir}")
    print(f"Seed: {args.seed}, Device: {args.device}")
    
    # Load data
    print("\nLoading data and masks...")
    df_masked, df_complete, mask, split_name = load_data_and_masks(
        dataset_dir=args.dataset_dir,
        mask_dir=args.mask_dir,
        mask_idx=args.mask_idx,
        use_train_set=args.use_train_set_for_conditioning
    )
    
    print(f"Loaded {split_name} split: {df_masked.shape[0]} samples, {df_masked.shape[1]} features")
    print(f"Missingness rate: {mask.sum() / mask.size:.2%}")
    
    # Run and save imputations
    print("\n" + "="*50)
    print("Running imputation methods...")
    print("="*50)

    dataset = os.path.normpath(args.dataset_dir).split(os.sep)[-1]
    mask_type = os.path.normpath(args.mask_dir).split(os.sep)[-1]
    missingness_rate = os.path.normpath(args.mask_dir).split(os.sep)[-2].strip("rate")

    file_stem = f"{dataset}_{mask_type}_{missingness_rate}_{args.mask_idx}.csv"
    
    saved_files = run_and_save_imputations(
        df_masked=df_masked,
        methods=methods,
        imputed_dir=args.imputed_dir,
        seed=args.seed,
        device=args.device,
        overwrite=args.overwrite,
        file_stem=file_stem,
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
