"""
Imputation Evaluation Script

Evaluates imputed datasets and computes metrics (RMSE, MAE, Accuracy, NRMSE).
Supports two modes:
1. Evaluate datasets from a structured directory tree
2. Evaluate specific files or directories

Directory structure expected:
    imputed_root/
    └── {dataset}/
        └── {split}/
            └── {missingness_type}/
                └── {rate}/
                    └── {mask_idx}/
                        └── {method}/
                            └── imputed_{method}_{dataset}_{type}_{rate}_{mask}.csv

Example:
    imputed/adult/test/MAR/rate30/mask_0/missforest/imputed_missforest_adult_MAR_30_0.csv

Dataset and mask structure:
    data/
    └── {dataset}/
        ├── train.csv
        ├── test.csv
        ├── info.json
        └── masks/
            └── {rate}/
                └── {missingness_type}/
                    ├── train_mask_{idx}.npy
                    └── test_mask_{idx}.npy

Example:
    data/adult/masks/rate30/MAR/test_mask_0.npy

Results are saved as:
    output_root/
    └── {dataset}/
        └── {split}_{missingness_type}_rate{rate}/
            └── metrics_mask{idx}_seed{seed}.csv

Usage:
    # Evaluate all imputed datasets in structured directory
    python evaluate_imputation.py \
        --imputed-dir /path/to/imputed \
        --data-root /path/to/data \
        --output-dir /path/to/results \
        --seed 42
    
    # Evaluate and aggregate all results
    python evaluate_imputation.py \
        --imputed-dir /path/to/imputed \
        --data-root /path/to/data \
        --output-dir /path/to/results \
        --aggregate \
        --seed 42
    
    # Evaluate a single imputed dataset file
    python evaluate_imputation.py \
        --data-dir /path/to/data/adult \
        --mask-dir /path/to/data/adult/masks/rate30/MAR \
        --mask-idx 0 \
        --imputed-file /path/to/imputed.csv \
        --output-dir /path/to/results
"""

import os
import re
import argparse
import gc
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


# Constants
DEFAULT_SEED = 42

# Pattern to extract metadata from directory structure
# Expects: {dataset}/{split}/{missingness_type}/{rate}/{mask_idx}/{method}/filename.csv
DIR_PATTERN = re.compile(
    r'(?P<dataset>[^/]+)/'
    r'(?P<split>train|test)/'
    r'(?P<missingness_type>[^/]+)/'
    r'(?P<rate>rate\d+)/'
    r'(?P<mask_idx>mask_\d+)/'
    r'(?P<method>[^/]+)/'
)


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


class ImputationValidator:
    """Validates imputation results for correctness."""
    
    def __init__(self, df_masked: pd.DataFrame, mask: np.ndarray):
        self.df_masked = df_masked
        self.mask = mask
        self.mask_df = pd.DataFrame(mask, columns=df_masked.columns)
    
    def check_validity(self, df_imputed: pd.DataFrame) -> Dict[str, Any]:
        """Check imputation validity."""
        masked_positions_still_nan = 0
        for col in self.df_masked.columns:
            col_mask = self.mask_df[col].values
            if col_mask.sum() > 0:
                masked_positions_still_nan += df_imputed.loc[self.mask_df[col], col].isna().sum()
        
        non_mask_changed = 0
        for col in self.df_masked.columns:
            non_mask_idx = ~self.mask_df[col].values
            if non_mask_idx.sum() > 0:
                original_vals = self.df_masked.loc[~self.mask_df[col], col].values
                imputed_vals = df_imputed.loc[~self.mask_df[col], col].values
                
                if pd.api.types.is_numeric_dtype(df_imputed[col]):
                    diff = np.abs(original_vals - imputed_vals) > 1e-6
                    non_mask_changed += np.sum(diff)
                else:
                    non_mask_changed += np.sum(original_vals != imputed_vals)
        
        return {
            "masked_positions_still_nan": int(masked_positions_still_nan),
            "non_masked_positions_changed": int(non_mask_changed),
            "imputation_valid": masked_positions_still_nan == 0 and non_mask_changed == 0
        }


class FeatureNormalizer:
    """
    Normalizes numeric features for comparable error metrics.
    
    Uses standard deviation normalization (NRMSE) computed on complete data.
    """
    
    def __init__(self, df_complete: pd.DataFrame, numeric_cols: List[str]):
        self.numeric_cols = numeric_cols
        self.params = {}
        
        # Fit normalization parameters on complete data
        for col in numeric_cols:
            std = df_complete[col].std()
            mean = df_complete[col].mean()
            
            # Handle zero std (constant columns)
            if std == 0 or pd.isna(std):
                std = 1.0
            
            self.params[col] = {
                'mean': mean,
                'std': std
            }
    
    def normalize(self, values: np.ndarray, col: str) -> np.ndarray:
        """Normalize values for a column."""
        if col not in self.params:
            return values
        
        params = self.params[col]
        return (values - params['mean']) / params['std']
    
    def get_params(self) -> Dict[str, Dict[str, float]]:
        """Get normalization parameters."""
        return self.params


class ImputationMetrics:
    """Computes imputation evaluation metrics including per-feature breakdown."""
    
    def __init__(self, df_complete: pd.DataFrame, mask: np.ndarray, info_path: Optional[str] = None):
        self.df_complete = df_complete
        self.mask = mask
        self.mask_df = pd.DataFrame(mask, columns=df_complete.columns)
        self.numeric_cols, self.categorical_cols = get_column_types(df_complete, info_path)
        self.validator = None
        
        # Initialize normalizer for NRMSE
        self.normalizer = FeatureNormalizer(df_complete, self.numeric_cols)
    
    def set_validator(self, validator: ImputationValidator):
        self.validator = validator
    
    def compute(self, df_imputed: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute imputation error metrics including per-feature breakdown.
        
        Returns:
            Dictionary with:
            - Aggregated metrics (rmse_mean, nrmse_mean, accuracy_mean, etc.)
            - Per-feature metrics (per_feature_rmse, per_feature_nrmse, per_feature_accuracy)
            - Validity metrics
        """
        metrics = {}
        per_feature_metrics = {}
        
        if self.validator:
            validity = self.validator.check_validity(df_imputed)
            metrics.update(validity)
        
        # Numeric metrics (RMSE, MAE, and NRMSE) with per-feature breakdown
        numeric_errors_rmse = []
        numeric_errors_mae = []
        numeric_errors_nrmse = []
        
        for col in self.numeric_cols:
            masked_idx = self.mask_df[col].values
            if masked_idx.sum() == 0:
                continue
            
            imputed_vals = df_imputed.loc[self.mask_df[col], col].values
            true_vals = self.df_complete.loc[self.mask_df[col], col].values
            
            # Handle potential NaN in imputed values
            valid_mask = ~(np.isnan(imputed_vals) | np.isnan(true_vals))
            if valid_mask.sum() == 0:
                continue
            
            imputed_valid = imputed_vals[valid_mask]
            true_valid = true_vals[valid_mask]
            
            # Raw RMSE and MAE
            rmse = np.sqrt(mean_squared_error(true_valid, imputed_valid))
            mae = mean_absolute_error(true_valid, imputed_valid)
            numeric_errors_rmse.append(rmse)
            numeric_errors_mae.append(mae)
            
            # Per-feature metrics
            per_feature_metrics[f"{col}_rmse"] = float(rmse)
            per_feature_metrics[f"{col}_mae"] = float(mae)
            per_feature_metrics[f"{col}_n_missing"] = int(masked_idx.sum())
            
            # NRMSE (normalized by feature std)
            imputed_normalized = self.normalizer.normalize(imputed_valid, col)
            true_normalized = self.normalizer.normalize(true_valid, col)
            nrmse = np.sqrt(mean_squared_error(true_normalized, imputed_normalized))
            numeric_errors_nrmse.append(nrmse)
            per_feature_metrics[f"{col}_nrmse"] = float(nrmse)
        
        if numeric_errors_rmse:
            metrics["rmse_mean"] = float(np.mean(numeric_errors_rmse))
            metrics["rmse_std"] = float(np.std(numeric_errors_rmse))
            metrics["mae_mean"] = float(np.mean(numeric_errors_mae))
            metrics["mae_std"] = float(np.std(numeric_errors_mae))
        
        if numeric_errors_nrmse:
            metrics["nrmse_mean"] = float(np.mean(numeric_errors_nrmse))
            metrics["nrmse_std"] = float(np.std(numeric_errors_nrmse))
        
        # Categorical metrics (Accuracy) with per-feature breakdown
        categorical_accuracies = []
        
        for col in self.categorical_cols:
            masked_idx = self.mask_df[col].values
            if masked_idx.sum() == 0:
                continue
            
            imputed_vals = df_imputed.loc[self.mask_df[col], col].values
            true_vals = self.df_complete.loc[self.mask_df[col], col].values
            
            # Handle potential NaN
            valid_mask = ~(pd.isna(imputed_vals) | pd.isna(true_vals))
            if valid_mask.sum() > 0:
                acc = accuracy_score(true_vals[valid_mask], imputed_vals[valid_mask])
                categorical_accuracies.append(acc)
                per_feature_metrics[f"{col}_accuracy"] = float(acc)
                per_feature_metrics[f"{col}_n_missing"] = int(masked_idx.sum())
        
        if categorical_accuracies:
            metrics["accuracy_mean"] = float(np.mean(categorical_accuracies))
            metrics["accuracy_std"] = float(np.std(categorical_accuracies))
        
        # Add per-feature metrics to main metrics dict
        metrics.update(per_feature_metrics)
        
        metrics["missingness_rate"] = float(self.mask.sum() / self.mask.size)
        metrics["n_numeric_features"] = len(numeric_errors_rmse)
        metrics["n_categorical_features"] = len(categorical_accuracies)
        
        return metrics


def extract_metadata_from_path(file_path: str, imputed_root: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from file path based on directory structure.
    
    Expected structure:
        {imputed_root}/{dataset}/{split}/{missingness_type}/{rate}/{mask_idx}/{method}/filename.csv
    """
    # Get relative path from imputed root
    rel_path = os.path.relpath(file_path, imputed_root)
    
    # Try to match pattern
    match = DIR_PATTERN.search(rel_path)
    if match:
        meta = match.groupdict()
        # Extract numeric mask index from "mask_0" format
        meta['mask_idx'] = int(meta['mask_idx'].replace('mask_', ''))
        # Extract rate value from "rate30" format
        meta['rate'] = meta['rate'].replace('rate', '')
        meta['file_path'] = file_path
        return meta
    
    return None


def find_imputed_files(imputed_root: str) -> List[Dict[str, Any]]:
    """
    Find all imputed dataset files in the structured directory.
    
    Returns list of dicts with file path and extracted metadata.
    """
    results = []
    imputed_path = Path(imputed_root)
    
    # Look for CSV files in the expected structure
    for csv_file in imputed_path.rglob('*.csv'):
        # Skip metrics files
        if 'metrics' in csv_file.name.lower():
            continue
        
        file_path = str(csv_file)
        metadata = extract_metadata_from_path(file_path, imputed_root)
        
        if metadata:
            results.append(metadata)
        else:
            print(f"Warning: Could not parse metadata for {file_path}")
            print(f"  Expected pattern: {{dataset}}/{{split}}/{{missingness_type}}/rate{{N}}/mask_{{N}}/{{method}}/{{filename}}.csv")
    
    return results


def evaluate_imputed_file(
    imputed_path: str,
    df_masked: pd.DataFrame,
    df_complete: pd.DataFrame,
    mask: np.ndarray,
    info_path: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a single imputed dataset file."""
    df_imputed = pd.read_csv(imputed_path)
    
    validator = ImputationValidator(df_masked, mask)
    metrics_computer = ImputationMetrics(df_complete, mask, info_path)
    metrics_computer.set_validator(validator)
    
    metrics = metrics_computer.compute(df_imputed)
    
    # Add metadata
    if seed is not None:
        metrics["seed"] = seed
    
    return metrics


def get_output_subdir(split: str, missingness_type: str, rate: str) -> str:
    """Generate output subdirectory name."""
    return f"{split}_{missingness_type}_rate{rate}"


def evaluate_structured_directory(
    imputed_root: str,
    data_root: str,
    output_root: str,
    seed: int = DEFAULT_SEED,
    overwrite: bool = False,
    aggregate: bool = False
) -> Optional[pd.DataFrame]:
    """
    Evaluate all imputed datasets in a structured directory.
    
    Args:
        imputed_root: Root directory containing imputed datasets
        data_root: Root directory containing original datasets with masks
        output_root: Root directory for saving metrics
        seed: Random seed
        overwrite: Whether to overwrite existing results
        aggregate: Whether to also create aggregated results file
    
    Returns:
        DataFrame with all results if aggregate=True, else None
    """
    # Find all imputed files
    print(f"Scanning for imputed datasets in: {imputed_root}")
    imputed_files = find_imputed_files(imputed_root)
    
    if not imputed_files:
        print(f"No imputed datasets found in {imputed_root}")
        print("Expected directory structure:")
        print("  {dataset}/{split}/{missingness_type}/rate{N}/mask_{N}/{method}/{filename}.csv")
        print("Example:")
        print("  adult/test/MAR/rate30/mask_0/missforest/imputed_missforest_adult_MAR_30_0.csv")
        return None
    
    print(f"Found {len(imputed_files)} imputed datasets")
    
    # Group files by (dataset, split, missingness_type, rate, mask_idx) for efficient processing
    grouped = defaultdict(list)
    
    for file_info in imputed_files:
        key = (
            file_info['dataset'],
            file_info['split'],
            file_info['missingness_type'],
            file_info['rate'],
            file_info['mask_idx']
        )
        grouped[key].append(file_info)
    
    print(f"Grouped into {len(grouped)} experiment configurations")
    
    # Process each group
    all_results = []
    dataset_results = defaultdict(list)  # Collect results per dataset for overview CSV
    
    for (dataset, split, missingness_type, rate, mask_idx), files in grouped.items():
        print(f"\n{'='*60}")
        print(f"Processing: {dataset} / {split} / {missingness_type} / rate{rate} / mask{mask_idx}")
        print(f"{'='*60}")
        
        # Load data and masks for this configuration
        # Dataset directory: {data_root}/{dataset}/
        dataset_dir = os.path.join(data_root, dataset)
        
        # Mask directory: {data_root}/{dataset}/masks/rate{rate}/{missingness_type}/
        mask_dir = os.path.join(data_root, dataset, "masks", f"rate{rate}", missingness_type)
        
        print(f"Dataset dir: {dataset_dir}")
        print(f"Mask dir: {mask_dir}")
        
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        if not os.path.exists(mask_dir):
            print(f"Warning: Mask directory not found: {mask_dir}")
            print(f"  Expected: {data_root}/{dataset}/masks/rate{rate}/{missingness_type}/")
            continue
        
        try:
            df_masked, df_complete, mask, split_name = load_data_and_masks(
                dataset_dir=dataset_dir,
                mask_dir=mask_dir,
                mask_idx=mask_idx,
                use_train_set=(split == 'train')
            )
            print(f"Loaded {split} split: {df_masked.shape[0]} samples, {df_masked.shape[1]} features")
            print(f"Missingness rate: {mask.sum() / mask.size:.2%}")
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        
        # Get info.json path if available
        info_path = os.path.join(dataset_dir, "info.json")
        if not os.path.exists(info_path):
            info_path = None
        
        # Determine output directory
        output_subdir = get_output_subdir(split, missingness_type, rate)
        output_dir = os.path.join(output_root, dataset, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"metrics_mask{mask_idx}_seed{seed}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Check if results already exist
        if os.path.exists(output_path) and not overwrite:
            print(f"Results already exist at {output_path}, skipping...")
            # Still load for aggregation
            try:
                existing_df = pd.read_csv(output_path)
                all_results.append(existing_df)
            except:
                pass
            continue
        
        # Evaluate each method
        results = []
        
        for file_info in files:
            method = file_info['method']
            imputed_path = file_info['file_path']
            
            print(f"\n  Evaluating method: {method}")
            
            try:
                metrics = evaluate_imputed_file(
                    imputed_path=imputed_path,
                    df_masked=df_masked,
                    df_complete=df_complete,
                    mask=mask,
                    info_path=info_path,
                    seed=seed
                )
                
                # Add metadata
                metrics['method'] = method
                metrics['dataset'] = dataset
                metrics['split'] = split
                metrics['mask_idx'] = mask_idx
                metrics['missingness_type'] = missingness_type
                metrics['rate'] = int(rate)
                metrics['seed'] = seed
                
                results.append(metrics)
                
                # Format aggregated metrics for display
                rmse_str = f"{metrics.get('rmse_mean', 'N/A'):.4f}" if 'rmse_mean' in metrics else 'N/A'
                mae_str = f"{metrics.get('mae_mean', 'N/A'):.4f}" if 'mae_mean' in metrics else 'N/A'
                nrmse_str = f"{metrics.get('nrmse_mean', 'N/A'):.4f}" if 'nrmse_mean' in metrics else 'N/A'
                acc_str = f"{metrics.get('accuracy_mean', 'N/A'):.4f}" if 'accuracy_mean' in metrics else 'N/A'
                
                print(f"    Aggregated: RMSE={rmse_str}, MAE={mae_str}, NRMSE={nrmse_str}, Acc={acc_str}")
                print(f"    Valid={metrics.get('imputation_valid', False)}, Features: {metrics.get('n_numeric_features', 0)} num, {metrics.get('n_categorical_features', 0)} cat")
                
                if not metrics.get('imputation_valid', True):
                    print(f"    WARNING: Validity check failed!")
                
                free_memory()
            
            except Exception as e:
                print(f"    Failed: {e}")
                results.append({
                    'method': method,
                    'dataset': dataset,
                    'split': split,
                    'mask_idx': mask_idx,
                    'missingness_type': missingness_type,
                    'rate': int(rate),
                    'seed': seed,
                    'error': str(e)
                })
        
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"\n  Saved metrics to: {output_path}")
            print(f"  Columns: {len(results_df.columns)} (includes per-feature metrics)")
            all_results.append(results_df)
            
            # Collect for per-dataset overview
            dataset_results[dataset].append(results_df)
    
    # Save per-dataset overview CSVs (with per-feature metrics)
    if dataset_results:
        print(f"\n{'='*60}")
        print("SAVING PER-DATASET OVERVIEW CSVs")
        print(f"{'='*60}")
        
        for dataset_name, dataset_dfs in dataset_results.items():
            dataset_overview = pd.concat(dataset_dfs, ignore_index=True)
            
            # Sort
            sort_cols = ['split', 'missingness_type', 'rate', 'mask_idx', 'method', 'seed']
            available_sort_cols = [c for c in sort_cols if c in dataset_overview.columns]
            if available_sort_cols:
                dataset_overview = dataset_overview.sort_values(available_sort_cols)
            
            # Reorder columns: metadata first, then aggregated metrics, then per-feature
            metadata_cols = ['split', 'missingness_type', 'rate', 'mask_idx', 'method', 'seed']
            aggregated_cols = ['rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'nrmse_mean', 'nrmse_std', 
                             'accuracy_mean', 'accuracy_std', 'missingness_rate', 'n_numeric_features', 
                             'n_categorical_features', 'imputation_valid', 'validity_issues']
            
            meta_cols = [c for c in metadata_cols if c in dataset_overview.columns]
            agg_cols = [c for c in aggregated_cols if c in dataset_overview.columns]
            other_cols = [c for c in dataset_overview.columns if c not in meta_cols and c not in agg_cols]
            
            dataset_overview = dataset_overview[meta_cols + agg_cols + other_cols]
            
            # Save to output_root/dataset/overview.csv
            dataset_output_dir = os.path.join(output_root, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            overview_path = os.path.join(dataset_output_dir, "overview.csv")
            dataset_overview.to_csv(overview_path, index=False)
            print(f"Saved {dataset_name} overview to: {overview_path}")
            print(f"  Experiments: {len(dataset_overview)}, Columns: {len(dataset_overview.columns)} (includes per-feature metrics)")
    
    # Aggregate if requested
    if aggregate and all_results:
        # Create overall all_results.csv with only aggregated metrics (no per-feature columns)
        print(f"\n{'='*60}")
        print("AGGREGATING ALL RESULTS (aggregated metrics only)")
        print(f"{'='*60}")
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Identify per-feature columns (contain feature names followed by metric type)
        # We keep: nrmse_mean, nrmse_std, accuracy_mean, accuracy_std (cross-dataset comparable metrics)
        # We also keep: missingness_rate, n_numeric_features, n_categorical_features, metadata columns
        # We remove: RMSE/MAE (scale-dependent) and per-feature columns
        # Note: Full RMSE/MAE metrics available in per-dataset overview.csv files
        
        aggregated_metric_cols = [
            'nrmse_mean', 'nrmse_std',
            'accuracy_mean', 'accuracy_std',
            'missingness_rate', 'n_numeric_features', 'n_categorical_features',
            'imputation_valid', 'validity_issues'
        ]
        metadata_cols = ['dataset', 'split', 'missingness_type', 'rate', 'mask_idx', 'method', 'seed']
        
        # Keep only aggregated metrics and metadata columns, with metadata first
        meta_cols = [c for c in metadata_cols if c in combined.columns]
        agg_cols = [c for c in aggregated_metric_cols if c in combined.columns]
        combined_agg = combined[meta_cols + agg_cols]
        
        # Sort
        sort_cols = ['dataset', 'split', 'missingness_type', 'rate', 'mask_idx', 'method', 'seed']
        available_sort_cols = [c for c in sort_cols if c in combined_agg.columns]
        if available_sort_cols:
            combined_agg = combined_agg.sort_values(available_sort_cols)
        
        # Save aggregated results
        agg_output_path = os.path.join(output_root, "all_results.csv")
        combined_agg.to_csv(agg_output_path, index=False)
        print(f"Saved aggregated results to: {agg_output_path}")
        print(f"Total experiments: {len(combined_agg)}")
        print(f"Columns: {len(combined_agg.columns)} (aggregated metrics only, no per-feature columns)")
        
        # Print summary
        if 'method' in combined_agg.columns:
            print("\nMethods evaluated:")
            print(combined_agg['method'].value_counts().to_string())
        
        if 'dataset' in combined_agg.columns:
            print("\nDatasets:")
            print(combined_agg['dataset'].value_counts().to_string())
        
        if 'rate' in combined_agg.columns:
            print("\nMissingness rates:")
            print(sorted(combined_agg['rate'].unique()))
        
        return combined_agg
    
    return None


def parse_eval_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate imputed datasets and compute metrics.'
    )
    
    # Structured directory mode
    parser.add_argument('--imputed-dir', type=str, default=None,
                        help='Root directory containing structured imputed datasets')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Root directory containing original datasets with masks (e.g., data/)')
    
    # Single file/directory mode
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing train.csv and test.csv (for single dataset evaluation)')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Directory containing mask files')
    parser.add_argument('--mask-idx', type=int, default=None,
                        help='Index of the mask to use')
    parser.add_argument('--imputed-file', type=str, default=None,
                        help='Single imputed CSV file to evaluate')
    parser.add_argument('--imputed-files', type=str, nargs='+', default=None,
                        help='List of imputed CSV files to evaluate')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save metrics')
    parser.add_argument('--aggregate', action='store_true',
                        help='Also create aggregated all_results.csv')
    parser.add_argument('--overwrite', '-o', action='store_true',
                        help='Overwrite existing metrics')
    
    # Other
    parser.add_argument('--use-train-set-for-conditioning', action='store_true',
                        help='Use train set instead of test set')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')
    
    return parser.parse_args()


def main():
    args = parse_eval_args()
    
    # Check which mode to use
    if args.imputed_dir:
        # Structured directory mode
        if not args.data_root:
            print("Error: --imputed-dir requires --data-root")
            return
        
        print(f"Evaluating structured imputed datasets...")
        print(f"Imputed root: {args.imputed_dir}")
        print(f"Data root: {args.data_root}")
        print(f"Output root: {args.output_dir}")
        
        evaluate_structured_directory(
            imputed_root=args.imputed_dir,
            data_root=args.data_root,
            output_root=args.output_dir,
            seed=args.seed,
            overwrite=args.overwrite,
            aggregate=args.aggregate
        )
    
    elif args.imputed_file or args.imputed_files:
        # Single file mode
        imputed_files = [args.imputed_file] if args.imputed_file else args.imputed_files
        
        if not args.data_dir or not args.mask_dir or args.mask_idx is None:
            print("Error: --imputed-file/--imputed-files requires --data-dir, --mask-dir, and --mask-idx")
            return
        
        print(f"Evaluating {len(imputed_files)} imputed file(s)...")
        
        # Load data
        df_masked, df_complete, mask, split_name = load_data_and_masks(
            dataset_dir=args.data_dir,
            mask_dir=args.mask_dir,
            mask_idx=args.mask_idx,
            use_train_set=args.use_train_set_for_conditioning
        )
        
        info_path = os.path.join(args.data_dir, "info.json")
        if not os.path.exists(info_path):
            info_path = None
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        results = []
        for imputed_path in imputed_files:
            if not os.path.exists(imputed_path):
                print(f"File not found: {imputed_path}")
                continue
            
            print(f"\nEvaluating: {imputed_path}")
            metrics = evaluate_imputed_file(
                imputed_path=imputed_path,
                df_masked=df_masked,
                df_complete=df_complete,
                mask=mask,
                info_path=info_path,
                seed=args.seed
            )
            results.append(metrics)
        
        if results:
            output_filename = f"metrics_seed{args.seed}.csv"
            output_path = os.path.join(args.output_dir, output_filename)
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"\nSaved metrics to: {output_path}")
            print(f"Columns: {len(pd.DataFrame(results).columns)} (includes per-feature metrics)")
    
    else:
        print("Error: Must specify either --imputed-dir (structured mode) or --imputed-file/--imputed-files")


if __name__ == "__main__":
    main()
