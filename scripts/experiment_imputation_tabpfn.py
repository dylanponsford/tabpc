import json
import argparse
import os
import time
import sys
import copy

sys.path.append(".")
sys.path.append("cirkit")

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from typing import Any
from tqdm import tqdm

from tabpfn_extensions import unsupervised
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised.unsupervised import TabPFNUnsupervisedModel, efficient_random_permutation, \
    _resolve_dag_order
from tabpfn_extensions.utils import get_max_num_classes
from tabpfn_extensions.unsupervised import simple_impute

from src.preprocessors.compose import Compose
from src.preprocessors.nan_handler import NanHandler
from src.preprocessors.df_type_conversion import TypeConversion
from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
from src.preprocessors.df_dequantization import Dequantization
from src.preprocessors.string_to_int import StringToInt
from src.preprocessors.to_tensor import ToTensor
from src.util import edit_json

TABPFN_MAX_CATS = get_max_num_classes(TabPFNClassifier())

def impute_dataset_tabpfn(
        data_root: str,
        dataset: str,
        mask_type: str,
        rate: int,
        mask_idx: int,
        imputed_dir: str,
        overwrite: bool,
        use_only_ordinal: bool,
        use_train_set_for_conditioning: bool = True,
        device: str | None = None,
    ) -> str:

    dataset_split = "train" if use_train_set_for_conditioning else "test"

    # Build method name with optional postfix
    method_name = "tabpfn_3_5"
    if use_only_ordinal:
        method_name += "_only_ordinal"

    # Construct paths matching baselines structure
    dataset_dir = os.path.join(data_root, dataset)
    mask_dir = os.path.join(data_root, dataset, "masks", f"rate{rate}", mask_type)
    dataset_mask_path = os.path.join(mask_dir, f"{dataset_split}_mask_{mask_idx}.npy")
    dataset_mask = np.load(dataset_mask_path)

    # Build directory structure matching baselines:
    # {imputed_dir}/{dataset}/{split}/{mask_type}/rate{rate}/mask_{mask_idx}/{method}/
    output_dir = os.path.join(
        imputed_dir,
        dataset,
        dataset_split,
        mask_type,
        f"rate{rate}",
        f"mask_{mask_idx}",
        method_name
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Filename matching baselines pattern: imputed_{method}_{dataset}_{type}_{rate}_{mask}.csv
    output_path = os.path.join(
        output_dir,
        f"imputed_{method_name}_{dataset}_{mask_type}_{rate}_{mask_idx}.csv"
    )

    # Check if sample already exists
    if os.path.exists(output_path) and not overwrite:
        print("Skipping generating existing file:", output_path)
        return output_dir

    base_df = pd.read_csv(f"{dataset_dir}/{dataset_split}.csv")
    df_to_impute = base_df.mask(dataset_mask)

    # Get dataset metadata
    with open(f"{dataset_dir}/info.json") as json_f:
        data_info_file = json.load(json_f)

    categorical_features = data_info_file["cat_col_idx"]
    numerical_features = data_info_file["num_col_idx"]

    # Add target label to list of categorical features if required
    task_type = data_info_file["task_type"]
    target_col_idx = data_info_file["target_col_idx"]
    if task_type == "binclass":
        categorical_features += target_col_idx

    # Set numerical features with sufficiently few values to be handled as categorical
    print(f"TabPFN maximum number of categories: {TABPFN_MAX_CATS}")
    for num_idx in numerical_features:
        if len(df_to_impute.iloc[:, num_idx].value_counts()) <= TABPFN_MAX_CATS:
            categorical_features.append(num_idx)

    # Encode the categorical data as integer values to pass to TabPFN
    if use_only_ordinal:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    else:
        preprocessors = [
            TypeConversion(cat_max_states=160,
                            int_max_states=None),
            StringToInt(),
            ToTensor(),
        ]
        encoder = Compose(preprocessors)

    df_encoded = df_to_impute.copy()

    if use_only_ordinal:
        df_encoded.iloc[:, categorical_features] = encoder.fit_transform(
            df_encoded.iloc[:, categorical_features]
        )
        X_preprocessed = torch.from_numpy(df_encoded.values.astype("float32")).float()
    else:
        X_preprocessed = encoder.fit_transform(df_encoded)

    synthetic_data_raw = simple_impute(
        X=X_preprocessed.numpy(),
        tabpfn_clf=TabPFNClassifier(device=device),
        tabpfn_reg=TabPFNRegressor(device=device),
        categorical_features=categorical_features,
        use_imputed=False,
    )

    # Restore the original categories
    if use_only_ordinal:
        synthetic_data = synthetic_data_raw.copy()
        synthetic_data.iloc[:, categorical_features] = encoder.inverse_transform(synthetic_data.iloc[:, categorical_features])
        synthetic_data_df = synthetic_data.copy()
    else:
        if type(synthetic_data_raw) == pd.DataFrame:
            synthetic_data = synthetic_data_raw.copy()
            synthetic_data_tensor = torch.tensor(synthetic_data.values)
        else:
            synthetic_data_tensor = torch.from_numpy(synthetic_data_raw).float()
        synthetic_data_df = encoder.reverse_transform(synthetic_data_tensor)

    # Save to csv
    synthetic_data_df.to_csv(output_path, index=False)

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data using TabPFN in the format expected by the repo")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Root data directory (e.g., data/)')
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
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Whether to overwrite existing samples with newly generated ones. (Default: False)")
    parser.add_argument("--use-only-ordinal", action="store_true", default=False,
                        help="Whether to use only ordinal encoder instead of TabPC preprocessing. (Default: False)")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="The maximum number of failed generations to allow before skipping to the next sample.")
    parser.add_argument('--use-train-set-for-conditioning', action='store_true', help='Whether to use the train set for conditioning (default: False, which means using the test set for conditioning)')
    parser.add_argument('--device', type=int, default=-1, help='Index for cuda device (-1 for CPU)')
    args = parser.parse_args()

    device_string = f"cuda:{args.device}" if args.device != -1 else "cpu"

    dataset_split = "train" if args.use_train_set_for_conditioning else "test"

    print(f"Data root: {args.data_root}. Dataset: {args.dataset}. Split: {dataset_split}.")
    print(f"Mask type: {args.mask_type}, Rate: {args.rate}, Mask idx: {args.mask_idx}.")
    print(f"Imputed dir: {args.imputed_dir}.")
    
    failure_count = 0
    while failure_count < args.max_failures:
        print(f"Failure_count: {failure_count}")
        try:
            start_time = time.time()
            output_dir = impute_dataset_tabpfn(
                data_root=args.data_root,
                dataset=args.dataset,
                mask_type=args.mask_type,
                rate=args.rate,
                mask_idx=args.mask_idx,
                imputed_dir=args.imputed_dir,
                overwrite=args.overwrite,
                use_only_ordinal=args.use_only_ordinal,
                use_train_set_for_conditioning=args.use_train_set_for_conditioning,
                device=device_string,
            )
            elapsed_time = time.time() - start_time
            dataset_split = "train" if args.use_train_set_for_conditioning else "test"
            # Build method name for timing file
            method_name = "tabpfn_3_5"
            if args.use_only_ordinal:
                method_name += "_only_ordinal"
            # Save timing with method name in filename
            timing_file = os.path.join(output_dir, f"generation_times_{method_name}.json")
            with edit_json(timing_file) as data:
                data[dataset_split] = elapsed_time
            break
        except Exception as e:
            print(f"Error for dataset {args.dataset} and mask type {args.mask_type}: {e}")
            failure_count += 1
