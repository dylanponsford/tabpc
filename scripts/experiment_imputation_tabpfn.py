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

from tabpfn_extensions import unsupervised
from tabpfn_extensions.unsupervised.experiments import GenerateSyntheticDataExperiment
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.unsupervised.unsupervised import TabPFNUnsupervisedModel
from tabpfn_extensions.utils import get_max_num_classes

from src.preprocessors.compose import Compose
from src.preprocessors.nan_handler import NanHandler
from src.preprocessors.df_type_conversion import TypeConversion
from src.preprocessors.df_quantile_normalizer import QuantileNormalizer
from src.preprocessors.df_dequantization import Dequantization
from src.preprocessors.string_to_int import StringToInt
from src.preprocessors.to_tensor import ToTensor
from src.util import edit_json

TABPFN_MAX_CATS = 160

# def generate_synthetic_dataset_tabpfn(
#         dataset_dir: str,
#         mask_dir: str,
#         mask_idx: int,
#         output_data_dir: str,
#         save_raw: bool,
#         overwrite: bool,
#         n_permutations: int,
#         use_only_ordinal: bool,
#         use_train_set_for_conditioning: bool = True,
#         device: str | None = None,
#     ) -> str:

#     train_mask_path = os.path.join(mask_dir, f"train_mask_{mask_idx}.npy")

#     mask_path_items = train_mask_path.split(os.sep)
#     mask_type = mask_path_items[-2]
#     missingness_rate = mask_path_items[-3].strip("rate")

#     dataset_name = dataset_dir.split(os.sep)[-1]

#     # Check if sample already exists
#     output_dir = os.path.join(output_data_dir, dataset_name, f"samples_{mask_type}_{missingness_rate}_{mask_idx}")
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"tabpfn_imputed_{dataset_name}_perms_{n_permutations}.csv")
#     if os.path.exists(output_path) and not overwrite:
#         print("Skipping generating existing file:", output_path)
#         return output_dir

#     dataset_split = "train" if use_train_set_for_conditioning else "test"
#     df_to_impute = pd.read_csv(f"{dataset_dir}/{dataset_split}.csv")

#     # TODO: actually mask the dataset
#     # TODO: make the below work with TabPFN impute functionality

#     # Get dataset metadata
#     with open(f"{dataset_dir}/info.json") as json_f:
#         data_info_file = json.load(json_f)

#     categorical_features = data_info_file["cat_col_idx"]
#     numerical_features = data_info_file["num_col_idx"]

#     # Add target label to list of categorical features if required
#     task_type = data_info_file["task_type"]
#     target_col_idx = data_info_file["target_col_idx"]
#     if task_type == "binclass":
#         categorical_features += target_col_idx

#     # Set numerical features with sufficiently few values to be handled as categorical
#     # int_features = data_info_file["int_col_idx"]
#     for num_idx in numerical_features:
#         if len(df_to_impute.iloc[:, num_idx].value_counts()) <= TABPFN_MAX_CATS:
#             categorical_features.append(num_idx)

#     # Encode the categorical data as integer values to pass to TabPFN
#     if use_only_ordinal:
#         encoder = OrdinalEncoder(
#             handle_unknown="use_encoded_value",
#             unknown_value=-1
#         )
#     else:
#         preprocessors = [
#             TypeConversion(cat_max_states=160, 
#                             int_max_states=None),
#             StringToInt(),
#             ToTensor(),
#         ]
#         encoder = Compose(preprocessors)

#     df_encoded = df_to_impute.copy()

#     if use_only_ordinal:
#         df_encoded.iloc[:, categorical_features] = encoder.fit_transform(df_encoded.iloc[:, categorical_features])
#         # X_preprocessed = df_X_encoded.to_numpy()
#         X_preprocessed = torch.from_numpy(df_encoded.values.astype("float32")).float()
        
#         feature_names = df_encoded.columns
#     else:
#         X_preprocessed = encoder.fit_transform(df_encoded)

#         feature_names = encoder.preprocessors[-1].parameters["names"]
    
#     feature_indices = [*range(X_preprocessed.shape[1])] # Select all indices for generation
#     # Now this can include features added after preprocessing (e.g. for inflated value handling)
    
#     y_preprocessed = torch.empty([])

#     # Initialize TabPFN models
#     model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
#         tabpfn_clf=TabPFNClassifier(), 
#         tabpfn_reg=TabPFNRegressor()
#     )

#     # Run synthetic data generation experiment
#     experiment = GenerateSyntheticDataExperiment(
#         task_type="unsupervised"
#     )

#     # import pdb; pdb.set_trace()

#     results = experiment.run(
#         tabpfn=model_unsupervised,
#         X=X_preprocessed,
#         y=y_preprocessed,
#         attribute_names=feature_names,
#         temp=1.0,                     # Temperature parameter for sampling
#         n_samples=df_to_impute.shape[0],     # Generate as many samples as original data
#         indices=feature_indices,
#         categorical_features=categorical_features, # Set categorical features
#         should_plot=False,
#         n_permutations=n_permutations,
#     )

#     data = experiment.data
#     synthetic_data_raw = data[data["real_or_synthetic"] == "Generated samples"].drop("real_or_synthetic", axis=1)

#     # Save a raw copy in case inverting the encoder doesn't work
#     if save_raw:
#         raw_output_path = os.path.join(output_dir, f"tabpfn_imputed_{dataset_name}_perms_{n_permutations}_raw.csv")
#         synthetic_data_raw.to_csv(raw_output_path)

#     # Restore the original categories
#     if use_only_ordinal:
#         synthetic_data = synthetic_data_raw.copy()
#         synthetic_data.iloc[:, categorical_features] = encoder.inverse_transform(synthetic_data.iloc[:, categorical_features])
#         synthetic_data_df = synthetic_data.copy()
#     else:
#         synthetic_data = synthetic_data_raw.copy()
#         synthetic_data_tensor = torch.tensor(synthetic_data.values)
#         synthetic_data_df = encoder.reverse_transform(synthetic_data_tensor)

#     # Save to csv
#     synthetic_data_df.to_csv(output_path, index=False)

#     return output_dir

def impute_single_permutation_(
        self,
        X: torch.Tensor,
        feature_permutation: list[int] | tuple[int, ...],
        t: float = 0.000000001,
        condition_on_all_features: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Impute missing values (np.nan) in X by sampling all cells independently from the trained models.

        :param X: Input data of the shape (num_examples, num_features) with missing values encoded as np.nan
        :param t: Temperature for sampling from the imputation distribution, lower values are more deterministic
        :return: Imputed data, with missing values replaced
        """
        X_fit = self.X_
        impute_X = copy.deepcopy(X)

        for i in range(len(feature_permutation)):
            column_idx = feature_permutation[i]

            if not condition_on_all_features:
                conditional_idx = feature_permutation[:i] if i > 0 else []
            else:
                conditional_idx = list(set(range(X.shape[1])) - {column_idx})

            y_predict = impute_X[:, column_idx]

            if torch.isnan(y_predict).sum() == 0:
                continue

            X_where_y_is_nan = impute_X[torch.isnan(y_predict)]
            X_where_y_is_nan = X_where_y_is_nan.reshape(-1, impute_X.shape[1])

            model, X_predict, _ = self.density_(
                X_where_y_is_nan,
                X_fit,
                conditional_idx,
                column_idx,
            )

            pred, pred_sampled = self.sample_from_model_prediction_(
                column_idx,
                X_fit,
                model,
                X_predict,
                t,
            )

            impute_X[torch.isnan(y_predict), column_idx] = pred_sampled.to(
                y_predict.dtype
            ).to(y_predict.device)

        return impute_X, pred


def use_classifier_(self, column_idx: int, y: torch.Tensor | np.ndarray) -> bool:
        """Determine whether to use a classifier or regressor for a feature.

        Args:
            column_idx: Index of the column to check
            y: Values of the feature

        Returns:
            bool: True if a classifier should be used, False for a regressor
        """
        is_categorical = column_idx in self.categorical_features
        if self.tabpfn_clf is None:
            # No classifier was provided: surface categorical columns so
            # density_ raises a clear "missing tabpfn_clf" error rather than
            # silently routing categorical data to the regressor; numerical
            # columns go to the regressor as usual.
            return is_categorical
        # Use the classifier only when both constraints hold:
        #   (a) the column is categorical, and
        #   (b) the classifier can actually predict that many classes
        #       (a TabPFN clf always reports a limit; None means no inherent
        #       limit, e.g. a non-TabPFN estimator).
        max_classes = get_max_num_classes(self.tabpfn_clf)
        # torch.unique stays on-device; np.unique raises on CUDA/MPS tensors.
        y_no_nan = y[~y.isnan()]
        n_unique = torch.unique(y_no_nan).numel() if torch.is_tensor(y_no_nan) else len(np.unique(y))
        return is_categorical and (max_classes is None or n_unique <= max_classes)


def impute_dataset_tabpfn(
        dataset_dir: str,
        mask_dir: str,
        mask_idx: int,
        output_data_dir: str,
        save_raw: bool,
        overwrite: bool,
        n_permutations: int,
        use_only_ordinal: bool,
        use_train_set_for_conditioning: bool = True,
        device: str | None = None,
    ) -> str:

    dataset_split = "train" if use_train_set_for_conditioning else "test"

    dataset_mask_path = os.path.join(mask_dir, f"{dataset_split}_mask_{mask_idx}.npy")
    dataset_mask = np.load(dataset_mask_path)
    
    mask_path_items = dataset_mask_path.split(os.sep)
    mask_type = mask_path_items[-2]
    missingness_rate = mask_path_items[-3].strip("rate")

    dataset_name = dataset_dir.split(os.sep)[-1]

    # Check if sample already exists
    output_dir = os.path.join(output_data_dir, dataset_name, f"samples_{mask_type}_{missingness_rate}_{mask_idx}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tabpfn_imputed_{dataset_name}_perms_{n_permutations}.csv")
    if os.path.exists(output_path) and not overwrite:
        print("Skipping generating existing file:", output_path)
        return output_dir

    base_df = pd.read_csv(f"{dataset_dir}/{dataset_split}.csv")

    df_to_impute = base_df.mask(dataset_mask)
    df_to_impute.to_csv(os.path.join(output_dir, f"masked_{dataset_split}.csv"), index=False)
    gt_values = base_df.mask(~dataset_mask)
    gt_values.to_csv(os.path.join(output_dir, f"gt_{dataset_split}.csv"), index=False)

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
    # int_features = data_info_file["int_col_idx"]
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
        df_encoded.iloc[:, categorical_features] = encoder.fit_transform(df_encoded.iloc[:, categorical_features])
        # X_preprocessed = df_X_encoded.to_numpy()
        X_preprocessed = torch.from_numpy(df_encoded.values.astype("float32")).float()
        
        feature_names = df_encoded.columns
    else:
        X_preprocessed = encoder.fit_transform(df_encoded)

        feature_names = encoder.preprocessors[-1].parameters["names"]
    
    # feature_indices = [*range(X_preprocessed.shape[1])] # Select all indices for generation
    # Now this can include features added after preprocessing (e.g. for inflated value handling)
    
    # y_preprocessed = torch.empty([])

    TabPFNUnsupervisedModel.impute_single_permutation_ = impute_single_permutation_
    TabPFNUnsupervisedModel.use_classifier_ = use_classifier_

    # Initialize TabPFN models
    model_unsupervised = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(device=device), 
        tabpfn_reg=TabPFNRegressor(device=device)
    )

    # # Run synthetic data generation experiment
    # experiment = GenerateSyntheticDataExperiment(
    #     task_type="unsupervised"
    # )

    # import pdb; pdb.set_trace()

    # results = experiment.run(
    #     tabpfn=model_unsupervised,
    #     X=X_preprocessed,
    #     y=y_preprocessed,
    #     attribute_names=feature_names,
    #     temp=1.0,                     # Temperature parameter for sampling
    #     n_samples=df_to_impute.shape[0],     # Generate as many samples as original data
    #     indices=feature_indices,
    #     categorical_features=categorical_features, # Set categorical features
    #     should_plot=False,
    #     n_permutations=n_permutations,
    # )

    # data = experiment.data
    # synthetic_data_raw = data[data["real_or_synthetic"] == "Generated samples"].drop("real_or_synthetic", axis=1)

    model_unsupervised.fit(X_preprocessed.numpy())

    model_unsupervised.set_categorical_features(categorical_features)

    # import pdb; pdb.set_trace()

    synthetic_data_raw = model_unsupervised.impute(
        X=X_preprocessed.numpy(),
        t=1.0, # Let's set the temperature to 1 for now such that we still get a distribution of values
        n_permutations=n_permutations,
    )

    # Save a raw copy in case inverting the encoder doesn't work
    if save_raw:
        raw_output_path = os.path.join(output_dir, f"tabpfn_imputed_{dataset_name}_perms_{n_permutations}_raw.csv")
        synthetic_data_raw.to_csv(raw_output_path)

    # Restore the original categories
    if use_only_ordinal:
        synthetic_data = synthetic_data_raw.copy()
        synthetic_data.iloc[:, categorical_features] = encoder.inverse_transform(synthetic_data.iloc[:, categorical_features])
        synthetic_data_df = synthetic_data.copy()
    else:
        synthetic_data = synthetic_data_raw.copy()
        synthetic_data_tensor = torch.tensor(synthetic_data.values)
        synthetic_data_df = encoder.reverse_transform(synthetic_data_tensor)

    # Save to csv
    synthetic_data_df.to_csv(output_path, index=False)

    return output_dir


if __name__ == "__main__":
    # Monkey patch the run method of the GenerateSyntheticDataExperiment class to fix current bug with treating features as categorical
    # GenerateSyntheticDataExperiment.run = run

    parser = argparse.ArgumentParser(description="Generate synthetic data using TabPFN in the format expected by the repo")
    # parser.add_argument("--dataset", type=str, required=True,
    #                     choices=["adult", "beijing", "default", "diabetes", "magic", "news", "shoppers", "all"],
    #                     help="Which dataset to generate a synthetic dataset for.")
    parser.add_argument('--dataset-dir', type=str, required=True, help='Folder of the dataset to run the experiment on.')
    parser.add_argument("--output-data-dir", required=True,
                        help="The directory in which to store the generated data.")
    parser.add_argument('--mask-dir', type=str, required=True, help='Folder of the missingness masks (train and test) to run the experiment on.')
    parser.add_argument('--mask-idx', type=int, required=True, help='Index of the missingness masks to use. (0-9)')
    # parser.add_argument("--num-repeats", type=int, default=5,
    #                     help="Number of times to regenerate a dataset to be able to compute standard deviations. (Default: 5)")
    parser.add_argument("--save-raw", action="store_true", default=False,
                        help="Whether to save the raw generated dataset before inverting the encoding. (Default: False)")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Whether to overwrite existing samples with newly generated ones. (Default: False)")
    parser.add_argument("--n-permutations", type=int, default=3,
                        help="The number of permutations to use for TabPFN's imputation generation. (Default: 3)")
    parser.add_argument("--use-only-ordinal", action="store_true", default=False,
                        help="Whether the use TabPC's preprocessing pipeline or just the sklearn OrdinalEncoder. (Default: False)")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="The maximum number of failed generations to allow before skipping to the next sample.")
    parser.add_argument('--use-train-set-for-conditioning', action='store_true', help='Whether to use the train set for conditioning (default: False, which means using the test set for conditioning)')
    parser.add_argument('--device', type=int, default=-1, help='Index for cuda device (-1 for CPU)')
    args = parser.parse_args()

    device_string = f"cuda:{args.device}" if args.device != -1 else "cpu"

    print(f"Dataset folder: {args.dataset_dir}. Mask folder: {args.mask_dir}. Mask index: {args.mask_idx}.")
    # failure_count = 0
    # while failure_count < args.max_failures:
    #     print(f"Failure_count: {failure_count}")
    #     try:
    #         start_time = time.time()
    #         output_dir = impute_dataset_tabpfn(
    #             dataset_dir=args.dataset_dir,
    #             mask_dir=args.mask_dir,
    #             mask_idx=args.mask_idx,
    #             output_data_dir=args.output_data_dir,
    #             save_raw=args.save_raw,
    #             overwrite=args.overwrite,
    #             n_permutations=args.n_permutations,
    #             use_only_ordinal=args.use_only_ordinal,
    #             use_train_set_for_conditioning=args.use_train_set_for_conditioning,
    #             device=device_string,
    #         )
    #         elapsed_time = time.time() - start_time
    #         # generation_times[dataset][repeat_idx] = elapsed_time
    #         # TODO: save elapsed time in output directory
    #         dataset_split = "train" if args.use_train_set_for_conditioning else "test"
    #         with edit_json(os.path.join(output_dir, "generation_times.json")) as data:
    #             data["dataset_split"] = elapsed_time
    #         break
    #     except Exception as e:
    #         print(f"Error for dataset {args.dataset_dir.split(os.sep)[-1]} and mask_dir {args.mask_dir}: {e}")
    #         failure_count += 1

    try:
        output_dir = impute_dataset_tabpfn(
            dataset_dir=args.dataset_dir,
            mask_dir=args.mask_dir,
            mask_idx=args.mask_idx,
            output_data_dir=args.output_data_dir,
            save_raw=args.save_raw,
            overwrite=args.overwrite,
            n_permutations=args.n_permutations,
            use_only_ordinal=args.use_only_ordinal,
            use_train_set_for_conditioning=args.use_train_set_for_conditioning,
            device=device_string,
        )
    except Exception:
        import pdb
        import traceback
        traceback.print_exc()
        pdb.post_mortem()