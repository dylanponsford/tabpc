import os
import gc
import sys
import argparse
import pandas as pd
import numpy as np
import torch

sys.path.append(".")
sys.path.append("cirkit")
from train_scripts.pc_imputation import main as train_pc
from src.util import Timer, get_available_device, edit_json
from src.experiment_config import ExperimentConfig, load_experiment
from src.models.probabilistic_circuit import ProbabilisticCircuit
from src.models.shallow_mixture import ShallowMixture

COND_MODELS_DIR = 'artifacts/tabpc_imputation'

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def get(d: dict, keys: tuple):
    try:
        for key in keys:
            d = d[key]
    except KeyError:
        return None
    return d

def approximate_MAP(
        experiment_path: str,
        overwrite: bool = False,
        device = None,
        n_samples: int = 1,
        cond_batch_size: int = 10, # Cond batch size has been chosen to fit on our GPUs, but can be tuned for better performance
        use_train_set_for_conditioning: bool = False # Whether to use the train set for conditioning (if False, use the test set)
        ):
    """
    TODO: write docstring
    """
    device = device if device is not None else get_available_device(mem_required=0.9, verbose=True)

    exp_config: ExperimentConfig = load_experiment(experiment_path)

    if exp_config.train_mask_path is None:
        raise ValueError("Mask path should be specified in experiment config \
                         for imputation.")
    train_mask_path = exp_config.train_mask_path

    mask_path_items = train_mask_path.split(os.sep)
    mask_idx = int(mask_path_items[-1].strip("train_mask_").strip(".npy"))
    mask_type = mask_path_items[-2]
    missingness_rate = mask_path_items[-3].strip("rate")

    # create folder for samples
    samples_path = os.path.join(experiment_path, f"imputed_samples")
    os.makedirs(samples_path, exist_ok=True)

    csv_samples_path = os.path.join(
        samples_path,
        f"samples_{mask_type}_{missingness_rate}_{mask_idx}"
    )
    os.makedirs(csv_samples_path, exist_ok=True)

    print(f"Computing aMAP {n_samples} times from model...")

    # load model
    model = exp_config.model

    if isinstance(model, ProbabilisticCircuit) or isinstance(model, ShallowMixture):
        model.circuit.to(device)

    times_file_name = os.path.join(
        samples_path,
        f"generation_times_{mask_type}_{missingness_rate}_{mask_idx}.json"
    )

    # Overwrite generation times
    if overwrite and os.path.exists(times_file_name):
        os.remove(times_file_name)

    # Initialize df_to_mask. Under the current setup, the dataset is returned pre-masked.
    df_to_mask = None
    if use_train_set_for_conditioning: # Get training set for conditioning if specified
        df_to_mask = exp_config.get_train_data()
    else: # Otherwise get test set
        df_to_mask = exp_config.get_test_data()

    # Save masked version of dataset
    dataset_split = "train" if use_train_set_for_conditioning else "test"
    df_to_mask.to_csv(os.path.join(csv_samples_path, f"masked_{dataset_split}.csv"), index=False)

    # Save ground-truth values
    dataset_path = exp_config.dataset_path
    base_df = pd.read_csv(os.path.join(dataset_path, f"{dataset_split}.csv"))
    gt_values = base_df.mask(~df_to_mask.isna())
    gt_values.to_csv(os.path.join(csv_samples_path, f"gt_{dataset_split}.csv"), index=False)

    # Generate a dataset of the same size as the chosen conditioning set (test or train)
    num_to_generate = len(df_to_mask)

    for i in range(n_samples):
        if use_train_set_for_conditioning:
            name = f"train_amap_{i}"
        else:
            name = f"test_amap_{i}"
        if not overwrite and os.path.exists(os.path.join(csv_samples_path, f"{name}.csv")):
            print(f"Imputation {i} already exists, skipping...")
            continue

        print(f"Generating sample {i+1}/{n_samples}...")
        with Timer() as generation_timer:
            if not isinstance(model, ProbabilisticCircuit):
                raise ValueError("Approximate MAP is only supported \
                                 for ProbabilisticCircuit models.")

            df = model.approximate_map(
                device=device,
                batch_size=cond_batch_size,
                condition=df_to_mask,
                num_to_generate=num_to_generate
            )

            # Save generated samples to csv
            df.to_csv(os.path.join(csv_samples_path, f"{name}_amap.csv"), index=False)

        with edit_json(filename=times_file_name) as data:
            data[i] = generation_timer.elapsed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use TabPC models for imputation of missing data.')
    parser.add_argument('--models-dir', type=str, help='Target directory for trained models', default=COND_MODELS_DIR)
    parser.add_argument('--seed', help='Random seed (int or None; default: 0)', default=0)
    parser.add_argument('--dataset-dir', type=str, required=True, help='Folder of the dataset to run the experiment on.')
    parser.add_argument('--mask-dir', type=str, required=True, help='Folder of the missingness masks (train and test) to run the experiment on.')
    parser.add_argument('--mask-idx', type=int, required=True, help='Index of the missingness masks to use. (0-9)')
    parser.add_argument('--train', '-t', action='store_true', help='Whether to run the training of models (default: False)')
    parser.add_argument('--sample', '-s', action='store_true', help='Whether to run the conditional sampling (default: False)')
    parser.add_argument('--overwrite', '-o', action='store_true', help='Whether to overwrite existing samples and times (default: False)')
    parser.add_argument('--cond-batch-size', type=int, default=10, help='Batch size for conditional sampling (default: 10, can be tuned for better performance)')
    parser.add_argument('--use-train-set-for-conditioning', action='store_true', help='Whether to use the train set for conditioning (default: False, which means using the test set for conditioning)')
    parser.add_argument('--num-units', type=int, default=1024, help='Number of units for the trained PC')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for the trained PC')
    parser.add_argument('--lr', type=float, default=0.25, help='Learning rate for the trained PC')

    args = parser.parse_args()

    # First get masked train and test datasets based on dataset-dir and mask-dir
    train_mask_filename = f'train_mask_{args.mask_idx}.npy'
    test_mask_filename = f'test_mask_{args.mask_idx}.npy'

    # Masks are stored as .npy files
    train_mask_path = os.path.join(args.mask_dir, train_mask_filename)
    test_mask_path = os.path.join(args.mask_dir, test_mask_filename)

    # Create specified folders if they don't exist
    os.makedirs(args.models_dir, exist_ok=True)
    # 1) Train SotA TabPC models so that they are in a separate folder
    if args.train:
        dataset_dir = args.dataset_dir
        print(f"Training TabPC on ...")
        train_pc(
            path=args.dataset_dir,
            num_units=args.num_units,
            batch_size=args.batch_size,
            lr =args.lr,
            seed=args.seed,
            target_folder=args.models_dir,
            train_mask_path=train_mask_path,
            test_mask_path=test_mask_path,
        )
        free_memory()

    experiment_paths = os.listdir(args.models_dir)
    experiment_paths = [os.path.join(args.models_dir, path) for path in experiment_paths]

    if args.sample:
        for experiment_path in experiment_paths:
            try:
                approximate_MAP(
                    experiment_path,
                    n_samples=1,
                    overwrite=args.overwrite,
                    cond_batch_size=args.cond_batch_size,
                    use_train_set_for_conditioning=args.use_train_set_for_conditioning
                    )
            except Exception as e:
                print(f"Error during conditional sampling for experiment {experiment_path}: {e}")
            free_memory()
