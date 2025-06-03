import os
import argparse
import yaml
import torch

from src.utils.downloader import download_training_data, download_test_faces
from src.utils.datasets import FaceDataset
from src.utils.evaluate import evaluate_on_training_data, evaluate_on_test_faces, get_test_faces
from src.utils.visualization import plot_demographic_results


def load_config(config_path):
    """Load YAML config and return as namespace object."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return argparse.Namespace(**cfg)


def validate_required_params(args, required_params):
    """Validate that all required parameters are present in config."""
    for param in required_params:
        if not hasattr(args, param):
            raise ValueError(f"Missing required parameter: {param}")


def setup_data_and_device(args):
    """
    Download data, create dataset, and setup device.

    Returns:
        tuple: (dataset, device, data_file, faces_root)
    """
    # Download data
    data_file = f"{args.data_dir}/train_face.h5"
    download_training_data(data_file)

    faces_root = f"{args.data_dir}/faces"
    download_test_faces(faces_root)

    # Create dataset
    print("Loading data into memory...")
    dataset = FaceDataset(data_file)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    return dataset, device, data_file, faces_root


def save_model(model, save_path):
    """Save model state dict to specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


def run_evaluation(model, dataset, faces_root, args, demographic_groups, model_name="Model"):
    """
    Run complete evaluation on training data and test demographics.

    Args:
        model: Trained model
        dataset: Training dataset
        faces_root: Path to test faces directory
        args: Configuration arguments
        demographic_groups: List of demographic group names
        model_name: Name for display purposes

    Returns:
        tuple: (train_accuracy, demographic_results)
    """
    print("\n" + "=" * 50)
    print("EVALUATION")
    print("=" * 50)

    # Evaluate on training data
    train_accuracy = evaluate_on_training_data(model, dataset, args.eval_samples)
    print(f"{model_name} accuracy on (potentially biased) training set: {train_accuracy:.4f}")

    # Evaluate on test faces for bias analysis
    test_faces = get_test_faces(faces_root, args.img_height, args.img_width, channels_last=False)
    demographic_results = evaluate_on_test_faces(model, test_faces, demographic_groups)

    return train_accuracy, demographic_results


def print_evaluation_results(demographic_results, model_name="Model"):
    """
    Print and plot demographic evaluation results.

    Args:
        demographic_results: Dictionary of demographic -> accuracy
        model_name: Name for display and plot title
    """
    print(f"{model_name} accuracy for different demographics:")
    for k, v in demographic_results.items():
        print(f"{k}: {v * 100:.2f}%")

    # plot_title = f"{model_name} Predictions by Demographic Group"
    # plot_demographic_results(demographic_results, title=plot_title)

