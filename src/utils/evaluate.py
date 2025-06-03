import glob

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2

from ..constants import DEMOGRAPHIC_CODES

def evaluate_on_training_data(model, dataset, num_samples, batch_size=32):
    """
    Evaluate model accuracy on a subset of the training data.
    """
    device = model.get_device()
    model.eval()

    # Create a subset for evaluation
    eval_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    eval_subset = torch.utils.data.Subset(dataset, eval_indices)
    eval_dataloader = DataLoader(eval_subset, batch_size=batch_size, shuffle=False)

    # Get predictions
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in eval_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model.predict(images)
            predictions = torch.round(torch.sigmoid(outputs.squeeze()))

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def get_test_faces(faces_dir, img_height, img_width, channels_last=True):
    """
    Load test faces from local files.

    Args:
        faces_dir (str): Directory containing face subdirectories
        channels_last (bool): Whether to return images in channels-last format
    """
    images = {code: [] for code in DEMOGRAPHIC_CODES}
    for key in images.keys():
        files = glob.glob(f"{faces_dir}/{key}/*.png")
        for file in sorted(files):
            image = cv2.resize(cv2.imread(file), (img_width, img_height))[:, :, ::-1] / 255.0
            if not channels_last:
                image = np.transpose(image, (2, 0, 1))
            images[key].append(image)

    return images["LF"], images["LM"], images["DF"], images["DM"]


def evaluate_on_test_faces(model, test_faces, demographic_names):
    """
    Evaluate model on test faces grouped by demographics.
    Args:
        model: Trained model
        test_faces: List of face arrays for each demographic group
        device: Device to run on
        demographic_names: List of demographic group names
    Returns:
        dict: Results with demographic names as keys and probabilities as values
    """
    device = model.get_device()
    model.eval()

    results = {}

    with torch.inference_mode():
        for i, faces in enumerate(test_faces):
            # Convert to tensor and move to device
            faces_tensor = torch.from_numpy(np.array(faces, dtype=np.float32)).to(device)

            # Get predictions
            logits = model.predict(faces_tensor)
            probs = torch.sigmoid(logits).squeeze()

            # Store probabilities for this demographic
            avg_prob = np.mean(probs.cpu().numpy())
            results[demographic_names[i]] = avg_prob

    return results

