import os
import requests

import torch

from ..constants import TRAIN_DATA_URL, GITHUB_REPO_OWNER, GITHUB_REPO_NAME, GITHUB_FACES_PATH

def download_github_directory(repo_owner, repo_name, directory_path, local_dir):
    """
    Recursively download all files from a GitHub repository directory

    Args:
        repo_owner (str): GitHub username/organization, e.g. "MITDeepLearning"
        repo_name (str): Repository name, e.g. "introtodeeplearning"
        directory_path (str): Path within the repo, e.g. "mitdeeplearning/data/faces"
        local_dir (str): Local path to save files under, e.g. "data/faces"
    """
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{directory_path}"
    resp = requests.get(api_url)
    resp.raise_for_status()
    items = resp.json()

    os.makedirs(local_dir, exist_ok=True)

    for item in items:
        item_type = item["type"]
        item_name = item["name"]
        item_path = item["path"]

        if item_type == "file":
            download_url = item["download_url"]
            local_path = os.path.join(local_dir, item_name)
            print(f"Downloading {item_path} → {local_path}")
            file_resp = requests.get(download_url, stream=True)
            file_resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in file_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        elif item_type == "dir":
            subdir_local = os.path.join(local_dir, item_name)
            download_github_directory(repo_owner, repo_name, item_path, subdir_local)
        # (ignore other types like symlinks)

def download_training_data(data_file):
    """Download the main training h5 file."""
    if os.path.exists(data_file):
        print(f"Using cached training data from {data_file}")
    else:
        os.makedirs(os.path.dirname(data_file), exist_ok=True)  # Add this
        print(f"Downloading training data to {data_file}")
        torch.hub.download_url_to_file(TRAIN_DATA_URL, data_file)

def download_test_faces(faces_root):
    """Download test face images."""
    if os.path.isdir(faces_root):
        print(f"Using face data from {faces_root}")
    else:
        print(f"Downloading face images to {faces_root}...")
        download_github_directory(
            repo_owner=GITHUB_REPO_OWNER,
            repo_name=GITHUB_REPO_NAME,
            directory_path=GITHUB_FACES_PATH,
            local_dir=faces_root,
        )
