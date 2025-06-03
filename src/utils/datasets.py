import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

class FaceDataset(Dataset):
    """
    A PyTorch Dataset that loads face/non-face examples from an HDF5 file.
    Each item is a tuple (image_tensor, label_tensor).

    Args:
        data_path (str): Path to the HDF5 file containing "images" (shape [N, H, W, C])
                         and "labels" (shape [N] or [N, 1]).
    """
    def __init__(self, data_path):
        super().__init__()

        # Open HDF5 file and load into memory
        try:
            with h5py.File(data_path, "r") as f:
                # Expect datasets "images" (N, H, W, C) and "labels" (N,) or (N, 1)
                self.images = f["images"][:]  # uint8 or float32
                self.labels = f["labels"][:]  # float32 or int
        except (OSError, KeyError) as e:
            raise ValueError(f"Could not load data from {data_path}: {e}")

        # Convert images to float32 in [0,1] if needed
        if self.images.dtype != np.float32:
            self.images = self.images.astype(np.float32) / 255.0

        # Ensure labels are shape (N,) and dtype float32
        self.labels = self.labels.astype(np.float32).squeeze()
        self.N = self.images.shape[0]

        # Separate positive and negative indices
        self.pos_indices = np.where(self.labels == 1.0)[0]
        self.neg_indices = np.where(self.labels == 0.0)[0]

        print(f"Dataset Loaded: {len(self.pos_indices)} faces, {len(self.neg_indices)} non-faces")

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        """
        Returns:
            img_tensor (Tensor): shape (C, H, W), dtype float32
            label_tensor (Tensor): scalar float32 (0.0 or 1.0)
        """
        img_np = self.images[index]       # (H, W, C), float32 in [0,1]
        lbl = float(self.labels[index])   # 0.0 or 1.0

        # Convert to torch.Tensor and permute to (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        label_tensor = torch.tensor(lbl, dtype=torch.float32)
        return img_tensor, label_tensor


    def get_all_face_images(self):
        """
        Get all face images (positive examples) from the dataset.

        Returns:
            numpy.ndarray: Array of face images with shape (N, H, W, C)
        """
        face_images = self.images[self.pos_indices]
        return face_images


    def get_batch_with_probabilities(self, batch_size, p_pos=None):
        """
        Sample a balanced training batch with adaptive resampling for debiasing.

        Creates batches with 50/50 face/non-face ratio while allowing custom sampling
        probabilities for face examples. This enables the debiasing algorithm to
        oversample faces with rare features (e.g., dark skin, accessories) during training.

        Sampling strategy:
        1. Sample half the batch from negative examples (uniform random)
        2. Sample half from face examples using provided probabilities
        3. Shuffle the combined batch to mix positive/negative examples

        Args:
            batch_size (int): Total batch size. Must be even to ensure balanced sampling.
            p_pos (numpy.ndarray, optional): Custom sampling probabilities for face examples.
                                            Should have same length as number of face images.
                                            If None, uses uniform sampling for faces.
                                            Higher probabilities = more likely to be selected.

        Returns:
            tuple: (images, labels) where:
                - images (numpy.ndarray): Batch of images with shape (batch_size, C, H, W)
                - labels (numpy.ndarray): Corresponding labels with shape (batch_size,)
                                        Values are 1.0 for faces, 0.0 for non-faces

        Raises:
            ValueError: If batch_size is odd (prevents balanced 50/50 sampling)
        """
        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced sampling")

        samples_per_class = batch_size // 2

        # Sample negative examples (uniform random)
        neg_batch_indices = np.random.choice(self.neg_indices, size=samples_per_class, replace=False)

        # Sample positive examples (with custom probabilities if provided)
        if p_pos is not None:
            pos_batch_indices = np.random.choice(
                self.pos_indices,
                size=samples_per_class,
                replace=False,
                p=p_pos
            )
        else:
            pos_batch_indices = np.random.choice(self.pos_indices, size=samples_per_class, replace=False)

        # Combine indices and shuffle
        batch_indices = np.concatenate([pos_batch_indices, neg_batch_indices])
        np.random.shuffle(batch_indices)

        # Get images and labels
        batch_images = self.images[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Convert to tensors and return
        img_tensors = torch.from_numpy(batch_images).permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        label_tensors = torch.from_numpy(batch_labels)

        return img_tensors.numpy(), label_tensors.numpy()

class BalancedSampler(Sampler):
    """
    Sampler that ensures each batch has equal numbers of positive and negative examples.

    Args:
        dataset: FaceDataset with pos_indices and neg_indices
        batch_size: Must be even for 50/50 sampling
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        if batch_size % 2 != 0:
            raise ValueError("Batch size must be even for balanced sampling")

        self.pos_indices = dataset.pos_indices
        self.neg_indices = dataset.neg_indices
        self.samples_per_class = batch_size // 2

        # Calculate number of batches we can make
        max_pos_batches = len(self.pos_indices) // self.samples_per_class
        max_neg_batches = len(self.neg_indices) // self.samples_per_class
        self.num_batches = min(max_pos_batches, max_neg_batches)

    def __iter__(self):
        # Shuffle indices for each epoch
        pos_shuffled = np.random.permutation(self.pos_indices)
        neg_shuffled = np.random.permutation(self.neg_indices)

        for i in range(self.num_batches):
            # Get half positive, half negative
            pos_batch = pos_shuffled[i * self.samples_per_class:(i + 1) * self.samples_per_class]
            neg_batch = neg_shuffled[i * self.samples_per_class:(i + 1) * self.samples_per_class]

            # Combine and shuffle within batch
            batch_indices = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch_indices)

            yield from batch_indices

    def __len__(self):
        return self.num_batches * self.batch_size

