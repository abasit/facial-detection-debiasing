import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.constants import DEMOGRAPHIC_GROUPS, DEFAULT_CONFIG
from src.models.dbvae import DBVAE
from src.utils.training import *
from src.utils.loss_functions import debiasing_loss_function

def get_latent_mu(images, dbvae, batch_size=64):
    """Get latent variable means for a batch of images."""
    device = dbvae.get_device()
    dbvae.eval()
    all_z_mean = []

    # Convert to tensor if needed
    images_t = torch.from_numpy(images).float()

    with torch.inference_mode():
        for start in range(0, len(images_t), batch_size):
            end = start + batch_size
            batch = images_t[start:end]
            batch = batch.permute(0, 3, 1, 2)
            batch = batch.to(device)

            # Forward pass on this chunk only
            _, z_mean, _ = dbvae.encode(batch)
            all_z_mean.append(z_mean.cpu())

    # Concatenate all results
    z_mean_full = torch.cat(all_z_mean, dim=0)
    return z_mean_full.numpy()

def get_training_sample_probabilities(images, dbvae, latent_dim, bins=10, smoothing_fac=0.001):
    """
    Compute adaptive sampling probabilities for debiasing facial detection training.

    This is the core debiasing algorithm that identifies underrepresented features
    in the latent space and increases their sampling probability during training.
    The method automatically discovers demographic biases without manual annotation.

    Algorithm:
    1. Encode all face images to latent space using trained VAE encoder
    2. For each latent dimension, create histogram of feature distribution
    3. Smooth histogram and invert density (rare features → high probability)
    4. Combine probabilities across all latent dimensions (taking maximum)
    5. Normalize to get final sampling probabilities

    Args:
        images (numpy.ndarray): Face images of shape (N, H, W, C)
        dbvae (DBVAE): Trained DB-VAE model with encoder
        latent_dim (int): Dimensionality of the latent space
        bins (int, optional): Number of histogram bins for distribution analysis.
                             More bins = finer granularity. Defaults to 10.
        smoothing_fac (float, optional): Smoothing factor to prevent division by zero
                                       and extreme reweighting. Defaults to 0.001.

    Returns:
        numpy.ndarray: Sampling probabilities for each input image, shape (N,).
                      Higher values indicate images with rarer features that should
                      be sampled more frequently during training.

    Note:
        This function is called each epoch during training as the learnt latent
        representations improve over time.
    """
    print("Recomputing the sampling probabilities")

    # Run the input batch and get the latent variable means
    mu = get_latent_mu(images, dbvae)

    # Initialize sampling probabilities
    training_sample_p = np.zeros(mu.shape[0], dtype=np.float64)

    # Consider distribution for each latent variable
    for i in range(latent_dim):
        latent_distribution = mu[:, i]

        # Generate histogram of latent distribution
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)

        # find which latent bin each data sample falls in
        bin_edges[0] = -float("inf")
        bin_edges[-1] = float("inf")
        bin_idx = np.digitize(latent_distribution, bin_edges)

        # Smooth the density function
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density / np.sum(hist_smoothed_density)

        # Invert density function for resampling
        p = 1.0 / (hist_smoothed_density[bin_idx - 1])
        p = p / np.sum(p)  # Normalize

        # Update sampling probabilities
        training_sample_p = np.maximum(training_sample_p, p)

    # Final normalization
    training_sample_p /= np.sum(training_sample_p)

    return training_sample_p


def train_one_epoch(model, dataset, face_images, optimizer, config):
    """Train the DB-VAE model for one epoch with adaptive resampling."""
    device = model.get_device()
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    # Recompute sampling probabilities for this epoch
    p_faces = get_training_sample_probabilities(face_images, model, config.latent_dim)

    # Calculate number of batches per epoch
    num_batches = len(dataset) // config.batch_size

    pbar = tqdm(range(num_batches), desc='Training')
    for batch_idx in pbar:
        # Get batch with adaptive sampling
        images, labels = dataset.get_batch_with_probabilities(config.batch_size, p_pos=p_faces)
        images = torch.from_numpy(images).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)

        # Forward pass
        optimizer.zero_grad()
        y_logit, z_mean, z_logvar, x_recon = model(images)

        # Compute loss
        loss, classification_loss = debiasing_loss_function(
            images, x_recon, labels, y_logit, z_mean, z_logvar, config.kl_weight
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        epoch_loss += loss.item()
        predicted = torch.round(torch.sigmoid(y_logit.squeeze()))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = epoch_loss / num_batches
    accuracy = 100 * correct / total

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

if __name__ == '__main__':
    # Parse arguments and load config file
    parser = argparse.ArgumentParser(description='Train DB-VAE for debiased facial detection')
    parser.add_argument('--config', type=str,
                        default=DEFAULT_CONFIG,
                        help='Path to a YAML configuration file')

    cmd_args = parser.parse_args()
    args = load_config(cmd_args.config)

    # Make sure we have some necessary parameters
    required_params = [
        'num_epochs', 'batch_size', 'lr',
        'img_channels', 'img_height', 'img_width', 'n_filters', 'hidden_dim',
        'latent_dim', 'kl_weight',
        'data_dir', 'results_dir', 'eval_samples'
    ]
    validate_required_params(args, required_params)

    dataset, device, data_file, faces_root = setup_data_and_device(args)
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataset) // args.batch_size}")

    # Create model and optimizer
    model = DBVAE(
        in_channels=args.img_channels,
        img_h=args.img_height,
        img_w=args.img_width,
        n_filters=args.n_filters,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        out_channels=args.img_channels
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Get all face images for adaptive resampling
    print("Loading face images for adaptive resampling...")
    face_images = dataset.get_all_face_images()

    # Training loop
    for epoch in range(args.num_epochs):
        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        metrics = train_one_epoch(model, dataset, face_images, optimizer, args)
        print(f'Loss: {metrics["loss"]:.4f}, Accuracy: {metrics["accuracy"]:.2f}%')

        # Model Evaluation
        train_acc, demographic_results = run_evaluation(
            model, dataset, faces_root, args, DEMOGRAPHIC_GROUPS, "DB-VAE"
        )
        print_evaluation_results(demographic_results, "DB-VAE")

    # Save model
    save_model(model, f'{args.results_dir}/dbvae.pth')

