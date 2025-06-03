import torch
import torch.nn.functional as F

def vae_loss_function(x, x_recon, mu, logvar, kl_weight):
    """
    Calculate VAE loss given input, reconstruction, and latent parameters.

    Args:
        x (Tensor): Original input images
        x_recon (Tensor): Reconstructed images
        mu (Tensor): Latent means
        logvar (Tensor): Latent log variation
        kl_weight (float): Weight for KL divergence term

    Returns:
        Tensor: VAE loss
    """
    # KL divergence loss: D_KL(q(z|x) || p(z))
    # For unit Gaussian prior: KL = 0.5 * sum(exp(log_var) + μ² - 1 - log_var)
    latent_loss = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, dim=1)

    # Reconstruction loss: L1 norm between input and reconstruction
    reconstruction_loss = torch.mean(torch.abs(x - x_recon), dim=(1, 2, 3))

    # Combined VAE loss
    vae_loss = kl_weight * latent_loss + reconstruction_loss

    return vae_loss


def debiasing_loss_function(x, x_pred, y, y_logit, mu, logvar, kl_weight):
    """
    Loss function for DB-VAE combining classification and VAE losses.

    Args:
        x (Tensor): True input images
        x_pred (Tensor): Reconstructed images
        y (Tensor): True labels (face or not face)
        y_logit (Tensor): Predicted classification logits
        mu (Tensor): Latent means
        logvar (Tensor): Latent log variance
        kl_weight (float): Weight for KL divergence term

    Returns:
        total_loss (Tensor): Combined DB-VAE loss
        classification_loss (Tensor): Classification component only
    """
    # VAE loss (reconstruction + KL divergence)
    vae_loss = vae_loss_function(x, x_pred, mu, logvar, kl_weight)

    # Classification loss using binary cross-entropy
    y = y.float()
    classification_loss = F.binary_cross_entropy_with_logits(
        y_logit.squeeze(), y, reduction="none"
    )

    # Face indicator: 1 for faces, 0 for non-faces
    # Only apply VAE loss to face images
    face_indicator = (y == 1.0).float()

    # Total loss: classification loss + VAE loss (only for faces)
    total_loss = torch.mean(classification_loss + face_indicator * vae_loss)

    return total_loss, classification_loss

