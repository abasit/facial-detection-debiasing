import torch
import torch.nn as nn

from .cnn import StandardCNN
from .decoder import FaceDecoder

class DBVAE(nn.Module):
    """Debiasing Variational Autoencoder."""

    def __init__(self, in_channels, img_h, img_w, n_filters, hidden_dim, latent_dim, out_channels):
        super(DBVAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder: outputs classification + latent parameters
        # n_outputs = 1 (classification) + 2*latent_dim (mean and log_var)
        self.encoder = StandardCNN(
            in_channels=in_channels, img_h=img_h, img_w=img_w,
            n_filters=n_filters, hidden_dim=hidden_dim,
            n_outputs=2 * latent_dim + 1
        )

        # Decoder: reconstructs images from latent codes
        self.decoder = FaceDecoder(
            latent_dim=latent_dim, n_filters=n_filters, out_channels=out_channels
        )

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        """
        VAE reparameterization: given mean and logvar, sample latent variables.
        reparameterization trick by sampling from an isotropic unit Gaussian.

        Args:
            z_mean, z_logvar (tensor): mean and log of variance of latent distribution (Q(z|X))

        Returns:
            z (tensor): sampled latent vector
        """
        # Generate random noise with the same shape as z_mean
        # sampled from a standard normal distribution (mean=0, std=1)
        eps = torch.randn_like(z_mean)

        # Reparameterization: z = μ + σ * ε = μ + e^(log σ) * ε
        z = z_mean + torch.exp(0.5 * z_logvar) * eps

        return z

    def encode(self, x):
        """
        Encode input images to classification logits and latent distribution parameters.

        Args:
            x (Tensor): Input images of shape [batch_size, channels, height, width]

        Returns:
            y_logit (Tensor): Classification logits
            z_mean (Tensor): Mean of latent distribution
            z_logvar (Tensor): Log variance of latent distribution
        """
        encoder_output = self.encoder(x)

        # Split encoder output
        y_logit = encoder_output[:, 0].unsqueeze(-1)  # Classification prediction
        z_mean = encoder_output[:, 1:self.latent_dim + 1]  # Latent means
        z_logvar = encoder_output[:, self.latent_dim + 1:]  # Latent log var

        return y_logit, z_mean, z_logvar

    def decode(self, z):
        """Decode latent space and output reconstruction."""
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass through the complete VAE.

        Args:
            x (Tensor): Input images

        Returns:
            y_logit (Tensor): Classification logits
            z_mean (Tensor): Latent means
            z_logvar (Tensor): Latent log variance
            recon (Tensor): Reconstructed images
        """
        # Encode input to a prediction and latent space
        y_logit, z_mean, z_logvar = self.encode(x)

        # Reparameterization
        z = self.reparameterize(z_mean, z_logvar)

        # Reconstruction
        recon = self.decode(z)

        return y_logit, z_mean, z_logvar, recon

    def predict(self, x):
        """Predict face or not face logit for given input x."""
        y_logit, _, _ = self.encode(x)
        return y_logit

    def get_device(self):
        """Get the device this model is on."""
        return next(self.parameters()).device
