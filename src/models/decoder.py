import torch.nn as nn

class FaceDecoder(nn.Module):
    """Decoder network for reconstructing face images from latent codes."""

    def __init__(self, latent_dim, n_filters, out_channels):
        super(FaceDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.n_filters = n_filters

        # Linear (fully connected) layer to project from latent space
        # to a 4 x 4 feature map with (6*n_filters) channels
        self.linear = nn.Sequential(
            nn.Linear(self.latent_dim, 4 * 4 * 6 * self.n_filters),
            nn.ReLU()
        )

        # Transposed convolutional layers for upsampling
        self.deconv = nn.Sequential(
            # [B, 6n_filters, 4, 4] -> [B, 4n_filters, 8, 8]
            nn.ConvTranspose2d(
                in_channels=6 * self.n_filters,
                out_channels=4 * self.n_filters,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            # [B, 4n_filters, 8, 8] -> [B, 2n_filters, 16, 16]
            nn.ConvTranspose2d(
                in_channels=4 * self.n_filters,
                out_channels=2 * self.n_filters,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(),
            # [B, 2n_filters, 16, 16] -> [B, n_filters, 32, 32]
            nn.ConvTranspose2d(
                in_channels=2 * self.n_filters,
                out_channels=self.n_filters,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.ReLU(),
            # [B, n_filters, 32, 32] -> [B, out_channels, 64, 64]
            nn.ConvTranspose2d(
                in_channels=self.n_filters,
                out_channels=out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            # Get values in the range (0, 1)
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (Tensor): Latent codes of shape [batch_size, latent_dim].

        Returns:
            Tensor of shape [batch_size, out_channels, 64, 64], representing reconstructed images.
        """
        x = self.linear(z)  # [B, 4*4*6*n_filters]
        x = x.view(-1, 6 * self.n_filters, 4, 4)  # [B, 6n_filters, 4, 4]
        x = self.deconv(x)  # [B, 3, 64, 64]
        return x

