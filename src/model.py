import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_PIXELS = 128 * 128


class VAE(nn.Module):
    """Fully connected VAE that turns a 128x128 image into a z vector and back into an image."""

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: compresses the image into a hidden vector
        self.encoder = nn.Sequential(
            nn.Flatten(), # Turn the image into one long row of pixels
            nn.Linear(IMAGE_PIXELS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Layers that output the mean and log variance of the z vector
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder: turns a z vector back into an image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, IMAGE_PIXELS),
            nn.Sigmoid(), # Keep pixel values between 0 and 1
            nn.Unflatten(1, (1, 128, 128)), # Turn the row of pixels back into an image
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)

        std_dev = torch.sqrt(torch.exp(log_var))
        noise = torch.randn_like(std_dev)
        z = mu + noise * std_dev # Create z vector

        decoded = self.decoder(z)

        return decoded, mu, log_var

    def sample(self, num_samples: int, device) -> torch.Tensor:
        """Generate sample by decoding a random Z vector."""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)
            samples = self.decoder(z)
        return samples

def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float):
    """Total loss is the reconstruction loss plus beta times the KL divergence. Returns (loss, recon, kl)."""
    # Reconstruction loss: how different the output image is from the real image
    reconstruction_loss = F.mse_loss(x_hat, x, reduction="sum") / len(x) # Squared error summed over each image's pixels

    # KL divergence: how far the z vectors drift from a standard Gaussian
    kl_terms = 1 + log_var - mu ** 2 - torch.exp(log_var)
    kl = -0.5 * kl_terms.sum() / len(x)

    loss = reconstruction_loss + beta * kl
    return loss, reconstruction_loss, kl
