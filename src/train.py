from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import load_folder
from model import VAE, vae_loss

DATA_DIR = Path("/home/cameron/workspaces/datasets/chest_xray") # Folder containing train/ and test/
OUT_DIR = Path("runs/vae_results")
EPOCHS = 300
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BETA = 1.0 # Weight on the KL divergence
LATENT_DIM = 256 # Size of the z vector


def plot_losses(epoch_losses: list[float], title: str, path: Path):
    """Save a graph of one loss over the epochs for the report."""
    epochs = range(1, len(epoch_losses) + 1)

    plt.figure() # Start a new graph so the lines do not pile onto the last one
    plt.plot(epochs, epoch_losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss per image")

    plt.savefig(path, dpi=150)
    plt.close()


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_images = load_folder(DATA_DIR / "train" / "NORMAL", 128)
    train_loader = DataLoader(train_images, batch_size=BATCH_SIZE, shuffle=True) # Shuffled batches of images

    model = VAE(LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    epoch_losses = []
    epoch_kls = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_kl = 0.0

        for data in train_loader:
            data = data.to(device) # Move the batch to the GPU

            decoded, mu, log_var = model(data)

            loss, kl = vae_loss(data, decoded, mu, log_var, BETA)

            optimizer.zero_grad() # Clear the old gradients
            loss.backward() # Compute new gradients for every weight
            optimizer.step() # Nudge each weight using its gradient

            total_loss += loss.item() * data.size(0) # Weight by batch size, since the last batch is smaller
            total_kl += kl.item() * data.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)

        epoch_kl = total_kl / len(train_loader.dataset)
        epoch_kls.append(epoch_kl)

        print(f"Epoch {epoch + 1}/{EPOCHS}: loss={epoch_loss:.4f} kl={epoch_kl:.4f}")

    torch.save(model.state_dict(), OUT_DIR / "vae.pt")

    plot_losses(epoch_losses, "Training loss", OUT_DIR / "loss_curve.png")
    plot_losses(epoch_kls, "KL divergence", OUT_DIR / "kl_curve.png")


if __name__ == "__main__":
    train()
