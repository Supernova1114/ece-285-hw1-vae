import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore
from torchvision.transforms.functional import normalize

from data import load_folder
from model import VAE
from train import DATA_DIR, LATENT_DIM, OUT_DIR

BATCH_SIZE = 50


def to_inception(images: torch.Tensor) -> torch.Tensor:
    """Turn grayscale 128x128 images into the color 299x299 images the Inception network expects."""
    images = F.interpolate(images, size=299, mode="bilinear") # Resize to 299x299
    images = images.repeat(1, 3, 1, 1) # Copy the gray channel into red, green and blue
    images = normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Same color scaling the Inception network was trained with
    return images


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(OUT_DIR / "vae.pt", map_location=device)) # Load the trained weights
    model.eval()

    real_images = load_folder(DATA_DIR / "train" / "NORMAL", 128).to(device)

    fid = FID(device=device)
    inception_score = InceptionScore(device=device)

    for real_batch in real_images.split(BATCH_SIZE):
        fake_batch = model.sample(len(real_batch), device) # Same number of generated images as real ones

        fake_batch = to_inception(fake_batch)
        real_batch = to_inception(real_batch)

        fid.update((fake_batch, real_batch))
        inception_score.update(fake_batch)

    print(f"FID: {fid.compute():.4f}")
    print(f"IS: {inception_score.compute():.4f}")


if __name__ == "__main__":
    evaluate()
