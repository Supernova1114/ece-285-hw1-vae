import torch
from torchvision.utils import save_image

from model import VAE
from train import LATENT_DIM, OUT_DIR

NUM_IMAGES = 1000
GENERATED_DIR = OUT_DIR / "generated"


def generate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    model = VAE(LATENT_DIM).to(device)
    model.load_state_dict(torch.load(OUT_DIR / "vae.pt", map_location=device)) # Load the trained weights
    model.eval()

    images = model.sample(NUM_IMAGES, device)

    for i, image in enumerate(images):
        save_image(image, GENERATED_DIR / f"generated_{i:04d}.png")

    save_image(images[:64], OUT_DIR / "generated_grid.png", nrow=8) # 8x8 preview grid for the report


if __name__ == "__main__":
    generate()
