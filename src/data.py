from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF


def load_folder(folder: Path, image_size: int) -> torch.Tensor:
    """Images load as a tensor in the form (image_count, pixel_range (0-1), image_size, image_size)"""

    paths = sorted(Path(folder).glob("*.jpeg"))
    
    image_tensors = []
    for path in paths:
        image = Image.open(path).convert("L") # Load and convert to greyscale.
        image = TF.resize(image, image_size)
        image = TF.center_crop(image, image_size)
        image_tensors.append(TF.to_tensor(image))

    return torch.stack(image_tensors) # Join list of image tensors into a single tensor
