import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

classes = ["normal_step", "sticker_or_callout", "true_bag_start"]
DEFAULT_IMAGES = [
    Path("training_data_crops/true_bag_start/21330_21330_01_page_013_top_left.png"),
    Path("training_data_crops/sticker_or_callout/21330_21330_01_page_110_top_left.png"),
    Path("training_data_crops/normal_step/21330_21330_01_page_181_top_left.png"),
]

# rebuild same model structure as training
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# load saved weights
state_dict = torch.load("bag_classifier.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# same transform as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]

    print(f"\nIMAGE {image_path}")
    for i, p in enumerate(probs):
        print(f"{classes[i]}: {p.item():.3f}")

    print("PRED:", classes[int(probs.argmax())])
    print("-" * 40)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict labels for crop images using the trained bag classifier.")
    parser.add_argument("image_paths", nargs="*", help="Crop image paths to classify.")
    args = parser.parse_args()

    image_paths = [Path(path) for path in args.image_paths] if args.image_paths else DEFAULT_IMAGES
    for image_path in image_paths:
        if not image_path.is_file():
            print(f"Skipping missing image: {image_path}")
            continue
        predict(image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
