"""
Extract image features using the MONET fine-tuned CLIP ViT-L/14 backbone.

Downloads MONET weights automatically from:
    https://aimslab.cs.washington.edu/MONET/weight_clip.pt

Saves features as an .npz file with keys:
    - embeddings : (N, 768) float32
    - isic_ids   : (N,) str

Usage
-----
    # Default: extract for all ISIC2024 images
    python scripts/extract_monet_features.py

    # Custom paths
    python scripts/extract_monet_features.py \
        --image-dir /path/to/images \
        --metadata  /path/to/final_metadata_all.csv \
        --output    /path/to/monet_features.npz \
        --device    cuda:0 \
        --batch-size 128
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

# Add MONET src to path so we can import clip
MONET_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MONET_ROOT / "src"))

import clip  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    """Simple dataset: reads images from a directory, returns (image_tensor, isic_id)."""

    def __init__(self, image_paths: dict, transform):
        self.ids = list(image_paths.keys())
        self.paths = [image_paths[k] for k in self.ids]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.ids[idx]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, dataloader, device):
    """Run all images through the CLIP visual encoder and collect embeddings."""
    model.eval()
    all_features = []
    all_ids = []

    for images, ids in tqdm(dataloader, desc="Extracting features"):
        features = model.encode_image(images.to(device)).float()
        # L2-normalize (same convention used elsewhere in MONET)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy())
        all_ids.extend(ids)

    embeddings = np.concatenate(all_features, axis=0)
    isic_ids = np.array(all_ids)
    return embeddings, isic_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract MONET ViT-L/14 features for ISIC2024 images."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/home/mecha109/PanDerm/Evaluation_datasets/processed_images_224x224",
        help="Directory containing .jpg images (filenames = ISIC IDs).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(MONET_ROOT / "data" / "isic2024" / "final_metadata_all.csv"),
        help="Path to final_metadata_all.csv. If provided, only images listed in "
             "the metadata are processed. Pass 'none' to process every image in --image-dir.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MONET_ROOT / "data" / "isic2024" / "monet_features.npz"),
        help="Output .npz path.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Build image path dict  {isic_id: full_path}
    # ------------------------------------------------------------------
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        sys.exit(f"Image directory not found: {image_dir}")

    all_images = {p.stem: str(p) for p in sorted(image_dir.glob("*.jpg"))}
    print(f"Found {len(all_images)} images in {image_dir}")

    if args.metadata.lower() != "none":
        meta = pd.read_csv(args.metadata, low_memory=False)
        valid_ids = set(meta["isic_id"].astype(str))
        image_paths = {k: v for k, v in all_images.items() if k in valid_ids}
        print(f"Filtered to {len(image_paths)} images matching metadata")
    else:
        image_paths = all_images

    if len(image_paths) == 0:
        sys.exit("No images to process.")

    # ------------------------------------------------------------------
    # 2. Load MONET ViT-L/14
    # ------------------------------------------------------------------
    print("Loading CLIP ViT-L/14 ...")
    clip_model, _ = clip.load("ViT-L/14", device="cpu", jit=False)

    print("Downloading MONET weights ...")
    state_dict = torch.hub.load_state_dict_from_url(
        "https://aimslab.cs.washington.edu/MONET/weight_clip.pt",
        map_location="cpu",
    )
    clip_model.load_state_dict(state_dict)
    clip_model = clip_model.to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # 3. Transforms (CLIP standard preprocessing for ViT-L/14 @ 224px)
    # ------------------------------------------------------------------
    transform = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    dataset = ImageFolderDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4. Extract & save
    # ------------------------------------------------------------------
    embeddings, isic_ids = extract_features(clip_model, dataloader, device)
    print(f"Embeddings shape: {embeddings.shape}")  # (N, 768)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), embeddings=embeddings, isic_ids=isic_ids)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
