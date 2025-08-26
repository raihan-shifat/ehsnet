import argparse
import os
import random
import glob
from pathlib import Path

def make_split(root, out_dir, seed=42, val_ratio=0.15, test_ratio=0.15):
    # Find all images in the images directory
    imgs = sorted(glob.glob(os.path.join(root, "images", "*")))
    assert imgs, f"No images found under {root}/images"
    
    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(imgs)
    
    # Calculate split sizes
    n = len(imgs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    
    # Split the dataset
    test, val, train = imgs[:n_test], imgs[n_test:n_test+n_val], imgs[n_test+n_val:]
    
    # Create output directory if it doesn't exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Write splits to text files (not JSON)
    for name, lst in [("train.txt", train), ("val.txt", val), ("test.txt", test)]:
        with open(os.path.join(out_dir, name), "w") as f:
            # Extract just the image name without path and extension
            for img_path in lst:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                f.write(f"{img_name}\n")
    
    print(f"Saved splits to {out_dir} | train {len(train)} val {len(val)} test {len(test)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="medical image/data/ISIC2017/train/images")
    ap.add_argument("--out", required=True, help="data/ISIC17")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    args = ap.parse_args()
    make_split(args.root, args.out, args.seed, args.val_ratio, args.test_ratio)