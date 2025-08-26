# EHSNet: Efficient Hybrid Segmentation Network

EHSNet is a deep learning model for semantic segmentation, focused on medical image tasks. It uses an EfficientNet-B4 encoder, hybrid attention decoder, multi-scale context modules, and a custom hybrid loss function (Dice + Focal + Boundary Loss) for high accuracy in segmenting medical images like those from the ISIC dataset.

This repository implements the model, training pipeline, evaluation, and inference for datasets like ISIC 2017 and 2018.

## Requirements

- Python 3.12 
- Dependencies (from `requirements.txt`):
  ```
  torch>=2.2
  torchvision>=0.17
  albumentations>=1.4.0
  opencv-python>=4.8
  numpy
  tqdm
  pyyaml
  scikit-image
  tensorboard
  ```

## Installation

1. Create a virtual environment:
   ```
   py -3.12 -m venv mimage1 
   ```

2. Activate the environment (on Windows):
   ```
   .\mimage1\Scripts\Activate.ps1
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset

Download the ISIC datasets (e.g., ISIC 2017 or 2018) from: [https://challenge.isic-archive.com/data/](https://challenge.isic-archive.com/data/)

Place the dataset in a directory like `data/ISIC2017/train` (adjust paths in commands as needed). Images should be resized to 256x256 as mentioned in the report.

## Usage

### 1. Prepare Dataset Split
Split the dataset into train/val/test sets (70%/15%/15%) and save to `.txt` files.

Example command (adjust paths):
```
python prepare_split.py --root "data/ISIC2017/train" --out "src/data/output"
```

### 2. Train the Model
Train using the config file.

```
python src/train.py --config configs/ehsnet.yaml
```

- Monitors DSC and mIoU; saves best model to `runs/isicXX_ehsnet/best_model.pth`.
- Uses data augmentation (flips, rotations, etc.) via Albumentations.

### 3. Evaluate the Model
Evaluate on test set.

For ISIC 2017:
```
python src/evaluate.py --config configs/ehsnet.yaml --checkpoint runs/isic17_ehsnet/best_model.pth
```

For ISIC 2018:
```
python src/evaluate.py --config configs/ehsnet.yaml --checkpoint runs/isic18_ehsnet/best_model.pth
```

Example Results:
- ISIC 2017: DSC 0.8732, mIoU 0.8028, Acc 0.9687, Spe 0.9785, Sen 0.8793
- ISIC 2018: DSC 0.8608, mIoU 0.7824, Acc 0.9542, Spe 0.9815, Sen 0.8679

### 4. Inference (Predict on New Images)
Run predictions on unseen images.

```
python src/utils/enf_pred.py
```

- Loads model and input image, outputs segmentation mask.

## Future Work
- Test on more datasets.
- Optimize for real-time inference.
- Integrate with clinical systems.

## Citation

If you use this code, please cite this repository. Add your paper once available.

```bibtex
@software{EHSNet_2025,
  title        = {EHSNet: Efficient Hybrid Segmentation Network},
  author       = {Raihan Shifat},
  year         = {2025},
  url          = {https://github.com/raihan-shifat/ehsnet}
}
```
