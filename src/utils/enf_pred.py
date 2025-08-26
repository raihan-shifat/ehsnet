import sys
import os

# ADD THIS AT THE VERY TOP - BEFORE ANY OTHER IMPORTS
# Add the parent directory (src) to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

# Now you can import your modules
import torch
import cv2
import numpy as np
from models.ehsnet import EHSNet
from data.isic_dataset import ISICDataset
from torchvision import transforms

# Rest of your code...
print("All imports successful!")
# rest of your imports...


def predict_and_save_mask(model, image_path, output_dir, threshold=0.5):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Apply necessary transformations (resize, normalization, etc.)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),  # Resize to the model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Apply transformations
    image_tensor = transform(image_rgb).unsqueeze(0).cuda()  # Add batch dimension and send to GPU
    
    # Perform inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)  # Get model output
        output = torch.sigmoid(output)  # Apply sigmoid activation
        mask = (output > threshold).cpu().numpy().squeeze()  # Apply threshold and remove batch dimension

    # Save the predicted mask
    mask = mask * 255  # Convert mask to 0-255 range for visibility
    mask = mask.astype(np.uint8)  # Ensure it's of type uint8
    mask_filename = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '_mask.png'))
    cv2.imwrite(mask_filename, mask)  # Save mask
    print(f"Mask saved to {mask_filename}")

def main(config_path, checkpoint_path, image_path, output_dir):
    # Set device (cuda or cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = EHSNet(num_classes=1, pretrained=False).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Make the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict and save the mask for the given image
    predict_and_save_mask(model, image_path, output_dir)

if __name__ == '__main__':
    # Paths
    config_path = "configs/ehsnet.yaml"  # If required
    checkpoint_path = "D:/Shifat Raihan/mimage/medical image/runs/isic18_ehsnet/best_model.pth"
    image_path = "D:/Shifat Raihan/mimage/medical image/output_masks/our test img 2017/img/ISIC_0000350.jpg" # Path to the input image
    output_dir = "output_masks"  # Directory to save the masks

    # Call the main function to perform inference and save the mask
    main(config_path, checkpoint_path, image_path, output_dir)
