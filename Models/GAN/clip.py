import os
from PIL import Image
import torch
from torchvision import transforms

def load_and_clip_images_from_folder(folder_path: str, output_folder: str, clip_min: float = -1.0, clip_max: float = 1.0):
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define the transformation to convert images to tensor
    transform_to_tensor = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and scales pixel values to [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Scales pixel values to [-1, 1] for RGB images
    ])

    # Define the reverse transformation to save back the image
    transform_to_image = transforms.Compose([
        transforms.Normalize(mean=[-1], std=[1/0.5]),  # Reverse normalization to [0, 1]
        transforms.ToPILImage()  # Converts tensor back to PIL image
    ])

    # Loop through each image in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img_path = os.path.join(folder_path, filename)
            
            # Load image
            image = Image.open(img_path).convert("L")
            
            # Convert image to tensor and normalize to [-1, 1]
            image_tensor = transform_to_tensor(image)

            # Apply clipping operation
            clipped_tensor = torch.clamp(image_tensor, clip_min, clip_max)

            # Convert back to image (reverse normalization)
            clipped_image = transform_to_image(clipped_tensor)

            # Save the processed image
            output_image_path = os.path.join(output_folder, filename)
            clipped_image.save(output_image_path)

            print(f"Processed and saved: {output_image_path}")

# Example usage:
# Specify the folder containing images and the folder to save the clipped images.
input_folder = './Models/GAN/img'
output_folder = './Models/GAN/clipped_img'

# Call the function to process the images
load_and_clip_images_from_folder(input_folder, output_folder)