# import libraries
import os
from PIL import Image
from torchvision import datasets, transforms

# Directory where the MNIST dataset will be downloaded
os.makedirs('data', exist_ok=True)

print("Loading MNIST dataset...")
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)


# Saving the original images in .png format to compute the FID score
print("Saving MNIST images in .png format...")

# Directory where the real images will be saved
real_images_dir = "real_images"
os.makedirs(real_images_dir, exist_ok=True)

# Save MNIST images as .png files
for i, (image, _) in enumerate(train_dataset):
    img = transforms.ToPILImage()(image) 
    img.save(f"{real_images_dir}/mnist_{i}.png")

print(f"Saved {i + 1} real MNIST images to {real_images_dir}")


