# import libraries
import os
import torch
from PIL import Image
from torchvision import datasets, transforms

from utils import load_model, observe_latent_space_color
from model import Generator

# Directory where the raw MNIST dataset will be downloaded
os.makedirs('data', exist_ok=True)

print("Loading MNIST dataset...")
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)



# ===================== Saving the MNIST images in .png format =====================
# Saving the original images in .png format to compute the FID score
# print("Saving MNIST images in .png format...")

# # Directory where the real images will be saved
# real_images_dir = "real_images"
# os.makedirs(real_images_dir, exist_ok=True)

# # Save MNIST images as .png files
# for i, (image, _) in enumerate(train_dataset):
#     img = transforms.ToPILImage()(image) 
#     img.save(f"{real_images_dir}/mnist_{i}.png")

# print(f"Saved {i + 1} real MNIST images to {real_images_dir}")



# ===================== Observing the Latent Space =====================
print('Model Loading...')
# Model Pipeline
mnist_dim = 784

model = Generator(g_output_dim = mnist_dim).cuda()
model = load_model(model, 'checkpoints')
model = torch.nn.DataParallel(model).cuda()
model.eval()

print('Model loaded.')

observe_latent_space_color(model, train_loader, n_samples=1000, method='pca')