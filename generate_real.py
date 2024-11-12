import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

if __name__ == '__main__':
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    
    for i, (images, _) in enumerate(train_loader):
        save_image(images, f'real_images/{i}.png')
        if i == 10000:
            break