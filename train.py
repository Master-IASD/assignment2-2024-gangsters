import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid


from model import Generator, Discriminator
from utils import D_train, G_train, save_models


# ============= Additional imports ==============
from PIL import Image
from perceptual import PerceptualLoss
from utils import G_train_perceptual



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")

    args = parser.parse_args()


    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Data Pipeline
    print('Dataset loading...')
    # MNIST Dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=args.batch_size, shuffle=False)
    print('Dataset Loaded.')


    print('Model Loading...')
    mnist_dim = 784
    G = torch.nn.DataParallel(Generator(g_output_dim = mnist_dim)).cuda()
    D = torch.nn.DataParallel(Discriminator(mnist_dim)).cuda()


    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 



    # define loss
    criterion = nn.BCELoss() 

    # define perceptual loss
    perceptual_loss_fn = PerceptualLoss() #.cuda()  

    # Try training with the hinge losss
    criterion2 = nn.HingeEmbeddingLoss()


    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr)

    # print("device", torch.cuda.current_device())

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            D_train(x, G, D, D_optimizer, criterion2)

            # === Classic Generator Training Function ===
            # G_train(x, G, D, G_optimizer, criterion2)

            # === Generator Training Function with Perceptual Loss ===
            G_loss = G_train_perceptual(x, G, D, G_optimizer, criterion, perceptual_loss_fn)

            with torch.no_grad():
                z = torch.randn(16, 20).cuda()
            grid = make_grid(G(z).view(-1, 1, 28, 28).cpu(), nrow=4, normalize=True)
            save_image(grid, f"images/{epoch}.png")

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints', epoch)
                
    print('Training done')

        
