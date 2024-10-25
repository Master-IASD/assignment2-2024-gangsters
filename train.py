import torch 
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid


from model import Generator, Discriminator
from utils import D_train, G_train, WD_train, WG_train, save_models, noise_generation




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0001,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--mode", type=str, default='GAN')

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
    G = Generator(g_output_dim = mnist_dim).cuda()
    D = Discriminator(mnist_dim).cuda()

    # model = DataParallel(model).cuda()
    print('Model loaded.')
    # Optimizer 
    

    # define optimizers
    G_optimizer = optim.Adam(G.parameters(), lr = args.lr,betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr = args.lr,betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    gpw = 10

    print('Start Training :')
    
    n_epoch = args.epochs
    for epoch in trange(1, n_epoch+1, leave=True):           
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, mnist_dim)
            
            if args.mode == "WGAN":
                if batch_idx % 2 == 0:
                    Dloss, Dlipp = WD_train(x, G, D, D_optimizer, gpw)
                
                if batch_idx % 1 == 0:
                    Gloss = WG_train(x, G, D, G_optimizer)
                
            else:
                Dloss = D_train(x, G, D, D_optimizer, criterion)
                Gloss = G_train(x, G, D, G_optimizer, criterion)
                
        if args.mode == "WGAN":
            print(f'Epoch: {epoch}, D_loss: {Dloss}, G_loss: {Gloss}, D_lipp: {Dlipp}')
        else:
            print(f'Epoch: {epoch}, D_loss: {Dloss}, G_loss: {Gloss}')
            
        # Generate images
        with torch.no_grad():
            z = noise_generation(16, 100)
        grid = make_grid(G(z).view(-1, 1, 28, 28).cpu(), nrow=4, normalize=True)
        print(f"predicted image class: {D(G(z)).argmax(1).detach().cpu().numpy()}")
        save_image(grid, f'images/{args.mode}_{epoch}.png')
            
        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')
                
    print('Training done')

        
