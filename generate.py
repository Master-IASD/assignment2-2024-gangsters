import torch 
import torchvision
import os
import argparse
import numpy as np

from model import Generator
from utils import load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("-d", "--dim", default=None, type=int)
    args = parser.parse_args()

    if args.dim== None:
        args.dim = ""

    gmm_params = np.load(f"checkpoints{args.dim}/gmm_parameters.npz")
    # Access the parameters
    means = gmm_params["means"][:,:-10]
    covariances = gmm_params["covariances"][:,:-10,:-10]
    clusters = gmm_params["clusters"]
    latent_dim = gmm_params["latent_dim"]

    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    model = Generator(latent_dim + 10, g_output_dim = mnist_dim).cuda()
    model = load_model(model, f'checkpoints{args.dim}')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')

    print('Start Generating')
    os.makedirs('samples', exist_ok=True)
    
    n_samples = 0
    for cluster in range(len(means)):
        # Sample latent vectors from each Gaussian distribution
        latent_vectors = np.random.multivariate_normal(
        mean=means[cluster], 
        cov=covariances[cluster], 
        size=1000
        )

        # Convert to tensor and move to device
        latent_vectors = torch.tensor(latent_vectors, dtype=torch.float32).cuda()
        one_hot = torch.zeros((latent_vectors.shape[0], 10)).cuda()
        one_hot[:,clusters[cluster]] = 1
        latent_vectors = torch.cat((latent_vectors, one_hot), dim=1)

        # Generate samples using the generator
        with torch.no_grad():
            generated_images = model(latent_vectors).view(-1, 1, 28, 28)
        for k in range(generated_images.shape[0]):
            torchvision.utils.save_image(generated_images[k:k+1], os.path.join('samples', f'{n_samples}.png'))
            n_samples += 1


    # n_samples = 0
    # with torch.no_grad():
    #     while n_samples<10000:
    #         z = torch.randn(args.batch_size, 100).cuda()
    #         x = model(z)
    #         x = x.reshape(args.batch_size, 28, 28)
    #         for k in range(x.shape[0]):
    #             if n_samples<10000:
    #                 torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
    #                 n_samples += 1

    print('Generating Done.')
    
