import torch 
import torchvision
import os
import argparse


from model import Generator
from utils import load_model, observe_latent_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Generator(g_output_dim = mnist_dim).cuda()
    model = load_model(model, 'checkpoints')
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    print('Model loaded.')



    print('Start Generating')
    os.makedirs('samples', exist_ok=True)

    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 20).cuda()
            x = model(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples', f'{n_samples}.png'))         
                    n_samples += 1

    print('Generating Done.')
    

# Observe the latent space embeddings of the Generator :
# observe_latent_space(model, n_samples=1000, method='pca')