import torch
import os



def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 20).cuda()
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).cuda()

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder, epoch):
    torch.save(G.state_dict(), os.path.join(folder,f'G{epoch}.pth'))
    torch.save(D.state_dict(), os.path.join(folder,f'D{epoch}.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G





# ========= function for Training the Generator on aversial loss + perceptual loss =========
def G_train_perceptual(x, G, D, G_optimizer, criterion, perceptual_loss_fn):
    """
    perceptual_loss_fn : perceptual loss function
    """
    # sending everything on the right device
    x = x.cuda()
    perceptual_loss_fn = perceptual_loss_fn.cuda()

    G_optimizer.zero_grad()

    # Sample random latent space vector
    z = torch.randn(x.size(0), 20).cuda()
    y = torch.ones(x.shape[0], 1).cuda()

    # Generate fake images
    G_output = G(z)
    # Discriminator forward pass
    D_output = D(G_output)

    # Compute GAN loss (adversarial loss)
    adversarial_loss = criterion(D_output, y)

    # Compute Perceptual Loss between generated images and real images
    percept_loss = perceptual_loss_fn(G_output.view(-1, 1, 28, 28), x.view(-1, 1, 28, 28)) # TODO: Ensure input images are normalized

    # Totam: Adversarial loss + weighted perceptual loss
    total_loss = adversarial_loss + 0.01 * percept_loss          # TODO : tune the weight of perceptual loss

    # Backpropagation and optimization
    total_loss.backward()
    G_optimizer.step()

    return total_loss.item()





# ========================= function for Observing the latent space =========================

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import datetime

def observe_latent_space(G, n_samples=100, method='pca'):
    """
    Observe the latent space embeddings of the Generator.
    
    :param G: Trained Generator model
    :param n_samples: Number of samples to generate from the latent space
    :param method: Dimensionality reduction method ('pca' or 'tsne')
    """
    # Generate latent vectors (random noise z)
    z = torch.randn(n_samples, 100).cuda()  # 100 is the latent space dimension, as can be seen in the function G_train
    G.eval()  # Set generator to evaluation mode
    
    with torch.no_grad():
        generated_images = G(z).cpu().numpy()
    
    # Flatten the generated images for dimensionality reduction (if needed)
    flattened_images = generated_images.reshape(n_samples, -1)

    # Apply dimensionality reduction (PCA or t-SNE)
    if method == 'pca':
        pca = PCA(n_components=2)
        reduced_latent = pca.fit_transform(flattened_images)
        title = "PCA of Latent Space"
    elif method == 'tsne':
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        reduced_latent = tsne.fit_transform(flattened_images)
        title = "t-SNE of Latent Space"
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    # Plot the reduced latent space embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], c='blue', alpha=0.7, s=40)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    # plt.show()

    # Directory of latent space plots
    os.makedirs('latent_space_plots', exist_ok=True)
    # Saving the plot
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'latent_space_plots/latent_space_{date}.png')
    print(f"Latent space plot saved as latent_space_{date}.png")





# ================== (Second) function to Observe the latent space with according label colours ==================

def observe_latent_space_color(G, dataloader, n_samples=100, method='pca'):
    """
    Observe the latent space embeddings of the Generator and color them by MNIST labels.
    
    :param G: Trained Generator model
    #########  :param D: Trained Discriminator model (optional if needed for some tasks)
    :param dataloader: Dataloader with MNIST dataset
    :param n_samples: Number of samples to generate or extract from the dataset
    :param method: Dimensionality reduction method ('pca' or 'tsne')
    """
    G.eval()  # Set generator to evaluation mode

    # Extract real images and labels from MNIST dataset
    images = []
    labels = []

    # Collect enough images from the dataloader to reach `n_samples`
    for batch_images, batch_labels in dataloader:
        images.append(batch_images)
        labels.append(batch_labels)
        
        # Stop when we've collected enough samples
        if len(torch.cat(images)) >= n_samples:
            break

    # Stack images and labels into single tensors
    images = torch.cat(images)[:n_samples].view(n_samples, -1).cuda()
    labels = torch.cat(labels)[:n_samples].cpu().numpy()  # Ensure labels are numpy for plotting


    # Generate latent vectors (random noise z) based on real data
    with torch.no_grad():
        z = torch.randn(n_samples, 100).cuda()  # latent space dimension
        generated_images = G(z).cpu().numpy()

    # Flatten the generated images for dimensionality reduction (if needed)
    flattened_images = generated_images.reshape(n_samples, -1)

    # Apply dimensionality reduction (PCA or t-SNE)
    if method == 'pca':
        pca = PCA(n_components=2)
        reduced_latent = pca.fit_transform(flattened_images)
        title = "PCA of Latent Space"
    elif method == 'tsne':
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        reduced_latent = tsne.fit_transform(flattened_images)
        title = "t-SNE of Latent Space"
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'tsne'.")

    # Plot the reduced latent space embeddings and color by label
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_latent[:, 0], reduced_latent[:, 1], c=labels, cmap='tab10', alpha=0.7, s=40)
    plt.colorbar(scatter, ticks=range(10))  # Add colorbar for labels (digits 0-9)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    # Save the plot
    os.makedirs('latent_space_plots', exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'latent_space_plots/latent_space_{date}.png')
    print(f"Latent space plot saved as latent_space_plots/latent_space_{date}.png")
