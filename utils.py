import torch
import os
import numpy as np
from torch.autograd import grad as torch_grad

def z_generation(samples, latent_dim):
    """
    Generate random noise vector z with 10 categorical sampled from gaussian centered to have value near 1.1 with std 0,1.
    
    :param batch_size: Number of samples to generate
    :param latent_dim: Dimension of the latent space
    :return: Random noise vector z
    """
    z = torch.normal(mean=0, std=1, size=(samples, latent_dim)) * 0.75
    zc = torch.zeros(samples, 10)
    zc_value = torch.normal(mean=1.1, std=0.15, size=(samples,))
    for i in range(10):
        # zc[i*samples//10:(i+1)*samples//10, i] = zc_value[i*samples//10:(i+1)*samples//10]
        zc[i*samples//10:(i+1)*samples//10, i] = 1
    z = torch.cat((z, zc), dim=1)
    return z

def noise_generation(batch_size, latent_dim):
    z = torch.randn(batch_size, latent_dim).cuda()
    return z

def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):

    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c) ), "Requested class %i outside bounds."%fix_class

    Tensor = torch.cuda.FloatTensor
    
    # Sample noise as generator input, zn
    zn = Tensor(0.75*np.random.normal(0, 1, (shape, latent_dim)))

    ######### zc, zc_idx variables with grads, and zc to one-hot vector
    # Pure one-hot vector generation
    zc_FT = Tensor(shape, n_c).fill_(0)
    zc_idx = torch.empty(shape, dtype=torch.long)

    if (fix_class == -1):
        zc_idx = zc_idx.random_(n_c).cuda()
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)
        #zc_idx = torch.empty(shape, dtype=torch.long).random_(n_c).cuda()
        #zc_FT = Tensor(shape, n_c).fill_(0).scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.cuda()
        zc_FT = zc_FT.cuda()

    zc = zc_FT

    ## Gaussian-noisey vector generation
    #zc = Variable(Tensor(np.random.normal(0, 1, (shape, n_c))), requires_grad=req_grad)
    #zc = softmax(zc)
    #zc_idx = torch.argmax(zc, dim=1)

    # Return components of latent space variable
    return zn, zc, zc_idx

def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 10

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()
    
    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated.view(b_size, -1))

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


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
        
    return D_loss.data.item(), D_real_score.mean().data.item(), D_fake_score.mean().data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 20).cuda()
    y = torch.ones(x.shape[0], 1).cuda()
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()


def save_models(G, D, E=None, folder=None, epoch=None):
    if E:
        torch.save(E.state_dict(), os.path.join(folder,f'E{epoch}.pth'))
    torch.save(G.state_dict(), os.path.join(folder,f'G{epoch}.pth'))
    torch.save(D.state_dict(), os.path.join(folder,f'D{epoch}.pth'))


def load_model(G, folder, epoch = ""):
    ckpt = torch.load(os.path.join(folder,'G'+ epoch + '.pth'))
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
    z = torch.randn(x.size(0), 100).cuda()
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
    os.makedirs('latent space plots', exist_ok=True)
    # Saving the plot
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'latent_space_plots/latent_space_{date}.png')
    print(f"Latent space plot saved as latent_space_{date}.png")





# ================== (Second) function to Observe the latent space with according label colours ==================

def observe_latent_space_color(G, latent_dim, n_samples=100, method='pca'):
    """
    Observe the latent space embeddings of the Generator and color them by MNIST labels.
    
    :param G: Trained Generator model
    #########  :param D: Trained Discriminator model (optional if needed for some tasks)
    :param dataloader: Dataloader with MNIST dataset
    :param n_samples: Number of samples to generate or extract from the dataset
    :param method: Dimensionality reduction method ('pca' or 'tsne')
    """
    G.eval()  # Set generator to evaluation mode

    # Generate latent vectors (random noise z) based on real data
    with torch.no_grad():
        z = noise_generation(n_samples, latent_dim)  # latent space dimension
        generated_images = G(z).cpu().numpy()

    z = z.cpu().numpy()
    # Apply dimensionality reduction (PCA or t-SNE)
    if method == 'pca':
        pca = PCA(n_components=2)
        reduced_latent = pca.fit_transform(z)
        title = "PCA of Latent Space"
    elif method == 'tsne':
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        reduced_latent = tsne.fit_transform(z)
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
    os.makedirs('images/latent_space_plots', exist_ok=True)
    date = datetime.datetime.now().strftime("%m-%d_%H-%M")
    plt.savefig(f'images/latent_space_plots/latent_dim_{latent_dim}_{date}.png')
    print(f"Latent space plot saved as images/latent_space_plots/latent_dim_{latent_dim}_{date}.png")
