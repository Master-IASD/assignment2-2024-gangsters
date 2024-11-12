import torch
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
from torchvision.utils import make_grid, save_image
from scipy.stats import multivariate_normal

def load_model(G, classifier, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    ckpt = torch.load(os.path.join(folder,'classifier.pth'))
    classifier.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G, classifier

def visualize_difference(generator, latent_vectors, original_noise, n_samples):
    """
    Visualize the difference between original and optimized latent vectors
    """
    z_vis = np.stack([latent_vectors[i:i+16, :] for i in range(0, n_samples, n_samples//10)], axis=0)
    z_vis = torch.from_numpy(z_vis).cuda()
    with torch.no_grad():
        grid = make_grid(generator(z_vis).view(-1, 1, 28, 28).cpu(), nrow=4, normalize=True)
        save_image(grid, f'images/predictions/new_pred.png')
        
    original_z_vis = np.stack([original_noise[i:i+16, :] for i in range(0, n_samples, n_samples//10)], axis=0)
    original_z_vis = torch.from_numpy(original_z_vis).cuda()
    with torch.no_grad():
        grid = make_grid(generator(original_z_vis).view(-1, 1, 28, 28).cpu(), nrow=4, normalize=True)
        save_image(grid, f'images/predictions/original_pred.png')
    

def analyze_latent_space(generator, D, classifier, latent_dim, n_samples=1000, lr=0.01, n_steps=100):
    """
    Analyze the latent space of a GAN using gradient ascent and visualization
    """
    # Generate random latent vectors
    z_list = []
    label_list = []
    original_z = []
    
    # Number of samples per class
    samples_per_class = n_samples // 10
    
    # For each digit class
    for target_class in range(10):
        # Generate random latent vectors
        z = torch.randn(samples_per_class, latent_dim, requires_grad=True, device='cuda')
        original_z.append(z.detach().cpu().numpy())
        
        # Initialize optimizer for this batch of z
        optimizer = torch.optim.Adam([z], lr=lr)
        
        # Gradient ascent to optimize z for target class
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # Generate images
            fake_images = generator(z)
            
            # Get classifier predictions
            predictions = classifier(fake_images.view(-1, 1, 28, 28))
            
            # Discriminator loss: maximize probability of fake images
            y_fake = torch.ones(samples_per_class, 1).cuda()
            D_output = D(fake_images)
            D_fake_loss = torch.nn.BCELoss()(D_output, y_fake)
            
            # Calculate loss: maximize probability of target class
            # while maintaining reasonable values in latent space
            class_loss = -torch.mean(predictions[:, target_class])
            regularization = 0.1 * torch.mean(torch.abs(z)) # L1 regularization
            loss = class_loss + regularization + D_fake_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Optional: Print progress
            if step % 20 == 0:
                with torch.no_grad():
                    current_pred = classifier(generator(z).view(-1, 1, 28, 28))
                    confidence = torch.mean(current_pred[:, target_class]).item()
                    print(f"Class {target_class}, Step {step}, Confidence: {confidence:.3f}, Loss: {D_fake_loss.item():.3f}")
        
        # Store optimized z and labels
        z_list.append(z.detach())
        label_list.extend([target_class] * samples_per_class)
    
    # Combine all optimized latent vectors
    final_z = torch.cat(z_list, dim=0)
    labels = np.array(label_list)
    original_noise = np.concatenate(original_z, axis=0)
    
    with torch.no_grad():
        final_images = generator(final_z).view(-1, 1, 28, 28)
        final_predictions = classifier(final_images)
        confidences = torch.max(final_predictions, dim=1)[0].cpu().numpy()
        
    return final_z.cpu().numpy(), labels, original_noise, confidences

def visualizing_clustering_latent_space(latent_vectors, labels, latent_dim, original_noise, method='pca'):
    """
    Visualize latent space using PCA or t-SNE
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2)
    
    old_new_vectors = np.concatenate((latent_vectors, original_noise), axis=0)
    reduced_old_new_vectors = reducer.fit_transform(old_new_vectors)
    reduced_vectors = reduced_old_new_vectors[:latent_vectors.shape[0], :]
    original_reduced_vectors = reduced_old_new_vectors[latent_vectors.shape[0]:, :]
    
    # Visualize the reduced latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'Latent Space Visualization using {method.upper()}')
    date = datetime.datetime.now().strftime("%d_%H-%M-%S")
    plt.savefig(f'images/latent_space_plots/latent_dim_{latent_dim}_{method}_{date}.png')
    
    # Visualize the original latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(original_reduced_vectors[:, 0], original_reduced_vectors[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f'Original Latent Space Visualization using {method.upper()}')
    plt.savefig(f'images/latent_space_plots/latent_dim_{latent_dim}_{method}_{date}_original.png')
    
    # Visualize the cluster of the latent space
    gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
    clusters = gmm.fit_predict(latent_vectors)
    means = gmm.means_  # Cluster means
    covariances = gmm.covariances_  # Cluster covariances
    
    # Step 3: Plot the clustered data in 2D for visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='tab10')
    plt.colorbar(scatter, label="Cluster ID")
    plt.title(f'Latent Space Clustering with {method.upper()} and GMM')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f'images/latent_space_plots/latent_dim_{latent_dim}_{method}_{date}_gmm.png')
    
    # return means, covariances

def generate_improved_samples(generator, gmm, latent_dim, n_samples=100, temperature=1.0):
    """
    Generate samples using the GMM-improved latent space with temperature control
    """
    # Sample from GMM with temperature scaling
    means = gmm.means_
    covs = gmm.covariances_ * temperature
    
    # Randomly select components based on weights
    component_indices = np.random.choice(
        len(gmm.weights_), 
        size=n_samples, 
        p=gmm.weights_
    )
    
    # Generate samples from selected components
    latent_samples = np.array([
        np.random.multivariate_normal(means[idx], covs[idx])
        for idx in component_indices
    ])
    
    # Convert to torch tensor
    latent_samples = torch.FloatTensor(latent_samples)
    
    with torch.no_grad():
        improved_images = generator(latent_samples)
    
    return improved_images, latent_samples