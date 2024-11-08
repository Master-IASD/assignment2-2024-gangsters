import os
import argparse
import numpy as np
import torch
import pandas as pd

from model import Encoder_CNN
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
import matplotlib
import matplotlib.cm as cm



def main():
    global args
    parser = argparse.ArgumentParser(description="TSNE generation script")
    parser.add_argument("-r", "--run_dir", dest="run_dir", help="Training run directory")
    parser.add_argument("-p", "--perplexity", dest="perplexity", default=-1, type=int,  help="TSNE perplexity")
    parser.add_argument("-n", "--n_samples", dest="n_samples", default=10000, type=int,  help="Number of samples")
    parser.add_argument("-d", "--latent_dim", default=100, type=int)
    parser.add_argument("--wass_metric", type=bool, default=False)
    args = parser.parse_args()

    # TSNE setup
    n_samples = args.n_samples
    perplexity = args.perplexity
    imgs_dir = 'images/latent'
    
    if args.wass_metric:
        ckpt_dir = 'checkpointsGAN'
    else:
        ckpt_dir = 'checkpoints' + str(args.latent_dim)
        
    os.makedirs(imgs_dir, exist_ok=True)

    n_c = 10
    batch_size = 64
    
    # Load encoder model
    encoder = Encoder_CNN(args.latent_dim, n_c).cuda()
    ckpt = torch.load(os.path.join(ckpt_dir,'E.pth'))
    encoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    encoder.eval()

    # Latent space info

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5), std=(0.5))])

    test_dataset = datasets.MNIST(root='data/MNIST/', train=False, transform=transform, download=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, shuffle=False)
    # Load TSNE
    if (perplexity < 0):
        tsne = TSNE(n_components=2, verbose=1, init='pca', random_state=0)
        fig_title = "PCA Initialization"
        if args.wass_metric:
            figname = os.path.join(imgs_dir, 'tsne-pca-wass-%i.png'%args.latent_dim)
        else:
            figname = os.path.join(imgs_dir, 'tsne-pca-%i.png'%args.latent_dim)
    else:
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
        fig_title = "Perplexity = $%d$"%perplexity
        if args.wass_metric:
            figname = os.path.join(imgs_dir, 'tsne-plex%i-wass-%i.png'%(args.latent_dim, perplexity))
        else:
            figname = os.path.join(imgs_dir, 'tsne-plex%i-%i.png'%(args.latent_dim,perplexity))

    full_enc = []
    full_labels = []
    total_samples = 0

    # Iterate over test_loader
    for imgs, labels in test_loader:
        # Stop when reaching n_samples
        if total_samples >= n_samples:
            break

        # Move images to GPU
        c_imgs = imgs.cuda()

        # Encode real images
        enc_zn, enc_zc, enc_zc_logits = encoder(c_imgs)

        # Stack latent space encoding
        enc = np.hstack((enc_zn.cpu().detach().numpy(), enc_zc_logits.cpu().detach().numpy()))
        full_enc.append(enc)
        full_labels.append(labels.cpu().numpy())

        # Update total samples
        total_samples += imgs.size(0)

    # Concatenate all encoded batches and labels
    full_enc = np.concatenate(full_enc, axis=0)
    full_labels = np.concatenate(full_labels, axis=0)

    # Cluster with TSNE
    tsne_enc = tsne.fit_transform(full_enc)

    # Convert full_labels to numpy for indexing purposes
    full_labels = np.array(full_labels)

    # Color and marker for each true class
    n_c = 10  # Number of classes
    colors = cm.rainbow(np.linspace(0, 1, n_c))
    markers = matplotlib.markers.MarkerStyle.filled_markers

    # Save TSNE figure to file
    fig, ax = plt.subplots(figsize=(16, 10))
    for iclass in range(n_c):
        # Get indices for each class
        idxs = full_labels == iclass
        # Scatter points for each class in TSNE dimensions
        ax.scatter(tsne_enc[idxs, 0],
                tsne_enc[idxs, 1],
                marker=markers[iclass % len(markers)],
                color=colors[iclass],
                label=f'Class {iclass}')

    # Set plot titles and labels
    ax.set_title(fig_title, fontsize=24)
    ax.set_xlabel(r'$X^{\mathrm{tSNE}}_1$', fontsize=18)
    ax.set_ylabel(r'$X^{\mathrm{tSNE}}_2$', fontsize=18)
    plt.legend(title='Class', loc='best', numpoints=1, fontsize=16)
    plt.tight_layout()
    fig.savefig(figname)
    
    
    # GMM clustering
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
    clusters = gmm.fit_predict(full_enc)
    means = gmm.means_  # Cluster means
    covariances = gmm.covariances_  # Cluster covariances
    
    # Step 3: Plot the clustered data in 2D for visualization
    plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(16, 10))
    for iclass in range(n_c):
        # Get indices for each class
        idxs = clusters == iclass
        # Scatter points for each class in TSNE dimensions
        ax.scatter(tsne_enc[idxs, 0],
                tsne_enc[idxs, 1],
                marker=markers[iclass % len(markers)],
                color=colors[iclass],
                label=f'Class {iclass}')

    plt.title(f'Latent Space Clustering with TSNE and GMM')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(f'images/latent/gmm-{args.latent_dim}.png')
    
    curves = means[:, -10:]

    max_values = np.max(curves, axis=1)
    sorted_curve_indices = np.argsort(max_values)  # Indices of curves sorted by max value

    # Step 3: Initialize a set to keep track of used indices and a dictionary to store final results
    used_indices = set()
    final_max_indices = {}

    # Step 4: Iterate over each curve, from the one with the smallest max value to the largest
    for curve_idx in sorted_curve_indices:
        curve = curves[curve_idx]
        
        # Sort indices of this curve by value in descending order
        sorted_indices = np.argsort(curve)[::-1]  # Indices sorted by value (largest first)
        
        # Find the first available index that hasn't been used yet
        for idx in sorted_indices:
            if idx not in used_indices:
                final_max_indices[curve_idx] = idx  # Assign this index as the max for this curve
                used_indices.add(idx)               # Mark this index as used
                break
    
    np.savez("checkpoints/gmm_parameters.npz", means=gmm.means_, covariances=gmm.covariances_, clusters=list(final_max_indices.values()), latent_dim=args.latent_dim)
    print("GMM parameters saved to checkpoints/gmm_parameters.npz")
    
if __name__ == "__main__":
    main()