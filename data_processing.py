from model import Generator, Discriminator
from processing.classifier import Classifier
from processing.utils_process import *


# ===================== Observing the Latent Space =====================
print('Model Loading...')
# Model Pipeline
mnist_dim = 784
latent_dim = 100
n_samples = 800
segment = n_samples // 10
torch.cuda.set_device(1)

G = Generator(g_output_dim = mnist_dim).cuda()
D = Discriminator(d_input_dim = mnist_dim).cuda()

classifier = Classifier().cuda()
G, classifier = load_model(G, classifier, 'checkpoints')
print('Model loaded.')

# Observing the latent space
latent_vectors, predicted_labels, original_noise, confidence = analyze_latent_space(G, D, classifier, latent_dim, n_samples, lr=0.1, n_steps=1000)

# Visualizing the latent space
reduced_vector = visualizing_clustering_latent_space(latent_vectors, predicted_labels, latent_dim, original_noise, method='tsne')
visualize_difference(G, latent_vectors, original_noise, n_samples)
