from model import Generator
from processing.classifier import Classifier
from processing.utils_process import *


# ===================== Observing the Latent Space =====================
print('Model Loading...')
# Model Pipeline
mnist_dim = 784
latent_dim = 100
n_samples = 2000
segment = n_samples // 10

generator = Generator(g_output_dim = mnist_dim).cuda()
classifier = Classifier().cuda()
generator, classifier = load_model(generator, classifier, 'checkpoints')
print('Model loaded.')

# Observing the latent space
latent_vectors, predicted_labels, original_noise = analyze_latent_space(generator, classifier, latent_dim, n_samples, lr=0.1, n_steps=1000)

# Visualizing the latent space
visualize_latent_space(latent_vectors, predicted_labels, latent_dim, original_noise, method='tsne')
visualize_difference(generator, latent_vectors, original_noise, n_samples)
