from model import Generator
from processing.classifier import Classifier
from processing.utils_process import *
from torchvision.utils import make_grid, save_image

# ===================== Observing the Latent Space =====================
print('Model Loading...')
# Model Pipeline
mnist_dim = 784
latent_dim = 100

generator = Generator(g_output_dim = mnist_dim).cuda()
classifier = Classifier().cuda()
generator, classifier = load_model(generator, classifier, 'checkpoints')
print('Model loaded.')

# Observing the latent space
latent_vector, predicted_labels, original_noise = analyze_latent_space(generator, classifier, latent_dim, n_samples=2000, lr=0.1, n_steps=2000)

# Visualizing the latent space
visualize_latent_space(latent_vector, predicted_labels, latent_dim, original_noise, method='tsne')

#Sample from new latent_vector
z_update = torch.from_numpy(latent_vector[:16, :]).cuda()
with torch.no_grad():
    grid = make_grid(generator(z_update).view(-1, 1, 28, 28).cpu(), nrow=4, normalize=True)
    save_image(grid, f'images/predictions/new_pred.png')

original_noise = torch.from_numpy(original_noise).cuda()
with torch.no_grad():
    grid = make_grid(generator(original_noise[:16, :]).view(-1, 1, 28, 28).cpu(), nrow=4, normalize=True)
    save_image(grid, f'images/predictions/original_pred.png')