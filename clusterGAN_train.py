import torch 
import os
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from itertools import chain as ichain
from tqdm import trange

from model import Generator, Discriminator, Encoder_CNN
from utils import sample_z, save_models, calc_gradient_penalty

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Normalizing Flow.')
    parser.add_argument("--epochs", type=int, default=251,
                        help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=0.0002,
                      help="The learning rate to use for training.")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Size of mini-batches for SGD")
    parser.add_argument("--latent_dim", type=int, default=100, 
                        help="dim of latent space")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="Exponential decay rate for the first moment estimates")
    parser.add_argument("--b2", type=float, default=0.9,
                        help="Exponential decay rate for the second moment estimates")
    parser.add_argument("--decay", type=float, default=2.5*1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--n_skip_iter", type=int, default=1,
                        help="Number of iterations to train GE for each D iteration")
    parser.add_argument("--wass_metric", type=bool, default=False,
                        help="Use Wasserstein metric for GAN loss")
    parser.add_argument("--name", type=str, default= "")

    args = parser.parse_args()
    torch.cuda.set_device(1)
    torch.cuda.empty_cache()
    print("device", torch.cuda.current_device())
    
    if args.wass_metric:
        imgs_dir = 'images/trainGAN' + str(args.latent_dim)
        ckpt_dir = 'checkpointsGAN' + str(args.latent_dim)
    else:
        imgs_dir = 'images/train' + str(args.latent_dim) + args.name
        ckpt_dir = 'checkpoints' + str(args.latent_dim) + args.name
    os.makedirs('data', exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    
    betan = 10
    betac = 10
    

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
    
    test_imgs, test_labels = next(iter(test_loader))
    test_imgs, test_labels = test_imgs.cuda(), test_labels.cuda()  # Added .cuda()
    
    mnist_dim = 784
    # Initialize models
    generator = Generator(g_input_dim=args.latent_dim + 10, g_output_dim = mnist_dim).cuda()
    discriminator = Discriminator(mnist_dim).cuda()
    encoder = Encoder_CNN(args.latent_dim, n_c=10).cuda()
    
    # Loss functions
    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    # Optimizers
    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())
    optimizer_GE = torch.optim.Adam(ge_chain, lr=1e-3, betas=(args.b1, args.b2), weight_decay=args.decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=3e-4, betas=(args.b1, args.b2))

    ge_l = []
    d_l = []
    
    c_zn = []
    c_zc = []
    c_i = []
    
    # Training loop 
    print('\nBegin training session with %i epochs...\n'%(args.epochs))
    for epoch in trange(args.epochs):
        for i, (real_imgs, itruth_label) in enumerate(train_loader):  # Changed from train_dataset to train_loader
            real_imgs, itruth_label = real_imgs.cuda(), itruth_label.cuda()  # Added .cuda()
           
            # Ensure generator/encoder are trainable
            generator.train()
            encoder.train()
            # Zero gradients for models
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            
            optimizer_GE.zero_grad()
            
            # Sample random latent variables
            zn, zc, zc_idx = sample_z(shape=real_imgs.shape[0],
                                      latent_dim=args.latent_dim,
                                      n_c=10)
            zn, zc, zc_idx = zn.cuda(), zc.cuda(), zc_idx.cuda()  # Added .cuda()
    
            # Generate a batch of images
            gen_imgs = generator(torch.cat((zn, zc), 1))
            
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs.view(real_imgs.size(0), -1))
            
            # Step for Generator & Encoder, n_skip_iter times less than for discriminator
            if (i % args.n_skip_iter == 0):
                # Encode the generated images
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs.view(-1, 1, 28, 28))
    
                # Calculate losses for z_n, z_c
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)
    
                # Check requested metric
                if args.wass_metric:
                    # Wasserstein GAN loss
                    ge_loss = torch.mean(D_gen) + betan * zn_loss + betac * zc_loss
                else:
                    # Vanilla GAN loss
                    valid = torch.ones(gen_imgs.size(0), 1).cuda()  # Added .cuda()
                    v_loss = bce_loss(D_gen, valid)
                    ge_loss = v_loss + betan * zn_loss + betac * zc_loss
    
                ge_loss.backward(retain_graph=True)
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
    
            optimizer_D.zero_grad()
    
            # Measure discriminator's ability to classify real from generated samples
            if args.wass_metric:
                # Gradient penalty term
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs.view(-1, 1, 28, 28))

                # Wasserstein GAN loss w/gradient penalty
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
                
            else:
                # Vanilla GAN loss
                fake = torch.zeros(gen_imgs.size(0), 1).cuda()  # Added .cuda()
                real_loss = bce_loss(D_real, valid)
                fake_loss = bce_loss(D_gen, fake)
                d_loss = (real_loss + fake_loss) / 2
    
            d_loss.backward()
            optimizer_D.step()

        # Save training losses
        d_l.append(d_loss.item())
        ge_l.append(ge_loss.item())
   
        # Generator in eval mode
        generator.eval()
        encoder.eval()

        # Cycle through test real -> enc -> gen
        t_imgs, t_label = test_imgs.data, test_labels
        e_tzn, e_tzc, e_tzc_logits = encoder(t_imgs)
        teg_imgs = generator(torch.cat((e_tzn, e_tzc), 1))
        img_mse_loss = mse_loss(t_imgs, teg_imgs.view(-1, 1, 28, 28))
        c_i.append(img_mse_loss.item())
        
        n_sqrt_samp = 5
        n_samp = n_sqrt_samp * n_sqrt_samp
       
        ## Cycle through randomly sampled encoding -> generator -> encoder
        zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_samp,
                                                 latent_dim=args.latent_dim,
                                                 n_c=10)
        zn_samp, zc_samp, zc_samp_idx = zn_samp.cuda(), zc_samp.cuda(), zc_samp_idx.cuda()  # Added .cuda()
        gen_imgs_samp = generator(torch.cat((zn_samp, zc_samp), 1))
        zn_e, zc_e, zc_e_logits = encoder(gen_imgs_samp.view(-1, 1, 28, 28))
        lat_mse_loss = mse_loss(zn_e, zn_samp)
        lat_xe_loss = xe_loss(zc_e_logits, zc_samp_idx)
        c_zn.append(lat_mse_loss.item())
        c_zc.append(lat_xe_loss.item())
      
        # Save cycled and generated examples
        r_imgs, i_label = real_imgs.data[:n_samp], itruth_label[:n_samp]
        e_zn, e_zc, e_zc_logits = encoder(r_imgs.view(-1, 1, 28, 28))
        reg_imgs = generator(torch.cat((e_zn, e_zc), 1)).view(-1, 1, 28, 28)
        save_image(r_imgs.data[:n_samp], '%s/real_%06i.png' %(imgs_dir, epoch), nrow=n_sqrt_samp, normalize=True)
        save_image(reg_imgs.data[:n_samp], '%s/reg_%06i.png' %(imgs_dir, epoch), nrow=n_sqrt_samp, normalize=True)
        save_image(gen_imgs_samp.data[:n_samp].view(-1, 1, 28, 28), '%s/gen_%06i.png' %(imgs_dir, epoch), nrow=n_sqrt_samp, normalize=True)
        
        ## Generate samples for specified classes
        class_samples = []
        for idx in range(10):
            zn_samp, zc_samp, zc_samp_idx = sample_z(shape=10, latent_dim=args.latent_dim, n_c=10, fix_class=idx)
            zn_samp, zc_samp = zn_samp.cuda(), zc_samp.cuda()  # Ensure tensors are on the GPU
            gen_imgs_samp = generator(torch.cat((zn_samp, zc_samp), 1))
            class_samples.append(gen_imgs_samp.view(-1, 1, 28, 28))

        # Concatenate all class samples along the batch dimension
        stack_imgs = torch.cat(class_samples, dim=0)

        save_image(stack_imgs,
                   '%s/gen_classes_%06i.png' %(imgs_dir, epoch), 
                   nrow=10, normalize=True)

        # Print the losses for each epoch
        print ("[Epoch %d/%d] \n"\
               "\tModel Losses: [D: %f] [GE: %f]" % (epoch, 
                                                     args.epochs, 
                                                     d_loss.item(),
                                                     ge_loss.item())
              )
        
        print("\tCycle Losses: [x: %f] [z_n: %f] [z_c: %f]" % (img_mse_loss.item(), 
                                                              lat_mse_loss.item(), 
                                                              lat_xe_loss.item())
             )
        if epoch % 10 == 0:
            # Save models
            save_models(generator, discriminator, encoder, ckpt_dir, epoch)