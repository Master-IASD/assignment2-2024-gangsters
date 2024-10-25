import torch
import os
import torch.nn as nn
from torchvision import transforms
import torchvision

def noise_generation(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()

def lipconstant(D,x,y):
    b = x.shape[0]
    n = y.shape[0]
    # shrink vectors if they are too large
    if n>b:
        y = y[0:b,:]
        n = b
    else:
        x = x[0:n,:]
        b = n
    
    # compute interpolated points
    alpha = torch.rand((b,1),device='cuda')
    interp = alpha * y + (1 - alpha) * x
    interp.requires_grad_()

    # Calculate probability of interpolated examples
    Di = D(interp).view(-1)

    # Calculate gradients of probabilities with respect to examples
    gradout = torch.ones(Di.size()).cuda()
    gradients = torch.autograd.grad(outputs=Di, inputs=interp, grad_outputs=gradout,
       create_graph=True, retain_graph=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))

    # Return gradient penalty
    return torch.mean(gradients_norm)

def compute_gradient_penalty(D, real_data, fake_data):
    batch_size = real_data.size(0)
    epsilon = torch.rand(batch_size, 1).cuda()
    
    # Interpolate between real and fake data
    epsilon = epsilon.expand_as(real_data)
    interpolated = epsilon * real_data + (1 - epsilon) * fake_data
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True).cuda()
    
    # Critic's prediction for interpolated data
    interpolated_output = D(interpolated)

    # Compute gradients of D output w.r.t. interpolated data
    gradients = torch.autograd.grad(outputs=interpolated_output, inputs=interpolated,
                     grad_outputs=torch.ones(interpolated_output.size()).cuda(),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    # Compute the gradient penalty (||gradient||_2 - 1)^2
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


def WD_train(x, G, D, D_optimizer, gpw=0.1):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    # x_real, y_real = x, torch.ones(x.shape[0], 1)
    # x_real, y_real = x_real.cuda(), y_real.cuda()
    
    x_real = x.cuda()
    z = noise_generation(x.shape[0], 100)
    x_fake = G(z).detach().cuda()

    real_output = D(x_real)
    fake_output =  D(x_fake) 
    
    D_real_loss = real_output.mean()       
    D_fake_loss = fake_output.mean()
    # D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_fake_loss - D_real_loss + gpw * compute_gradient_penalty(D, x_real, x_fake)
    D_loss.backward()
    D_optimizer.step()
    
    Dlipp = lipconstant(D,x_real,x_fake)
        
    return  D_loss.data.item(), Dlipp


def WG_train(x, G, D, G_optimizer):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = noise_generation(x.shape[0], 100)
    # y = torch.ones(x.shape[0], 1).cuda()
                 
    fake = G(z)
    D_output = D(fake)
    G_loss = - D_output.mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G


def D_train(x, G, D, D_optimizer, criterion):
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.cuda(), y_real.cuda()

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).cuda()
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