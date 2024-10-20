import torch
import os
import torch.nn as nn
from torchvision import transforms
import torchvision

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def lipconstant(args,D,x,y):
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
    alpha = torch.rand((b,1,1,1),device=args.device)
    interp = alpha * y + (1 - alpha) * x
    interp.requires_grad_()

    # Calculate probability of interpolated examples
    Di = D(interp).view(-1)

    # Calculate gradients of probabilities with respect to examples
    gradout = torch.ones(Di.size()).to(args.device)
    gradients = torch.autograd.grad(outputs=Di, inputs=interp, grad_outputs=gradout,
       create_graph=True, retain_graph=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))

    # Return gradient penalty
    return torch.mean(gradients_norm)

def gradient_penalty(args,D,x,y):
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
    alpha = torch.rand((b,1,1,1)).to(args.device)
    interp = (alpha * y + (1 - alpha) * x)
    interp.requires_grad_()

    # Calculate probability of interpolated examples
    Di = D(interp).view(-1)

    # Calculate gradients of probabilities with respect to examples
    gradout = torch.ones(Di.size()).to(args.device)
    gradients = torch.autograd.grad(outputs=Di, inputs=interp, grad_outputs=gradout,
       create_graph=True, retain_graph=True)[0]

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return ((gradients_norm - 1) ** 2).mean()


def D_train(args,x, G, D, D_optimizer, gpw=0.1):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(args.device), y_real.to(args.device)

    D_output = D(x_real)
    D_real_loss = D_output.mean()
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).to(args.device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(args.device)

    D_output =  D(x_fake)
    
    D_fake_loss = D_output.mean()
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = - D_real_loss + D_fake_loss + gpw * gradient_penalty(args, D, x_real, x_fake)
    D_loss.backward()
    D_optimizer.step()
    
    # Dlipp = lipconstant(args, D,x_real,x_fake)
        
    return  D_loss.data.item()


def G_train(args,x, G, D, G_optimizer):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(args.device)
    y = torch.ones(x.shape[0], 1).to(args.device)
                 
    G_output = G(z)
    D_output = D(G_output)
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
