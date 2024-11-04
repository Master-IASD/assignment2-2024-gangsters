import torch
import torchvision.models as models
import torch.nn as nn

# Load a pre-trained VGG network (usually VGG16 or VGG19)
class VGGPerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(VGGPerceptualLoss, self).__init__()
        # Load pre-trained VGG16 model (we can also load VGG19 model)
        vgg = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        
        # Choose specific layers that capture perceptual features
        for x in range(4):  # conv1_1 and conv1_2 layers
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):  # conv2_1 and conv2_2 layers
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):  # conv3_1 and conv3_2 layers
            self.slice3.add_module(str(x), vgg[x])
        
        # Freeze VGG parameters since we don't want to update the weights
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        return h_relu1, h_relu2, h_relu3



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGPerceptualLoss().cuda()  # TODO :Ensure it's on the correct device
        self.criterion = nn.MSELoss()  # we use a L2 loss but we also can use an L1 loss

    def forward(self, generated_img, real_img):
        # TODO : Ensure input images are normalized in the same way as VGG was trained
        # Convert 1-channel grayscale images to 3-channel RGB images
        generated_img_rgb = generated_img.repeat(1, 3, 1, 1)  # From [batch_size, 1, 28, 28] -> [batch_size, 3, 28, 28]
        real_img_rgb = real_img.repeat(1, 3, 1, 1)  # From [batch_size, 1, 28, 28] -> [batch_size, 3, 28, 28]

        real_features = self.vgg(real_img_rgb)
        generated_features = self.vgg(generated_img_rgb)
        
        # Compute perceptual loss as the sum of the differences between feature maps
        perceptual_loss = 0
        for gen_feat, real_feat in zip(generated_features, real_features):
            perceptual_loss += self.criterion(gen_feat, real_feat)
        
        return perceptual_loss