#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#


# Import Libraries
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as utils

# Setting some hyperparameters
BATCH_SIZE = 64
IMAGE_SIZE = 64

# Creating the transformer
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading the Dataset & defining dataloader
dataset = dataset.CIFAR10(
    root='/data',
    download=True,
    transform=transform
)
dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=BATCH_SIZE,
                                         shuffle=True)

# Defining weights_init function
def weights_init(m):
    """
    Initializes the weights of the network layers to ensure proper weight initialization 
    before training. This method applies specific initialization strategies based on 
    the type of layer:

    - For convolutional layers (layers containing 'Conv' in their class name), the weights 
      are initialized from a normal distribution with mean 0.0 and standard deviation 0.02.

    - For batch normalization layers (layers containing 'BatchNorm' in their class name), 
      the weights are initialized from a normal distribution with mean 1.0 and standard deviation 0.02. 
      The bias of the batch normalization layer is set to 0.

    Args:
        m: A layer of the neural network (could be a convolutional layer, batch normalization 
           layer, or any other layer in the network).

    Example:
        # Apply weights_init to a network
        model = MyNeuralNetwork()
        model.apply(weights_init)

    Explanation:
        Proper weight initialization can improve the model's convergence during training. 
        For convolutional layers, initializing the weights with small random values prevents 
        vanishing or exploding gradients, which can occur if weights are too small or too large. 
        For batch normalization layers, initializing the weights close to 1.0 helps maintain 
        stable outputs, while setting the bias to 0 ensures no shift is applied initially.

    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Defining the Generator class
class G(nn.Module):
    """
    A Generator network that generates images from a noise vector (latent space) 
    using a series of transposed convolutional layers (ConvTranspose2d). The input 
    is a noise vector of size 100, and the output is an RGB image of size 64x64.

    Architecture:
        - The input is a noise vector of shape (100, 1, 1).
        - The network upsamples the noise vector through five transposed convolution 
          layers, progressively increasing the spatial size and decreasing the 
          depth of the feature maps.
        - The final output is a 64x64 image with 3 channels (RGB image).

    Layer-by-layer explanation:
        1. First layer: 
            - Transforms the input noise vector (100 channels) into a 4x4 feature map 
              with 512 channels using a kernel size of 4, stride of 1, and padding of 0.
        2. Second layer: 
            - Upsamples the 512-channel 4x4 feature map to an 8x8 feature map with 256 channels.
        3. Third layer: 
            - Upsamples the 256-channel 8x8 feature map to a 16x16 feature map with 128 channels.
        4. Fourth layer: 
            - Upsamples the 128-channel 16x16 feature map to a 32x32 feature map with 64 channels.
        5. Fifth layer:
            - Upsamples the 64-channel 32x32 feature map to a 64x64 RGB image with 3 channels.
            - A Tanh activation function is used in the final layer to scale the output image 
              pixel values to the range [-1, 1].

    Output:
        - The output is a 3-channel (RGB) image of size 64x64.

    Example usage:
        # Initialize generator model
        G_model = G()
        
        # Generate a random noise vector
        noise = torch.randn(batch_size, 100, 1, 1)
        
        # Generate a batch of images
        images = G_model(noise)
    """
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100,
                               out_channels=512,
                                kernel_size=4,
                                stride=1,
                                padding=0,
                                bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, 
                               out_channels=128, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=64, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, 
                               out_channels=3, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Defining the Discriminator
class D(nn.Module):
    """
    A Discriminator network that classifies images as real or fake using a series of 
    convolutional layers (Conv2d). The input is an RGB image of size 64x64, and the 
    output is a single scalar representing the probability that the image is real (1) 
    or fake (0).

    Architecture:
        - The input is an RGB image of shape (3, 64, 64).
        - The network downsamples the image through five convolutional layers, progressively 
          reducing the spatial size and increasing the depth of the feature maps.
        - The final output is a single scalar value (using a Sigmoid activation) that 
          indicates the discriminator's confidence in whether the input image is real or fake.

    Layer-by-layer explanation:
        1. First layer: 
            - Takes the input RGB image (3 channels) and reduces it to a 32x32 feature map with 64 channels.
        2. Second layer: 
            - Downsamples the 64-channel 32x32 feature map to a 16x16 feature map with 128 channels.
        3. Third layer: 
            - Downsamples the 128-channel 16x16 feature map to an 8x8 feature map with 256 channels.
        4. Fourth layer: 
            - Downsamples the 256-channel 8x8 feature map to a 4x4 feature map with 512 channels.
        5. Fifth layer: 
            - Outputs a single scalar (1 channel) using a 4x4 convolution that flattens the feature map 
              into a single value, followed by a Sigmoid activation to produce a probability.

    Output:
        - The output is a scalar value between 0 and 1, where 1 indicates that the discriminator 
          classifies the input as a real image, and 0 indicates it is fake.

    Example usage:
        # Initialize discriminator model
        D_model = D()

        # Create a random batch of images (batch_size, channels, height, width)
        images = torch.randn(batch_size, 3, 64, 64)
        
        # Classify the batch of images
        real_or_fake = D_model(images)
    """
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, 
                      out_channels=64, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, 
                      out_channels=128, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, 
                      out_channels=256, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, 
                      out_channels=512, 
                      kernel_size=4, 
                      stride=2, 
                      padding=1, 
                      bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, 
                      out_channels=1, 
                      kernel_size=4, 
                      stride=1, 
                      padding=0, 
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

# Creating Generator & Discriminator instances
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G_net = G().to(device)
G_net.apply(weights_init)
D_net = D().to(device)
D_net.apply(weights_init)

# Defining Criterion and Optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G_net.parameters(), 
                               lr=0.0002, 
                               betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D_net.parameters(), 
                               lr=0.0002, 
                               betas=(0.5, 0.999))

# Create results directory if it doesn't exist
if not os.path.exists('./results'):
    os.makedirs('./results')

# Training the DCGANs
for epoch in range(26):
    for i, data in enumerate(dataloader, 0):
        # Step 1: Update weights of Discriminator
        D_net.zero_grad()
        real, _ = data
        real = real.to(device)
        target = torch.ones(real.size(0)).to(device)
        output = D_net(real)
        errD_real = criterion(output, target)

        # Training the discriminator with a fake image generated by the generator
        noise = torch.randn(real.size(0), 100, 1, 1).to(device)
        fake = G_net(noise)
        target = torch.zeros(real.size(0)).to(device)
        output = D_net(fake.detach())
        errD_fake = criterion(output, target)

        # Backpropagating the total error
        errD = errD_real + errD_fake
        errD.backward()
        optimizer_D.step()

        # Step 2: Updating the weights of the generator
        G_net.zero_grad()
        target = torch.ones(real.size(0)).to(device)
        output = D_net(fake)
        errG = criterion(output, target)
        errG.backward()
        optimizer_G.step()
        
        # Step 3: Printing the losses and saving the images every 100 steps
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.item(), errG.item()))
            utils.save_image(real, '%s/real_samples.png' % "./results", normalize=True)
            fake = G_net(noise)
            utils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize=True)