import torch 
import torch.nn as nn

class conv_block(nn.Module):
    """
    Convolution block containing a convolution layer, ReLU activation and Batchnorm layer.
    """
    def __init__(self, input_channels, output_channels, kernel_size=5, stride=1, padding=0, upsample=False):
        super().__init__()
        if upsample:
            self.conv = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.bn = torch.nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Encoder(nn.Module):
    """
    Encode the input into a latent representation consisting of a mean and std.
    """
    def __init__(self, input_channels, n_layers=8):
        super().__init__()
        self.layer_list = [conv_block(input_channels*2**(n), input_channels*2**(n+1), 3) for n in range(n_layers)]
        self.layers = nn.Sequential(*self.layer_list)
        self.fc_layer = nn.Linear(6400, input_channels*2**(n_layers))
    
    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_layer(x)

        means = x[:, :x.shape[1]//2]
        stds = torch.exp(x[:, x.shape[1]//2:]) # ensure positive std
        
        # means/stds: [batch, channels (latent dims)]
        return means, stds


class Decoder(nn.Module):
    """
    Take the latent space activation and pass it through a series of upsampling+convolution layers.
    """
    def __init__(self, input_channels, n_layers=8):
        super().__init__()
        self.input_channels = input_channels
        self.n_layers = n_layers
        self.fc_layer = nn.Linear(input_channels*2**(n_layers)//2, 6400)
        self.layer_list = [conv_block(input_channels*2**(n+1), input_channels*2**(n), 3, upsample=True) for n in range(n_layers)]
        self.layers = nn.Sequential(*self.layer_list[::-1])
        self.tanh = nn.Tanh()

    def forward(self, means, stds):
        # reparametrization trick:
        z = torch.normal(0, 1, stds.shape).to(stds.device)
        sampled_latent_vector = means + z*stds
        x = self.fc_layer(sampled_latent_vector)
        x = x.reshape(x.shape[0], self.input_channels*2**(self.n_layers), -1)
        x = x.reshape(x.shape[0], x.shape[1], int(x.shape[2]**(1/2)), int(x.shape[2]**(1/2)))
        # decoder layers
        x = self.layers(x)
        x = self.tanh(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_channels, n_layers):
        super().__init__()
        self.encoder = Encoder(input_channels, n_layers//2)
        self.decoder = Decoder(input_channels, n_layers//2)

    def forward(self, x):
        means, stds = self.encoder(x)
        out = self.decoder(means, stds)
        return out, means, stds

def VAE_loss(reconstruction, original, means, stds, h=0.005):
    """
    reconstruction: [batches, channels, height, width]
    original: [batches, channels, height, width]
    means: [batches, means]
    stds: [batches, stds]

    The VAE ELBO loss can be written as  - E_q [log p(x | z)] - KL(q(z | x) || p(z)).
    * We assume p(z) ~ N(0, I)
    """
    # maximize log probabality => minimize mse
    recon = torch.mean((reconstruction-original)**2)
    KL = torch.mean(1/2*(stds.sum(dim=1)+torch.sum(means**2, dim=1)-torch.log(torch.prod(stds, dim=1))))

    return recon + h*KL

