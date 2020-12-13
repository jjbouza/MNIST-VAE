import torch
from torchvision.utils import save_image, make_grid

from vae import VAE, VAE_loss

if __name__=='__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = VAE(input_channels=1, n_layers=8).to(device)
    model.load_state_dict(torch.load('/blue/vemuri/josebouza/projects/VAE/model.pt'))

    batches = 256
    means = torch.zeros([batches, 8]).to(device)
    stds = torch.ones([batches, 8]).to(device)
    out = model.decoder(means, stds)
    
    image_grid = make_grid(out)
    save_image(image_grid, '/blue/vemuri/josebouza/projects/VAE/sample.png')
