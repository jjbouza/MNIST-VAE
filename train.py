import torch
import torchvision
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from vae import VAE, VAE_loss
from tqdm import tqdm
import datetime

def log(writer, original, reconstruction, loss, it):
    og_grid = torchvision.utils.make_grid(original)
    re_grid = torchvision.utils.make_grid(reconstruction)

    writer.add_image('Input Image', og_grid*0.5+0.5, it)
    writer.add_image('Reconstruction Image', re_grid*0.5+0.5, it)
    writer.add_scalar('Loss', loss, it)


def train(model, optimizer, dataloader, loss_fn, device, writer, epochs=50):
    """
    Training loop.
    """
    total_it = 0
    for epoch in range(epochs):
        iterator_bar = tqdm(dataloader)
        for sample, _ in iterator_bar:
            optimizer.zero_grad()
            sample = sample.to(device)
            reconstruction, means, stds = model(sample)
            loss = loss_fn(reconstruction, sample, means, stds)
            iterator_bar.set_description('Epoch {} - Training Loss: {}'.format(epoch, loss))
            iterator_bar.refresh()
            log(writer, sample, reconstruction, loss, total_it)
            loss.backward()
            optimizer.step()
            total_it += 1

    return model


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Training starting on device {}".format(device))
    writer = SummaryWriter('/blue/vemuri/josebouza/projects/VAE/logs/runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))

    dataset = MNIST('/blue/vemuri/josebouza/data/mnist/', 
                    download=True,
                    transform=transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.5,), std=(0.5,))]))
    dataloader = DataLoader(dataset, 
                            batch_size=256,
                            num_workers=4)
    model = VAE(input_channels=1, n_layers=8).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    train(model, optimizer, dataloader, VAE_loss, device, writer)
    torch.save(model.state_dict(), '/blue/vemuri/josebouza/projects/VAE/model.pt')
