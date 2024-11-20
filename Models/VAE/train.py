import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(models_dir)
from Util.loadData import create_hanzi_dataloaders

latent_dims = 2
num_epochs = 1
batch_size = 32
image_size = 32
capacity = 64
learning_rate = 1e-3
variational_beta = 1
use_gpu = True
exp_name = datetime.datetime.now().strftime("%m%d-%H%M%S")
exp_path = f'./Models/VAE/checkpoints/{exp_name}'
os.makedirs(exp_path, exist_ok=True)
img_transform = transforms.Compose([
    transforms.ToTensor(),
])


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 16 x 16
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 8 x 8
        self.fc_mu = nn.Linear(in_features=c*2*8*8, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=c*2*8*8, out_features=latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        c = capacity
        self.fc = nn.Linear(in_features=latent_dims, out_features=c*2*8*8)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), capacity*2, 8, 8)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x

class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 1024), x.view(-1, 1024), reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + variational_beta * kldivergence


if __name__=="__main__":
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    vae = VariationalAutoencoder().to(device)
    train_dataloader = create_hanzi_dataloaders(root_dir="./Dataset/hanzi/processed", batch_size=batch_size,preprocess=img_transform)
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)
    vae.train()

    train_loss_avg = []
    print('Training ...')
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        train_loss_avg.append(0)
        num_batches = 0
        for image_batch, _ in train_dataloader:
            image_batch = image_batch.to(device)
            image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
            loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_avg[-1] += loss.item()
            num_batches += 1
        train_loss_avg[-1] /= num_batches

    with open(f'{exp_path}/training_loss.txt', 'w') as f:
        for loss in train_loss_avg:
            f.write(f"{loss}\n")
    torch.save(vae.state_dict(), f'{exp_path}/my_vae.pth')
    
    
    fig = plt.figure()
    plt.plot(train_loss_avg)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'{exp_path}/training_loss.png')

    print(f'Training Complete. \nLoss saved to training_loss.txt\nModel saved to saves/my_vae.pth')