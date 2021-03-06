#%%
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import tqdm

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
# %%
transform = transforms.Compose([
        #transforms.Grayscale(),
        transforms.ToTensor()
        ,transforms.Normalize(mean=(0.5,), std=(0.5,))
])
# %%
batch_size = 32
data_loader = torch.utils.data.DataLoader(
    MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)


# %%
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self._model = nn.Sequential(
            nn.Linear(1034, 128),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm2d(128),
            nn.Flatten(start_dim=1),
            nn.Linear(7*7*128, 1024),
            nn.BatchNorm1d(1024),
        )

        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        #x = self.encoder(x)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


# %%
class View(nn.Module):
    #Changing the Shape
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

#%%
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self._model = nn.Sequential(
            nn.Linear(110, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7*7*128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(7*7*128),
            View((-1, 128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
        )

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
        #return out


# %%
generator = Generator().to(device)
discriminator = Discriminator().to(device)


# %%
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)


# %%
writer = SummaryWriter()


# %%
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


# %%
def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()


# %%
def training():
    num_epochs = 50
    n_critic = 5
    display_step = 50
    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch), end=' ')
        for i, (images, labels) in tqdm(enumerate(data_loader)):
            
            step = epoch * len(data_loader) + i + 1
            real_images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            generator.train()
            
            d_loss = discriminator_train_step(len(real_images), discriminator,
                                            generator, d_optimizer, criterion,
                                            real_images, labels)
            

            g_loss = generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion)
            
            writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)  
            
            if step % display_step == 0:
                generator.eval()
                z = Variable(torch.randn(9, 100)).to(device)
                labels = Variable(torch.LongTensor(np.arange(9))).to(device)
                sample_images = generator(z, labels).unsqueeze(1)
                grid = make_grid(sample_images, nrow=3, normalize=True)
                writer.add_image('sample_image', grid, step)
        print('Done!')


# %%
# TRAINING
pretrained = False
if not pretrained:
    training()
    torch.save(generator.state_dict(), 'generator_state.pt')
else:
    generator.load_state_dict(torch.load("generator_state.pt"))

# %%
z = Variable(torch.randn(100, 100)).to(device)
labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).to(device)


# %%
images = generator(z, labels).unsqueeze(1)
print(images.size())

# %%
grid = make_grid(images, nrow=10, normalize=True)
save_image(grid, "grid_result.jpg")

# %%
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
ax.axis('off')


# %%
def generate_digit(generator, digit):
    z = Variable(torch.randn(1, 100)).to(device)
    label = torch.LongTensor([digit]).to(device)
    img = generator(z, label).data.cpu()
    img = 0.5 * img + 0.5
    return transforms.ToPILImage()(img)


# %%
generate_digit(generator, 8)


# %%


