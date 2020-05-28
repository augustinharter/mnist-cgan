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
from scene_extractor import Extractor
import random
import cv2

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
WIDTH = 12
NOISE_DIM = 10
DATA_SIZE = 5000
S_CHANNELS = 4
A_CHANNELS = 2
loader = Extractor("rollouts/test")
batch_size = 32
X, Y = loader.extract(n=DATA_SIZE, stride=12, n_channels=3, 
                    size=(WIDTH, WIDTH), r_fac=4.5, grayscale=True)
data_set = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False)

# %%
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        global WIDTH
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
            nn.Linear(WIDTH**2*(S_CHANNELS+A_CHANNELS), 1024),
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
    
    def forward(self, actions, scenes):
        actions = actions.view(actions.size(0), WIDTH**2*A_CHANNELS)
        c = scenes.view(scenes.size(0), WIDTH**2*S_CHANNELS)
        #x = self.encoder(x)
        x = torch.cat([actions, c], 1)
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
    global WIDTH, NOISE_DIM
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
            nn.Linear(NOISE_DIM + WIDTH**2*S_CHANNELS, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, WIDTH**2*A_CHANNELS),
            nn.Tanh()
        )

    def forward(self, z, scenes):
        z = z.view(z.size(0), NOISE_DIM)
        c = scenes.view(scenes.size(0), WIDTH**2*S_CHANNELS)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), A_CHANNELS, WIDTH, WIDTH)
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
def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, scenes):
    global NOISE_DIM
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, NOISE_DIM)).to(device)
    actions = generator(z, scenes)
    validity = discriminator(actions, scenes)
    #tqdm.write(str(actions[0]))
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


# %%
def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, actions, scenes):
    global NOISE_DIM
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(actions, scenes)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    
    # train with fake images
    z = Variable(torch.randn(batch_size, NOISE_DIM)).to(device)
    gen_actions = generator(z, scenes)
    fake_validity = discriminator(gen_actions, scenes)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()

#%%
def generate():
    num_cells = 4*8
    z = Variable(torch.randn(num_cells, NOISE_DIM)).to(device)
    selection = np.random.randint(len(X), size=(num_cells,))
    scenes = torch.tensor(np.array(X)[selection], dtype=torch.float).to(device)
    actions= np.array(Y)[selection]

    gen_actions = generator(z, scenes).abs()

    s = scenes.detach().numpy()
    g = gen_actions.detach().numpy()
    green = np.max(np.stack((0.5*s[:,0],s[:,1],0.5*s[:,2]), axis=-1), axis=-1).reshape(num_cells,1,12,12)
    blue = s[:,3].reshape(num_cells,1,12,12)
    red = np.max(np.stack((0.5*actions[:,0],actions[:,1]), axis=-1), axis=-1).reshape(num_cells,1,12,12)
    orig = np.concatenate((red, green, blue), axis=1)
    red = np.max(np.stack((0.5*g[:,0],g[:,1]), axis=-1), axis=-1).reshape(num_cells,1,12,12)
    gen = np.concatenate((red, green, blue), axis=1)
    #print(combined)
    combined = np.concatenate((orig, gen), axis=1).reshape(2*num_cells,3,12,12)
    grid = make_grid(torch.tensor(combined), nrow=8, normalize=True)
    plt.imshow(grid[0])
    plt.show(block=False)
    save_image(grid, "action_result.png")

# %%
def training(safe=True):
    num_epochs = 100
    n_critic = 5
    display_step = 10
    for epoch in tqdm(range(num_epochs)):
        #tqdm.write(f'Starting epoch {epoch}...')
        for i, (scenes, actions) in enumerate(data_loader):
            step = epoch * len(data_loader) + i + 1
            real_scenes = Variable(scenes.float()).to(device)
            real_actions = Variable(actions.float()).to(device)
            generator.train()

            #print(real_actions[0,0])
            
            d_loss = discriminator_train_step(len(real_scenes), discriminator,
                                            generator, d_optimizer, criterion,
                                            real_actions, real_scenes)
            

            g_loss = generator_train_step(len(real_scenes), discriminator, generator,
                                            g_optimizer, criterion, real_scenes)
            
            writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)  
        
        tqdm.write(f"{d_loss} {g_loss}")
        
        if epoch>0 and epoch%display_step ==0:
            if safe:
                torch.save(discriminator.state_dict(), 'disc_state.pt')
                torch.save(generator.state_dict(), 'gen_state.pt')
            generate()
        #tqdm.write('Done!')


# %%
# TRAINING
only_generate = False
if only_generate:
    generator.load_state_dict(torch.load("gen_state.pt"))
    discriminator.load_state_dict(torch.load("disc_state.pt"))
    generate()
else:
    #generator.load_state_dict(torch.load("gen_state.pt"))
    #discriminator.load_state_dict(torch.load("disc_state.pt"))
    training()
# %%