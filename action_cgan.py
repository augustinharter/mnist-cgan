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
DATA_SIZE = 1000
loader = Extractor("rollouts/test")
batch_size = 32
X, Y = loader.extract(n=DATA_SIZE, stride=12, n_channels=3, 
                    size=(WIDTH, WIDTH), r_fac=4.5, grayscale=False)
X_test, Y_test = X[:10], Y[:10]
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
            nn.Linear(WIDTH**2*5, 1024),
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
        actions = actions.view(actions.size(0), WIDTH**2)
        c = scenes.view(scenes.size(0), WIDTH**2*4)
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
            nn.Linear(NOISE_DIM + WIDTH**2*4, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, WIDTH**2),
            nn.Tanh()
        )

    def forward(self, z, scenes):
        z = z.view(z.size(0), NOISE_DIM)
        c = scenes.view(scenes.size(0), WIDTH**2*4)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), WIDTH, WIDTH)
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
    action = generator(z, scenes)
    validity = discriminator(action, scenes)
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
    num_ex = 10
    z = Variable(torch.randn(num_ex, NOISE_DIM)).to(device)
    scenes = torch.tensor(X.sample(10), dtype=torch.float).to(device)
    actions= torch.tensor(Y.sample(10), dtype=torch.float).to(device)

    # %%
    gen_actions = generator(z, scenes)[:,None,:]

    # %%
    combined = Variable(torch.cat((scenes, actions, gen_actions), dim=1).view(-1,1,12,12))
    grid = make_grid(combined, nrow=6, normalize=True)
    #plt.imshow(grid)
    save_image(grid, "action_result.jpg")

# %%
def training():
    num_epochs = 500
    n_critic = 5
    display_step = 50
    for epoch in tqdm(range(num_epochs)):
        #tqdm.write(f'Starting epoch {epoch}...')
        for i, (scenes, actions) in enumerate(data_loader):
            step = epoch * len(data_loader) + i + 1
            real_scenes = Variable(scenes.float()).to(device)
            real_actions = Variable(actions.float()).to(device)
            generator.train()
            
            d_loss = discriminator_train_step(len(real_scenes), discriminator,
                                            generator, d_optimizer, criterion,
                                            real_actions, real_scenes)
            

            g_loss = generator_train_step(len(real_scenes), discriminator, generator,
                                            g_optimizer, criterion, real_scenes)
            
            writer.add_scalars('scalars', {'g_loss': g_loss, 'd_loss': d_loss}, step)  
        
        if epoch>0 and epoch%display_step ==0:
            generate(X_test, Y_test)
        #tqdm.write('Done!')


# %%
# TRAINING
pretrained = False
if not pretrained:
    training()
    torch.save(generator.state_dict(), 'cgan_state.pt')
else:
    generator.load_state_dict(torch.load("generator_state.pt"))

# %%

generate(X_test, Y_test)