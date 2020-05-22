#%%
import torch
from torch import nn
import tqdm
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision.datasets import MNIST
from matplotlib import pyplot as plt
from PIL import Image

#%%
batch_size = 32
data_loader = torch.utils.data.DataLoader(
  MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
  batch_size=batch_size, shuffle=True)

#%%
def check_input():
  data_loader = list(data_loader)
  print(data_loader[0][0].shape)
  grid = make_grid(data_loader[0][0])
  plt.imshow(grid.transpose(0,2).transpose(0,1))
#check_input()
