#%%
import action_cgan
import torch
import torch as T
from scene_extractor import Extractor
from matplotlib import pyplot as plt
import numpy as np
from simulator import Simulator
from random import sample
#%%
def setup(n_samples):
    WIDTH = 12
    NOISE_DIM = 10
    S_CHAN = 4
    A_CHAN = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = action_cgan.Generator(WIDTH, NOISE_DIM, S_CHAN, A_CHAN).to(device)
    #discriminator = action_cgan.Discriminator(WIDTH, S_CHAN, A_CHAN).to(device)
    generator.load_state_dict(torch.load("gen_state.pt"))
    generator.noise_dim = 10
    #discriminator.load_state_dict(torch.load("disc_state.pt"))
    loader = Extractor("rollouts/solutions")
    X, Y, info = loader.extract(n=n_samples, stride=12, n_channels=3, 
                    size=(WIDTH, WIDTH), r_fac=4.5, grayscale=True)
    return generator, np.array(X), info

def gen_actions(gen, X):
    X = T.tensor(X, dtype=T.float)
    z = torch.randn(X.shape[0], gen.noise_dim)

    A = gen(z,X).detach().abs().numpy()
    # Thresholding
    A *= A>0.15
    return A

def pic_to_values(pic):
    X, Y = 0, 0
    for y in range(pic.shape[0]):
        for x in range(pic.shape[1]):
            if pic[y,x]:
                X += pic[y,x]*x
                Y += pic[y,x]*y
    summed = np.sum(pic)
    X /= pic.shape[0]*summed
    Y /= pic.shape[0]*summed
    R = np.sqrt(summed)/pic.shape[0]
    return X, Y, R

def actions_to_solutions(A, info):
    V = []
    for i, (a1, a2) in enumerate(A):
        inf = info[i]
        rad, con, init = inf['r'], inf['con_pos'], inf['init_pos']
        o_radius, a_radius = inf['orig_radius'], inf['action_radius']
        #plt.imshow(a1)
        #plt.show()
        x1, y1, r1 = pic_to_values(a1)
        x2, y2, r2 = pic_to_values(a2)
        r = (r1+r2)/2

        #print("diff:", int(x2*2*rad-rad), int(y2*2*rad-rad))
        x, y = con[0]
        #print("r_diff:", abs(r1-r2))
        x1 = int(x-rad+(rad*2*x1))
        y1 = int(y-rad+(2*rad*(1-y1)))
        x2 = int(x-rad+(rad*2*x2))
        y2 = int(y-rad+(2*rad*(1-y2)))
        r = int(2*rad*r)
        #print("pred:", x2, y2, r2)
        #x, y = con[1]
        #print("orig:", int(x), int(y), int(a_radius))
        V.append(((x1,y1), (x2, y2), r))
    return V

def simulate_solutions(V, info):
    sim = Simulator()
    for i, inf in enumerate(info):
        rad, con, init = inf['r'], inf['con_pos'], inf['init_pos']
        o_radius, a_radius = inf['orig_radius'], inf['action_radius']
        pos1, pos2, r = V[i]
        sim.setup_space()
        sim.add_ball(o_radius, con[0], color=[0,200,0,255])
        sim.add_ball(r, pos1, color=[200,0,0,255])
        sim.run(n_frames=200)
    sim.quit()
    


#%%
if __name__ == "__main__":
    generator, X, info = setup(100)
    selection = np.random.randint(len(X), size=(10,))
    X = X[selection]
    info = [info[i] for i in selection]
    A = gen_actions(generator, X)
    V = actions_to_solutions(A, info)
    simulate_solutions(V, info)

# %%


# %%
