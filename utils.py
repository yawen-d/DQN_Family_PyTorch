import gym
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T


# input extraction

# define the resize transfomation
resize = T.Compose([T.ToPILImage(),
                    T.Resize((84,84), interpolation=Image.CUBIC),
                    T.Grayscale(num_output_channels=1),
                    T.ToTensor()])

# get a resized screen
def get_screen(env):
    '''
    return a resized screen (tensor) in (BCHW)
    '''
    # Returned screen is size (1000,1000,3), need to transpose to 
    # torch order (CHW)
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    # print(type(screen), screen.shape)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    
    # Resize, and add a batch dimension (HW)
    screen = resize(screen).squeeze(0)
    return screen
    
# show the screen
def show_screen(screen):
    plt.figure()
    plt.imshow(screen.cpu().numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()