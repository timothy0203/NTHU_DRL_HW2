import torch
import torch.nn as nn
import random
from tqdm import tqdm
import pickle
import numpy as np
import collections 
import cv2
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

class DDQNSolver(nn.Module):

    def __init__(self, input_shape, n_actions):
        super(DDQNSolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
    

STATE_SPACE = (1, 84, 84)
ACTION_SAPCE = 12
EXPLORATION = 0.001
class Agent:

    def __init__(self):

        self.state_space = STATE_SPACE
        self.action_space = ACTION_SAPCE
        self.exploration_rate = EXPLORATION
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model_path = "./DDQN_6129.pt"
        # self.model_path = "./DDQN_7769_e0.pt" # 5193
        # self.model_path = "./DDQN_7772_e0.pt" # 6119
        # self.model_path = "./DDQN_7732_e0.pt" # 4348
        # self.model_path = "./DDQN_8020_e0.pt" # 
        # self.model_path = "./DDQN_7888_e0.pt" # 6987
        # self.model_path = "./DDQN_8263_e0.pt" # 5329.5
        # self.model_path = "./DDQN_8619_e0.pt" # 5329.5
        # self.model_path = "./DDQN_8010_e0.pt" # 
        # self.model_path = "./DDQN_7894.pt" # 4986
        # self.model_path = "./DDQN_6708.pt"
        # self.model_path = "./DDQN_8163_e0.pt" # 0, 6161.92
        # self.model_path = "./DDQN_7646_epi_50.pt" # 7457
        self.model_path = "./112062530_hw2_data" # 7457
        self.keep_action = 0

        self.local_net = DDQNSolver(STATE_SPACE, ACTION_SAPCE)
        # self.local_net = DDQNSolver(state_space, action_space).to(self.device)
        # self.target_net = DDQNSolver(state_space, action_space).to(self.device)
            
        self.local_net.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
        # self.target_net.load_state_dict(torch.load("dq2.pt", map_location=torch.device(self.device)))
                    
        # self.optimizer = torch.optim.Adam(self.local_net.parameters(), lr=lr)
        # self.copy = 5000  # Copy the local model weights into the target network every 5000 steps
        self.step = 0

        # self.memory_sample_size = batch_size
        
        # Learning parameters
        # self.gamma = gamma
        # self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss
        # self.exploration_max = exploration_max
        # self.exploration_min = exploration_min
        # self.exploration_decay = exploration_decay

    def preprocess_state(self, frame):
        # Step 1: ProcessFrame - Resize and grayscale
        if frame.size == 240 * 256 * 3:
            img = np.reshape(frame, [240, 256, 3]).astype(np.float32)
        else:
            raise ValueError("Unknown resolution.")
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])

        # Step 2: ImageToPyTorch - Change channel ordering
        x_t = np.moveaxis(x_t, -1, 0)  # Change from HxWxC to CxHxW

        # Step 3: ScaledFloatFrame - Scale pixels
        x_t = x_t.astype(np.float32) / 255.0

        return x_t

    # checker version
    def act(self, state):
        # Epsilon-greedy action
        if self.step % 4 == 0:
            state = self.preprocess_state(state)
            state = torch.Tensor(np.array([state]))
            if random.random() < self.exploration_rate:  
                action = torch.tensor([[random.randrange(self.action_space)]])
            # Local net is used for the policy
            else:
                action =  torch.argmax(self.local_net(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()
            self.keep_action = int(action[0])
        self.step += 1
        return self.keep_action