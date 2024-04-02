import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from skimage import color
from skimage.transform import resize
from tensorflow.keras.models import load_model


FRAME_SKIP = 4  # Number of frames to skip/repeat action
FRAME_STACK_SIZE = 4  # Number of frames to stack
STATE_SHAPE = (84, 84, FRAME_STACK_SIZE)  # Adjust the channel dimension
# ACTION_SIZE = env.action_space.n
# BATCH_SIZE = 8192
# BATCH_SIZE = 1024
BATCH_SIZE = 2048
# BATCH_SIZE = 32
MEMORY_SIZE = 10000
GAMMA = 0.99
EPSILON_MAX = 0.5
# EPSILON_MAX = 0.25
# EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.99995
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 0.00025

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # self.action_space = gym.spaces.Discrete(12)
        self.state_shape = STATE_SHAPE
        self.action_size = 12
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = 0.4
        self.target_update_counter = 0
        self.stacked_frames_var = np.zeros(STATE_SHAPE)
        self.model = self.build_model()
        # self.model.load_weights("../DDQN_model/112062530_hw2_data")
        self.model = load_model("./112062530_hw2_data_2618_e49.h5")


    def act(self, observation):
        state = self.preprocess_state(observation)
        if np.array_equal(self.stacked_frames_var, np.zeros(STATE_SHAPE)):
            self.stacked_frames_var = self.stack_frames(np.zeros(STATE_SHAPE), state, True)
        else:
            self.stacked_frames_var = self.stack_frames(self.stacked_frames_var, state, False)

        action = self.choose_action(np.expand_dims(self.stacked_frames_var, axis=0))
        return action

    def build_model(self):
        model = Sequential([
            Conv2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape),
            Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size)
        ])
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def preprocess_state(self, state):
        state = color.rgb2gray(state)
        state = resize(state, (84, 84), anti_aliasing=True)
        state = np.expand_dims(state, axis=-1)  # Keep the last dimension for stacking
        return state

    def stack_frames(self, stacked_frames, new_frame, is_new_episode):
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = np.zeros(STATE_SHAPE)
            # Because we're in a new episode, copy the same frame 4x
            for _ in range(FRAME_STACK_SIZE):
                stacked_frames[:, :, _] = new_frame[:, :, 0]
        else:
            # Shift the oldest frame out and append the new frame at the end
            stacked_frames = np.roll(stacked_frames, -1, axis=2)
            stacked_frames[:, :, -1] = new_frame[:, :, 0]
        return stacked_frames

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.model.predict(state, verbose=0)[0])