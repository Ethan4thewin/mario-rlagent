import torch
from pathlib import Path
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from torch import nn
from torchvision import transforms as T
import numpy as np
import time
import random

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            total_reward += reward
            if done:
                break
        
        if len(step_result) == 5:
            return obs, total_reward, terminated, truncated, info
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

def play_trained_mario(checkpoint_path, episodes=1):
    MOVEMENT_OPTIONS = [
        ["right"],
        ["right", "A"]
    ]
    
    # Set up environment
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        render_mode="human",
        apply_api_compatibility=True
    )
    
    # Configure environment with action options
    env = JoypadSpace(env, MOVEMENT_OPTIONS)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create and load the model
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    model = MarioNet(state_dim, action_dim).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    try:
        # Since the model always does the same thing, we can just do one good run
        print("\nStarting Mario's Run")
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        
        total_reward = 0
        done = False
        steps = 0
        stuck_counter = 0
        last_x_pos = 0
        
        while not done:
            state_t = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state_t = torch.tensor(state_t, device=device).unsqueeze(0)

            with torch.no_grad():
                action_values = model(state_t, model="online")
                # For 2 actions, we'll sometimes randomly pick between them if they're close in value
                top_action = torch.argmax(action_values, dim=1).item()
                second_best = 1 - top_action  # Since we only have 2 actions (0 or 1)
                
                # If the actions are close in value, randomly choose between them
                if abs(action_values[0][top_action] - action_values[0][second_best]) < 5.0:
                    action_idx = random.choice([top_action, second_best])
                else:
                    action_idx = top_action

            step_result = env.step(action_idx)
            
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            # Get Mario's x position
            current_x_pos = info.get('x_pos', 0)
            
            # Check if Mario is stuck
            if abs(current_x_pos - last_x_pos) < 1:
                stuck_counter += 1
            else:
                stuck_counter = 0
            
            # If Mario is stuck for too long, try a different action
            if stuck_counter > 20:
                action_idx = random.randint(0, action_dim - 1)
                stuck_counter = 0
            
            last_x_pos = current_x_pos
            total_reward += reward
            state = next_state
            steps += 1
            
            # Print progress
            if steps % 100 == 0:
                print(f'Steps: {steps}, X Position: {current_x_pos}, Reward: {total_reward}')
            
            if done or info.get('flag_get', False):
                print(f'\nRun finished!')
                print(f'Total steps: {steps}')
                print(f'Final reward: {total_reward}')
                print(f'Final x position: {current_x_pos}')
                if info.get('flag_get', False):
                    print('Mario reached the flag!')
                elif info.get('life', 2) < 2:
                    print('Mario lost a life')
                break
            
    except Exception as e:
        print(f"Error during gameplay: {e}")
    finally:
        # Add a delay before closing
        time.sleep(2)
        env.close()

if __name__ == "__main__":
    print("Starting Mario gameplay...")
    model_path = "trained_mario.chkpt"
    play_trained_mario(model_path, episodes=1)
    print("Gameplay finished.")