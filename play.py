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
import sys
import traceback

# Capture screen
import cv2
import win32gui
from PIL import ImageGrab
import cv2

# Helper function to give window's properties
def diagnose_window(window_name="SuperMarioBros-1-1-v0"):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        print("\nWindow Diagnostics:")
        print(f"Window Handle: {hwnd}")
        
        # Get window properties
        rect = win32gui.GetWindowRect(hwnd)
        print(f"Window Rect: {rect}")
        
        client_rect = win32gui.GetClientRect(hwnd)
        print(f"Client Rect: {client_rect}")
        
        # Get window style
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
        print(f"Window Style: {hex(style)}")
        
        # Get extended style
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        print(f"Extended Style: {hex(ex_style)}")
        
        # Get window class name
        class_name = win32gui.GetClassName(hwnd)
        print(f"Window Class: {class_name}")
    else:
        print("Window not found")

# Helper function to capture the game window
def capture_game_window(window_name="SuperMarioBros-1-1-v0"):
    """Capture the game window using PIL's ImageGrab."""
    try:
        # Find window by exact name
        hwnd = win32gui.FindWindow(None, window_name)
        
        if not hwnd:
            print(f"Could not find window: {window_name}")
            return None

        # Get window rectangle
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        # Get client area dimensions
        client_rect = win32gui.GetClientRect(hwnd)
        client_width = client_rect[2] - client_rect[0]
        client_height = client_rect[3] - client_rect[1]
        
        # Calculate borders
        border_x = (width - client_width) // 2
        border_y = height - client_height - border_x

        try:
            # Capture the entire window area
            screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
            
            # Convert PIL image to numpy array
            frame = np.array(screenshot)
            
            # Convert RGB to BGR (for OpenCV compatibility)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Crop to client area
            cropped = frame[border_y:border_y + client_height, 
                          border_x:border_x + client_width]

            return cropped

        except Exception as e:
            print(f"Error during capture: {e}")
            return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

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
    # Define the expanded action space
    MOVEMENT_OPTIONS = [
        ["NOOP"],
        ["right"],
        ["right", "B"],
        ["right", "A"],
        ["right", "A", "B"],
        ["B"]
    ]
    
    # Set up environment
    env = gym_super_mario_bros.make(
        "SuperMarioBros-1-1-v0",
        render_mode="human",
        apply_api_compatibility=True
    )
    
    # Configure environment
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

    # Initialize state and give the window time to appear
    print("\nStarting Mario's Run")
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    
    # Wait longer for the window to appear and render
    print("Waiting for game window to initialize...")
    for i in range(5):
        print(f"Initialization attempt {i+1}/5...")
        time.sleep(1)
        if win32gui.FindWindow(None, "SuperMarioBros-1-1-v0"):
            print("Window found!")
            time.sleep(1)
            break

    # Initialize video capture
    video_writer = None
    print("\nInitializing video capture...")
    
    # Try to capture frame multiple times
    for i in range(5):
        test_frame = capture_game_window()
        if test_frame is not None:
            break
        time.sleep(1)

    if test_frame is not None:
        height, width = test_frame.shape[:2]
        print(f"Captured frame dimensions: {width}x{height}")
        
        # Create output directory if it doesn't exist
        output_dir = Path("gameplay_videos")
        output_dir.mkdir(exist_ok=True)
        
        # Create video writer with unique timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = output_dir / f"mario_gameplay_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc, 
            30.0,
            (width, height)
        )
        
        if video_writer.isOpened():
            print(f"Successfully initialized video recording to: {video_path}")
        else:
            print("Failed to initialize video writer")
            video_writer = None
    else:
        print("Failed to capture game window after multiple attempts")
        video_writer = None

    try:
        total_reward = 0
        done = False
        steps = 0
        frames_written = 0
        
        # Dictionary to map action indices to descriptions
        action_names = {
            0: "NOOP",
            1: "Walk Right",
            2: "Run Right",
            3: "Walk Jump",
            4: "Run Jump",
            5: "Charge Run"
        }
        
        while not done:
            # Get model's action prediction
            state_t = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state_t = torch.tensor(state_t, device=device).unsqueeze(0)

            # Capture and save frame
            if video_writer is not None:
                frame = capture_game_window()
                if frame is not None:
                    video_writer.write(frame)
                    frames_written += 1
                    if frames_written % 100 == 0:
                        print(f"Frames written: {frames_written}")

            with torch.no_grad():
                action_values = model(state_t, model="online")
                action_idx = torch.argmax(action_values, axis=1).item()

            step_result = env.step(action_idx)
            
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if steps % 20 == 0:
                print(f'Step: {steps:4d} | Position: {info["x_pos"]:4d} | ' \
                      f'Action: {action_names[action_idx]:10s} | ' \
                      f'Q-values: {action_values[0].cpu().numpy()}')
            
            if done or info.get('flag_get', False):
                print(f'\nRun Complete!')
                print(f'Total steps: {steps}')
                print(f'Final reward: {total_reward}')
                print(f'Final x position: {info["x_pos"]}')
                if info.get('flag_get', False):
                    print('Mario reached the flag!')
                elif info.get('life', 2) < 2:
                    print('Mario lost a life')
                print("\nFinal statistics:")
                print(f'Average speed: {info["x_pos"] / steps:.2f} pixels/step')
                break
            
    except Exception as e:
        print(f"Error during gameplay: {e}")
        traceback.print_exc()
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"\nVideo saved with {frames_written} frames")
        env.close()

if __name__ == "__main__":
    print("Starting Mario gameplay...")
    model_path = "models/continued_training_mario.chkpt"
    play_trained_mario(model_path, episodes=1)
    print("Gameplay finished.")