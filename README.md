# Mario Reinforcement Learning Project

🎮 An experimental project exploring reinforcement learning by teaching an AI agent to play Super Mario Bros.

## Project Overview

This project uses Double Deep Q-Learning (DDQN) to train an AI agent to play Super Mario Bros. 
The agent learns through trial and error, developing strategies to progress through the game's levels without explicit programming of game mechanics.

### Current Status: Exploration Phase
- Successfully implemented basic DDQN architecture
- Agent can interact with the game environment
- Currently experimenting with different learning parameters to finish World 1-1
- Working on improving the agent's performance and learning efficiency

## Technical Details

### Technologies Used
- Python
- PyTorch for deep learning
- OpenAI Gym for environment interaction
- NES-py for Nintendo emulation
- gym-super-mario-bros environment

### Key Features
- Custom CNN architecture for processing game frames
- Experience replay buffer with prioritized sampling
- Frame skipping and image preprocessing for efficient learning
- Epsilon-greedy exploration strategy
- Reward shaping based on game progress

### Model Architecture
- Input: 4 stacked grayscale frames (84x84 pixels)
- Convolutional layers for feature extraction
- Fully connected layers for Q-value prediction
- Outputs: Action values for possible game controls

## Project Structure
```
├── play.py           # Script for running trained models
├── test.ipynb        # Training notebook and experiments
└── README.md
```

## Current Challenges & Next Steps
- [ ] Fine-tuning hyperparameters for better learning
- [ ] Implementing additional learning strategies
- [ ] Improving exploration vs exploitation balance
- [ ] Optimizing reward function
- [ ] Enhancing model architecture

## Installation & Requirements
(Coming soon)

## Results
The project is in the exploration phase, with the agent currently learning basic movement and obstacle avoidance. 
Performance metrics and videos will be added as the project progresses.

## Acknowledgments
This project builds upon various RL resources and implementations from the research community thanks to:
- OpenAI Gym
- gym-super-mario-bros environment
- PyTorch community

## License
MIT License
