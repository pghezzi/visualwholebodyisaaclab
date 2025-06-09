# MultiState Environment for Isaac Lab

This repository contains a migrated version of the MultiState environment from IsaacGymEnvs to Isaac Lab. The environment is designed for training reinforcement learning agents to control a robot with multiple states.

## Installation

1. Install Isaac Lab following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

2. Install additional dependencies:
```bash
pip install torch numpy gym skrl
```

## Project Structure

```
high-level-isaaclab/
├── config/
│   └── multistate_cfg.yaml    # Environment configuration
├── envs/
│   └── multistate_env.py      # Environment implementation
├── train_multistate.py        # Training script
└── README.md                  # This file
```

## Configuration

The environment can be configured through the `config/multistate_cfg.yaml` file. Key configuration parameters include:

- Simulation settings (timestep, gravity, etc.)
- Scene settings (ground plane, lighting, etc.)
- Environment settings (number of environments, episode length, etc.)
- Robot settings (URDF path, initial pose, joint limits, etc.)
- Feature settings (number of features, encoding dimension, etc.)
- Training settings (wandb integration, experiment directory, etc.)

## Usage

### Training

To train a new agent:

```bash
python train_multistate.py
```

### Evaluation

To evaluate a trained agent:

```bash
python train_multistate.py --eval --checkpoint path/to/checkpoint.pt
```

## Key Changes from IsaacGymEnvs

1. Configuration System:
   - Replaced YAML-based configuration with Isaac Lab's `configclass` system
   - Moved simulation parameters to individual actor configurations
   - Simplified configuration structure

2. Environment Implementation:
   - Using Isaac Lab's `DirectRLEnv` base class
   - Leveraging Isaac Lab's scene and simulation management
   - Improved sensor integration

3. Training Pipeline:
   - Maintained compatibility with skrl training framework
   - Updated to use Isaac Lab's device management
   - Improved experiment tracking with wandb integration

## Notes

- The robot URDF path needs to be updated in the configuration file
- Joint limits should be updated based on your specific robot
- Camera and sensor configurations can be adjusted based on your needs

## License

This project is licensed under the same terms as Isaac Lab. 
