from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.config import configclass
from isaaclab.envs.mdp import MDPCfg

import torch
import torch.nn as nn
import os
import numpy as np

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

@configclass
class MultiStateEnvCfg(DirectRLEnvCfg):
    # Simulation config
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1/120,
        gravity=(0.0, 0.0, -9.81)
    )
    
    # Scene config
    scene: InteractiveSceneCfg = InteractiveSceneCfg()
    
    # Environment config
    num_envs: int = 34
    max_episode_length: int = 1500
    control_frequency: int = 60
    
    # Task specific config
    enable_debug_vis: bool = False
    camera_mode: str = "full"
    small_value_set_zero: bool = False
    last_commands: bool = False
    record_video: bool = False
    use_tanh: bool = False
    near_goal_stop: bool = False
    obj_move_prob: float = 0.0
    
    # Robot config
    robot_start_pose: tuple = (-2.00, 0, 0.55)
    
    # Feature config
    num_features: int = 1024
    encode_dim: int = 128
    no_feature: bool = False
    
    # Training config
    wandb: bool = False
    wandb_project: str = "isaac-lab-multistate"
    wandb_name: str = "multistate-training"
    experiment_dir: str = "experiments"

# Policy network definition
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim, use_tanh=False, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", deterministic=False):
        Model.__init__(self, observation_space, action_space, device)
        transform_func = torch.distributions.transforms.TanhTransform() if use_tanh else None
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, transform_func=transform_func, deterministic=deterministic)

        self.num_features = num_features
        self.encode_dim = encode_dim
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.ELU(),
                nn.Linear(512, self.encode_dim)
            )
            
        self.net = nn.Sequential(
            nn.Linear(self.num_observations - self.num_features + self.encode_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        if self.num_features > 0:
            features_encode = self.feature_encoder(inputs["states"][..., :self.num_features])
            actions = self.net(torch.cat([inputs["states"][..., self.num_features:], features_encode], dim=-1))
        else:
            actions = self.net(inputs["states"])
        return actions, self.log_std_parameter, {}

# Value network definition
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.num_features = num_features
        self.encode_dim = encode_dim
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.num_features, 512),
                nn.ELU(),
                nn.Linear(512, self.encode_dim)
            )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations - self.num_features + self.encode_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        if self.num_features > 0:
            feature_encode = self.feature_encoder(inputs["states"][..., :self.num_features])
            return self.net(torch.cat([inputs["states"][..., self.num_features:], feature_encode], dim=-1)), {}
        else:
            return self.net(inputs["states"]), {}

def get_trainer(is_eval=False):
    set_seed(43)
    
    # Load configuration
    cfg = MultiStateEnvCfg()
    
    # Create environment
    env = create_env(cfg)
    device = env.rl_device
    
    # Create memory buffer
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
    
    # Create models
    num_features = 0 if cfg.no_feature else cfg.num_features
    encode_dim = 0 if cfg.no_feature else cfg.encode_dim
    
    models_ppo = {
        "policy": Policy(
            env.observation_space, 
            env.action_space, 
            device, 
            num_features=num_features, 
            encode_dim=encode_dim, 
            use_tanh=cfg.use_tanh, 
            clip_actions=cfg.use_tanh, 
            deterministic=is_eval
        ),
        "value": Value(
            env.observation_space, 
            env.action_space, 
            device, 
            num_features=num_features, 
            encode_dim=encode_dim
        )
    }
    
    # Configure PPO
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo.update({
        "rollouts": 24,
        "learning_epochs": 5,
        "mini_batches": 6,
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": 5e-4,
        "learning_rate_scheduler": KLAdaptiveRL,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "value_loss_scale": 1.0,
        "kl_threshold": 0,
        "rewards_shaper": None,
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": device},
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "experiment": {
            "write_interval": 24,
            "checkpoint_interval": 500,
            "directory": cfg.experiment_dir,
            "experiment_name": cfg.wandb_name,
            "wandb": cfg.wandb
        }
    })
    
    if cfg.wandb:
        cfg_ppo["experiment"]["wandb_kwargs"] = {
            "project": cfg.wandb_project,
            "tensorboard": False,
            "name": cfg.wandb_name
        }
    
    # Create agent
    agent = PPO(
        models=models_ppo,
        memory=memory,
        cfg=cfg_ppo,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # Configure trainer
    cfg_trainer = {
        "timesteps": 1000000,  # Default value, can be overridden
        "headless": True
    }
    
    # Create trainer
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    return trainer

if __name__ == "__main__":
    trainer = get_trainer()
    trainer.train() 