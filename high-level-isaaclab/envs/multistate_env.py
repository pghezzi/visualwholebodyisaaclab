from isaaclab.envs import DirectRLEnv
from isaaclab.envs.mdp import MDPCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ArticulationCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import torch_utils

import torch
import numpy as np
import gym.spaces as spaces
import os
import random

class MultiStateEnv(DirectRLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Initialize simulation
        self.sim = SimulationContext(cfg.sim)
        
        # Create scene
        self.scene = InteractiveScene(cfg.scene)
        
        # Add ground plane
        self.scene.add_ground_plane()
        
        # Add robot
        robot_urdf_path = os.path.join(cfg.asset_root, cfg.asset_file_robot)
        self.robot_cfg = ArticulationCfg(
            urdf_path=robot_urdf_path,
            initial_position=cfg.robot_start_pose,
            initial_orientation=(0, 0, 0, 1),
            fixed_base=not cfg.floating_base,
            joint_damping=cfg.control["damping"]["joint"],
            joint_friction=0.1
        )
        self.robot = self.scene.add_articulation(self.robot_cfg)
        
        # Add sensors
        if cfg.sensor["enable_camera"]:
            camera_cfg = CameraCfg(
                width=cfg.sensor["resized_resolution"][0],
                height=cfg.sensor["resized_resolution"][1],
                fov=60.0,
                near=0.1,
                far=100.0
            )
            self.camera = self.scene.add_camera(camera_cfg)
            
        # Add contact sensors
        contact_cfg = ContactSensorCfg()
        self.contact_sensor = self.scene.add_contact_sensor(contact_cfg)
        
        # Initialize object assets
        self._init_object_assets(cfg)
        
        # Initialize buffers
        self.num_envs = cfg.num_envs
        self.max_episode_length = cfg.max_episode_length
        self.control_frequency = cfg.control_frequency_low
        
        # Initialize observation and action spaces
        self._setup_spaces()
        
        # Initialize state buffers
        self._init_buffers()
        
        # Initialize object instances
        self.objects = {}
        self.object_heights = {}
        self.object_positions = {}
        self.object_orientations = {}
        self.object_scales = {}
        
    def _init_object_assets(self, cfg):
        """Initialize object assets from configuration."""
        self.object_assets = {}
        obj_set_path = os.path.join(cfg.asset_root, cfg.asset_file_obj)
        
        for obj_name, obj_cfg in cfg.asset_multi.items():
            obj_urdf_path = os.path.join(obj_set_path, f"{obj_name}.urdf")
            if os.path.exists(obj_urdf_path):
                self.object_assets[obj_name] = {
                    "urdf_path": obj_urdf_path,
                    "height": obj_cfg["height"],
                    "orientation": obj_cfg["orientation"],
                    "scale": obj_cfg["scale"]
                }
                
    def _setup_spaces(self):
        # Define observation space
        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space
        action_dim = self._get_action_dim()
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
    def _init_buffers(self):
        # Initialize state buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.observation_space.shape[0]),
            device=self.device
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device)
        
        # Initialize object state buffers
        self.object_positions = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_heights = torch.zeros(self.num_envs, device=self.device)
        self.object_scales = torch.ones(self.num_envs, device=self.device)
        
    def _get_obs_dim(self):
        # Calculate observation dimension based on robot state and sensors
        robot_state_dim = self.robot.get_state_dim()
        if self.cfg.sensor["enable_camera"]:
            camera_dim = self.camera.get_observation_dim()
        else:
            camera_dim = 0
        return robot_state_dim + camera_dim
        
    def _get_action_dim(self):
        # Return number of controllable joints
        return self.robot.get_num_actions()
        
    def reset(self):
        # Reset environment
        self.progress_buf.zero_()
        self.reset_buf.ones_()
        
        # Reset robot
        self.robot.reset()
        
        # Reset objects
        self._reset_objects()
        
        # Get initial observation
        obs = self._get_obs()
        
        return obs
        
    def _reset_objects(self):
        """Reset object positions and states."""
        # Randomly select objects for each environment
        for env_idx in range(self.num_envs):
            obj_name = random.choice(list(self.object_assets.keys()))
            obj_cfg = self.object_assets[obj_name]
            
            # Set object properties
            self.object_heights[env_idx] = obj_cfg["height"]
            self.object_scales[env_idx] = obj_cfg["scale"]
            self.object_orientations[env_idx] = torch.tensor(obj_cfg["orientation"], device=self.device)
            
            # Randomize object position within bounds
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(-1.0, 1.0)
            z = self.object_heights[env_idx] / 2  # Place on ground
            self.object_positions[env_idx] = torch.tensor([x, y, z], device=self.device)
            
            # Create or reset object in scene
            if env_idx not in self.objects:
                obj_cfg = ArticulationCfg(
                    urdf_path=obj_cfg["urdf_path"],
                    initial_position=self.object_positions[env_idx].cpu().numpy(),
                    initial_orientation=self.object_orientations[env_idx].cpu().numpy(),
                    fixed_base=True,
                    scale=self.object_scales[env_idx].item()
                )
                self.objects[env_idx] = self.scene.add_articulation(obj_cfg)
            else:
                self.objects[env_idx].reset(
                    position=self.object_positions[env_idx].cpu().numpy(),
                    orientation=self.object_orientations[env_idx].cpu().numpy()
                )
        
    def step(self, actions):
        # Apply actions
        self.robot.apply_actions(actions)
        
        # Step simulation
        self.sim.step()
        
        # Get observations
        obs = self._get_obs()
        
        # Calculate rewards
        reward = self._get_rewards()
        
        # Check termination
        done = self._get_dones()
        
        # Update progress
        self.progress_buf += 1
        
        return obs, reward, done, {}
        
    def _get_obs(self):
        # Get robot state
        robot_state = self.robot.get_state()
        
        # Get camera observations if enabled
        if self.cfg.sensor["enable_camera"]:
            camera_obs = self.camera.get_observation()
            obs = torch.cat([robot_state, camera_obs], dim=-1)
        else:
            obs = robot_state
            
        return obs
        
    def _get_rewards(self):
        # Get reward components from configuration
        reward_scales = self.cfg.reward["scales"]
        
        # Initialize reward buffer
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Calculate reward components
        reward += reward_scales["approaching"] * self._get_approaching_reward()
        reward += reward_scales["lifting"] * self._get_lifting_reward()
        reward += reward_scales["pick_up"] * self._get_pick_up_reward()
        reward += reward_scales["acc_penalty"] * self._get_acceleration_penalty()
        reward += reward_scales["command_penalty"] * self._get_command_penalty()
        reward += reward_scales["action_rate"] * self._get_action_rate_penalty()
        reward += reward_scales["ee_orn"] * self._get_end_effector_orientation_reward()
        reward += reward_scales["base_dir"] * self._get_base_direction_reward()
        reward += reward_scales["base_approaching"] * self._get_base_approaching_reward()
        reward += reward_scales["grasp_base_height"] * self._get_grasp_base_height_reward()
        
        return reward
        
    def _get_approaching_reward(self):
        """Reward for approaching the target object."""
        robot_pos = self.robot.get_base_position()
        object_pos = self.object_positions
        
        # Calculate distance to object
        dist = torch.norm(robot_pos - object_pos, dim=-1)
        
        # Reward is negative distance (closer is better)
        return -dist
        
    def _get_lifting_reward(self):
        """Reward for lifting the object."""
        object_pos = self.object_positions
        object_heights = self.object_heights
        
        # Calculate how much the object is lifted above its initial height
        lift_amount = object_pos[:, 2] - object_heights / 2
        
        # Only give positive reward when object is lifted
        return torch.clamp(lift_amount, min=0.0)
        
    def _get_pick_up_reward(self):
        """Reward for successfully picking up the object."""
        # Check if object is being held (using contact sensor)
        is_held = self.contact_sensor.get_contact_state()
        
        # Check if object is lifted above threshold
        object_pos = self.object_positions
        object_heights = self.object_heights
        is_lifted = object_pos[:, 2] > (object_heights / 2 + self.cfg.lifted_success_threshold)
        
        return (is_held & is_lifted).float()
        
    def _get_acceleration_penalty(self):
        """Penalty for high accelerations."""
        robot_vel = self.robot.get_base_velocity()
        return -torch.norm(robot_vel, dim=-1)
        
    def _get_command_penalty(self):
        """Penalty for large command values."""
        return -torch.norm(self.robot.get_last_actions(), dim=-1)
        
    def _get_action_rate_penalty(self):
        """Penalty for large action changes."""
        action_diff = self.robot.get_last_actions() - self.robot.get_previous_actions()
        return -torch.norm(action_diff, dim=-1)
        
    def _get_end_effector_orientation_reward(self):
        """Reward for maintaining desired end-effector orientation."""
        ee_orn = self.robot.get_end_effector_orientation()
        target_orn = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        return -torch.norm(ee_orn - target_orn, dim=-1)
        
    def _get_base_direction_reward(self):
        """Reward for moving base towards object."""
        robot_pos = self.robot.get_base_position()
        robot_vel = self.robot.get_base_velocity()
        object_pos = self.object_positions
        
        # Calculate direction to object
        to_object = object_pos - robot_pos
        to_object = to_object / (torch.norm(to_object, dim=-1, keepdim=True) + 1e-6)
        
        # Project velocity onto direction to object
        return torch.sum(robot_vel * to_object, dim=-1)
        
    def _get_base_approaching_reward(self):
        """Reward for base approaching object."""
        robot_pos = self.robot.get_base_position()
        object_pos = self.object_positions
        
        # Calculate distance to object
        dist = torch.norm(robot_pos - object_pos, dim=-1)
        
        # Reward is negative distance (closer is better)
        return -dist
        
    def _get_grasp_base_height_reward(self):
        """Reward for maintaining desired base height during grasp."""
        robot_pos = self.robot.get_base_position()
        target_height = self.cfg.reward["base_height_target"]
        
        # Calculate height difference
        height_diff = torch.abs(robot_pos[:, 2] - target_height)
        
        # Reward is negative height difference (closer to target is better)
        return -height_diff
        
    def _get_dones(self):
        # Check episode termination conditions
        done = torch.zeros(self.num_envs, device=self.device)
        
        # Check episode length
        done = torch.where(
            self.progress_buf >= self.max_episode_length,
            torch.ones_like(done),
            done
        )
        
        # Check if object is lifted above success threshold
        object_pos = self.object_positions
        object_heights = self.object_heights
        is_lifted = object_pos[:, 2] > (object_heights / 2 + self.cfg.lifted_success_threshold)
        
        # Check if object is held for required number of steps
        is_held = self.contact_sensor.get_contact_state()
        hold_steps = torch.zeros(self.num_envs, device=self.device)
        hold_steps = torch.where(is_held, hold_steps + 1, torch.zeros_like(hold_steps))
        
        # Terminate if object is lifted and held for required steps
        done = torch.where(
            (is_lifted & (hold_steps >= self.cfg.hold_steps)),
            torch.ones_like(done),
            done
        )
        
        return done 