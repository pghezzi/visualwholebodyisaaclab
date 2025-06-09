from isaaclab.config import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.actuators import ArticulationCfg
from isaaclab.sensors import CameraCfg

@configclass
class MultiStateEnvCfg(DirectRLEnvCfg):
    # Simulation config
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=0.005,
        gravity=(0.0, 0.0, -9.81),
        physx=dict(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            contact_offset=0.02,
            rest_offset=0.0,
            bounce_threshold_velocity=0.2,
            max_depenetration_velocity=50.0,
            gpu_max_rigid_contact_count=1000000
        )
    )
    
    # Scene config
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        ground_plane=dict(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0
        )
    )
    
    # Environment config
    num_envs: int = 10240
    num_agents: int = 1
    env_spacing: float = 5.0
    enable_debug_vis: bool = False
    is_flagrun: bool = False
    max_episode_length: int = 150
    pd_control: bool = True
    power_scale: float = 1.0
    control_frequency_inv: int = 4
    control_frequency_low: int = 8
    img_delay_frame: int = 4
    floating_base: bool = True
    
    # Task specific config
    lifted_success_threshold: float = 0.35
    lifted_init_threshold: float = 0.05
    base_object_dis_threshold: float = 0.6
    hold_steps: int = 25
    last_commands: bool = False
    
    # Asset config
    asset_root: str = "data/asset"
    asset_file_robot: str = "b1z1-float/urdf/b1z1.urdf"
    asset_file_obj: str = "obj_set/"
    
    # Robot control config
    control: dict = dict(
        stiffness=dict(joint=80, z1=5),  # [N*m/rad]
        damping=dict(joint=2.0, z1=0.5)   # [N*m*s/rad]
    )
    
    # Reward config
    reward: dict = dict(
        base_height_target=0.55,
        only_positive_rewards=False,
        scales=dict(
            approaching=0.5,
            lifting=1.0,
            pick_up=3.5,
            acc_penalty=-0.001,
            command_penalty=-1.0,
            command_reward=0.25,
            standpick=0.25,
            action_rate=-0.001,
            ee_orn=0.01,
            base_dir=0.25,
            rad_penalty=0.0,
            base_ang_pen=0.0,
            base_approaching=0.01,
            grasp_base_height=0.5,
            gripper_rate=0.0
        )
    )
    
    # Sensor config
    sensor: dict = dict(
        depth_clip_lower=0.15,
        depth_clip_rand_range=(0.18, 0.25),
        enable_camera=False,
        resized_resolution=(96, 54)
    )
    
    # Object assets config
    asset_multi: dict = dict(
        green_bowl=dict(
            dict_idx=0,
            height=0.026,
            orientation=(0.0, 0.0, 0.0, 1.0),
            scale=1.0
        ),
        sugar_box=dict(
            dict_idx=1,
            height=0.088,
            orientation=(0.0, 0.707, 0.0, 0.707),
            scale=1.0
        ),
        # Add other objects as needed...
    ) 