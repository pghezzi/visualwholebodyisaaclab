from isaacgym import gymapi, gymutil
import math

# Initialize
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create environment
env = gym.create_env(sim, gymapi.Vec3(0, 0, 0), gymapi.Vec3(1, 1, 1), 1)

# Set asset options
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.armature = 0.01
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

# Load URDF
asset_root = "low-level/resources/robots/b1z1/urdf"  # where the URDF is
urdf_file = "b1z1_copy.urdf"
asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)

# Add asset to env
pose = gymapi.Transform()
pose.p = gymapi.Vec3(10.0, 0.0, 0.5)
actor = gym.create_actor(env, asset, pose, "robot", 0, 1)

# Get number of DOFs
num_dofs = gym.get_asset_dof_count(asset)

# Set up joint properties
props = gym.get_actor_dof_properties(env, actor)
props["driveMode"].fill(gymapi.DOF_MODE_POS)
props["stiffness"].fill(100.0)
props["damping"].fill(2.0)
gym.set_actor_dof_properties(env, actor, props)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# Set target positions for joints (example values - adjust based on your robot)
target_positions = [0.0] * num_dofs  # Initialize with zeros
gym.set_actor_dof_position_targets(env, actor, target_positions)

# Main simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Update target positions (example: oscillating motion)
    for i in range(num_dofs):
        target_positions[i] = 0.5 * math.sin(gym.get_sim_time(sim) + i)
    
    # Set new target positions
    gym.set_actor_dof_position_targets(env, actor, target_positions)
    
    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
