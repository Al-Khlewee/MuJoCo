"""
Convert MJX/Brax Policy to Animation Data for Blender
======================================================

This script loads a trained policy (e.g., mjx_brax_quadruped_policy) and runs
a simulation to extract joint positions over time. The output is a JSON file
compatible with Blender animation scripts.

Author: AI Assistant  
Date: October 24, 2025
License: MIT

Usage:
    python convert_policy_to_animation.py --policy mjx_brax_quadruped_policy --output barkour_animation.txt
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import from main scripts
sys.path.append(str(Path(__file__).parent.parent))

# Import MJX/Brax libraries
try:
    import jax
    import jax.numpy as jnp
    from brax import envs
    from brax.io import model
    from brax.training.agents.ppo import networks as ppo_networks
    import functools
    
    # Import the custom Barkour environment from training script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from train_barkour_local import BarkourEnv
    
except ImportError as e:
    print(f"ERROR: Missing dependencies. Install with:")
    print(f"  pip install jax jaxlib brax mujoco")
    print(f"\nAlso ensure train_barkour_local.py is in the parent directory")
    sys.exit(1)


def load_policy(policy_path):
    """Load trained policy parameters."""
    print(f"Loading policy from: {policy_path}")
    params = model.load_params(policy_path)
    print(f"✓ Policy loaded successfully")
    return params


def create_inference_function(env, params, policy_hidden_layers=(128, 128, 128, 128)):
    """Create inference function from policy parameters."""
    
    # Import normalization function
    from brax.training.acme import running_statistics
    
    # Create normalization preprocessing
    normalize_fn = running_statistics.normalize
    normalizer_params = params[0]  # RunningStatisticsState
    
    def preprocess_observations_fn(obs, rng):
        return normalize_fn(obs, normalizer_params)
    
    # Create network factory
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=policy_hidden_layers
    )
    
    # Build PPO network
    ppo_network = make_networks_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_observations_fn
    )
    
    # Create inference function
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params)
    
    return inference_fn


def reorder_joint_angles_to_mjcf(joint_angles):
    """Reorder joint angles to match MJCF actuator order (FL, HL, FR, HR)."""
    return jnp.concatenate([
        joint_angles[0:3],   # Front Left remains
        joint_angles[6:9],   # Hind Left moves next
        joint_angles[3:6],   # Front Right moves after hind left
        joint_angles[9:12],  # Hind Right remains last
    ])


def extract_joint_positions(state):
    """
    Extract joint positions from Brax state.
    
    Returns:
        dict: Joint positions including base position/orientation and joint angles
    """
    qpos = state.pipeline_state.q
    joint_angles = reorder_joint_angles_to_mjcf(qpos[7:])
    
    # Barkour VB has:
    # - 7 values for floating base (x, y, z, qw, qx, qy, qz)
    # - 12 joint angles (3 per leg: hip_x, hip_y, knee)
    
    result = {
        'dt': state.info.get('dt', 0.02),  # Default timestep
        'base_pos': qpos[:3].tolist(),  # x, y, z
        'base_quat': qpos[3:7].tolist(),  # qw, qx, qy, qz
        'joints': joint_angles.tolist(),  # All joint angles reordered to MJCF order
    }
    
    return result


def run_simulation(policy_path, duration=5.0, fps=60, command=None):
    """
    Run simulation with trained policy and extract animation data.
    
    Args:
        policy_path: Path to trained policy file
        duration: Simulation duration in seconds
        fps: Frames per second for output
        command: Joystick command [vx, vy, vyaw] (default: forward walk)
        
    Returns:
        dict: Animation data with joint positions over time
    """
    
    # Load policy
    params = load_policy(policy_path)
    
    # Create environment (Barkour VB) - use custom environment
    print(f"Creating Barkour environment...")
    
    # Register the custom environment
    envs.register_environment('barkour', BarkourEnv)
    
    # Get menagerie path
    menagerie_path = Path(__file__).parent.parent / "mujoco_menagerie" / "google_barkour_vb"
    
    env = envs.get_environment('barkour', menagerie_path=menagerie_path)
    print(f"✓ Environment created: barkour")
    
    # Create inference function
    print(f"Building inference function...")
    inference_fn = create_inference_function(env, params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    print(f"✓ Inference function ready")
    
    # Set command (default: walk forward)
    if command is None:
        command = jnp.array([1.0, 0.0, 0.0])  # [vx, vy, vyaw]
    else:
        command = jnp.array(command)
    
    print(f"\nRunning simulation...")
    print(f"  Duration: {duration}s")
    print(f"  FPS: {fps}")
    print(f"  Command: vx={command[0]:.2f}, vy={command[1]:.2f}, vyaw={command[2]:.2f}")
    
    # Initialize simulation
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    # Set the command in state.info
    state.info['command'] = command
    
    # Calculate number of frames
    dt = 0.02  # Brax default timestep
    num_steps = int(duration / dt)
    output_every_n = max(1, int(1.0 / (fps * dt)))  # Sample at desired FPS
    
    # Storage for animation data
    frames = []
    
    # Run simulation
    for step in range(num_steps):
        # Get action from policy
        rng, rng_act = jax.random.split(rng)
        
        # The observation already includes the command from the environment
        obs = state.obs
        
        act, _ = jit_inference_fn(obs, rng_act)  # Returns (action, extra_info)
        
        # Step environment
        state = jit_step(state, act)
        
        # Save frame at desired FPS
        if step % output_every_n == 0:
            frame_data = extract_joint_positions(state)
            frames.append(frame_data)
            
            if (step + 1) % 100 == 0:
                progress = (step + 1) / num_steps * 100
                print(f"  Progress: {progress:.1f}% ({len(frames)} frames)")
    
    print(f"✓ Simulation complete: {len(frames)} frames generated")
    
    # Get joint names from environment
    joint_names = get_joint_names(env)
    
    # Package animation data
    animation_data = {
        "Format": "qpos_trajectory",
        "Robot": "google_barkour_vb",
        "Duration": duration,
        "FPS": fps,
        "JointNames": joint_names,
        "Command": command.tolist(),
        "Frames": frames,
    }
    
    return animation_data


def get_joint_names(env):
    """Extract joint names from environment."""
    
    # Barkour VB joint structure
    # Based on google_barkour_vb.xml in mujoco_menagerie
    
    base_joints = [
        "floating_base",  # Virtual joint for base
    ]
    
    leg_joints = []
    for leg in ['fl', 'hl', 'fr', 'hr']:  # match MJCF actuator order
        leg_joints.extend([
            f"{leg}_hx",  # Hip X (abduction/adduction)
            f"{leg}_hy",  # Hip Y (flexion/extension)  
            f"{leg}_kn",  # Knee
        ])
    
    return base_joints + leg_joints


def save_animation_data(animation_data, output_path):
    """Save animation data to JSON file."""
    
    output_path = Path(output_path)
    
    print(f"\nSaving animation data...")
    print(f"  Output: {output_path}")
    print(f"  Frames: {len(animation_data['Frames'])}")
    print(f"  Duration: {animation_data['Duration']}s")
    
    with open(output_path, 'w') as f:
        json.dump(animation_data, f, indent=2)
    
    print(f"✓ Animation data saved successfully")
    
    # Print file size
    file_size = output_path.stat().st_size
    if file_size < 1024:
        size_str = f"{file_size} B"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.2f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"
    
    print(f"  File size: {size_str}")


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Convert trained MJX/Brax policy to Blender animation data"
    )
    
    parser.add_argument(
        '--policy',
        type=str,
        required=True,
        help='Path to trained policy file (e.g., mjx_brax_quadruped_policy)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='barkour_animation.txt',
        help='Output file path (default: barkour_animation.txt)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=5.0,
        help='Simulation duration in seconds (default: 5.0)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='Output frames per second (default: 60)'
    )
    
    parser.add_argument(
        '--vx',
        type=float,
        default=1.0,
        help='Forward velocity command (default: 1.0)'
    )
    
    parser.add_argument(
        '--vy',
        type=float,
        default=0.0,
        help='Lateral velocity command (default: 0.0)'
    )
    
    parser.add_argument(
        '--vyaw',
        type=float,
        default=0.0,
        help='Yaw rate command (default: 0.0)'
    )
    
    args = parser.parse_args()
    
    # Check if policy file exists
    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"ERROR: Policy file not found: {policy_path}")
        sys.exit(1)
    
    print("="*70)
    print("MJX/Brax Policy to Blender Animation Converter")
    print("="*70)
    print()
    
    # Set command
    command = [args.vx, args.vy, args.vyaw]
    
    try:
        # Run simulation and extract data
        animation_data = run_simulation(
            policy_path=policy_path,
            duration=args.duration,
            fps=args.fps,
            command=command
        )
        
        # Save to file
        save_animation_data(animation_data, args.output)
        
        print()
        print("="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print()
        print("Next steps:")
        print(f"  1. Open Blender")
        print(f"  2. Run: blender/mujoco_barkour_importer.py")
        print(f"  3. Run: blender/animate_barkour.py with {args.output}")
        print()
        
    except Exception as e:
        print()
        print("="*70)
        print("❌ ERROR")
        print("="*70)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
