"""
Standalone script to run the trained Barkour quadruped policy locally.

This script loads a trained policy and simulates the google_barkour_vb quadruped
with customizable velocity commands.

Author: Generated for local MJX/Brax simulation
"""

import os
import sys
import functools
from pathlib import Path
from typing import Any, Dict, List, Sequence
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import jax
from jax import numpy as jp
import numpy as np
import mediapy as media
from ml_collections import config_dict
import imageio

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import mjcf, model


# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE PATHS AND COMMANDS
# ============================================================================

# File paths (use absolute paths for Windows)
BASE_DIR = Path(r"c:\users\hatem\Desktop\MuJoCo")
MODEL_PATH = BASE_DIR / "mjx_brax_quadruped_policy"
MENAGERIE_PATH = BASE_DIR / "mujoco_menagerie" / "google_barkour_vb"
OUTPUT_VIDEO = BASE_DIR / "barkour_simulation.mp4"

# Simulation parameters
X_VEL = 1.0      # Forward velocity (m/s) - range: [-0.6, 1.5]
Y_VEL = 0.0      # Sideways velocity (m/s) - range: [-0.8, 0.8]
ANG_VEL = -0.5   # Yaw rate (rad/s) - range: [-0.7, 0.7]

N_STEPS = 500    # Number of simulation steps
RENDER_EVERY = 2 # Render every N frames
RANDOM_SEED = 0  # Random seed for reproducibility

# ============================================================================
# ENVIRONMENT DEFINITION
# ============================================================================

def get_config():
    """Returns reward config for barkour quadruped environment."""
    
    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                scales=config_dict.ConfigDict(
                    dict(
                        tracking_lin_vel=1.5,
                        tracking_ang_vel=0.8,
                        lin_vel_z=-2.0,
                        ang_vel_xy=-0.05,
                        orientation=-5.0,
                        torques=-0.0002,
                        action_rate=-0.01,
                        feet_air_time=0.2,
                        stand_still=-0.5,
                        termination=-1.0,
                        foot_slip=-0.1,
                    )
                ),
                tracking_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )
    return default_config


class BarkourEnv(PipelineEnv):
    """Environment for the Barkour quadruped with joystick policy."""

    def __init__(
        self,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        scene_file: str = 'scene_mjx.xml',
        menagerie_path: Path = None,
        **kwargs,
    ):
        if menagerie_path is None:
            menagerie_path = MENAGERIE_PATH
        
        path = menagerie_path / scene_file
        if not path.exists():
            raise FileNotFoundError(
                f"Scene file not found: {path}\n"
                f"Please ensure mujoco_menagerie is in the correct location."
            )
        
        sys = mjcf.load(path.as_posix())
        self._dt = 0.02  # 50 fps
        sys = sys.tree_replace({'opt.timestep': 0.004})

        # Override menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend='mjx', n_frames=n_frames)

        self.reward_config = get_config()
        for k, v in kwargs.items():
            if k.endswith('_scale'):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'torso'
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
        self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
        self.lowers = jp.array([-0.7, -1.0, 0.05] * 4)
        self.uppers = jp.array([0.52, 2.1, 2.1] * 4)
        
        feet_site = [
            'foot_front_left', 'foot_hind_left',
            'foot_front_right', 'foot_hind_right',
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
        self._feet_site_id = np.array(feet_site_id)
        
        lower_leg_body = [
            'lower_leg_front_left', 'lower_leg_hind_left',
            'lower_leg_front_right', 'lower_leg_hind_right',
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.6, 1.5]
        lin_vel_y = [-0.8, 0.8]
        ang_vel_yaw = [-0.7, 0.7]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)
        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            'rng': rng,
            'last_act': jp.zeros(12),
            'last_vel': jp.zeros(12),
            'command': self.sample_command(key),
            'last_contact': jp.zeros(4, dtype=bool),
            'feet_air_time': jp.zeros(4),
            'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            'kick': jp.array([0.0, 0.0]),
            'step': 0,
        }

        obs_history = jp.zeros(15 * 31)
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jp.zeros(2)
        metrics = {'total_dist': 0.0}
        for k in state_info['rewards']:
            metrics[k] = state_info['rewards'][k]
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

        # Apply kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info['step'], push_interval) == 0
        qvel = state.pipeline_state.qvel
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({'pipeline_state.qvel': qvel})

        # Physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # Observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # Foot contact detection
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3
        contact_filt_mm = contact | state.info['last_contact']
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
        first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
        state.info['feet_air_time'] += self.dt

        # Termination conditions
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # Calculate rewards
        rewards = {
            'tracking_lin_vel': self._reward_tracking_lin_vel(state.info['command'], x, xd),
            'tracking_ang_vel': self._reward_tracking_ang_vel(state.info['command'], x, xd),
            'lin_vel_z': self._reward_lin_vel_z(xd),
            'ang_vel_xy': self._reward_ang_vel_xy(xd),
            'orientation': self._reward_orientation(x),
            'torques': self._reward_torques(pipeline_state.qfrc_actuator),
            'action_rate': self._reward_action_rate(action, state.info['last_act']),
            'stand_still': self._reward_stand_still(state.info['command'], joint_angles),
            'feet_air_time': self._reward_feet_air_time(
                state.info['feet_air_time'], first_contact, state.info['command']
            ),
            'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
            'termination': self._reward_termination(done, state.info['step']),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # Update state
        state.info['kick'] = kick
        state.info['last_act'] = action
        state.info['last_vel'] = joint_vel
        state.info['feet_air_time'] *= ~contact_filt_mm
        state.info['last_contact'] = contact
        state.info['rewards'] = rewards
        state.info['step'] += 1
        state.info['rng'] = rng

        state.info['command'] = jp.where(
            state.info['step'] > 500,
            self.sample_command(cmd_rng),
            state.info['command'],
        )
        state.info['step'] = jp.where(
            done | (state.info['step'] > 500), 0, state.info['step']
        )

        state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info['rewards'])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def _get_obs(
        self, pipeline_state: base.State, state_info: Dict[str, Any], obs_history: jax.Array
    ) -> jax.Array:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jp.concatenate([
            jp.array([local_rpyrate[2]]) * 0.25,
            math.rotate(jp.array([0, 0, -1]), inv_torso_rot),
            state_info['command'] * jp.array([2.0, 2.0, 0.25]),
            pipeline_state.q[7:] - self._default_pose,
            state_info['last_act'],
        ])

        obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
            state_info['rng'], obs.shape, minval=-1, maxval=1
        )
        obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)
        return obs

    # Reward functions
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(-lin_vel_error / self.reward_config.rewards.tracking_sigma)
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= math.normalize(commands[:2])[1] > 0.05
        return rew_air_time

    def _reward_stand_still(self, commands: jax.Array, joint_angles: jax.Array) -> jax.Array:
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        pos = pipeline_state.site_xpos[self._feet_site_id]
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def render(
        self, trajectory: List[base.State], camera: str | None = None,
        width: int = 240, height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or 'track'
        return super().render(trajectory, camera=camera, width=width, height=height)


# ============================================================================
# MAIN SIMULATION FUNCTION
# ============================================================================

def check_environment():
    """Check if all required files and paths exist."""
    print("=" * 60)
    print("CHECKING ENVIRONMENT")
    print("=" * 60)
    
    errors = []
    
    if not MODEL_PATH.exists():
        errors.append(f"Model not found: {MODEL_PATH}")
    else:
        print(f"✓ Model found: {MODEL_PATH}")
    
    if not MENAGERIE_PATH.exists():
        errors.append(f"Menagerie path not found: {MENAGERIE_PATH}")
    else:
        print(f"✓ Menagerie path found: {MENAGERIE_PATH}")
        
        scene_file = MENAGERIE_PATH / "scene_mjx.xml"
        if not scene_file.exists():
            errors.append(f"Scene file not found: {scene_file}")
        else:
            print(f"✓ Scene file found: {scene_file}")
    
    if errors:
        print("\n" + "=" * 60)
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  ✗ {error}")
        print("=" * 60)
        return False
    
    print("=" * 60)
    print("All files found successfully!")
    print("=" * 60 + "\n")
    return True


def main():
    """Main function to run the Barkour quadruped simulation."""
    
    print("\n" + "=" * 60)
    print("BARKOUR QUADRUPED SIMULATION")
    print("=" * 60 + "\n")
    
    # Check environment
    if not check_environment():
        print("\nPlease fix the file path issues and try again.")
        sys.exit(1)
    
    # Configure JAX/XLA
    print("Configuring JAX...")
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
    
    # Check for GPU
    try:
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        if any('gpu' in str(d).lower() for d in devices):
            print("✓ GPU detected - using GPU acceleration")
        else:
            print("⚠ No GPU detected - using CPU (will be slower)")
    except Exception as e:
        print(f"⚠ Could not check devices: {e}")
    
    print("\n" + "=" * 60)
    print("LOADING MODEL AND ENVIRONMENT")
    print("=" * 60)
    
    # Register environment
    envs.register_environment('barkour', BarkourEnv)
    
    # Create environment
    print("Creating environment...")
    env = envs.get_environment('barkour', menagerie_path=MENAGERIE_PATH)
    print(f"✓ Environment created (action size: {env.action_size}, obs size: {env.observation_size})")
    
    # Load policy
    print(f"Loading policy from: {MODEL_PATH}")
    try:
        params = model.load_params(MODEL_PATH.as_posix())
        print("✓ Policy loaded successfully")
    except Exception as e:
        print(f"✗ Error loading policy: {e}")
        sys.exit(1)
    
    # Create inference function
    # Create the inference function with observation normalization
    # The policy was trained with normalize_observations=True
    # The params tuple contains: (normalizer_state, network_params, ...)
    print(f"Policy structure: tuple with {len(params)} elements")
    print(f"  Element 0: {type(params[0]).__name__} (observation normalizer)")
    print(f"  Element 1: {type(params[1]).__name__} (network parameters)")
    
    # Import the normalization function
    from brax.training import acting
    from brax.training.acme import running_statistics
    
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128)
    )
    
    # Create preprocessing function that uses the saved normalization state
    normalize_fn = running_statistics.normalize
    normalizer_params = params[0]  # RunningStatisticsState
    
    def preprocess_observations_fn(obs, rng):
        return normalize_fn(obs, normalizer_params)
    
    ppo_network = make_networks_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=preprocess_observations_fn
    )
    
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    print("✓ Using observation normalization (as in training)")
    print("✓ Inference function ready")
    
    # JIT compile step functions
    print("JIT compiling environment functions (this may take a moment)...")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # Warm-up JIT compilation
    rng = jax.random.PRNGKey(RANDOM_SEED)
    _ = jit_reset(rng)
    print("✓ Environment functions compiled")
    
    print("\n" + "=" * 60)
    print("RUNNING SIMULATION")
    print("=" * 60)
    print(f"Commands: x_vel={X_VEL:.2f}, y_vel={Y_VEL:.2f}, ang_vel={ANG_VEL:.2f}")
    print(f"Steps: {N_STEPS}, Render every: {RENDER_EVERY}")
    print("=" * 60 + "\n")
    
    # Set command
    the_command = jp.array([X_VEL, Y_VEL, ANG_VEL])
    
    # Initialize state
    rng = jax.random.PRNGKey(RANDOM_SEED)
    state = jit_reset(rng)
    state.info['command'] = the_command
    rollout = [state.pipeline_state]
    
    # Run simulation
    print("Simulating...")
    for i in range(N_STEPS):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        
        if (i + 1) % 100 == 0:
            print(f"  Step {i + 1}/{N_STEPS}")
        
        if state.done:
            print(f"  Episode terminated early at step {i + 1}")
            break
    
    print(f"✓ Simulation complete ({len(rollout)} frames)")
    
    print("\n" + "=" * 60)
    print("RENDERING VIDEO")
    print("=" * 60)
    
    # Render video
    print("Rendering frames...")
    video = env.render(rollout[::RENDER_EVERY], camera='track')
    fps = 1.0 / env.dt / RENDER_EVERY
    print(f"✓ Rendered {len(video)} frames at {fps:.1f} FPS")
    
    # Save video
    print(f"Saving video to: {OUTPUT_VIDEO}")
    video_saved = False
    
    # Try method 1: imageio with ffmpeg
    try:
        print("  Attempting to save with imageio...")
        imageio.mimwrite(OUTPUT_VIDEO.as_posix(), video, fps=fps, quality=8)
        print(f"✓ Video saved successfully!")
        print(f"  File: {OUTPUT_VIDEO}")
        print(f"  Size: {OUTPUT_VIDEO.stat().st_size / 1024 / 1024:.2f} MB")
        video_saved = True
    except Exception as e:
        print(f"  imageio failed: {e}")
    
    # Try method 2: Save as GIF
    if not video_saved:
        try:
            print("  Trying to save as GIF...")
            gif_path = OUTPUT_VIDEO.with_suffix('.gif')
            imageio.mimsave(gif_path.as_posix(), video, fps=int(fps), loop=0)
            print(f"✓ Saved as GIF: {gif_path}")
            print(f"  Size: {gif_path.stat().st_size / 1024 / 1024:.2f} MB")
            video_saved = True
        except Exception as e2:
            print(f"  GIF save failed: {e2}")
    
    # Try method 3: Save individual frames
    if not video_saved:
        try:
            print("  Saving individual frames...")
            frames_dir = BASE_DIR / "barkour_frames"
            frames_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(video):
                imageio.imwrite(frames_dir / f"frame_{i:04d}.png", frame)
            print(f"✓ Saved {len(video)} frames to: {frames_dir}")
            video_saved = True
        except Exception as e3:
            print(f"  Frame save failed: {e3}")
    
    # Last resort: Save first and last frame
    if not video_saved:
        try:
            img_path = OUTPUT_VIDEO.with_suffix('.png')
            imageio.imwrite(img_path.as_posix(), video[0])
            print(f"✓ Saved first frame as: {img_path}")
        except:
            print("  Could not save any output")
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)
    print(f"\nTo run with different commands, modify the following variables:")
    print(f"  X_VEL (current: {X_VEL})")
    print(f"  Y_VEL (current: {Y_VEL})")
    print(f"  ANG_VEL (current: {ANG_VEL})")
    print("\n")


if __name__ == "__main__":
    main()
