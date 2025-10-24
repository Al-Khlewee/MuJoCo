"""
Training script for Barkour quadruped from scratch using PPO.

This script replicates the training process from the Google Colab notebook
but is optimized for local Windows execution with proper checkpointing,
monitoring, and GPU utilization.

Training time: ~30-60 minutes on RTX 3080/4090, ~2-6 hours on older GPUs
GPU Memory: Requires 12GB+ VRAM (adjust num_envs if you have less)

Author: Generated for local MJX/Brax training
"""

import os
import sys
import functools
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple
import warnings

warnings.filterwarnings('ignore')

import jax
from jax import numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from ml_collections import config_dict
import imageio

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import mjcf, model
from etils import epath
from flax.training import orbax_utils
from orbax import checkpoint as ocp


# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE SETTINGS
# ============================================================================

# File paths (use absolute paths for Windows)
BASE_DIR = Path(r"c:\users\hatem\Desktop\MuJoCo")
MENAGERIE_PATH = BASE_DIR / "mujoco_menagerie" / "google_barkour_vb"
CHECKPOINT_DIR = BASE_DIR / "training_checkpoints" / "barkour_joystick"
OUTPUT_MODEL_PATH = BASE_DIR / "trained_barkour_policy"
LOG_DIR = BASE_DIR / "training_logs"

# Training hyperparameters (GPU-optimized, will work on CPU but slower)
NUM_TIMESTEPS = 1_000_000    # 1M steps (Quick test: ~10 min on GPU, ~1 hour on CPU)
NUM_EVALS = 5                # Number of evaluation checkpoints
NUM_ENVS = 2048              # Parallel environments (GPU-optimized, reduce to 256 if CPU)
BATCH_SIZE = 256             # PPO batch size (GPU-optimized)
EPISODE_LENGTH = 1000        # Max steps per episode
LEARNING_RATE = 3e-4         # PPO learning rate
SEED = 0                     # Random seed

# GPU/Performance settings
# Attempt to use GPU (may not work on Windows - JAX Windows builds are CPU-only)
USE_GPU = True
# GPU-optimized settings (will fall back to CPU if GPU unavailable)
NUM_MINIBATCHES = 32         # Training minibatches (GPU-optimized)
UNROLL_LENGTH = 20           # Rollout length (GPU-optimized)

# ============================================================================
# ENVIRONMENT DEFINITION (Same as inference script)
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
# DOMAIN RANDOMIZATION (From notebook)
# ============================================================================

def domain_randomize(sys, rng):
    """Randomizes the mjx.Model for robust training."""
    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # Randomize friction
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = sys.geom_friction.at[:, 0].set(friction)
        # Randomize actuator gain/bias
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = jax.random.uniform(
            key, (1,), minval=gain_range[0], maxval=gain_range[1]
        ) + sys.actuator_gainprm[:, 0]
        gain = sys.actuator_gainprm.at[:, 0].set(param)
        bias = sys.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'actuator_gainprm': 0,
        'actuator_biasprm': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })

    return sys, in_axes


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories for training."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"✓ Log directory: {LOG_DIR}")


def save_checkpoint(current_step, make_policy, params):
    """Save training checkpoint."""
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = CHECKPOINT_DIR / f'step_{current_step}'
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)
    print(f"  → Checkpoint saved: {path}")


def create_progress_plot():
    """Initialize training progress plot."""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    return fig, ax1, ax2


def update_progress_plot(ax1, ax2, x_data, y_data, y_err, times, num_timesteps):
    """Update the training progress visualization."""
    ax1.clear()
    ax2.clear()
    
    # Plot 1: Reward over time
    ax1.errorbar(x_data, y_data, yerr=y_err, capsize=3)
    ax1.set_xlim([0, num_timesteps * 1.05])
    ax1.set_ylim([0, 50])
    ax1.set_xlabel('Environment Steps')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title(f'Training Progress (Current: {y_data[-1]:.1f})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps per second
    if len(times) > 1:
        time_diffs = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        step_diffs = [x_data[i+1] - x_data[i] for i in range(len(x_data)-1)]
        steps_per_sec = [s / t if t > 0 else 0 for s, t in zip(step_diffs, time_diffs)]
        
        ax2.plot(x_data[1:], steps_per_sec, 'g-', linewidth=2)
        ax2.set_xlabel('Environment Steps')
        ax2.set_ylabel('Steps per Second')
        ax2.set_title(f'Training Speed (Current: {steps_per_sec[-1]:.0f} SPS)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.pause(0.01)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_barkour():
    """Main training function - replicates Google Colab notebook."""
    
    print("\n" + "=" * 70)
    print("BARKOUR QUADRUPED TRAINING FROM SCRATCH")
    print("=" * 70 + "\n")
    
    # Setup
    setup_directories()
    
    # Configure JAX
    print("\nConfiguring JAX/XLA...")
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
    
    if USE_GPU:
        print("🔍 Attempting to use GPU...")
        # Try to detect GPU
        try:
            # Don't force CPU - let JAX try to use GPU
            pass
        except Exception as e:
            print(f"⚠ GPU initialization failed: {e}")
            print("⚠ Falling back to CPU")
    else:
        print("⚠ GPU disabled - forcing CPU")
        jax.config.update('jax_platform_name', 'cpu')
    
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if any('gpu' in str(d).lower() for d in devices):
        print("✓ GPU detected - training will be fast!")
    else:
        print("⚠ No GPU detected - training will be SLOW (reduce NUM_TIMESTEPS)")
    
    # Register environment
    print("\n" + "=" * 70)
    print("ENVIRONMENT SETUP")
    print("=" * 70)
    envs.register_environment('barkour', BarkourEnv)
    env = envs.get_environment('barkour', menagerie_path=MENAGERIE_PATH)
    eval_env = envs.get_environment('barkour', menagerie_path=MENAGERIE_PATH)
    print(f"✓ Environment registered: {env.__class__.__name__}")
    print(f"  Action size: {env.action_size}")
    print(f"  Observation size: {env.observation_size}")
    
    # Create network factory
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128)
    )
    
    # Setup training configuration (matching Google Colab)
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Total timesteps: {NUM_TIMESTEPS:,}")
    print(f"Parallel environments: {NUM_ENVS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Episode length: {EPISODE_LENGTH}")
    print(f"Number of evaluations: {NUM_EVALS}")
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")
    
    # Estimate training time
    if 'gpu' in str(devices[0]).lower():
        estimated_minutes = NUM_TIMESTEPS / 1_000_000 * 0.6  # ~0.6 min per 1M steps on good GPU
        print(f"\n⏱ Estimated training time: {estimated_minutes:.1f} minutes (~{estimated_minutes/60:.1f} hours)")
    else:
        estimated_minutes = NUM_TIMESTEPS / 1_000_000 * 5  # Much slower on CPU
        print(f"\n⏱ Estimated training time: {estimated_minutes:.1f} minutes (~{estimated_minutes/60:.1f} hours)")
        print("⚠ WARNING: Training on CPU will be VERY slow! Consider using GPU or reducing NUM_TIMESTEPS.")
    
    # Training progress tracking
    x_data = []
    y_data = []
    y_err = []
    times = [datetime.now()]
    
    # Create progress plot
    fig, ax1, ax2 = create_progress_plot()
    
    def progress_callback(num_steps, metrics):
        """Called during training to report progress."""
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        y_err.append(metrics['eval/episode_reward_std'])
        
        # Update plot
        update_progress_plot(ax1, ax2, x_data, y_data, y_err, times, NUM_TIMESTEPS)
        
        # Print progress
        elapsed = (times[-1] - times[0]).total_seconds()
        progress_pct = (num_steps / NUM_TIMESTEPS) * 100
        print(f"\n{'='*70}")
        print(f"Step {num_steps:,} / {NUM_TIMESTEPS:,} ({progress_pct:.1f}%)")
        print(f"Reward: {y_data[-1]:.2f} ± {y_err[-1]:.2f}")
        print(f"Elapsed time: {elapsed/60:.1f} minutes")
        if len(times) > 1:
            steps_remaining = NUM_TIMESTEPS - num_steps
            time_per_step = elapsed / num_steps
            eta_seconds = steps_remaining * time_per_step
            print(f"ETA: {eta_seconds/60:.1f} minutes")
        print(f"{'='*70}")
    
    # Configure PPO training (matching Google Colab parameters exactly)
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=NUM_TIMESTEPS,
        num_evals=NUM_EVALS,
        reward_scaling=1,
        episode_length=EPISODE_LENGTH,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=UNROLL_LENGTH,
        num_minibatches=NUM_MINIBATCHES,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=LEARNING_RATE,
        entropy_cost=1e-2,
        num_envs=NUM_ENVS,
        batch_size=BATCH_SIZE,
        network_factory=make_networks_factory,
        randomization_fn=domain_randomize,
        policy_params_fn=save_checkpoint,
        seed=SEED
    )
    
    # Start training
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("This will take a while. Progress will be shown above.")
    print("You can monitor the plots and checkpoints.\n")
    
    start_time = datetime.now()
    
    try:
        # Train the policy
        make_inference_fn, params, metrics = train_fn(
            environment=env,
            progress_fn=progress_callback,
            eval_env=eval_env
        )
        
        training_successful = True
        
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user!")
        training_successful = False
        params = None
        make_inference_fn = None
    
    except Exception as e:
        print(f"\n\n✗ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        training_successful = False
        params = None
        make_inference_fn = None
    
    finally:
        plt.ioff()
        
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    if training_successful:
        print("TRAINING COMPLETE!")
    else:
        print("TRAINING ENDED (may be incomplete)")
    print(f"{'='*70}")
    print(f"Total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    if len(times) > 2:
        print(f"JIT compilation time: {(times[1] - times[0]).total_seconds():.1f} seconds")
        print(f"Actual training time: {(times[-1] - times[1]).total_seconds()/60:.1f} minutes")
    
    # Save final model
    if training_successful and params is not None:
        print(f"\n{'='*70}")
        print("SAVING FINAL MODEL")
        print(f"{'='*70}")
        OUTPUT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        model.save_params(OUTPUT_MODEL_PATH.as_posix(), params)
        print(f"✓ Final model saved to: {OUTPUT_MODEL_PATH}")
        print(f"  You can now use this with run_barkour_local.py!")
        
        # Save training plot
        plot_path = LOG_DIR / "training_progress.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training plot saved to: {plot_path}")
        
        # Save training summary
        summary_path = LOG_DIR / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("BARKOUR TRAINING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total time: {total_time/60:.1f} minutes\n")
            f.write(f"Total steps: {NUM_TIMESTEPS:,}\n")
            f.write(f"Final reward: {y_data[-1]:.2f} ± {y_err[-1]:.2f}\n")
            f.write(f"Peak reward: {max(y_data):.2f}\n")
            f.write(f"\nHyperparameters:\n")
            f.write(f"  Environments: {NUM_ENVS}\n")
            f.write(f"  Batch size: {BATCH_SIZE}\n")
            f.write(f"  Learning rate: {LEARNING_RATE}\n")
            f.write(f"  Episode length: {EPISODE_LENGTH}\n")
            f.write(f"\nCheckpoints saved to: {CHECKPOINT_DIR}\n")
            f.write(f"Final model: {OUTPUT_MODEL_PATH}\n")
        print(f"✓ Training summary saved to: {summary_path}")
        
        return make_inference_fn, params, training_successful
    else:
        print("\n⚠ Model not saved (training incomplete or failed)")
        return None, None, False


# ============================================================================
# INFERENCE TESTING
# ============================================================================

def run_inference_demo(make_inference_fn, params, env=None):
    """Test the trained policy with visualization."""
    print("\n" + "=" * 70)
    print("TESTING TRAINED POLICY")
    print("=" * 70)
    
    # Create environment if not provided
    if env is None:
        envs.register_environment('barkour', BarkourEnv)
        env = envs.get_environment('barkour', menagerie_path=MENAGERIE_PATH)
    
    # Create inference function
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    # Test configurations
    test_commands = [
        (1.5, 0.0, 0.0, "Forward at 1.5 m/s"),
        (0.5, 0.5, 0.0, "Forward-Right diagonal"),
        (0.0, 0.0, 0.7, "Rotate in place (yaw)"),
        (1.0, 0.0, -0.5, "Forward with left turn"),
    ]
    
    print("\nRunning multiple test scenarios...")
    print(f"Each scenario: 500 steps (~10 seconds simulated)")
    
    for idx, (x_vel, y_vel, ang_vel, description) in enumerate(test_commands, 1):
        print(f"\n--- Test {idx}/{len(test_commands)}: {description} ---")
        print(f"Command: x={x_vel:.1f} m/s, y={y_vel:.1f} m/s, ang={ang_vel:.1f} rad/s")
        
        # Initialize
        rng = jax.random.PRNGKey(idx * 100)
        state = jit_reset(rng)
        state.info['command'] = jp.array([x_vel, y_vel, ang_vel])
        
        rollout = [state.pipeline_state]
        rewards_collected = []
        
        # Run simulation
        n_steps = 500
        for step_i in range(n_steps):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)
            rewards_collected.append(float(state.reward))
            
            if state.done:
                print(f"  Episode terminated early at step {step_i}")
                break
        
        # Calculate statistics
        total_reward = sum(rewards_collected)
        avg_reward = total_reward / len(rewards_collected) if rewards_collected else 0
        
        # Get final position
        final_pos = rollout[-1].x.pos[0]  # torso position
        distance_traveled = np.linalg.norm(final_pos[:2])
        
        print(f"  ✓ Completed {len(rollout)-1} steps")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Distance traveled: {distance_traveled:.2f} m")
        
        # Save video
        video_path = LOG_DIR / f"test_{idx}_{description.replace(' ', '_')}.mp4"
        try:
            print(f"  Rendering video...")
            frames = env.render(rollout[::2], camera='track', width=640, height=480)
            
            # Save video with imageio
            with imageio.get_writer(video_path, fps=25, codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            print(f"  ✓ Video saved: {video_path}")
        except Exception as e:
            print(f"  ✗ Video rendering failed: {e}")
    
    print("\n" + "=" * 70)
    print("INFERENCE TESTING COMPLETE")
    print("=" * 70)
    print(f"\nTest videos saved to: {LOG_DIR}")
    print("\nResults show how well your trained policy performs!")
    print("Compare these with the pre-trained policy to see the improvement.")


# ============================================================================
# STANDALONE INFERENCE MODE
# ============================================================================

def run_standalone_inference():
    """Run inference on a previously trained model."""
    print("\n" + "=" * 70)
    print("BARKOUR INFERENCE MODE")
    print("=" * 70)
    
    # Check if trained model exists
    if not OUTPUT_MODEL_PATH.exists():
        print(f"\n✗ ERROR: No trained model found at {OUTPUT_MODEL_PATH}")
        print("\nPlease train a model first by running this script normally.")
        print("Or update OUTPUT_MODEL_PATH to point to your trained model.")
        return
    
    print(f"\nLoading trained model from: {OUTPUT_MODEL_PATH}")
    
    # Setup environment
    envs.register_environment('barkour', BarkourEnv)
    env = envs.get_environment('barkour', menagerie_path=MENAGERIE_PATH)
    
    # Load model
    try:
        params = model.load_params(OUTPUT_MODEL_PATH.as_posix())
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create inference function
    # We need to recreate the network factory to get make_inference_fn
    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128)
    )
    
    # Create a dummy training setup just to get the inference function structure
    # This is a bit hacky but necessary since we only saved params
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=1000,  # dummy value
        num_evals=1,
        reward_scaling=1,
        episode_length=EPISODE_LENGTH,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=4,
        num_updates_per_batch=4,
        discounting=0.97,
        learning_rate=LEARNING_RATE,
        entropy_cost=1e-2,
        num_envs=128,  # small for quick init
        batch_size=64,
        network_factory=make_networks_factory,
        seed=SEED
    )
    
    print("Initializing network architecture...")
    # This will JIT compile but not actually train
    make_inference_fn, _, _ = train_fn(environment=env)
    print("✓ Network initialized")
    
    # Run inference demo
    run_inference_demo(make_inference_fn, params, env)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Barkour quadruped training and inference')
    parser.add_argument('--inference-only', action='store_true', 
                       help='Skip training and only run inference on existing model')
    parser.add_argument('--no-prompt', action='store_true',
                       help='Skip confirmation prompt before training')
    args = parser.parse_args()
    
    # Check if files exist
    if not MENAGERIE_PATH.exists():
        print(f"✗ ERROR: Menagerie path not found: {MENAGERIE_PATH}")
        print("Please ensure mujoco_menagerie is cloned to the correct location.")
        sys.exit(1)
    
    scene_file = MENAGERIE_PATH / "scene_mjx.xml"
    if not scene_file.exists():
        print(f"✗ ERROR: Scene file not found: {scene_file}")
        sys.exit(1)
    
    # Inference-only mode
    if args.inference_only:
        run_standalone_inference()
        sys.exit(0)
    
    # Training mode
    print("\n" + "=" * 70)
    print("BARKOUR QUADRUPED - TRAINING FROM SCRATCH")
    print("Replicating Google Colab notebook training process")
    print("=" * 70 + "\n")
    
    print("✓ All required files found")
    print("\nThis will train a quadruped locomotion policy from scratch.")
    print(f"Training for {NUM_TIMESTEPS:,} steps with {NUM_ENVS} parallel environments.")
    print("\nPress Ctrl+C at any time to stop training (checkpoints will be saved).\n")
    
    if not args.no_prompt:
        input("Press Enter to start training, or Ctrl+C to cancel...")
    
    # Run training
    make_inference_fn, params, success = train_barkour()
    
    if success:
        print("\n" + "=" * 70)
        print("TRAINING SUCCESSFUL!")
        print("=" * 70)
        print(f"\nYour trained model is ready at: {OUTPUT_MODEL_PATH}")
        
        # Ask if user wants to test the policy
        print("\n" + "=" * 70)
        print("INFERENCE TEST")
        print("=" * 70)
        test_policy = input("\nWould you like to test the trained policy now? (y/n): ").lower().strip()
        
        if test_policy == 'y':
            run_inference_demo(make_inference_fn, params, None)
        else:
            print("\nTo test your trained model later, run:")
            print(f"  python train_barkour_local.py --inference-only")
            print(f"  or: python run_barkour_local.py")
        
        print("\n" + "=" * 70 + "\n")
    else:
        print("\nTraining did not complete successfully.")
        print("Check the error messages above for details.")
        sys.exit(1)
