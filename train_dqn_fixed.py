#!/usr/bin/env python3
"""
Fixed DQN training script that works with the direct ScalingEnv approach.
No gym wrappers, no numpy compatibility issues.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS
from gym_scaling.load_generators import LOAD_PATTERNS


class GymnasiumWrapper(gym.Env):
    """Wrap ScalingEnv to be compatible with Gymnasium and Stable-Baselines3."""
    
    def __init__(self, load_pattern='SINE_CURVE'):
        super().__init__()
        
        # Create the underlying environment
        self.env = ScalingEnv()
        
        # Set load pattern
        if load_pattern in LOAD_PATTERNS:
            pattern = LOAD_PATTERNS[load_pattern]
            def load_func(step, max_influx, offset):
                return pattern['function'](step, max_influx, offset, **pattern['options'])
            self.env.scaling_env_options['input'] = {
                'function': load_func,
                'options': {}
            }
        elif load_pattern == 'SINE_CURVE':
            self.env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
        else:
            self.env.scaling_env_options['input'] = INPUTS['RANDOM']
        
        self.env.change_rate = 1
        
        # Define action and observation spaces for Gymnasium
        self.action_space = spaces.Discrete(self.env.num_actions)
        
        # Observation space: [normalized_instances, load, capacity, influx, queue]
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(5,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        obs = self.env.reset()
        obs = obs.astype(np.float32)
        
        return obs, {}
    
    def step(self, action):
        """Take a step in the environment."""
        obs, reward, done, info = self.env.step(action)
        obs = obs.astype(np.float32)
        
        # Gymnasium API: return (obs, reward, terminated, truncated, info)
        terminated = done
        truncated = False
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()


def make_env(load_pattern='SINE_CURVE'):
    """Create and wrap environment."""
    def _init():
        env = GymnasiumWrapper(load_pattern=load_pattern)
        env = Monitor(env)
        return env
    return _init


def train_dqn(load_pattern='SINE_CURVE', total_timesteps=50000, model_name=None):
    """
    Train DQN agent with fixed environment wrapper.
    
    Args:
        load_pattern: Load pattern to use
        total_timesteps: Total training timesteps
        model_name: Name for saved model
    
    Returns:
        Trained model and save path
    """
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dqn_{load_pattern.lower()}_{timestamp}"
    
    print("=" * 70)
    print("TRAINING DQN AGENT FOR CLOUD AUTOSCALING")
    print("=" * 70)
    print(f"Load pattern: {load_pattern}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Model name: {model_name}")
    print(f"Estimated time: ~{total_timesteps // 1000} minutes")
    print("=" * 70)
    
    # Create directories
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('models/best', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create vectorized environment
    env = DummyVecEnv([make_env(load_pattern)])
    
    # Create DQN model
    print("\nCreating DQN model...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        seed=42
    )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='models/checkpoints',
        name_prefix=model_name
    )
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(load_pattern)])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/best',
        log_path='logs/dqn_eval',
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Train
    print("\nStarting training...")
    print("Progress will be shown below:")
    print("-" * 70)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        model_path = f"models/{model_name}.zip"
        model.save(model_path)
        
        print("\n" + "=" * 70)
        print("✓ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Best model saved to: models/best/")
        print(f"✓ Checkpoints saved to: models/checkpoints/")
        print("=" * 70)
        
        env.close()
        eval_env.close()
        
        return model, model_path
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        model_path = f"models/{model_name}_interrupted.zip"
        model.save(model_path)
        print(f"Model saved to: {model_path}")
        
        env.close()
        eval_env.close()
        
        return model, model_path


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent (fixed version)')
    parser.add_argument('--pattern', type=str, default='SINE_CURVE',
                       choices=['SINE_CURVE', 'SINUSOIDAL', 'STEADY', 'SPIKE', 'POISSON', 'RANDOM'],
                       help='Load pattern for training')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps (default: 50000)')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name for saving')
    
    args = parser.parse_args()
    
    print("\n🚀 Starting DQN Training")
    print(f"   Pattern: {args.pattern}")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Estimated time: ~{args.timesteps // 1000} minutes\n")
    
    # Train
    model, model_path = train_dqn(
        load_pattern=args.pattern,
        total_timesteps=args.timesteps,
        model_name=args.name
    )
    
    print("\n✓ Training script completed successfully!")
    print(f"\nTo test your model, run:")
    print(f"  python -c \"from dqn_demo import load_dqn_model, create_env_with_pattern, run_dqn_agent; "
          f"model = load_dqn_model('{model_path}'); "
          f"env = create_env_with_pattern('SINE_CURVE'); "
          f"metrics = run_dqn_agent(model, env, 200); "
          f"print(f'Avg reward: {{sum(metrics[\\\"rewards\\\"]):.2f}}')\"")
