#!/usr/bin/env python3
"""
Train DQN agent using Stable-Baselines3 for cloud autoscaling.
Simplified version for the autoscaling project.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS
from gym_scaling.load_generators import LOAD_PATTERNS


def create_env(load_pattern='SINE_CURVE'):
    """Create environment with specified load pattern."""
    env = ScalingEnv()
    
    # Set load pattern
    if load_pattern in LOAD_PATTERNS:
        pattern = LOAD_PATTERNS[load_pattern]
        def load_func(step, max_influx, offset):
            return pattern['function'](step, max_influx, offset, **pattern['options'])
        env.scaling_env_options['input'] = {
            'function': load_func,
            'options': {}
        }
    elif load_pattern == 'SINE_CURVE':
        env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    else:
        env.scaling_env_options['input'] = INPUTS['RANDOM']
    
    env.change_rate = 1  # Change influx every step
    
    return env


def train_dqn(load_pattern='SINE_CURVE', total_timesteps=50000, model_name=None):
    """
    Train DQN agent.
    
    Args:
        load_pattern: Load pattern to use ('SINE_CURVE', 'SINUSOIDAL', 'SPIKE', etc.)
        total_timesteps: Total training timesteps
        model_name: Name for saved model (auto-generated if None)
    
    Returns:
        Trained model and save path
    """
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dqn_{load_pattern.lower()}_{timestamp}"
    
    print("=" * 60)
    print("Training DQN Agent for Cloud Autoscaling")
    print("=" * 60)
    print(f"Load pattern: {load_pattern}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Model name: {model_name}")
    print("=" * 60)
    
    # Create environment
    env = create_env(load_pattern)
    env = Monitor(env)  # Wrap for logging
    
    # Create DQN model
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
    os.makedirs('models/checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='models/checkpoints',
        name_prefix=model_name
    )
    
    # Create eval environment
    eval_env = create_env(load_pattern)
    eval_env = Monitor(eval_env)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='models/best',
        log_path='logs/dqn_eval',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10
    )
    
    # Save final model
    model_path = f"models/{model_name}.zip"
    model.save(model_path)
    
    print("\n" + "=" * 60)
    print(f"✓ Training complete!")
    print(f"✓ Model saved to: {model_path}")
    print("=" * 60)
    
    env.close()
    eval_env.close()
    
    return model, model_path


def evaluate_dqn(model_path, load_pattern='SINE_CURVE', num_episodes=5, max_steps=200):
    """
    Evaluate trained DQN model.
    
    Args:
        model_path: Path to saved model
        load_pattern: Load pattern to test on
        num_episodes: Number of episodes to run
        max_steps: Steps per episode
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating DQN model: {model_path}")
    print(f"Load pattern: {load_pattern}")
    
    # Load model
    env = create_env(load_pattern)
    model = DQN.load(model_path)
    
    all_metrics = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        
        episode_metrics = {
            'rewards': [],
            'instances': [],
            'load': [],
            'queue_size': [],
            'influx': [],
            'actions': []
        }
        
        for step in range(max_steps):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # Collect metrics
            episode_metrics['rewards'].append(reward)
            episode_metrics['instances'].append(len(env.instances))
            episode_metrics['load'].append(env.load)
            episode_metrics['queue_size'].append(env.queue_size)
            episode_metrics['influx'].append(env.influx)
            episode_metrics['actions'].append(env.actions[action])
            
            if done:
                break
        
        all_metrics.append(episode_metrics)
        
        print(f"  Episode {episode + 1}: "
              f"Reward={sum(episode_metrics['rewards']):.2f}, "
              f"Avg Load={np.mean(episode_metrics['load']):.1f}%, "
              f"Avg Queue={np.mean(episode_metrics['queue_size']):.2f}")
    
    # Aggregate metrics
    aggregated = {
        'avg_reward': np.mean([sum(m['rewards']) for m in all_metrics]),
        'avg_load': np.mean([np.mean(m['load']) for m in all_metrics]),
        'avg_queue': np.mean([np.mean(m['queue_size']) for m in all_metrics]),
        'avg_instances': np.mean([np.mean(m['instances']) for m in all_metrics]),
        'episodes': all_metrics
    }
    
    env.close()
    
    return aggregated


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for autoscaling')
    parser.add_argument('--pattern', type=str, default='SINE_CURVE',
                       choices=['SINE_CURVE', 'SINUSOIDAL', 'STEADY', 'SPIKE', 'POISSON', 'RANDOM'],
                       help='Load pattern for training')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--name', type=str, default=None,
                       help='Model name for saving')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate after training')
    
    args = parser.parse_args()
    
    # Train
    model, model_path = train_dqn(
        load_pattern=args.pattern,
        total_timesteps=args.timesteps,
        model_name=args.name
    )
    
    # Evaluate if requested
    if args.eval:
        metrics = evaluate_dqn(model_path, load_pattern=args.pattern)
        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {metrics['avg_reward']:.2f}")
        print(f"  Average Load: {metrics['avg_load']:.1f}%")
        print(f"  Average Queue: {metrics['avg_queue']:.2f}")
        print(f"  Average Instances: {metrics['avg_instances']:.1f}")
