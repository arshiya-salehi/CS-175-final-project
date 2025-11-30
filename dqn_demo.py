"""
DQN Demo functions for Jupyter notebook.
Load and test pre-trained DQN models.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS
from gym_scaling.load_generators import LOAD_PATTERNS


def load_dqn_model(model_path='models/dqn_test.zip'):
    """Load a trained DQN model."""
    try:
        model = DQN.load(model_path)
        print(f"✓ DQN model loaded from: {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


def create_env_with_pattern(load_pattern='SINE_CURVE'):
    """Create environment with specified load pattern."""
    env = ScalingEnv()
    
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
    
    env.change_rate = 1
    return env


def run_dqn_agent(model, env, num_steps=200):
    """
    Run DQN agent and collect metrics.
    
    Args:
        model: Trained DQN model
        env: Environment instance
        num_steps: Number of steps to run
    
    Returns:
        Dictionary with metrics
    """
    obs = env.reset()
    
    metrics = {
        'rewards': [],
        'instances': [],
        'load': [],
        'queue_size': [],
        'influx': [],
        'actions': []
    }
    
    for step in range(num_steps):
        # Get action from DQN model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Store metrics
        metrics['rewards'].append(reward)
        metrics['instances'].append(len(env.instances))
        metrics['load'].append(env.load)
        metrics['queue_size'].append(env.queue_size)
        metrics['influx'].append(env.influx)
        metrics['actions'].append(env.actions[action])
        
        if done:
            print(f"Episode ended at step {step} (queue overflow)")
            break
    
    return metrics


def plot_dqn_performance(metrics, title="DQN Agent Performance"):
    """Plot DQN agent performance metrics."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(title, fontsize=16)
    
    # Rewards
    axes[0, 0].plot(metrics['rewards'], color='blue')
    axes[0, 0].set_title('Reward per Step')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Instances
    axes[0, 1].plot(metrics['instances'], color='green')
    axes[0, 1].set_title('Number of Instances')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Instances')
    axes[0, 1].grid(True)
    
    # Load
    axes[1, 0].plot(metrics['load'], color='orange')
    axes[1, 0].set_title('CPU Load (%)')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Load %')
    axes[1, 0].axhline(y=80, color='r', linestyle='--', alpha=0.3, label='High threshold')
    axes[1, 0].axhline(y=40, color='b', linestyle='--', alpha=0.3, label='Low threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Queue Size
    axes[1, 1].plot(metrics['queue_size'], color='red')
    axes[1, 1].set_title('Queue Size')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Queue Size')
    axes[1, 1].grid(True)
    
    # Influx
    axes[2, 0].plot(metrics['influx'], color='purple')
    axes[2, 0].set_title('Request Influx')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Requests')
    axes[2, 0].grid(True)
    
    # Actions
    axes[2, 1].plot(metrics['actions'], color='brown', marker='o', markersize=2)
    axes[2, 1].set_title('Scaling Actions')
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('Action (-1, 0, +1)')
    axes[2, 1].set_yticks([-1, 0, 1])
    axes[2, 1].set_yticklabels(['Remove', 'Nothing', 'Add'])
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def compare_policies(metrics_dict):
    """
    Compare multiple policies side by side.
    
    Args:
        metrics_dict: Dictionary mapping policy names to their metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Policy Comparison', fontsize=16)
    
    # Cumulative reward
    for name, metrics in metrics_dict.items():
        axes[0, 0].plot(np.cumsum(metrics['rewards']), label=name, linewidth=2)
    axes[0, 0].set_title('Cumulative Reward')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Cumulative Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Average queue size
    names = list(metrics_dict.keys())
    avg_queues = [np.mean(metrics_dict[name]['queue_size']) for name in names]
    axes[0, 1].bar(names, avg_queues, color=['blue', 'green', 'red', 'orange'][:len(names)])
    axes[0, 1].set_title('Average Queue Size')
    axes[0, 1].set_ylabel('Queue Size')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Average instances
    avg_instances = [np.mean(metrics_dict[name]['instances']) for name in names]
    axes[0, 2].bar(names, avg_instances, color=['blue', 'green', 'red', 'orange'][:len(names)])
    axes[0, 2].set_title('Average Instances')
    axes[0, 2].set_ylabel('Instances')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Average load
    avg_loads = [np.mean(metrics_dict[name]['load']) for name in names]
    axes[1, 0].bar(names, avg_loads, color=['blue', 'green', 'red', 'orange'][:len(names)])
    axes[1, 0].set_title('Average Load (%)')
    axes[1, 0].set_ylabel('Load %')
    axes[1, 0].axhline(y=80, color='r', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(y=40, color='b', linestyle='--', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Load over time comparison
    for name, metrics in metrics_dict.items():
        axes[1, 1].plot(metrics['load'], label=name, alpha=0.7)
    axes[1, 1].set_title('Load Over Time')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Load %')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Instances over time comparison
    for name, metrics in metrics_dict.items():
        axes[1, 2].plot(metrics['instances'], label=name, alpha=0.7)
    axes[1, 2].set_title('Instances Over Time')
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Instances')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Policy':<20} {'Avg Reward':<15} {'Avg Queue':<15} {'Avg Instances':<15} {'Avg Load':<15}")
    print("-" * 80)
    for name in names:
        metrics = metrics_dict[name]
        print(f"{name:<20} {np.mean(metrics['rewards']):<15.4f} "
              f"{np.mean(metrics['queue_size']):<15.2f} "
              f"{np.mean(metrics['instances']):<15.2f} "
              f"{np.mean(metrics['load']):<15.2f}")
    print("=" * 80)


def demo_dqn_on_patterns(model, patterns=['SINE_CURVE', 'SINUSOIDAL', 'SPIKE'], num_steps=200):
    """
    Test DQN model on multiple load patterns.
    
    Args:
        model: Trained DQN model
        patterns: List of load pattern names
        num_steps: Steps per pattern
    
    Returns:
        Dictionary mapping patterns to metrics
    """
    results = {}
    
    for pattern in patterns:
        print(f"\nTesting on {pattern} pattern...")
        env = create_env_with_pattern(pattern)
        metrics = run_dqn_agent(model, env, num_steps)
        results[pattern] = metrics
        
        print(f"  Average reward: {np.mean(metrics['rewards']):.4f}")
        print(f"  Average load: {np.mean(metrics['load']):.1f}%")
        print(f"  Average queue: {np.mean(metrics['queue_size']):.2f}")
        print(f"  Average instances: {np.mean(metrics['instances']):.1f}")
        
        env.close()
    
    return results


# Quick test function
def quick_dqn_test():
    """Quick test of DQN model - use this in notebook."""
    print("=" * 60)
    print("DQN Agent Quick Test")
    print("=" * 60)
    
    # Load model
    model = load_dqn_model('models/dqn_test.zip')
    if model is None:
        print("\n✗ No trained model found!")
        print("  Run train_dqn_sb3.py first to train a model.")
        return None
    
    # Test on SINE_CURVE
    print("\nTesting on SINE_CURVE workload...")
    env = create_env_with_pattern('SINE_CURVE')
    metrics = run_dqn_agent(model, env, num_steps=200)
    env.close()
    
    # Print results
    print(f"\n✓ Test complete!")
    print(f"  Average reward: {np.mean(metrics['rewards']):.4f}")
    print(f"  Average load: {np.mean(metrics['load']):.1f}%")
    print(f"  Average queue: {np.mean(metrics['queue_size']):.2f}")
    print(f"  Average instances: {np.mean(metrics['instances']):.1f}")
    
    # Plot
    plot_dqn_performance(metrics, "DQN Agent on Sine Wave Workload")
    
    return metrics
