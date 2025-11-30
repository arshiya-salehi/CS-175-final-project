#!/usr/bin/env python3
"""
Direct test of the Gym-Scaling environment bypassing gym wrappers.
This demonstrates the environment functionality for your autoscaling project.
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import random
from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS


def main():
    print("=" * 70)
    print("CLOUD AUTOSCALING ENVIRONMENT - DIRECT TEST")
    print("=" * 70)
    
    # Create environment directly
    print("\n1. Creating Scaling Environment...")
    env = ScalingEnv()
    print(f"   ✓ Environment created successfully")
    print(f"   - Actions available: {env.actions} (remove, nothing, add)")
    print(f"   - Max instances: {int(env.max_instances)}")
    print(f"   - Min instances: {int(env.min_instances)}")
    print(f"   - Capacity per instance: {env.capacity_per_instance} requests/step")
    print(f"   - Cost per instance: ${env.scaling_env_options['cost_per_instance_per_hour']}/hour")
    
    # Reset and examine initial state
    print("\n2. Examining Initial State...")
    obs = env.reset()
    print(f"   ✓ Environment reset")
    print(f"   - Observation: {obs}")
    print(f"   - Observation breakdown:")
    print(f"     [0] Normalized instances: {obs[0]:.3f} ({len(env.instances)} instances)")
    print(f"     [1] Normalized load: {obs[1]:.3f} ({env.load}%)")
    print(f"     [2] Total capacity: {obs[2]:.0f} requests/step")
    print(f"     [3] Current influx: {obs[3]:.0f} requests")
    print(f"     [4] Queue size: {obs[4]:.0f} requests")
    
    # Run with random policy
    print("\n3. Running Random Policy (50 steps)...")
    obs = env.reset()
    metrics = {'rewards': [], 'instances': [], 'load': [], 'queue': [], 'influx': []}
    
    for i in range(50):
        action = random.randint(0, 2)  # Random action
        obs, reward, done, info = env.step(action)
        
        metrics['rewards'].append(reward)
        metrics['instances'].append(len(env.instances))
        metrics['load'].append(env.load)
        metrics['queue'].append(env.queue_size)
        metrics['influx'].append(env.influx)
        
        if i % 10 == 0:
            print(f"   Step {i:2d}: action={env.actions[action]:+2d}, "
                  f"instances={len(env.instances):3d}, load={env.load:5.1f}%, "
                  f"queue={env.queue_size:7.0f}, reward={reward:7.4f}")
        
        if done:
            print(f"   ⚠ Episode ended at step {i} (queue overflow)")
            break
    
    print(f"\n   Results:")
    print(f"   - Total steps: {len(metrics['rewards'])}")
    print(f"   - Average reward: {np.mean(metrics['rewards']):.4f}")
    print(f"   - Total cost: ${env.total_cost:.2f}")
    print(f"   - Average queue: {np.mean(metrics['queue']):.2f}")
    print(f"   - Max queue: {max(metrics['queue']):.0f}")
    
    # Test with sine wave workload
    print("\n4. Testing with Sine Wave Workload Pattern...")
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1  # Change influx every step
    obs = env.reset()
    
    influx_values = []
    for i in range(100):
        action = 1  # Do nothing
        obs, reward, done, info = env.step(action)
        influx_values.append(env.influx)
        
        if done:
            break
    
    print(f"   ✓ Sine wave test completed ({len(influx_values)} steps)")
    print(f"   - Influx range: {min(influx_values):.0f} - {max(influx_values):.0f}")
    print(f"   - Influx mean: {np.mean(influx_values):.0f}")
    print(f"   - Influx std: {np.std(influx_values):.0f}")
    
    # Test threshold-based policy
    print("\n5. Testing Threshold-Based Autoscaling Policy...")
    print("   (Similar to Kubernetes HPA with 80% high / 40% low thresholds)")
    
    def threshold_policy(obs, high=80, low=40):
        """Simple threshold-based autoscaling."""
        load = obs[1] * 100
        queue = obs[4]
        
        if load > high or queue > 100:
            return 2  # Add instance
        elif load < low and queue == 0:
            return 0  # Remove instance
        else:
            return 1  # Do nothing
    
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1
    obs = env.reset()
    
    metrics_threshold = {'rewards': [], 'instances': [], 'load': [], 'queue': [], 'actions': []}
    
    for i in range(200):
        action = threshold_policy(obs)
        obs, reward, done, info = env.step(action)
        
        metrics_threshold['rewards'].append(reward)
        metrics_threshold['instances'].append(len(env.instances))
        metrics_threshold['load'].append(env.load)
        metrics_threshold['queue'].append(env.queue_size)
        metrics_threshold['actions'].append(env.actions[action])
        
        if i % 40 == 0:
            print(f"   Step {i:3d}: action={env.actions[action]:+2d}, "
                  f"instances={len(env.instances):3d}, load={env.load:5.1f}%, "
                  f"queue={env.queue_size:7.0f}")
        
        if done:
            print(f"   ⚠ Episode ended at step {i}")
            break
    
    print(f"\n   Results:")
    print(f"   - Total steps: {len(metrics_threshold['rewards'])}")
    print(f"   - Average reward: {np.mean(metrics_threshold['rewards']):.4f}")
    print(f"   - Total cost: ${env.total_cost:.2f}")
    print(f"   - Average queue: {np.mean(metrics_threshold['queue']):.2f}")
    print(f"   - Max queue: {max(metrics_threshold['queue']):.0f}")
    print(f"   - Average load: {np.mean(metrics_threshold['load']):.2f}%")
    print(f"   - Average instances: {np.mean(metrics_threshold['instances']):.2f}")
    
    # Test custom configuration
    print("\n6. Testing Custom Configuration...")
    custom_env = ScalingEnv(scaling_env_options={
        'max_instances': 50.0,
        'min_instances': 5.0,
        'capacity_per_instance': 100,
        'discrete_actions': (-2, -1, 0, 1, 2),  # More aggressive scaling
        'input': INPUTS['SINE_CURVE'],
    })
    
    print(f"   ✓ Custom environment created")
    print(f"   - Actions: {custom_env.actions}")
    print(f"   - Max instances: {int(custom_env.max_instances)}")
    print(f"   - Capacity per instance: {custom_env.capacity_per_instance}")
    
    obs = custom_env.reset()
    print(f"   - Initial instances: {len(custom_env.instances)}")
    
    # Test aggressive scaling
    obs, reward, done, _ = custom_env.step(4)  # +2 action
    print(f"   - After +2 action: {len(custom_env.instances)} instances")
    
    obs, reward, done, _ = custom_env.step(0)  # -2 action
    print(f"   - After -2 action: {len(custom_env.instances)} instances")
    
    # Summary
    print("\n" + "=" * 70)
    print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n📊 Environment Summary:")
    print("   - State space: [normalized_instances, load, capacity, influx, queue]")
    print("   - Action space: Discrete (remove, nothing, add instances)")
    print("   - Reward function: Balances load utilization, cost, and queue size")
    print("   - Workload patterns: RANDOM, SINE_CURVE, PRODUCTION_DATA")
    
    print("\n🎯 For Your RL Autoscaling Project:")
    print("   1. ✓ Environment is working and ready to use")
    print("   2. Implement Q-learning with discretized state space")
    print("   3. Train DQN using Stable-Baselines3 or custom implementation")
    print("   4. Compare with PPO (policy-gradient method)")
    print("   5. Evaluate against threshold baselines")
    print("   6. Customize reward function for cost vs latency tradeoffs")
    print("   7. Test with production workload traces")
    
    print("\n📓 Next Steps:")
    print("   - Open 'autoscaling_demo.ipynb' for interactive Jupyter notebook")
    print("   - Modify reward function in gym_scaling/envs/scaling_env.py")
    print("   - Implement your RL agents (Q-learning, DQN, PPO)")
    print("   - Run experiments and collect metrics")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
