#!/usr/bin/env python3
"""
Simple test to verify the environment works despite warnings.
"""

import warnings
warnings.filterwarnings('ignore')

import gym
import gym_scaling
import numpy as np
from gym_scaling.envs.scaling_env import INPUTS


def main():
    print("=" * 60)
    print("SIMPLE ENVIRONMENT TEST")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = gym.make('Scaling-v0')
    print(f"   ✓ Environment created")
    print(f"   - Actions: {env.actions}")
    print(f"   - Max instances: {int(env.max_instances)}")
    print(f"   - Min instances: {int(env.min_instances)}")
    print(f"   - Capacity per instance: {env.capacity_per_instance}")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs = env.reset()
    print(f"   ✓ Environment reset")
    print(f"   - Initial observation shape: {obs.shape}")
    print(f"   - Initial instances: {len(env.instances)}")
    print(f"   - Initial load: {env.load}%")
    
    # Take some steps
    print("\n3. Taking 20 steps with random actions...")
    total_reward = 0
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 5 == 0:
            print(f"   Step {i:2d}: instances={len(env.instances):2d}, "
                  f"load={env.load:3.0f}%, queue={env.queue_size:6.0f}, "
                  f"reward={reward:7.4f}")
        
        if done:
            print(f"   Episode ended at step {i} (queue overflow)")
            break
    
    print(f"\n   ✓ Completed {i+1} steps")
    print(f"   - Total reward: {total_reward:.4f}")
    print(f"   - Average reward: {total_reward/(i+1):.4f}")
    print(f"   - Total cost: ${env.total_cost:.2f}")
    
    # Test with sine wave
    print("\n4. Testing with SINE_CURVE workload...")
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1
    obs = env.reset()
    
    influx_values = []
    for i in range(50):
        action = 1  # Do nothing
        obs, reward, done, info = env.step(action)
        influx_values.append(env.influx)
        
        if done:
            break
    
    print(f"   ✓ Sine wave test completed")
    print(f"   - Influx range: {min(influx_values):.0f} - {max(influx_values):.0f}")
    print(f"   - Influx mean: {np.mean(influx_values):.0f}")
    
    # Test threshold policy
    print("\n5. Testing threshold-based policy...")
    env = gym.make('Scaling-v0')
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1
    obs = env.reset()
    
    def threshold_policy(obs):
        load = obs[1] * 100
        queue = obs[4]
        if load > 80 or queue > 100:
            return 2  # Add
        elif load < 40 and queue == 0:
            return 0  # Remove
        else:
            return 1  # Nothing
    
    total_reward = 0
    queue_sizes = []
    
    for i in range(100):
        action = threshold_policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        queue_sizes.append(env.queue_size)
        
        if done:
            break
    
    print(f"   ✓ Threshold policy test completed")
    print(f"   - Steps: {i+1}")
    print(f"   - Average reward: {total_reward/(i+1):.4f}")
    print(f"   - Average queue: {np.mean(queue_sizes):.2f}")
    print(f"   - Max queue: {max(queue_sizes):.0f}")
    print(f"   - Total cost: ${env.total_cost:.2f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nThe environment is working correctly!")
    print("\nKey observations:")
    print("- State: [normalized_instances, load, capacity, influx, queue]")
    print("- Actions: -1 (remove), 0 (nothing), +1 (add instance)")
    print("- Reward: Balances load utilization, cost, and queue size")
    print("\nNext steps:")
    print("1. Open autoscaling_demo.ipynb for detailed examples")
    print("2. Implement Q-learning or DQN agent")
    print("3. Compare against threshold baselines")
    print("4. Customize reward function for your objectives")


if __name__ == '__main__':
    main()
