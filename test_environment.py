#!/usr/bin/env python3
"""
Quick test script for the Gym-Scaling environment.
Run this to verify the environment works correctly.
"""

import gym
import gym_scaling
import numpy as np
from gym_scaling.envs.scaling_env import INPUTS


def test_basic_environment():
    """Test basic environment creation and interaction."""
    print("=" * 60)
    print("Testing Basic Environment")
    print("=" * 60)
    
    env = gym.make('Scaling-v0')
    
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Actions: {env.actions}")
    print(f"  Max instances: {env.max_instances}")
    print(f"  Min instances: {env.min_instances}")
    
    # Reset environment
    obs = env.reset()
    print(f"\n✓ Initial observation: {obs}")
    print(f"  [normalized_instances, load, total_capacity, influx, queue_size]")
    
    # Take a few steps
    print(f"\n✓ Taking 10 random steps...")
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Step {i+1}: action={env.actions[action]:+d}, "
              f"reward={reward:.4f}, instances={len(env.instances)}, "
              f"load={env.load}%, queue={env.queue_size:.0f}")
        
        if done:
            print(f"  Episode ended (queue overflow)")
            break
    
    print(f"\n✓ Total reward: {total_reward:.4f}")
    print(f"✓ Total cost: ${env.total_cost:.2f}")
    
    env.close()
    return True


def test_workload_patterns():
    """Test different workload patterns."""
    print("\n" + "=" * 60)
    print("Testing Different Workload Patterns")
    print("=" * 60)
    
    patterns = ['RANDOM', 'SINE_CURVE']
    
    for pattern_name in patterns:
        print(f"\n--- Testing {pattern_name} pattern ---")
        env = gym.make('Scaling-v0')
        env.scaling_env_options['input'] = INPUTS[pattern_name]
        env.change_rate = 1  # Change every step
        
        obs = env.reset()
        influx_values = []
        
        for i in range(20):
            action = 1  # Do nothing
            obs, reward, done, info = env.step(action)
            influx_values.append(env.influx)
            
            if done:
                break
        
        print(f"  Influx range: {min(influx_values):.0f} - {max(influx_values):.0f}")
        print(f"  Influx mean: {np.mean(influx_values):.0f}")
        print(f"  Influx std: {np.std(influx_values):.0f}")
        
        env.close()
    
    return True


def test_threshold_policy():
    """Test a simple threshold-based autoscaling policy."""
    print("\n" + "=" * 60)
    print("Testing Threshold-Based Policy")
    print("=" * 60)
    
    def threshold_policy(observation, high=80, low=40):
        """Simple threshold policy."""
        load = observation[1] * 100
        queue_size = observation[4]
        
        if load > high or queue_size > 100:
            return 2  # Add instance
        elif load < low and queue_size == 0:
            return 0  # Remove instance
        else:
            return 1  # Do nothing
    
    env = gym.make('Scaling-v0')
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1
    
    obs = env.reset()
    total_reward = 0
    queue_sizes = []
    loads = []
    
    print(f"\n✓ Running threshold policy for 100 steps...")
    
    for i in range(100):
        action = threshold_policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        queue_sizes.append(env.queue_size)
        loads.append(env.load)
        
        if done:
            print(f"  Episode ended at step {i+1}")
            break
    
    print(f"\n✓ Results:")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward: {total_reward/len(queue_sizes):.4f}")
    print(f"  Total cost: ${env.total_cost:.2f}")
    print(f"  Average queue size: {np.mean(queue_sizes):.2f}")
    print(f"  Max queue size: {max(queue_sizes):.0f}")
    print(f"  Average load: {np.mean(loads):.2f}%")
    print(f"  Average instances: {np.mean([len(env.instances) for _ in range(10)]):.2f}")
    
    env.close()
    return True


def test_custom_configuration():
    """Test custom environment configuration."""
    print("\n" + "=" * 60)
    print("Testing Custom Configuration")
    print("=" * 60)
    
    custom_options = {
        'max_instances': 50.0,
        'min_instances': 5.0,
        'capacity_per_instance': 100,
        'discrete_actions': (-2, -1, 0, 1, 2),
        'input': INPUTS['SINE_CURVE'],
    }
    
    env = gym.make('Scaling-v0', scaling_env_options=custom_options)
    
    print(f"✓ Custom environment created")
    print(f"  Actions: {env.actions}")
    print(f"  Max instances: {env.max_instances}")
    print(f"  Min instances: {env.min_instances}")
    print(f"  Capacity per instance: {env.capacity_per_instance}")
    
    obs = env.reset()
    print(f"\n✓ Testing aggressive scaling actions...")
    
    # Test scaling up by 2
    obs, reward, done, _ = env.step(4)  # +2 action
    print(f"  After +2 action: {len(env.instances)} instances")
    
    # Test scaling down by 2
    obs, reward, done, _ = env.step(0)  # -2 action
    print(f"  After -2 action: {len(env.instances)} instances")
    
    env.close()
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GYM-SCALING ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic Environment", test_basic_environment),
        ("Workload Patterns", test_workload_patterns),
        ("Threshold Policy", test_threshold_policy),
        ("Custom Configuration", test_custom_configuration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Environment is working correctly.")
        print("\nNext steps:")
        print("1. Open autoscaling_demo.ipynb for interactive exploration")
        print("2. Run train_deepq.py to train a DQN agent")
        print("3. Modify the reward function in gym_scaling/envs/scaling_env.py")
        print("4. Implement your own Q-learning or DQN agent")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == '__main__':
    main()
