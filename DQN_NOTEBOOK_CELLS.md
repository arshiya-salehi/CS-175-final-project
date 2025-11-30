# DQN Cells to Add to Notebook

Add these cells to your `autoscaling_demo.ipynb` notebook to test the DQN agent.

## Cell 1: Install Stable-Baselines3 (if needed)

```python
# Install stable-baselines3 if not already installed
# Uncomment the line below if you need to install it
# !pip install stable-baselines3
```

## Cell 2: Import DQN Demo Functions

```python
# Import DQN demo functions
import sys
sys.path.insert(0, '.')

from dqn_demo import (
    load_dqn_model,
    create_env_with_pattern,
    run_dqn_agent,
    plot_dqn_performance,
    compare_policies,
    demo_dqn_on_patterns,
    quick_dqn_test
)

print("✓ DQN demo functions imported successfully")
```

## Cell 3: Quick DQN Test

```python
# Quick test of pre-trained DQN model
dqn_metrics = quick_dqn_test()
```

**Expected Output:**
- Loads the pre-trained DQN model from `models/dqn_test.zip`
- Tests it on SINE_CURVE workload for 200 steps
- Displays performance metrics
- Shows visualization plots

## Cell 4: Test DQN on Multiple Load Patterns

```python
# Load DQN model
dqn_model = load_dqn_model('models/dqn_test.zip')

if dqn_model is not None:
    # Test on multiple patterns
    patterns = ['SINE_CURVE', 'SINUSOIDAL', 'SPIKE', 'STEADY']
    dqn_pattern_results = demo_dqn_on_patterns(dqn_model, patterns, num_steps=200)
    
    # Plot results for each pattern
    for pattern, metrics in dqn_pattern_results.items():
        plot_dqn_performance(metrics, f"DQN Agent on {pattern} Workload")
```

**What this does:**
- Tests DQN on 4 different workload patterns
- Shows how well the agent generalizes
- Plots performance for each pattern

## Cell 5: Compare DQN vs Threshold vs Q-Learning

```python
# Compare all three policies on SINE_CURVE workload
print("Comparing DQN, Threshold, and Q-Learning policies...")

# 1. DQN agent
env_dqn = create_env_with_pattern('SINE_CURVE')
dqn_model = load_dqn_model('models/dqn_test.zip')
dqn_comparison_metrics = run_dqn_agent(dqn_model, env_dqn, num_steps=200)
env_dqn.close()

# 2. Threshold policy (from earlier cell)
env_threshold = ScalingEnv()
env_threshold.scaling_env_options['input'] = INPUTS['SINE_CURVE']
env_threshold.change_rate = 1

def threshold_policy(obs, high=80, low=40):
    load = obs[1] * 100
    queue = obs[4]
    if load > high or queue > 100:
        return 2  # Add
    elif load < low and queue == 0:
        return 0  # Remove
    else:
        return 1  # Nothing

obs = env_threshold.reset()
threshold_comparison_metrics = {
    'rewards': [], 'instances': [], 'load': [],
    'queue_size': [], 'influx': [], 'actions': []
}

for step in range(200):
    action = threshold_policy(obs)
    obs, reward, done, info = env_threshold.step(action)
    threshold_comparison_metrics['rewards'].append(reward)
    threshold_comparison_metrics['instances'].append(len(env_threshold.instances))
    threshold_comparison_metrics['load'].append(env_threshold.load)
    threshold_comparison_metrics['queue_size'].append(env_threshold.queue_size)
    threshold_comparison_metrics['influx'].append(env_threshold.influx)
    threshold_comparison_metrics['actions'].append(env_threshold.actions[action])
    if done:
        break

env_threshold.close()

# 3. Q-Learning agent (if available)
try:
    import pickle
    with open('models/qlearning_model.pkl', 'rb') as f:
        q_table_dict = pickle.load(f)
    
    def discretize_state(obs, bins=10):
        instances_bin = min(int(obs[0] * bins), bins - 1)
        load_bin = min(int(obs[1] * bins), bins - 1)
        if obs[4] == 0:
            queue_bin = 0
        elif obs[4] < 100:
            queue_bin = 1
        elif obs[4] < 500:
            queue_bin = 2
        else:
            queue_bin = 3
        return (instances_bin, load_bin, queue_bin)
    
    env_qlearning = ScalingEnv()
    env_qlearning.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env_qlearning.change_rate = 1
    
    obs = env_qlearning.reset()
    qlearning_comparison_metrics = {
        'rewards': [], 'instances': [], 'load': [],
        'queue_size': [], 'influx': [], 'actions': []
    }
    
    for step in range(200):
        state = discretize_state(obs)
        if state in q_table_dict:
            action = np.argmax(q_table_dict[state])
        else:
            action = 1
        
        obs, reward, done, info = env_qlearning.step(action)
        qlearning_comparison_metrics['rewards'].append(reward)
        qlearning_comparison_metrics['instances'].append(len(env_qlearning.instances))
        qlearning_comparison_metrics['load'].append(env_qlearning.load)
        qlearning_comparison_metrics['queue_size'].append(env_qlearning.queue_size)
        qlearning_comparison_metrics['influx'].append(env_qlearning.influx)
        qlearning_comparison_metrics['actions'].append(env_qlearning.actions[action])
        if done:
            break
    
    env_qlearning.close()
    
    # Compare all three
    comparison_dict = {
        'DQN': dqn_comparison_metrics,
        'Threshold': threshold_comparison_metrics,
        'Q-Learning': qlearning_comparison_metrics
    }
    
except FileNotFoundError:
    print("Q-Learning model not found, comparing DQN vs Threshold only")
    comparison_dict = {
        'DQN': dqn_comparison_metrics,
        'Threshold': threshold_comparison_metrics
    }

# Plot comparison
compare_policies(comparison_dict)
```

**What this does:**
- Runs all three policies on the same workload
- Compares performance side-by-side
- Shows which policy performs best

## Cell 6: Train Your Own DQN Model (Optional)

```python
# Train a new DQN model (takes ~10-15 minutes for 50k timesteps)
# Uncomment to train

# from train_dqn_sb3 import train_dqn

# print("Training new DQN model...")
# print("This will take approximately 10-15 minutes for 50,000 timesteps")
# print("You can reduce timesteps for faster training (e.g., 10000)")

# model, model_path = train_dqn(
#     load_pattern='SINE_CURVE',
#     total_timesteps=50000,  # Reduce to 10000 for quick test
#     model_name='my_dqn_model'
# )

# print(f"\n✓ Training complete! Model saved to: {model_path}")

# # Test the newly trained model
# env_test = create_env_with_pattern('SINE_CURVE')
# test_metrics = run_dqn_agent(model, env_test, num_steps=200)
# env_test.close()

# plot_dqn_performance(test_metrics, "Newly Trained DQN Agent")
```

## Cell 7: Analyze DQN Decision Making

```python
# Analyze how DQN makes decisions
print("Analyzing DQN decision-making patterns...")

if dqn_model is not None:
    env_analysis = create_env_with_pattern('SINE_CURVE')
    obs = env_analysis.reset()
    
    decision_data = []
    
    for step in range(100):
        # Get Q-values for current state
        action, _ = dqn_model.predict(obs, deterministic=True)
        
        # Record state and action
        decision_data.append({
            'step': step,
            'instances': len(env_analysis.instances),
            'load': env_analysis.load,
            'queue': env_analysis.queue_size,
            'influx': env_analysis.influx,
            'action': env_analysis.actions[action]
        })
        
        obs, reward, done, info = env_analysis.step(action)
        if done:
            break
    
    env_analysis.close()
    
    # Analyze patterns
    import pandas as pd
    df = pd.DataFrame(decision_data)
    
    print("\nDQN Action Distribution:")
    print(df['action'].value_counts())
    
    print("\nAverage Load by Action:")
    print(df.groupby('action')['load'].mean())
    
    print("\nAverage Queue by Action:")
    print(df.groupby('action')['queue'].mean())
    
    # Plot decision patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Load vs Action
    for action in [-1, 0, 1]:
        action_data = df[df['action'] == action]
        axes[0, 0].scatter(action_data['load'], action_data['instances'], 
                          label=f'Action {action}', alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Load (%)')
    axes[0, 0].set_ylabel('Instances')
    axes[0, 0].set_title('DQN Actions: Load vs Instances')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Queue vs Action
    for action in [-1, 0, 1]:
        action_data = df[df['action'] == action]
        axes[0, 1].scatter(action_data['queue'], action_data['instances'],
                          label=f'Action {action}', alpha=0.6, s=50)
    axes[0, 1].set_xlabel('Queue Size')
    axes[0, 1].set_ylabel('Instances')
    axes[0, 1].set_title('DQN Actions: Queue vs Instances')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Action timeline
    axes[1, 0].plot(df['step'], df['action'], marker='o', markersize=4)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Action')
    axes[1, 0].set_title('DQN Actions Over Time')
    axes[1, 0].set_yticks([-1, 0, 1])
    axes[1, 0].set_yticklabels(['Remove', 'Nothing', 'Add'])
    axes[1, 0].grid(True)
    
    # Load and instances together
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    ax1.plot(df['step'], df['load'], 'b-', label='Load', alpha=0.7)
    ax2.plot(df['step'], df['instances'], 'g-', label='Instances', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Load (%)', color='b')
    ax2.set_ylabel('Instances', color='g')
    ax1.set_title('Load and Instances Over Time')
    ax1.grid(True)
    
    plt.tight_layout()
    plt.show()
```

**What this does:**
- Analyzes DQN's decision-making patterns
- Shows when it scales up/down/nothing
- Visualizes the relationship between state and actions

## Summary

After adding these cells, your notebook will have:

1. ✅ **Quick DQN test** - Load and test pre-trained model
2. ✅ **Multi-pattern testing** - Test DQN on different workloads
3. ✅ **Policy comparison** - Compare DQN vs Threshold vs Q-Learning
4. ✅ **Training capability** - Train your own DQN model
5. ✅ **Decision analysis** - Understand how DQN makes decisions

## Files You Need

Make sure these files exist:
- ✅ `models/dqn_test.zip` - Pre-trained DQN model (copied from old:imported-code)
- ✅ `dqn_demo.py` - DQN demo functions (created)
- ✅ `train_dqn_sb3.py` - DQN training script (created)
- ✅ `gym_scaling/load_generators.py` - Load pattern generators (copied)
- ✅ `models/qlearning_model.pkl` - Q-learning model (from earlier training)

## Installation

If you don't have stable-baselines3 installed:

```bash
pip install stable-baselines3
```

That's it! You're ready to explore DQN for cloud autoscaling! 🚀
