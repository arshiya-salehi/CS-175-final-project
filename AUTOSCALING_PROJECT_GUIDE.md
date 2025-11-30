# RL-Based Cloud Autoscaling Project Guide

## Overview

This project implements reinforcement learning agents (Q-learning and DQN) for cost-optimized cloud autoscaling using the Adobe Gym-Scaling environment. The goal is to learn policies that dynamically adjust compute instances in response to varying workloads while minimizing cost and preventing SLA violations.

## Environment Details

### State Space (Observation)
The environment provides a 5-dimensional observation vector:
- `[0]` **Normalized instances**: Current instances / max_instances (0.0 - 1.0)
- `[1]` **Normalized load**: CPU load / 100 (0.0 - 1.0)
- `[2]` **Total capacity**: Total processing capacity (instances × capacity_per_instance)
- `[3]` **Influx**: Current request influx rate
- `[4]` **Queue size**: Number of requests waiting in queue

### Action Space
Discrete actions for scaling:
- **0**: Remove one instance (-1)
- **1**: Do nothing (0)
- **2**: Add one instance (+1)

Can be customized to more aggressive scaling: `(-2, -1, 0, 1, 2)`

### Reward Function
The reward balances three objectives:

```python
reward = (-1 * (1 - normalized_load)) * num_instances_normalized  # Load utilization
       + boundary_penalty                                          # -0.1 if violating min/max
       - inverse_odds(queue_size)                                  # Queue penalty
```

**Key insights:**
- Penalizes underutilization (low load with many instances = high cost)
- Heavily penalizes queue buildup (SLA violations)
- Encourages maintaining ~80-90% load with minimal queue

### Workload Patterns
1. **RANDOM**: Random influx between offset and max_influx
2. **SINE_CURVE**: Sinusoidal pattern simulating daily traffic cycles
3. **PRODUCTION_DATA**: Load from real production traces (requires data file)

## Quick Start

### 1. Test the Environment

Run the direct test to verify everything works:

```bash
python direct_test.py
```

This will:
- Create and test the environment
- Run random and threshold-based policies
- Display performance metrics
- Test different workload patterns

### 2. Interactive Exploration

Open the Jupyter notebook for detailed examples:

```bash
jupyter notebook autoscaling_demo.ipynb
```

The notebook includes:
- Environment setup and configuration
- Random agent baseline
- Threshold-based policy implementation
- Visualization of metrics
- Policy comparison framework
- DQN training examples (optional)

### 3. Run Existing Scripts

Test with pre-trained model (if available):
```bash
python enjoy_deepq.py
```

Train a new DQN model:
```bash
python train_deepq.py
```

## Implementation Roadmap

### Phase 1: Baseline Policies
1. **Random Policy** ✓ (implemented in notebook)
   - Random action selection
   - Establishes lower bound performance

2. **Threshold-Based Policy** ✓ (implemented in notebook)
   - Scale up if load > 80% or queue > threshold
   - Scale down if load < 40% and queue == 0
   - Similar to Kubernetes HPA

3. **Static Policy**
   - Fixed number of instances
   - Useful for cost comparison

### Phase 2: Q-Learning Agent
Implement tabular Q-learning with discretized state space:

```python
# Discretize continuous state
def discretize_state(obs, bins):
    instances_bin = int(obs[0] * bins)
    load_bin = int(obs[1] * bins)
    queue_bin = min(int(obs[4] / 100), bins-1)
    return (instances_bin, load_bin, queue_bin)

# Q-learning update
Q[state, action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state, action])
```

**Hyperparameters to tune:**
- Learning rate (α): 0.1 - 0.5
- Discount factor (γ): 0.95 - 0.99
- Exploration (ε): Start 1.0, decay to 0.01
- State discretization bins: 10-20 per dimension

### Phase 3: Deep Q-Network (DQN)
Use neural network for Q-value approximation:

**Option A: Stable-Baselines3 (Recommended)**
```python
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment
env = make_vec_env(lambda: ScalingEnv(), n_envs=1)

# Train DQN
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.02,
    verbose=1
)

model.learn(total_timesteps=100000)
```

**Option B: Custom DQN Implementation**
- Network: 2-3 hidden layers, 64-128 units each
- Experience replay buffer: 10k-50k transitions
- Target network update frequency: 500-1000 steps
- Batch size: 32-64

### Phase 4: Policy Gradient (PPO)
Compare value-based (DQN) vs policy-gradient methods:

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1
)

model.learn(total_timesteps=100000)
```

## Evaluation Metrics

Track these metrics for each policy:

### Performance Metrics
- **Cumulative Reward**: Sum of rewards over episode
- **Average Reward**: Mean reward per step
- **Episode Length**: Steps before termination

### Cost Efficiency
- **Total Cost**: Dollar cost of instances used
- **Cost per Request**: Total cost / total requests processed
- **Instance Utilization**: Average load percentage

### Queue Stability
- **Average Queue Size**: Mean queue length
- **Max Queue Size**: Peak queue length
- **Queue Overflow Rate**: Episodes ending due to overflow

### SLA Compliance
- **Response Time**: Queue size as proxy for latency
- **SLA Violations**: Steps with queue > threshold
- **Violation Rate**: Percentage of steps violating SLA

## Customization Guide

### Modify Reward Function

Edit `gym_scaling/envs/scaling_env.py`, method `__get_reward()`:

```python
def __get_reward(self):
    # Example: Emphasize cost reduction
    cost_penalty = -0.01 * len(self.instances)
    
    # Example: Strict SLA enforcement
    sla_penalty = -10.0 if self.queue_size > 500 else 0.0
    
    # Example: Encourage high utilization
    utilization_reward = 0.5 if 0.7 <= self.load/100 <= 0.9 else 0.0
    
    total_reward = cost_penalty + sla_penalty + utilization_reward
    self.collected_rewards.append(total_reward)
    return total_reward
```

### Add Custom Workload Pattern

Edit `gym_scaling/envs/scaling_env.py`, add to `INPUTS` dict:

```python
'CUSTOM_PATTERN': {
    'function': lambda step, max_influx, offset: your_function(step),
    'options': {},
}
```

### Adjust Environment Parameters

```python
custom_options = {
    'max_instances': 100.0,          # Maximum instances allowed
    'min_instances': 2.0,            # Minimum instances required
    'capacity_per_instance': 87,     # Requests per instance per step
    'cost_per_instance_per_hour': 0.192,  # AWS c3.large pricing
    'discrete_actions': (-1, 0, 1),  # Scaling actions
    'change_rate': 10000,            # Steps between influx changes
    'offset': 500,                   # Minimum influx
}

env = ScalingEnv(scaling_env_options=custom_options)
```

## Experiment Design

### Experiment 1: Baseline Comparison
Compare random, static, and threshold policies:
- Run each for 1000 steps
- Use SINE_CURVE workload
- Record all metrics
- Establish performance bounds

### Experiment 2: Q-Learning Evaluation
- Train for 50k-100k steps
- Test on RANDOM and SINE_CURVE
- Compare against baselines
- Analyze learned policy behavior

### Experiment 3: DQN vs Q-Learning
- Train both for equal timesteps
- Evaluate on unseen workload patterns
- Compare sample efficiency
- Analyze convergence speed

### Experiment 4: DQN vs PPO
- Compare value-based vs policy-gradient
- Evaluate stability and variance
- Test generalization to new patterns
- Analyze computational efficiency

### Experiment 5: Production Workload
- Load real production traces
- Test all trained policies
- Measure cost savings vs baselines
- Validate SLA compliance

## Visualization

Create plots for analysis:

```python
import matplotlib.pyplot as plt

def plot_training_progress(rewards, window=100):
    """Plot learning curve with moving average."""
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(moving_avg)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Progress')
    plt.show()

def plot_policy_comparison(metrics_dict):
    """Compare multiple policies side-by-side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for name, metrics in metrics_dict.items():
        axes[0, 0].plot(np.cumsum(metrics['rewards']), label=name)
        # Add more plots...
    
    plt.legend()
    plt.show()
```

## Troubleshooting

### Issue: Environment crashes with numpy errors
**Solution**: The environment uses older gym/numpy versions. Use `direct_test.py` which bypasses gym wrappers.

### Issue: Training is unstable
**Solution**: 
- Reduce learning rate
- Increase exploration period
- Use larger replay buffer
- Normalize observations

### Issue: Agent always scales up/down
**Solution**:
- Check reward function balance
- Adjust cost penalties
- Increase exploration
- Verify state normalization

### Issue: Poor generalization
**Solution**:
- Train on multiple workload patterns
- Add noise to training data
- Use larger network capacity
- Implement domain randomization

## File Structure

```
.
├── gym_scaling/                    # Environment implementation
│   ├── __init__.py
│   └── envs/
│       ├── scaling_env.py         # Main environment (MODIFY REWARD HERE)
│       ├── helpers.py
│       └── rendering.py
├── autoscaling_demo.ipynb         # Interactive Jupyter notebook
├── direct_test.py                 # Quick environment test
├── train_deepq.py                 # DQN training script
├── enjoy_deepq.py                 # Test pre-trained model
├── models/                        # Saved models
│   └── scaling_model.pkl
└── AUTOSCALING_PROJECT_GUIDE.md   # This file
```

## References

- Adobe Gym-Scaling: https://github.com/adobe/gym-scaling
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- OpenAI Gym: https://www.gymlibrary.dev/
- RL Autoscaling Survey: Garí et al., 2020
- DQN Paper: Mnih et al., 2015

## Next Steps

1. ✅ Environment is working and tested
2. ⬜ Implement Q-learning agent
3. ⬜ Train and evaluate DQN
4. ⬜ Implement PPO for comparison
5. ⬜ Customize reward function for your objectives
6. ⬜ Run experiments with different workloads
7. ⬜ Collect and analyze results
8. ⬜ Write up findings and comparisons

Good luck with your RL autoscaling project! 🚀
