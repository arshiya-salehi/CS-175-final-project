# PPO Training Guide

## What is PPO?

**PPO (Proximal Policy Optimization)** is a policy-gradient reinforcement learning algorithm that:
- Learns a policy (probability distribution over actions) directly
- More stable than older policy gradient methods
- Often works better than DQN for continuous control
- Requires more training samples than DQN

## PPO vs DQN vs Q-Learning

| Feature | Q-Learning | DQN | PPO |
|---------|-----------|-----|-----|
| **Type** | Value-based (tabular) | Value-based (deep) | Policy-gradient |
| **Learning** | Q-table lookup | Q-value network | Policy network |
| **Training Time** | Fast (~3 min) | Medium (~1-2 hours) | Slow (~2-4 hours) |
| **Sample Efficiency** | High | Medium | Low |
| **Stability** | High | Medium | High |
| **Best For** | Discrete states | Complex states | Continuous actions |

## Training PPO

### Quick Training (20k timesteps - ~40 minutes)

```bash
python train_ppo_fixed.py --pattern SINE_CURVE --timesteps 20000
```

**What to expect:**
- Takes ~40-60 minutes
- Basic learning
- Should achieve:
  - Average load: 70-80%
  - Queue size: < 100
  - Reward: > -0.3

### Medium Training (50k timesteps - ~2 hours)

```bash
python train_ppo_fixed.py --pattern SINE_CURVE --timesteps 50000
```

**What to expect:**
- Takes ~2 hours
- Good learning
- Should achieve:
  - Average load: 80-85%
  - Queue size: < 50
  - Reward: > -0.15

### Full Training (100k timesteps - ~4 hours)

```bash
python train_ppo_fixed.py --pattern SINE_CURVE --timesteps 100000
```

**What to expect:**
- Takes ~4 hours
- Best learning
- Should achieve:
  - Average load: 85-90%
  - Queue size: near 0
  - Reward: > -0.1

## Why PPO Takes Longer

PPO needs more samples because:

1. **On-policy learning**: Can only learn from recent experiences
   - DQN: Uses replay buffer (learns from old experiences)
   - PPO: Must collect new experiences for each update

2. **Multiple epochs**: Updates policy multiple times per batch
   - Each batch of 2048 steps → 10 epochs of training
   - More computation per sample

3. **Policy optimization**: More complex than Q-value updates
   - Must balance exploration vs exploitation
   - Clips updates to prevent large policy changes

## Training Progress

You'll see output like:

```
---------------------------------
| rollout/                |     |
|    ep_len_mean          | 156 |  ← Episode length
|    ep_rew_mean          | -0.4|  ← Average reward (should improve)
| time/                   |     |
|    fps                  | 45  |  ← Frames per second
|    iterations           | 10  |  ← Training iterations
|    time_elapsed         | 456 |  ← Seconds elapsed
|    total_timesteps      | 20480| ← Total timesteps
| train/                  |     |
|    approx_kl            | 0.01|  ← KL divergence (should be small)
|    clip_fraction        | 0.1 |  ← Fraction of clipped updates
|    explained_variance   | 0.8 |  ← How well value function fits
|    learning_rate        | 3e-4|  ← Learning rate
|    loss                 | 0.2 |  ← Policy loss
|    policy_gradient_loss | -0.01| ← Policy gradient loss
|    value_loss           | 0.5 |  ← Value function loss
---------------------------------
```

**Key metrics to watch:**
- `ep_rew_mean` - Should increase (become less negative)
- `explained_variance` - Should be high (>0.7)
- `approx_kl` - Should be small (<0.05)
- `clip_fraction` - Should be moderate (0.1-0.3)

## After Training

### Test Your PPO Model

```python
# In your notebook
from ppo_demo import load_ppo_model, create_env_with_pattern, run_ppo_agent, plot_ppo_performance
import numpy as np

# Load model (use actual filename)
ppo_model = load_ppo_model('models/ppo_sine_curve_20251123_180000.zip')

# Test it
env = create_env_with_pattern('SINE_CURVE')
ppo_metrics = run_ppo_agent(ppo_model, env, 200)
env.close()

# Check results
print(f"Steps completed: {len(ppo_metrics['rewards'])}")
print(f"Average reward: {np.mean(ppo_metrics['rewards']):.4f}")
print(f"Average load: {np.mean(ppo_metrics['load']):.1f}%")
print(f"Average queue: {np.mean(ppo_metrics['queue_size']):.2f}")

# Visualize
plot_ppo_performance(ppo_metrics, "PPO Agent (20k timesteps)")
```

### Compare All Four Policies

```python
from ppo_demo import compare_all_policies

# Compare all policies
comparison = {
    'Threshold': threshold_metrics,
    'Q-Learning': qlearning_metrics,
    'DQN': dqn_metrics,
    'PPO': ppo_metrics
}

compare_all_policies(comparison)
```

## Training Tips

### 1. Start Small
```bash
# Test with 5k timesteps first (~10 min)
python train_ppo_fixed.py --pattern SINE_CURVE --timesteps 5000
```

This helps verify:
- Training works without errors
- Environment is compatible
- Approximate training time

### 2. Monitor Progress
Watch the terminal output:
- `ep_rew_mean` should gradually increase
- `fps` tells you training speed
- `time_elapsed` shows how long it's been running

### 3. Use Checkpoints
Training saves checkpoints every 10,000 steps:
- Location: `models/checkpoints/`
- Can resume from checkpoint if interrupted
- Best model saved to `models/best/`

### 4. Interrupt if Needed
Press `Ctrl+C` to stop training:
- Model will be saved as `*_interrupted.zip`
- Can still use partially trained model
- Better than nothing!

## Expected Performance

### After 20k timesteps (~40 min):
```
✓ Completed 200 steps
✓ Average reward: -0.20 to -0.30
✓ Average load: 70-80%
✓ Average queue: 0-100
✓ Performance: Decent, better than random
```

### After 50k timesteps (~2 hours):
```
✓ Completed 200 steps
✓ Average reward: -0.10 to -0.20
✓ Average load: 80-85%
✓ Average queue: 0-50
✓ Performance: Good, competitive with DQN
```

### After 100k timesteps (~4 hours):
```
✓ Completed 200 steps
✓ Average reward: -0.05 to -0.15
✓ Average load: 85-90%
✓ Average queue: 0-10
✓ Performance: Excellent, may beat DQN
```

## Comparison with Other Methods

| Method | Training Time | Performance | Best Use Case |
|--------|--------------|-------------|---------------|
| **Threshold** | None | Good baseline | Simple, predictable |
| **Q-Learning** | ~3 min | Excellent | Fast training, discrete states |
| **DQN** | ~1-2 hours | Excellent | Complex states, value-based |
| **PPO** | ~2-4 hours | Excellent | Stable learning, policy-based |

## When to Use PPO

**Use PPO when:**
- ✅ You have time for longer training
- ✅ You want stable, reliable learning
- ✅ You want to compare policy-gradient vs value-based
- ✅ You need smooth, continuous policies

**Skip PPO if:**
- ❌ You're short on time (use Q-Learning instead)
- ❌ You only need one RL method (Q-Learning is faster)
- ❌ Your computer is slow (training will take very long)

## Troubleshooting

### Training is too slow
- Reduce timesteps: `--timesteps 10000`
- This is normal for PPO
- Consider using Q-Learning or DQN instead

### Model performs poorly
- Train longer: `--timesteps 50000`
- Check that training completed without errors
- Verify `ep_rew_mean` improved during training

### Out of memory
- PPO uses more memory than DQN
- Close other applications
- Reduce batch size (edit train_ppo_fixed.py)

### Want faster results
- Use your Q-Learning model (already excellent!)
- Q-Learning: 88% load, 0 queue, trained in 3 min
- PPO is just an additional comparison

## Summary

**PPO Training:**
1. Takes longer than DQN (2-4 hours for good results)
2. More stable and reliable learning
3. Policy-gradient approach (different from Q-Learning/DQN)
4. Good for comparing different RL paradigms

**Recommendation:**
- **If you have time**: Train PPO with 20k-50k timesteps
- **If you're short on time**: Focus on Q-Learning (already excellent)
- **For comparison**: Train both DQN and PPO to show different approaches

Your Q-Learning results are already strong enough to demonstrate RL success! PPO adds another comparison point but isn't essential.
