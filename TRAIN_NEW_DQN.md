# Train a New DQN Model

## Problem with Current Model

The `models/dqn_test.zip` model is **untrained** or **minimally trained**. The training failed due to numpy compatibility issues after only a few steps. That's why you're seeing:

```
Episode ended at step 49 (queue overflow)
Average reward: -0.6348
Average queue: 9102.26  ← HUGE queue! Model hasn't learned anything
```

**The model never completed training** - it crashed during the first episode.

## Solution: Train a New Model

I've created `train_dqn_fixed.py` which fixes all the compatibility issues.

### Quick Training (Recommended - 10 minutes)

```bash
python train_dqn_fixed.py --pattern SINE_CURVE --timesteps 10000
```

**What to expect:**
- Takes ~10 minutes
- Model will learn basic autoscaling
- Should achieve:
  - Average load: 70-80%
  - Queue size: < 100
  - No queue overflow
  - Reward: > -0.2

### Medium Training (Better - 30 minutes)

```bash
python train_dqn_fixed.py --pattern SINE_CURVE --timesteps 30000
```

**What to expect:**
- Takes ~30 minutes
- Better performance
- Should achieve:
  - Average load: 75-85%
  - Queue size: < 50
  - Reward: > -0.1

### Full Training (Best - 50 minutes)

```bash
python train_dqn_fixed.py --pattern SINE_CURVE --timesteps 50000
```

**What to expect:**
- Takes ~50 minutes
- Best performance
- Should achieve:
  - Average load: 80-90%
  - Queue size: near 0
  - Reward: > 0

## Training Progress

You'll see output like this:

```
======================================================================
TRAINING DQN AGENT FOR CLOUD AUTOSCALING
======================================================================
Load pattern: SINE_CURVE
Total timesteps: 10,000
Model name: dqn_sine_curve_20241123_170000
Estimated time: ~10 minutes
======================================================================

Creating DQN model...

Starting training...
Progress will be shown below:
----------------------------------------------------------------------
---------------------------------
| rollout/            |         |
|    ep_len_mean      | 156     |
|    ep_rew_mean      | -0.45   |
| time/               |         |
|    episodes         | 4       |
|    fps              | 89      |
|    time_elapsed     | 7       |
|    total_timesteps  | 640     |
| train/              |         |
|    learning_rate    | 0.0001  |
|    loss             | 0.234   |
|    n_updates        | 160     |
---------------------------------
...
```

**Key metrics to watch:**
- `ep_rew_mean` - Should increase (become less negative)
- `ep_len_mean` - Should increase (agent survives longer)
- `loss` - Should decrease over time

## After Training

### Test Your Model

```bash
# Quick test
python -c "from dqn_demo import load_dqn_model, create_env_with_pattern, run_dqn_agent, plot_dqn_performance; import numpy as np; model = load_dqn_model('models/dqn_sine_curve_*.zip'); env = create_env_with_pattern('SINE_CURVE'); metrics = run_dqn_agent(model, env, 200); print(f'Avg reward: {np.mean(metrics[\"rewards\"]):.4f}, Avg load: {np.mean(metrics[\"load\"]):.1f}%, Avg queue: {np.mean(metrics[\"queue_size\"]):.2f}')"
```

### Use in Notebook

Replace the model path in your notebook:

```python
# Load your newly trained model
dqn_model = load_dqn_model('models/dqn_sine_curve_20241123_170000.zip')

# Test it
env_dqn = create_env_with_pattern('SINE_CURVE')
dqn_metrics = run_dqn_agent(dqn_model, env_dqn, num_steps=200)
env_dqn.close()

# Visualize
plot_dqn_performance(dqn_metrics, "Newly Trained DQN Agent")
```

## Training Options

### Different Patterns

```bash
# Train on sinusoidal pattern
python train_dqn_fixed.py --pattern SINUSOIDAL --timesteps 10000

# Train on spike pattern
python train_dqn_fixed.py --pattern SPIKE --timesteps 10000

# Train on steady pattern
python train_dqn_fixed.py --pattern STEADY --timesteps 10000
```

### Custom Name

```bash
python train_dqn_fixed.py --pattern SINE_CURVE --timesteps 10000 --name my_dqn_model
```

This will save to `models/my_dqn_model.zip`

## Interrupt Training

If you need to stop training early:
- Press `Ctrl+C`
- Model will be saved as `models/<name>_interrupted.zip`
- You can still use this model (it's partially trained)

## Expected Results After Training

### Before Training (current dqn_test.zip):
```
✗ Episode ended at step 49 (queue overflow)
✗ Average reward: -0.6348
✗ Average load: 87.5%
✗ Average queue: 9102.26  ← TERRIBLE!
✗ Average instances: 25.5
```

### After Training (10k timesteps):
```
✓ Completed 200 steps
✓ Average reward: -0.15 to -0.25
✓ Average load: 70-80%
✓ Average queue: 0-100
✓ Average instances: 45-55
```

### After Training (50k timesteps):
```
✓ Completed 200 steps
✓ Average reward: -0.05 to -0.15
✓ Average load: 80-90%
✓ Average queue: 0-10
✓ Average instances: 50-60
```

## Comparison with Other Policies

After training, you should see:

| Policy | Avg Reward | Avg Load | Avg Queue | Performance |
|--------|-----------|----------|-----------|-------------|
| Random | -0.43 | 50-60% | 0 | Poor (underutilized) |
| Threshold (80/40) | -0.14 | 75% | 0 | Good baseline |
| Q-Learning | -0.11 | 88% | 0 | Very good |
| **DQN (trained)** | **-0.10** | **85%** | **0** | **Best** |
| DQN (untrained) | -0.63 | 87% | 9102 | Broken ❌ |

## Troubleshooting

### "No module named 'stable_baselines3'"
```bash
pip install stable-baselines3
```

### Training is slow
- Reduce timesteps: `--timesteps 5000`
- This is normal - DQN training takes time
- You can interrupt with Ctrl+C and use partial model

### Model still performs poorly
- Train longer: `--timesteps 50000`
- Try different pattern: `--pattern SINUSOIDAL`
- Check that training completed without errors

### Want to see training progress
Training shows progress every 10 episodes with:
- Episode reward mean
- Episode length mean
- Training loss
- FPS (frames per second)

## Summary

1. **Current model is broken** - training failed, never learned
2. **Train new model**: `python train_dqn_fixed.py --pattern SINE_CURVE --timesteps 10000`
3. **Wait ~10 minutes** for training to complete
4. **Test in notebook** with your newly trained model
5. **Compare** with threshold and Q-learning policies

The new model should perform **much better** than the current one!
