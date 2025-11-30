# Troubleshooting Guide

## Common Issues and Solutions

### 1. NumPy AttributeError: 'numpy' has no attribute 'bool8'

**Error:**
```
AttributeError: module 'numpy' has no attribute 'bool8'
```

**Cause:** The old `gym` library (0.14.0) has compatibility issues with newer NumPy versions (2.0+).

**Solution:** Use the environment directly instead of `gym.make()`:

❌ **Don't do this:**
```python
import gym
import gym_scaling
env = gym.make('Scaling-v0')  # This causes the error
```

✅ **Do this instead:**
```python
from gym_scaling.envs.scaling_env import ScalingEnv
env = ScalingEnv()  # This works!
```

**Files already fixed:**
- ✅ `direct_test.py` - Uses `ScalingEnv()` directly
- ✅ `train_qlearning.py` - Uses `ScalingEnv()` directly
- ✅ `autoscaling_demo.ipynb` - Updated to use `ScalingEnv()` directly

### 2. TypeError: 'float' object cannot be interpreted as an integer

**Error:**
```
TypeError: 'float' object cannot be interpreted as an integer
```

**Cause:** `random.randint()` requires integer arguments but receives floats.

**Solution:** Already fixed in `gym_scaling/envs/scaling_env.py`:
```python
'RANDOM': {
    'function': lambda step, max_influx, offset: random.randint(int(offset), int(max_influx)),
    ...
}
```

If you see this error, the file was reverted. Re-apply the fix above.

### 3. Gym Deprecation Warnings

**Warning:**
```
Gym has been unmaintained since 2022 and does not support NumPy 2.0...
```

**Cause:** The environment uses old gym (0.14.0).

**Solution:** These are just warnings and can be safely ignored. To suppress them:

```python
import warnings
warnings.filterwarnings('ignore')
```

This is already included in all test scripts.

### 4. Environment Reset Signature Error

**Error:**
```
TypeError: ScalingEnv.reset: `seed` is not present.
```

**Cause:** Old gym environments didn't have `seed` parameter in `reset()`.

**Solution:** Already fixed in `gym_scaling/envs/scaling_env.py`:
```python
def reset(self, seed=None, options=None):
    if seed is not None:
        numpy.random.seed(seed)
        random.seed(seed)
    # ... rest of reset code
```

### 5. Training is Unstable / Rewards Fluctuate

**Symptoms:**
- Rewards jump around wildly
- Agent doesn't seem to learn
- Performance degrades over time

**Solutions:**

1. **Reduce learning rate:**
   ```python
   agent = QLearningAgent(learning_rate=0.05)  # Try 0.05 instead of 0.1
   ```

2. **Slower epsilon decay:**
   ```python
   agent = QLearningAgent(epsilon_decay=0.999)  # Explore longer
   ```

3. **Train longer:**
   ```python
   train_qlearning(n_episodes=5000)  # More episodes
   ```

4. **Check state discretization:**
   ```python
   # Try fewer bins for simpler state space
   def discretize_state(obs, bins=5):  # Instead of 10
       ...
   ```

### 6. Agent Always Scales Up or Down

**Symptoms:**
- Agent only adds instances
- Agent only removes instances
- Agent never uses "do nothing" action

**Solutions:**

1. **Check reward balance:**
   - Cost penalty might be too high/low
   - Queue penalty might be too high/low
   - Edit `gym_scaling/envs/scaling_env.py` in `__get_reward()`

2. **Increase exploration:**
   ```python
   agent = QLearningAgent(
       epsilon_start=1.0,
       epsilon_end=0.05,  # Higher minimum exploration
       epsilon_decay=0.999  # Slower decay
   )
   ```

3. **Verify action boundaries:**
   ```python
   # Check that min/max instances are enforced
   print(f"Min: {env.min_instances}, Max: {env.max_instances}")
   ```

### 7. Poor Generalization to New Workloads

**Symptoms:**
- Agent works on SINE_CURVE but fails on RANDOM
- Performance drops on unseen patterns

**Solutions:**

1. **Train on multiple patterns:**
   ```python
   # Alternate between patterns during training
   if episode % 2 == 0:
       env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
   else:
       env.scaling_env_options['input'] = INPUTS['RANDOM']
   ```

2. **Add noise to training:**
   ```python
   # Randomize change_rate
   env.change_rate = random.randint(1, 10)
   ```

3. **Use DQN instead of Q-learning:**
   - Neural networks generalize better than tables
   - See `AUTOSCALING_PROJECT_GUIDE.md` for DQN implementation

### 8. Jupyter Notebook Kernel Crashes

**Symptoms:**
- Kernel dies during training
- "Kernel Restarting" message

**Solutions:**

1. **Reduce episode length:**
   ```python
   run_random_agent(env, num_steps=100)  # Instead of 1000
   ```

2. **Train in script instead:**
   ```bash
   python train_qlearning.py  # More stable than notebook
   ```

3. **Clear output regularly:**
   - In Jupyter: Cell → All Output → Clear

### 9. Model File Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/qlearning_model.pkl'
```

**Solution:**

1. **Train the model first:**
   ```bash
   python train_qlearning.py
   ```

2. **Check models directory exists:**
   ```bash
   mkdir -p models
   ```

3. **Verify file path:**
   ```python
   import os
   print(os.path.exists('models/qlearning_model.pkl'))
   ```

### 10. Matplotlib Not Available

**Error:**
```
ImportError: No module named 'matplotlib'
```

**Solution:**

```bash
pip install matplotlib
```

Or skip plotting:
```python
# Comment out plot_metrics() calls
# plot_metrics(metrics, "Title")
```

## Quick Fixes Summary

### If you get numpy errors:
```python
# Use this:
from gym_scaling.envs.scaling_env import ScalingEnv
env = ScalingEnv()

# Not this:
# env = gym.make('Scaling-v0')
```

### If training is unstable:
```python
agent = QLearningAgent(
    learning_rate=0.05,      # Lower
    epsilon_decay=0.999,     # Slower
    epsilon_end=0.05         # Higher minimum
)
```

### If agent doesn't learn:
```python
# Train longer
train_qlearning(n_episodes=5000, max_steps=500)

# Simplify state space
def discretize_state(obs, bins=5):  # Fewer bins
    ...
```

### If you want to skip gym entirely:
All scripts already use `ScalingEnv()` directly:
- ✅ `direct_test.py`
- ✅ `train_qlearning.py`
- ✅ `autoscaling_demo.ipynb` (updated)

## Still Having Issues?

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.7+
   ```

2. **Verify installation:**
   ```bash
   python -c "from gym_scaling.envs.scaling_env import ScalingEnv; print('OK')"
   ```

3. **Run the test script:**
   ```bash
   python direct_test.py
   ```
   If this works, your environment is fine.

4. **Check the files:**
   - `gym_scaling/envs/scaling_env.py` should have the fixes
   - Look for `def reset(self, seed=None, options=None):`
   - Look for `random.randint(int(offset), int(max_influx))`

## Getting Help

If you're still stuck:

1. Check which file is causing the issue
2. Look at the error message carefully
3. Compare with working examples in `direct_test.py`
4. Make sure you're using `ScalingEnv()` directly, not `gym.make()`

The environment is working correctly - all test scripts pass. Most issues come from trying to use `gym.make()` instead of `ScalingEnv()` directly.
