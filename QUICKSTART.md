# Quick Start Guide - RL Cloud Autoscaling

## 🎯 Goal
Train reinforcement learning agents (Q-learning, DQN, PPO) to perform cost-optimized cloud autoscaling using the Adobe Gym-Scaling environment.

## ⚡ Quick Start (5 minutes)

### 1. Test the Environment
```bash
python direct_test.py
```
**Expected output**: Environment tests pass, showing random and threshold policies working.

### 2. Train Q-Learning Agent
```bash
python train_qlearning.py
```
**Expected output**: Agent trains for 1000 episodes, saves model to `models/qlearning_model.pkl`

### 3. Explore Interactively
```bash
jupyter notebook autoscaling_demo.ipynb
```
**What you'll see**: Interactive examples, visualizations, and policy comparisons.

## 📁 Key Files

| File | Purpose |
|------|---------|
| `direct_test.py` | Quick environment verification |
| `train_qlearning.py` | Q-learning implementation (WORKING ✓) |
| `autoscaling_demo.ipynb` | Interactive Jupyter notebook |
| `AUTOSCALING_PROJECT_GUIDE.md` | Comprehensive project guide |
| `DEMO_SUMMARY.md` | Detailed summary of what's set up |
| `gym_scaling/envs/scaling_env.py` | Environment code (modify reward here) |

## 🎓 What You Have

### Working Environment ✓
- **State**: [instances, load, capacity, influx, queue]
- **Actions**: Remove (-1), Nothing (0), Add (+1) instance
- **Reward**: Balances load utilization, cost, and queue size
- **Workloads**: RANDOM, SINE_CURVE, PRODUCTION_DATA

### Trained Q-Learning Agent ✓
- Trained for 1000 episodes
- Achieves 88% average load with zero queue
- Model saved and ready to use
- Training visualization saved to `qlearning_training.png`

### Documentation ✓
- Complete project guide with implementation roadmap
- Jupyter notebook with interactive examples
- Customization instructions for reward function
- Evaluation metrics and experiment design

## 🚀 Next Steps

### For Your Project

1. **Compare Baselines** (1-2 hours)
   - Run threshold policies with different thresholds
   - Implement static policy
   - Compare metrics: reward, cost, queue, load

2. **Optimize Q-Learning** (2-3 hours)
   - Tune hyperparameters (learning rate, epsilon decay)
   - Experiment with state discretization
   - Test on different workload patterns

3. **Implement DQN** (3-4 hours)
   ```bash
   pip install stable-baselines3
   ```
   - Use neural network for Q-value approximation
   - Compare with tabular Q-learning
   - Evaluate sample efficiency

4. **Add PPO** (2-3 hours)
   - Implement policy-gradient method
   - Compare value-based (DQN) vs policy-gradient (PPO)
   - Analyze stability and convergence

5. **Custom Reward** (1-2 hours)
   - Modify `gym_scaling/envs/scaling_env.py`
   - Emphasize cost vs latency tradeoffs
   - Test different reward formulations

6. **Production Evaluation** (2-3 hours)
   - Load real workload traces
   - Measure cost savings vs baselines
   - Validate SLA compliance

## 📊 Current Results

### Q-Learning Performance
```
Training: 1000 episodes
Final avg reward: -9.025 (improved from -61.241)
Evaluation: 88.4% avg load, 0 queue, $126,407 cost
Consistency: Perfect (0.0 std across 10 episodes)
```

### Threshold Policy (80/40) Performance
```
Average reward: -0.1398
Average load: 75.06%
Average queue: 0.00
Total cost: $579,890
```

**Observation**: Q-learning achieves higher load utilization (88% vs 75%) with lower cost per episode.

## 🔧 Common Tasks

### Modify Reward Function
Edit `gym_scaling/envs/scaling_env.py`, method `__get_reward()`:
```python
def __get_reward(self):
    # Add your custom reward components here
    cost_penalty = -0.01 * len(self.instances)
    sla_penalty = -10 if self.queue_size > 500 else 0
    # ...
```

### Change Workload Pattern
```python
from gym_scaling.envs.scaling_env import INPUTS

env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
env.change_rate = 1  # Change influx every step
```

### Adjust Environment Parameters
```python
custom_options = {
    'max_instances': 100.0,
    'min_instances': 2.0,
    'capacity_per_instance': 87,
    'discrete_actions': (-2, -1, 0, 1, 2),  # More aggressive
}
env = ScalingEnv(scaling_env_options=custom_options)
```

### Train Longer
```python
agent, metrics = train_qlearning(
    n_episodes=5000,  # More episodes
    max_steps=500,    # Longer episodes
    verbose=True
)
```

## 📈 Evaluation Metrics

Track these for each policy:

**Performance**
- Cumulative reward
- Average reward per step
- Episode length

**Cost Efficiency**
- Total cost ($)
- Cost per request
- Instance utilization (%)

**Queue Stability**
- Average queue size
- Max queue size
- Overflow rate (%)

**SLA Compliance**
- SLA violation rate
- Response time (queue as proxy)
- Steps with queue > threshold

## 💡 Tips

1. **Start Simple**: Test with threshold baselines before RL
2. **Visualize**: Use the notebook to understand environment behavior
3. **Tune Carefully**: Small changes in hyperparameters matter
4. **Compare Fairly**: Use same workload and episode length
5. **Track Everything**: Log all metrics for analysis
6. **Iterate**: Start with Q-learning, then move to DQN/PPO

## 🐛 Troubleshooting

**NumPy errors**: Use `ScalingEnv()` directly, not `gym.make()` (already fixed in all scripts)
**Gym warnings**: Ignore them, environment works fine
**Training unstable**: Reduce learning rate, increase exploration
**Poor performance**: Train longer, tune hyperparameters
**Agent always scales up**: Adjust cost penalties in reward

See `TROUBLESHOOTING.md` for detailed solutions.

## 📚 Documentation

- **Quick Start**: This file
- **Project Guide**: `AUTOSCALING_PROJECT_GUIDE.md` (comprehensive)
- **Demo Summary**: `DEMO_SUMMARY.md` (what's been set up)
- **Jupyter Notebook**: `autoscaling_demo.ipynb` (interactive)

## ✅ Checklist

- [x] Environment installed and tested
- [x] Q-learning agent implemented
- [x] Model trained and saved
- [x] Documentation complete
- [ ] Baseline comparison
- [ ] Hyperparameter tuning
- [ ] DQN implementation
- [ ] PPO implementation
- [ ] Custom reward function
- [ ] Production evaluation

## 🎉 You're Ready!

Everything is set up and working. The Q-learning agent successfully learned to autoscale with 88% load and zero queue. Now you can:

1. Compare against baselines
2. Implement DQN and PPO
3. Customize for your objectives
4. Evaluate on production workloads

**Start with**: `python direct_test.py` to verify, then open `autoscaling_demo.ipynb` to explore!

Good luck with your RL autoscaling project! 🚀
