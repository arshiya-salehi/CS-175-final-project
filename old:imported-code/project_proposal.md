Page 1 of 2

# Project Proposal for CS 175

**Project Title:** Reinforcement Learning for Cloud Autoscaling (Gym-Scaling)

**List of Team Members:**
- Name1, StudentID1, uci_email_address
- Name2, StudentID2, uci_email_address
- [Name3, StudentID3, uci_email_address]

---

1. Project Summary

This project builds and evaluates reinforcement learning (RL) methods to perform cloud autoscaling. We adapt and extend the Gym-Scaling environment to train DQN (and optionally PPO) agents that decide when to add/remove instances in response to incoming load; evaluation is performed by measuring average reward, cost, queue size, and scaling events across synthetic and production-derived load traces.

2. Problem Definition

Cloud autoscaling is the problem of dynamically adjusting the number of running instances to match incoming workload while minimizing cost and ensuring service quality. Inputs to our system are time series of request influx (various load patterns: RANDOM, SINE_CURVE, production traces) and environment state (instances, capacity, queue size, load). The output is a sequence of scaling actions (add, remove, or no-op) chosen at fixed control intervals. The goal is to learn policies that maximize cumulative reward (trade-off of utilization and cost) and minimize queue/backlog and SLA violations.

Prior work uses heuristics and control-theoretic methods (threshold-based autoscalers) and RL methods (Q-learning, policy gradients) for similar problems; we build on those by using modern deep RL (DQN / SB3) in a simulated, reproducible environment and compare against static and rule‑based baselines. Representative references: (1) Auto-scaling with RL literature; (2) Stable Baselines3 and Gymnasium migration guides (we use Gymnasium compatibility).

3. Proposed Technical Approach

We will treat autoscaling as a discrete-action MDP and train a DQN agent (stable-baselines3) to choose actions from the discrete set (-1, 0, +1). The pipeline:

- Environment: `gym_scaling.envs.ScalingEnv` (simulates influx, queue, instances, cost). We will use `src/env_wrapper.py` to ensure observations match SB3 expectations and set different `scaling_env_options` (input functions and parameters).
- Policy Learning: Train DQN using `src/train_dqn_sb3.py` with varied load patterns. Optionally train PPO (`train_ppo_sb3.py`) for comparison.
- Baselines: Implement a static autoscaler (`src/static_autoscaler.py`) and simple heuristic policies (threshold-based) included in the repo for direct comparison.
- Evaluation: Use `src/evaluate.py` and `src/visualize.py` to compute metrics and plot results; repeat runs with different random seeds and load patterns.

Each component takes inputs and produces outputs as follows: the load generator produces time-series influx; the environment consumes influx and current instances to produce observations and rewards; the RL agent consumes observations and outputs actions; the evaluation scripts consume saved policies and environment traces to compute metrics and figures.

4. Data Sets

We use simulated load patterns included in the repo and a small production-like trace:

- Synthetic patterns (in `src/load_generators.py` and `gym_scaling/envs/scaling_env.py`): `RANDOM`, `SINE_CURVE`, and other patterns described in `LOAD_PATTERNS`.
- Production-like trace: `data/worker_one.xlsx` (referenced by `INPUTS['PRODUCTION_DATA']` in `scaling_env.py`). This file contains historical influx values used to replay realistic workloads.

No external large dataset is required; we will rely on these synthetic and small production traces. The `models/` directory contains a saved smoke-test model (`dqn_test.zip`) which can be used to validate evaluation and visualization code.

5. Experiments and Evaluation

Planned experiments:
- Train DQN across multiple load patterns (RANDOM, SINE_CURVE, production trace) and random seeds.
- Baseline comparisons: static autoscaler (fixed instances), threshold-based heuristic, and optionally PPO.
- Metrics: cumulative reward, average queue size, average number of instances (cost proxy), number of scaling events, and SLA violations (e.g., queue > threshold).
- Evaluation methodology: For each trained policy and baseline, run N evaluation episodes (e.g., 10) per load pattern with fixed seeds and report mean ± std for each metric. No cross-validation per se; we treat each episode as an independent run and aggregate statistics.

6. Software

Publicly available code we will use:
- Python 3.10+ (repo uses Python; current environment is Anaconda)
- `gymnasium` (preferred over `gym`) for environment API
- `stable-baselines3` (DQN, PPO implementations)
- `PyTorch` (backend for SB3)
- `numpy`, `pandas`, `matplotlib`, `seaborn` for data processing and plotting
- `openpyxl` (if reading `worker_one.xlsx`)

Code we will write (in this repo):
- Adaptations to `gym_scaling` environment where needed (fixes and Gymnasium compatibility).
- `train_dqn_sb3.py` and `train_ppo_sb3.py` training scripts (already present; we will extend as needed for experiments and hyperparameter sweeps).
- Evaluation harness (`src/evaluate.py`) and plotting (`src/visualize.py`) scripts to compute metrics and generate result figures.
- Additional load-generators or wrappers to run batch experiments and aggregate results.

We will use GitHub for coordination (the current repo is organized with training, evaluation, and visualization code). Each team member will implement and test a distinct component and push code with clear PRs.

7. Individual Student Responsibilities

- Name1: Environment & Simulation (modify `gym_scaling/envs/scaling_env.py`, ensure Gymnasium compatibility, implement additional load patterns, maintain `data/` ingestion). Integrate `INPUTS['PRODUCTION_DATA']` usage and test resets.
- Name2: RL Algorithms & Training (adapt and run `src/train_dqn_sb3.py`, perform hyperparameter sweeps, run DQN/PPO experiments, save trained models to `models/`, and assist with reproducibility and seeding).
- Name3: Evaluation & Visualization (use `src/evaluate.py` and `src/visualize.py` to compute metrics, produce figures and tables for the report, and run statistical comparisons across policies).

---

Notes and next steps:
- Fill in team member names, student IDs, and UCI emails at the top of this document.
- To reproduce experiments locally, create/activate a Python environment and install pinned dependencies. Example (conda):

```bash
conda create -n gym-scaling python=3.10 -y
conda activate gym-scaling
pip install -r requirements.txt
# if using gymnasium-compatible SB3, ensure versions are compatible
```

- Quick test command (already used in the repo):

```bash
python src/train_dqn_sb3.py --pattern RANDOM --timesteps 1000 --name dqn_test
```

- We can extend this document with a timeline and milestones after team roles are finalized.


(End of proposal)
