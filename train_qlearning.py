#!/usr/bin/env python3
"""
Q-Learning implementation for cloud autoscaling.
This is a starter template for your RL autoscaling project.
"""

import numpy as np
import random
import pickle
from collections import defaultdict
from gym_scaling.envs.scaling_env import ScalingEnv, INPUTS


class QLearningAgent:
    """Tabular Q-Learning agent with discretized state space."""
    
    def __init__(self, n_actions=3, learning_rate=0.1, discount_factor=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: defaultdict returns 0 for unseen state-action pairs
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def discretize_state(self, obs, bins=10):
        """
        Discretize continuous observation into discrete state.
        
        Args:
            obs: [normalized_instances, load, capacity, influx, queue_size]
            bins: Number of bins per dimension
        
        Returns:
            Tuple representing discrete state
        """
        # Focus on most important features
        instances_bin = min(int(obs[0] * bins), bins - 1)
        load_bin = min(int(obs[1] * bins), bins - 1)
        
        # Discretize queue size (0, small, medium, large)
        if obs[4] == 0:
            queue_bin = 0
        elif obs[4] < 100:
            queue_bin = 1
        elif obs[4] < 500:
            queue_bin = 2
        else:
            queue_bin = 3
        
        return (instances_bin, load_bin, queue_bin)
    
    def get_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Discrete state tuple
            training: If True, use epsilon-greedy; if False, use greedy
        
        Returns:
            Action index
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state][action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), pickle.load(f))
        print(f"Model loaded from {filepath}")


def train_qlearning(n_episodes=1000, max_steps=200, verbose=True):
    """
    Train Q-learning agent on autoscaling environment.
    
    Args:
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        verbose: Print progress
    
    Returns:
        Trained agent and training metrics
    """
    # Create environment
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1
    
    # Create agent
    agent = QLearningAgent(
        n_actions=3,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_costs = []
    
    print("=" * 60)
    print("Training Q-Learning Agent")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Learning rate: {agent.lr}")
    print(f"Discount factor: {agent.gamma}")
    print(f"Epsilon decay: {agent.epsilon_decay}")
    print("=" * 60)
    
    for episode in range(n_episodes):
        obs = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        episode_cost = 0
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.get_action(state, training=True)
            next_obs, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            # Track metrics
            episode_reward += reward
            episode_cost = env.total_cost
            
            state = next_state
            
            if done:
                break
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        episode_costs.append(episode_cost)
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_cost = np.mean(episode_costs[-100:])
            print(f"Episode {episode + 1:4d} | "
                  f"Avg Reward: {avg_reward:7.3f} | "
                  f"Avg Length: {avg_length:6.1f} | "
                  f"Avg Cost: ${avg_cost:8.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    print("=" * 60)
    print("Training completed!")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table size: {len(agent.q_table)} states")
    print("=" * 60)
    
    return agent, {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'costs': episode_costs
    }


def evaluate_agent(agent, n_episodes=10, max_steps=200):
    """
    Evaluate trained agent.
    
    Args:
        agent: Trained Q-learning agent
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
    
    Returns:
        Evaluation metrics
    """
    env = ScalingEnv()
    env.scaling_env_options['input'] = INPUTS['SINE_CURVE']
    env.change_rate = 1
    
    episode_rewards = []
    episode_costs = []
    episode_queues = []
    episode_loads = []
    
    print("\n" + "=" * 60)
    print("Evaluating Agent")
    print("=" * 60)
    
    for episode in range(n_episodes):
        obs = env.reset()
        state = agent.discretize_state(obs)
        
        episode_reward = 0
        queues = []
        loads = []
        
        for step in range(max_steps):
            # Greedy action selection (no exploration)
            action = agent.get_action(state, training=False)
            next_obs, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_obs)
            
            episode_reward += reward
            queues.append(env.queue_size)
            loads.append(env.load)
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_costs.append(env.total_cost)
        episode_queues.append(np.mean(queues))
        episode_loads.append(np.mean(loads))
        
        print(f"Episode {episode + 1}: "
              f"Reward={episode_reward:7.2f}, "
              f"Cost=${env.total_cost:8.2f}, "
              f"Avg Queue={np.mean(queues):6.2f}, "
              f"Avg Load={np.mean(loads):5.1f}%")
    
    print("=" * 60)
    print("Evaluation Results:")
    print(f"  Average Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Average Cost: ${np.mean(episode_costs):.2f} ± ${np.std(episode_costs):.2f}")
    print(f"  Average Queue: {np.mean(episode_queues):.2f} ± {np.std(episode_queues):.2f}")
    print(f"  Average Load: {np.mean(episode_loads):.1f}% ± {np.std(episode_loads):.1f}%")
    print("=" * 60)
    
    return {
        'rewards': episode_rewards,
        'costs': episode_costs,
        'queues': episode_queues,
        'loads': episode_loads
    }


def main():
    """Main training and evaluation pipeline."""
    
    # Train agent
    agent, train_metrics = train_qlearning(
        n_episodes=1000,
        max_steps=200,
        verbose=True
    )
    
    # Save trained agent
    agent.save('models/qlearning_model.pkl')
    
    # Evaluate agent
    eval_metrics = evaluate_agent(agent, n_episodes=10, max_steps=200)
    
    # Plot training progress (optional)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Training rewards
        window = 50
        moving_avg = np.convolve(train_metrics['rewards'], 
                                np.ones(window)/window, mode='valid')
        axes[0, 0].plot(moving_avg)
        axes[0, 0].set_title('Training Rewards (Moving Average)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Training costs
        axes[0, 1].plot(train_metrics['costs'])
        axes[0, 1].set_title('Training Costs')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cost ($)')
        axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(train_metrics['lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Evaluation rewards
        axes[1, 1].bar(range(len(eval_metrics['rewards'])), eval_metrics['rewards'])
        axes[1, 1].set_title('Evaluation Rewards')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('qlearning_training.png')
        print("\nTraining plot saved to qlearning_training.png")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    
    print("\n✓ Training and evaluation complete!")
    print("  Model saved to: models/qlearning_model.pkl")
    print("\nNext steps:")
    print("  1. Tune hyperparameters (learning rate, epsilon decay, etc.)")
    print("  2. Experiment with different state discretization")
    print("  3. Compare with threshold baseline")
    print("  4. Implement DQN for comparison")


if __name__ == '__main__':
    main()
