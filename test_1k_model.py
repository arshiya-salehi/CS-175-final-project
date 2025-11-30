#!/usr/bin/env python3
"""Test the 1,000 timestep DQN model."""

import warnings
warnings.filterwarnings('ignore')

from dqn_demo import load_dqn_model, create_env_with_pattern, run_dqn_agent
import numpy as np

print('Testing 1,000 timestep DQN model...')
print('=' * 60)

model = load_dqn_model('models/dqn_sine_curve_20251123_174533.zip')
if model:
    env = create_env_with_pattern('SINE_CURVE')
    metrics = run_dqn_agent(model, env, 200)
    env.close()
    
    print(f'\nResults:')
    print(f'  Steps completed: {len(metrics["rewards"])}')
    print(f'  Average reward: {np.mean(metrics["rewards"]):.4f}')
    print(f'  Total reward: {sum(metrics["rewards"]):.2f}')
    print(f'  Average load: {np.mean(metrics["load"]):.1f}%')
    print(f'  Average queue: {np.mean(metrics["queue_size"]):.2f}')
    print(f'  Max queue: {max(metrics["queue_size"]):.0f}')
    print(f'  Average instances: {np.mean(metrics["instances"]):.1f}')
    
    # Check if it completed without overflow
    if len(metrics['rewards']) == 200:
        print(f'\n✓ SUCCESS: Completed all 200 steps without overflow!')
    else:
        print(f'\n⚠ Episode ended early at step {len(metrics["rewards"])}')
    
    print('\n' + '=' * 60)
    print('Comparison with broken dqn_test.zip:')
    print('=' * 60)
    print(f'{"Metric":<20} {"Old (broken)":<20} {"New (1k steps)":<20}')
    print('-' * 60)
    print(f'{"Steps completed":<20} {"49":<20} {len(metrics["rewards"]):<20}')
    print(f'{"Avg reward":<20} {"-0.6348":<20} {np.mean(metrics["rewards"]):<20.4f}')
    print(f'{"Avg queue":<20} {"9102.26":<20} {np.mean(metrics["queue_size"]):<20.2f}')
    print(f'{"Max queue":<20} {"~10000":<20} {max(metrics["queue_size"]):<20.0f}')
    print(f'{"Avg load":<20} {"87.5%":<20} {f"{np.mean(metrics["load"]):.1f}%":<20}')
    print('=' * 60)
