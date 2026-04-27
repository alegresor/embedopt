#!/usr/bin/env python3
"""
Visualize reward traces from reward_history.json files.

This script reads reward_history.json files and plots reward as a function of step.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_reward_history(json_path):
    """
    Load reward history from JSON file.
    
    Args:
        json_path: Path to reward_history.json file
        
    Returns:
        List of dictionaries with reward history data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    steps = [entry['step'] for entry in data]
    rewards = [entry['reward'] for entry in data]
    return steps, rewards


# ============================================================================
# CONFIGURATION
# ============================================================================
system = "6V7W_B"
lr=0.1
name1 = f"embedopt_{lr}_k20"
name2 = f"dps_{lr}_k20"
seed = 42

# Base directory
base_dir = Path(__file__).parent / "steering_output_k20"
path1 = base_dir / system / f"{system}_{name1}" / f"seed_{seed}" / "reward_history.json"
path2 = base_dir / system / f"{system}_{name2}" / f"seed_{seed}" / "reward_history.json"
# ============================================================================

steps1, rewards1 = load_reward_history(path1)
steps2, rewards2 = load_reward_history(path2)

plt.figure(figsize=(4, 4), dpi=200)
plt.plot(steps2, rewards2, label="DPS", color="#2E8B57", lw=3, alpha=0.7)
plt.plot(steps1, rewards1, label="EmbedOpt", color="#FF6347", lw=3, alpha=0.7)
plt.yticks([])
plt.xticks([])
plt.legend()
plt.ylabel("Reward")
plt.xlabel("Step")
plt.title("Reward Trace")
output_dir = Path(__file__).parent / "reward_trace_visualization"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / f"{system}_{name1}_vs_{name2}_reward_trace.png", dpi=200)