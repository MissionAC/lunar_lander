#!/bin/bash

# ====================================================================
# Shell script to automate hyperparameter tuning for SB3 PPO
# This script runs different configurations sequentially.
# ====================================================================

# Create necessary directories
mkdir -p results/models
mkdir -p results/plots

echo "Starting Hyperparameter Tuning Experiments..."

# Experiment 1: Baseline (Using defaults or standard values)
echo "Running Experiment 1: Baseline"
python main.py --lr 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --net_arch 128 128

# Experiment 2: Tuning Learning Rate (Higher LR)
echo "Running Experiment 2: Higher Learning Rate"
python main.py --lr 0.001 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --net_arch 128 128

# Experiment 3: Tuning Discount Factor (gamma)
echo "Running Experiment 3: Higher Gamma (focus on long-term reward)"
python main.py --lr 0.0003 --gamma 0.999 --gae_lambda 0.95 --clip_range 0.2 --net_arch 128 128

# Experiment 4: Tuning Network Architecture (Deeper/Wider network)
echo "Running Experiment 4: Deeper Network Architecture"
python main.py --lr 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --net_arch 256 256

# Experiment 5: Tuning Clipping Boundary
echo "Running Experiment 5: Tighter Clipping Boundary"
python main.py --lr 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.1 --net_arch 128 128

echo "All experiments finished! Check the 'results/models' and 'results/plots' folders."