#!/bin/bash

# ====================================================================
# Shell script to automate hyperparameter tuning for SB3 PPO
# This script runs different configurations sequentially.
# ====================================================================

# Create necessary directories
mkdir -p results/models
mkdir -p results/plots
mkdir -p results/logs

echo "Starting Hyperparameter Tuning Experiments..."

# Experiment 1: Baseline (Using defaults or standard values with 64 64 arch)
echo "Running Experiment 1: Baseline"
python -u main.py --lr 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --net_arch 64 64 2>&1 | tee results/logs/exp1_baseline.log

# Experiment 2: Tuning Learning Rate (Higher LR, kept other params at baseline)
echo "Running Experiment 2: Higher Learning Rate"
python -u main.py --lr 0.001 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --net_arch 64 64 2>&1 | tee results/logs/exp2_lr.log

# Experiment 3: Tuning Discount Factor (Higher gamma, kept other params at baseline)
echo "Running Experiment 3: Higher Gamma (focus on long-term reward)"
python -u main.py --lr 0.0003 --gamma 0.999 --gae_lambda 0.95 --clip_range 0.2 --net_arch 64 64 2>&1 | tee results/logs/exp3_gamma.log

# Experiment 4: Tuning Network Architecture (Deeper/Wider network compared to baseline)
echo "Running Experiment 4: Wider Network Architecture (128 128)"
python -u main.py --lr 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --net_arch 128 128 2>&1 | tee results/logs/exp4_arch_128.log

# Experiment 5: Tuning Network Architecture (Deeper/Wider network compared to baseline)
echo "Running Experiment 5: Deeper Network Architecture (64 64 64)"
python -u main.py --lr 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.2 --net_arch 64 64 64 2>&1 | tee results/logs/exp5_arch_64_64_64.log

# Experiment 6: Tuning Clipping Boundary (Tighter clip, kept other params at baseline)
echo "Running Experiment 6: Tighter Clipping Boundary"
python -u main.py --lr 0.0003 --gamma 0.99 --gae_lambda 0.95 --clip_range 0.1 --net_arch 64 64 2>&1 | tee results/logs/exp6_clip.log

echo "All experiments finished! Check the 'results' folder for saved models, plots, and logs."