# LunarLander-v3 PPO Agent 🚀

This repository contains the implementation of a Proximal Policy Optimization (PPO) agent trained to solve the `LunarLander-v3` (Continuous) environment from Gymnasium. It was developed as part of a Reinforcement Learning assignment.

The project not only trains a baseline PPO agent but also includes a comprehensive hyperparameter tuning suite to analyze the effects of various configurations (Learning Rate, Gamma, Network Architecture, and Clipping Boundary) on the agent's performance.

## 📁 Repository Structure

```text
lunar_lander/
│
├── config.py                 # Centralized configuration for environment and PPO baseline params
├── main.py                   # Main training script with argparse for hyperparameter tuning
├── utils.py                  # Custom SB3 Callbacks (Top-3 model saving) & plotting functions
├── run_experiments.sh        # Bash script to automate the hyperparameter tuning pipeline
├── visualize_models.ipynb    # Interactive script to load and render trained models
├── hyperparameter_report.tex # LaTeX source code for the analysis report
│
├── results/                  # Auto-generated directory for outputs
│   ├── logs/                 # Console logs for each experiment
│   ├── models/               # Saved .zip models (keeps top 3 per experiment)
│   └── plots/                # Training vs Evaluation learning curves
│
└── fig/                      # Directory containing plots used in the LaTeX report
```

## 🛠️ Installation

Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment (e.g., Anaconda or venv).

1. Clone the repository:
```bash
git clone git@github.com:MissionAC/lunar_lander.git
cd lunar_lander
```

2. Install the required dependencies:
```bash
pip install gymnasium[box2d] stable-baselines3[extra] matplotlib
```
*(Note: `box2d` is required for the LunarLander environment).*

## 🚀 Usage

### 1. Run a Single Training Session (Baseline)

You can train a single PPO agent using the default configurations defined in `config.py`:
```bash
python main.py
```

To override specific hyperparameters, use the command-line arguments:
```bash
python main.py --lr 0.001 --gamma 0.999 --clip_range 0.1 --net_arch 128 128
```

### 2. Run the Hyperparameter Tuning Suite

To reproduce the experiments for the hyperparameter analysis (Question 1B), run the bash script. This will sequentially train models with different configurations, log the console outputs, save the top 3 best-performing models for each run, and generate learning curves.
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

### 3. Visualize Trained Models

To watch the trained agent land the spacecraft, run the visualization script. It automatically scans the `results/` directory for `.zip` models and allows you to select which one to render.
```bash
python visualize_models.py
```
*(Note: If you are running this on WSL/Linux without an audio driver, the script automatically suppresses ALSA audio warnings by setting a dummy audio driver).*

## 📊 Features & Implementation Details

* **Episode-based Logging:** The assignment specifically requests plotting the X-axis by *episode number* (not timesteps). A custom `EpisodeLoggerCallback` was implemented to track and log rewards precisely at the end of each episode.

* **Top-K Model Saving:** The callback dynamically tracks the evaluation performance and saves only the **Top 3** models for each hyperparameter configuration to save disk space while keeping the best policies.

* **Smoothing:** The generated plots include an Exponential Moving Average (EMA) smoothed curve for training rewards to better visualize the learning trend amidst high variance.

* **Control Variable Analysis:** The `run_experiments.sh` script employs a strict control variable approach, changing only one hyperparameter at a time against an optimized `[64, 64]` baseline to cleanly observe its effect.

## 📝 Summary of Findings

A detailed analysis of how hyperparameters (LR, Gamma, Capacity, Clip Range) affect the agent's performance is provided in the submitted PDF report. Below is a summary of the key findings:

* **Network Capacity:** Wider networks (e.g., `[128, 128]`) significantly improve learning speed and stability compared to the default `[64, 64]`.

* **Gamma:** While `0.999` offers theoretical long-term stability, it severely hinders early-stage learning (credit assignment problem). The baseline `0.99` is highly superior for sample efficiency.

* **Learning Rate & Clipping:** Aggressive learning rates (`0.001`) cause catastrophic forgetting, while overly tight clipping (`0.1`) restricts exploration and slows down convergence.