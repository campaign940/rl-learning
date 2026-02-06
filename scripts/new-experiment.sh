#!/bin/bash
# Create a new experiment directory with proper structure
#
# Usage: ./scripts/new-experiment.sh "Experiment Name"
# Example: ./scripts/new-experiment.sh "DQN Breakout"

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_success() {
    echo -e "${BLUE}[SUCCESS]${NC} $1"
}

# Check arguments
if [ $# -ne 1 ]; then
    print_error "Invalid number of arguments"
    echo "Usage: $0 \"Experiment Name\""
    echo "Example: $0 \"DQN Breakout\""
    exit 1
fi

EXPERIMENT_NAME=$1

# Get repository root (assumes script is in scripts/ directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXPERIMENTS_DIR="$REPO_ROOT/experiments"

print_info "Repository root: $REPO_ROOT"

# Create experiments directory if it doesn't exist
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    print_info "Creating experiments directory..."
    mkdir -p "$EXPERIMENTS_DIR"
fi

# Find next experiment number
print_info "Finding next experiment number..."
LAST_EXP_NUM=0

for dir in "$EXPERIMENTS_DIR"/exp-*; do
    if [ -d "$dir" ]; then
        exp_num=$(basename "$dir" | sed 's/exp-\([0-9]\{3\}\).*/\1/')
        if [ "$exp_num" -gt "$LAST_EXP_NUM" ]; then
            LAST_EXP_NUM=$exp_num
        fi
    fi
done

NEXT_EXP_NUM=$(printf "%03d" $((LAST_EXP_NUM + 1)))
print_info "Next experiment number: $NEXT_EXP_NUM"

# Create slug from experiment name
SLUG=$(echo "$EXPERIMENT_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9-]//g')
EXP_DIR_NAME="exp-${NEXT_EXP_NUM}-${SLUG}"
TARGET_DIR="$EXPERIMENTS_DIR/$EXP_DIR_NAME"

print_info "Creating experiment directory: $EXP_DIR_NAME"

# Create directory structure
mkdir -p "$TARGET_DIR"
mkdir -p "$TARGET_DIR/figures"
mkdir -p "$TARGET_DIR/logs"
mkdir -p "$TARGET_DIR/checkpoints"
mkdir -p "$TARGET_DIR/src"

# Get current date
CURRENT_DATE=$(date +%Y-%m-%d)

# Create config.yaml
print_info "Creating config.yaml..."
cat > "$TARGET_DIR/config.yaml" << EOF
# Experiment Configuration
experiment:
  name: "$EXPERIMENT_NAME"
  number: $NEXT_EXP_NUM
  date_created: $CURRENT_DATE
  description: |
    Brief description of the experiment.

  tags:
    - rl
    - experiment

  reproducibility:
    random_seed: 42
    deterministic: true

# Environment Configuration
environment:
  name: ""
  type: ""  # gym, atari, custom, etc.
  params:
    render_mode: "rgb_array"

# Agent Configuration
agent:
  algorithm: ""  # DQN, A2C, PPO, etc.
  network:
    architecture: ""
    hidden_layers: []
    activation: "relu"

  hyperparameters:
    learning_rate: 0.001
    gamma: 0.99
    batch_size: 32
    buffer_size: 10000

# Training Configuration
training:
  total_timesteps: 100000
  eval_frequency: 1000
  save_frequency: 5000
  log_frequency: 100

  early_stopping:
    enabled: false
    patience: 10
    min_improvement: 0.01

# Evaluation Configuration
evaluation:
  num_episodes: 10
  deterministic: true

# Logging Configuration
logging:
  tensorboard: true
  wandb: false
  csv: true

  metrics:
    - episode_reward
    - episode_length
    - loss
    - learning_rate

# Output Configuration
output:
  figures_dir: "figures/"
  logs_dir: "logs/"
  checkpoints_dir: "checkpoints/"
EOF

# Create results.md
print_info "Creating results.md..."
cat > "$TARGET_DIR/results.md" << EOF
# Experiment $NEXT_EXP_NUM: $EXPERIMENT_NAME

**Date:** $CURRENT_DATE
**Status:** Not Started
**Duration:**

## Hypothesis

What we expect to happen:
-
-

## Experiment Setup

### Environment
- **Name:**
- **Type:**
- **Description:**

### Agent
- **Algorithm:**
- **Architecture:**
- **Key Hyperparameters:**
  - Learning rate:
  - Batch size:
  - Gamma:

### Training Configuration
- **Total timesteps:**
- **Evaluation frequency:**
- **Hardware:**

## Results

### Training Performance

#### Metrics
| Metric | Initial | Final | Best |
|--------|---------|-------|------|
| Episode Reward | | | |
| Episode Length | | | |
| Loss | | | |
| Success Rate | | | |

#### Learning Curves
![Training Reward](figures/training_reward.png)
![Training Loss](figures/training_loss.png)

### Evaluation Performance

| Episode | Reward | Length | Success |
|---------|--------|--------|---------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| Average | | | |

### Visualizations

#### Sample Episodes
![Episode 1](figures/episode_1.gif)

#### Attention/Activations
![Activations](figures/activations.png)

## Analysis

### Key Findings
1.
2.
3.

### Comparison to Baseline
-
-

### Unexpected Behaviors
-
-

### Performance Bottlenecks
-
-

## Conclusions

### Summary
-

### What Worked
-
-

### What Didn't Work
-
-

### Lessons Learned
1.
2.
3.

## Next Steps

### Immediate Follow-ups
- [ ]
- [ ]

### Future Experiments
- [ ]
- [ ]

### Improvements to Try
- [ ]
- [ ]

## References

### Code
- Main implementation: \`src/\`
- Configuration: \`config.yaml\`
- Logs: \`logs/\`

### Related Experiments
- Experiment XXX: [link]

### Papers/Resources
-
-

## Reproducibility

### Environment
\`\`\`bash
# Python version
python --version

# Key dependencies
pip freeze > requirements.txt
\`\`\`

### Command to Reproduce
\`\`\`bash
python src/train.py --config config.yaml --seed 42
\`\`\`

### Notes
-
-

## Appendix

### Full Configuration
See [config.yaml](config.yaml)

### Complete Logs
See [logs/](logs/)

### Additional Figures
See [figures/](figures/)
EOF

# Create README.md
print_info "Creating README.md..."
cat > "$TARGET_DIR/README.md" << EOF
# Experiment $NEXT_EXP_NUM: $EXPERIMENT_NAME

**Status:** 🔴 Not Started
**Created:** $CURRENT_DATE
**Last Updated:** $CURRENT_DATE

## Quick Links
- [Configuration](config.yaml)
- [Results](results.md)
- [Figures](figures/)
- [Logs](logs/)
- [Checkpoints](checkpoints/)

## Description

Brief description of what this experiment aims to test.

## Structure

\`\`\`
$EXP_DIR_NAME/
├── README.md              # This file
├── config.yaml           # Experiment configuration
├── results.md            # Detailed results and analysis
├── src/                  # Source code
│   ├── train.py         # Training script
│   ├── eval.py          # Evaluation script
│   └── utils.py         # Helper functions
├── figures/              # Generated figures and plots
├── logs/                 # Training logs
└── checkpoints/          # Model checkpoints
\`\`\`

## Quick Start

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py --config config.yaml

# Evaluate
python src/eval.py --checkpoint checkpoints/best_model.pt

# Generate plots
python src/plot_results.py --log logs/training.csv
\`\`\`

## Status Updates

### $CURRENT_DATE - Created
- Initialized experiment structure
- Created configuration template
- Ready to start implementation

---

**Note:** Update this README and [results.md](results.md) as the experiment progresses.
EOF

# Create basic source files
print_info "Creating source files..."

# train.py
cat > "$TARGET_DIR/src/train.py" << 'EOF'
#!/usr/bin/env python3
"""
Training script for the experiment.
"""

import argparse
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train agent')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration: {config['experiment']['name']}")

    # TODO: Implement training logic
    print("Training not implemented yet")


if __name__ == '__main__':
    main()
EOF

# eval.py
cat > "$TARGET_DIR/src/eval.py" << 'EOF'
#!/usr/bin/env python3
"""
Evaluation script for the experiment.
"""

import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Evaluate agent')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of evaluation episodes')
    args = parser.parse_args()

    # TODO: Implement evaluation logic
    print("Evaluation not implemented yet")


if __name__ == '__main__':
    main()
EOF

# utils.py
cat > "$TARGET_DIR/src/utils.py" << 'EOF'
"""
Utility functions for the experiment.
"""

import numpy as np
from pathlib import Path


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    # TODO: Set seeds for torch, tensorflow, etc.


def save_checkpoint(model, path: Path):
    """Save model checkpoint."""
    # TODO: Implement checkpoint saving
    pass


def load_checkpoint(path: Path):
    """Load model checkpoint."""
    # TODO: Implement checkpoint loading
    pass
EOF

# Create .gitignore for experiment
print_info "Creating .gitignore..."
cat > "$TARGET_DIR/.gitignore" << EOF
# Logs
logs/**/*.csv
logs/**/*.txt
logs/**/*.log

# Checkpoints (keep best model only)
checkpoints/*
!checkpoints/.gitkeep
!checkpoints/best_model.*

# Generated figures (except final results)
figures/tmp_*
figures/debug_*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# OS
.DS_Store
Thumbs.db
EOF

# Create .gitkeep files
touch "$TARGET_DIR/figures/.gitkeep"
touch "$TARGET_DIR/logs/.gitkeep"
touch "$TARGET_DIR/checkpoints/.gitkeep"

# Create requirements.txt
print_info "Creating requirements.txt..."
cat > "$TARGET_DIR/requirements.txt" << EOF
# Core RL libraries
gymnasium
stable-baselines3
torch
numpy

# Visualization
matplotlib
seaborn
tensorboard

# Utilities
pyyaml
pandas
tqdm
EOF

print_success "Experiment created successfully!"
echo ""
print_info "Experiment Details:"
echo "  Number: $NEXT_EXP_NUM"
echo "  Name: $EXPERIMENT_NAME"
echo "  Directory: $TARGET_DIR"
echo ""
print_info "Next steps:"
echo "  1. cd $TARGET_DIR"
echo "  2. Edit config.yaml with your experiment parameters"
echo "  3. Implement training logic in src/train.py"
echo "  4. Run: python src/train.py --config config.yaml"
echo "  5. Document results in results.md"
echo ""
print_warning "Don't forget to update the experiments index!"
