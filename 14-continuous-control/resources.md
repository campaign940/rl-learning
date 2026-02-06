# Week 14 Resources: Continuous Control (DDPG, TD3, SAC)

## Key Papers (Must Read)

### 1. Silver et al. 2014: Deterministic Policy Gradient (DPG)
**Paper**: [ICML 2014](http://proceedings.mlr.press/v32/silver14.pdf)

**Contribution**: Theoretical foundation for deterministic policy gradients, proved DPG theorem.

**Essential sections**:
- Section 2: Background
- Section 3: Gradients of Deterministic Policies
- Section 4: Deterministic Actor-Critic Algorithms

**Why read**: Foundation for DDPG/TD3. Understanding DPG makes everything else clearer.

### 2. Lillicrap et al. 2015: Continuous Control with Deep RL (DDPG)
**Paper**: [ArXiv](https://arxiv.org/abs/1509.02971)

**Contribution**: Combined DPG with deep networks, replay buffer, target networks from DQN.

**Essential sections**:
- Section 3: Algorithm
- Section 4: Results
- Appendix: Implementation details

**Why read**: First successful deep RL for continuous control. Combines many important ideas.

### 3. Fujimoto et al. 2018: Addressing Function Approximation Error (TD3)
**Paper**: [ArXiv](https://arxiv.org/abs/1802.09477)

**Contribution**: Identified overestimation in DDPG, proposed three fixes (twin Q, delayed updates, target smoothing).

**Essential sections**:
- Section 3: Background (overestimation in actor-critic)
- Section 4: Twin Delayed DDPG
- Section 5: Experiments
- Appendix: Hyperparameters

**Why read**: TD3 is more robust than DDPG. Understanding why matters for implementation.

### 4. Haarnoja et al. 2018: Soft Actor-Critic (SAC)
**Paper**: [ArXiv](https://arxiv.org/abs/1801.01290) (Original)

**Paper**: [ArXiv](https://arxiv.org/abs/1812.05905) (With Automatic Temperature Tuning)

**Contribution**: Maximum entropy RL framework, automatic temperature tuning, state-of-the-art performance.

**Essential sections**:
- Section 3: Soft Policy Iteration
- Section 4: Soft Actor-Critic
- Section 5: Automating Entropy Adjustment

**Why read**: SAC is current SOTA for continuous control. Used widely in robotics.

## Primary Lectures

### CS285 (Berkeley) - Lectures 10-11
**Lecture 10**: [Optimal Control and Planning](https://www.youtube.com/watch?v=AKbX1Zvo7r8&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=10)

**Lecture 11**: [Model-Free RL with Continuous Actions](https://www.youtube.com/watch?v=Ds1trXd6pos&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=11)

**Coverage**:
- Continuous action spaces
- Deterministic policy gradients
- DDPG, TD3, SAC

**Why watch**: Best explanation of continuous control algorithms from theory to practice.

### OpenAI Spinning Up
**DDPG**: [spinningup.openai.com/en/latest/algorithms/ddpg.html](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

**TD3**: [spinningup.openai.com/en/latest/algorithms/td3.html](https://spinningup.openai.com/en/latest/algorithms/td3.html)

**SAC**: [spinningup.openai.com/en/latest/algorithms/sac.html](https://spinningup.openai.com/en/latest/algorithms/sac.html)

**What's included**:
- Algorithm descriptions
- PyTorch implementations
- Hyperparameter suggestions
- Performance benchmarks

**Why read**: Best practical guide. Code is clean and well-documented.

## Blog Posts and Tutorials

### Lilian Weng: Policy Gradient Algorithms
**URL**: [lilianweng.github.io/posts/2018-04-08-policy-gradient/](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**Coverage**: Section on DDPG, deterministic policies.

### Jonathan Hui: RL - DDPG & TD3
**URL**: [jonathan-hui.medium.com/rl-deep-deterministic-policy-gradient-ddpg-1e7ac3b6e4ba](https://jonathan-hui.medium.com/rl-deep-deterministic-policy-gradient-ddpg-1e7ac3b6e4ba)

**Coverage**:
- DDPG intuition
- TD3 improvements
- Visual explanations

### Ben Lansdell: Introduction to SAC
**URL**: [towardsdatascience.com/soft-actor-critic-demystified-b8427df61665](https://towardsdatascience.com/soft-actor-critic-demystified-b8427df61665)

**Coverage**:
- Maximum entropy RL
- SAC algorithm
- Reparameterization trick

### Reinforcement Learning Tips and Tricks (TD3 author)
**Blog**: [Scott Fujimoto's Website](https://sfujim.github.io/)

**Coverage**: Practical advice on implementing and debugging continuous control algorithms.

## Code Repositories

### CleanRL - DDPG, TD3, SAC
**Repo**: [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)

**Files**:
- `cleanrl/ddpg_continuous_action.py`
- `cleanrl/td3_continuous_action.py`
- `cleanrl/sac_continuous_action.py`

**Why use**: Single-file implementations, modern PyTorch, well-documented.

**Getting started**:
```bash
pip install cleanrl[mujoco]
python cleanrl/sac_continuous_action.py --env-id HalfCheetah-v4
```

### Stable-Baselines3 - DDPG, TD3, SAC
**Repo**: [github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

**Why use**: Production-ready, easy API, active maintenance.

**Example**:
```python
from stable_baselines3 import SAC

model = SAC("MlpPolicy", "Pendulum-v1", verbose=1)
model.learn(total_timesteps=50000)
model.save("sac_pendulum")
```

### Spinning Up - Reference Implementations
**Repo**: [github.com/openai/spinningup](https://github.com/openai/spinningup)

**Files**:
- `spinup/algos/pytorch/ddpg/`
- `spinup/algos/pytorch/td3/`
- `spinup/algos/pytorch/sac/`

**Why use**: Reference implementations from experts, good for understanding algorithms deeply.

### Haarnoja's SAC Implementation
**Repo**: [github.com/haarnoja/sac](https://github.com/haarnoja/sac)

**Note**: Original implementation by SAC authors. TensorFlow, somewhat dated, but authoritative.

## Advanced Topics

### Maximum Entropy RL

**Paper**: [Ziebart et al. 2008 - "Maximum Entropy Inverse RL"](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf)

**Topic**: Foundation of maximum entropy framework.

**Paper**: [Levine 2018 - "Reinforcement Learning and Control as Probabilistic Inference"](https://arxiv.org/abs/1805.00909)

**Topic**: Unifies RL and probabilistic inference, foundation for soft Q-learning and SAC.

### Distributional RL for Continuous Control

**Paper**: [Barth-Maron et al. 2018 - "Distributed Distributional Deterministic Policy Gradients (D4PG)"](https://arxiv.org/abs/1804.08617)

**Contribution**: Distributional RL + DDPG for improved performance.

### Off-Policy with Function Approximation

**Paper**: [Mahmood et al. 2014 - "Weighted Importance Sampling for Off-Policy Learning"](https://proceedings.neurips.cc/paper/2014/file/be3087e74e9100d4bc4c6268cdbe8456-Paper.pdf)

**Topic**: Theory of off-policy learning, importance sampling.

## Exploration in Continuous Control

### Ornstein-Uhlenbeck Process

**Paper**: [Uhlenbeck & Ornstein 1930](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)

**Usage in DDPG**: Temporally correlated noise for smoother exploration.

**Implementation**:
```python
class OUNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
```

**Modern alternative**: Gaussian noise (simpler, works as well).

### Parameter Space Noise

**Paper**: [Plappert et al. 2017 - "Parameter Space Noise for Exploration"](https://arxiv.org/abs/1706.01905)

**Contribution**: Add noise to network parameters instead of actions. More structured exploration.

## Robotics Applications

### OpenAI Robotics

**Blog**: [OpenAI - Learning Dexterity](https://openai.com/blog/learning-dexterity/)

**Project**: Robot hand solves Rubik's Cube using SAC + domain randomization.

**Paper**: [Andrychowicz et al. 2020](https://arxiv.org/abs/1808.00177)

### ANYmal Quadruped

**Paper**: [Lee et al. 2020 - "Learning Quadrupedal Locomotion over Challenging Terrain"](https://arxiv.org/abs/2010.11251)

**Application**: SAC for quadruped robot locomotion on rough terrain.

**Video**: [YouTube](https://www.youtube.com/watch?v=JbBpUPFjNUc)

### Manipulation

**Paper**: [Kalashnikov et al. 2018 - "QT-Opt: Scalable Deep RL for Vision-Based Robotic Manipulation"](https://arxiv.org/abs/1806.10293)

**Application**: Large-scale robot grasping using off-policy Q-learning (similar to DDPG/SAC).

## Debugging and Best Practices

### Debugging Continuous Control

**Common issues**:
1. **Q-values explode**: Check target updates, learning rates
2. **Policy collapses**: Increase exploration noise or entropy
3. **No learning**: Check reward scale, normalize observations
4. **Unstable training**: Lower learning rates, increase replay buffer size

**Diagnostic plots**:
- Q-value distribution over time
- Policy entropy (for SAC)
- Gradient norms (actor and critic)
- Replay buffer diversity
- Episode returns (smoothed)

### Hyperparameter Tuning

**TD3 Defaults** (work for many tasks):
```python
{
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_delay': 2,
    'target_policy_noise': 0.2,
    'target_noise_clip': 0.5,
    'exploration_noise': 0.1,
}
```

**SAC Defaults**:
```python
{
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'target_entropy': -dim(action_space),  # Automatic
    'auto_tune_alpha': True,
}
```

**Tuning guide**:
- If slow learning: Increase learning_rate (1e-3)
- If unstable: Decrease learning_rate (1e-4), increase tau (0.01)
- If insufficient exploration: Increase exploration_noise (0.2) or entropy coefficient
- If memory issues: Reduce buffer_size

### RL Baselines Zoo - Tuned Hyperparameters
**Repo**: [github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

**Files**:
- `hyperparams/ddpg.yml`
- `hyperparams/td3.yml`
- `hyperparams/sac.yml`

**What's included**: Hyperparameters for 50+ continuous control environments.

## Environments for Practice

### Easy (Start Here)
- **Pendulum-v1**: Classic, fast, good for debugging
- **MountainCarContinuous-v0**: Harder, tests exploration
- **LunarLanderContinuous-v2**: Space lander with continuous thrusters

### Medium
- **BipedalWalker-v3**: 2D walker, challenging but solvable
- **Reacher-v4**: Robot arm reaching (MuJoCo)
- **Pusher-v4**: Pushing object to target

### Hard
- **HalfCheetah-v4**: Fast quadruped running
- **Ant-v4**: Quadruped with complex dynamics
- **Humanoid-v4**: Full humanoid, 17D actions
- **HandManipulateBlock-v0**: Dexterous manipulation (very hard!)

### Real-World Inspired
- **PyBullet**: Free physics simulator (alternative to MuJoCo)
  - `HalfCheetahBulletEnv-v0`
  - `AntBulletEnv-v0`
  - `HumanoidBulletEnv-v0`

## Advanced Variants and Extensions

### Model-Based + Model-Free

**Paper**: [Janner et al. 2019 - "When to Trust Your Model: Model-Based Policy Optimization (MBPO)"](https://arxiv.org/abs/1906.08253)

**Contribution**: Combines world model with SAC for improved sample efficiency.

### Hindsight Experience Replay (HER)

**Paper**: [Andrychowicz et al. 2017](https://arxiv.org/abs/1707.01495)

**Contribution**: Learn from failures in goal-conditioned tasks. Compatible with DDPG/TD3/SAC.

**Use case**: Sparse reward robotics (e.g., reaching, pushing).

### Residual RL

**Paper**: [Silver et al. 2018 - "Residual Policy Learning"](https://arxiv.org/abs/1812.03201)

**Contribution**: Learn corrections on top of existing controller. Combines classical control + RL.

### Multi-task and Meta-RL

**Paper**: [Yu et al. 2020 - "Meta-World: A Benchmark for Multi-Task and Meta-RL"](https://arxiv.org/abs/1910.10897)

**Application**: Single policy for multiple manipulation tasks.

## Study Plan

### Week 14 Day 1-2: Foundations
1. Read DPG paper (Silver et al. 2014)
2. Read DDPG paper (Lillicrap et al. 2015)
3. Watch CS285 Lecture 11
4. Understand deterministic policy gradient theorem

### Week 14 Day 3-4: TD3 and Improvements
5. Read TD3 paper (Fujimoto et al. 2018)
6. Understand overestimation bias
7. Study CleanRL TD3 implementation
8. Implement DDPG or TD3 on Pendulum

### Week 14 Day 5-6: SAC and Maximum Entropy
9. Read SAC papers (both versions)
10. Understand maximum entropy RL
11. Study reparameterization trick
12. Implement SAC on BipedalWalker

### Week 14 Day 7: Applications
13. Try MuJoCo environments (HalfCheetah)
14. Compare DDPG vs TD3 vs SAC
15. Read about robotics applications
16. Explore advanced topics (HER, model-based)

## Quick Reference

### Key Equations

**Deterministic Policy Gradient**:
```
∇_θ J = E_s[∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s; θ)]
```

**DDPG Critic Update**:
```
y = r + γ Q'(s', μ'(s'))
L = (Q(s,a) - y)²
```

**TD3 Target**:
```
y = r + γ min(Q1', Q2')(s', μ'(s') + clip(ε, -c, c))
```

**SAC Objective**:
```
J(π) = E[Σ(r + α H(π))]
     = E[Q(s,a) - α log π(a|s)]
```

### Typical Hyperparameters

**For continuous control (MuJoCo)**:
```python
{
    'algorithm': 'SAC',  # or TD3
    'learning_rate': 3e-4,
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'learning_starts': 10000,
    'train_freq': 1,
    'gradient_steps': 1,
}
```

## Community and Discussion

### Reddit
- r/reinforcementlearning (general RL)
- r/learnmachinelearning (beginner-friendly)
- r/robotics (applications)

### Twitter
Follow:
- @scottfujimoto (TD3 author)
- @tuomaso (SAC author)
- @sferika (Sergey Levine, maximum entropy RL)
- @OpenAI (applications)

### Discord
- OpenAI Scholars
- Reinforcement Learning Discord

### Conferences
- **CoRL**: Conference on Robot Learning (RL applications)
- **ICRA/IROS**: Robotics conferences (RL for robots)
- **NeurIPS/ICML**: ML conferences (RL theory and algorithms)

## Practical Tips

1. **Start with SAC**: Most robust, works out-of-the-box
2. **Use Stable-Baselines3**: For baselines and comparisons
3. **Normalize observations**: Critical for continuous control
4. **Check reward scale**: Should be roughly O(1)
5. **Monitor Q-values**: Should be stable, not exploding
6. **Be patient**: 1M-10M steps typical for complex tasks
7. **Try PyBullet first**: Free alternative to MuJoCo
8. **Read RL Baselines Zoo hyperparams**: Good starting points

## What's Next?

After mastering continuous control, explore:

**Model-Based RL**: Learn dynamics model, plan with it
- PETS, MBPO, Dreamer, PlaNet

**Offline RL**: Learn from fixed datasets (no online interaction)
- CQL, IQL, TD3+BC

**Multi-Agent RL**: Multiple agents interacting
- MADDPG, QMIX, MAPPO

**Hierarchical RL**: Temporal abstraction, options
- HIRO, HAC

**Real Robots**: Sim-to-real transfer, domain randomization

## Final Thoughts

Continuous control is where RL shines:
- **DDPG**: Historical importance, foundation
- **TD3**: Practical improvement, robust
- **SAC**: Current SOTA, default choice

**For your projects**: Start with SAC from Stable-Baselines3. Once working, optimize if needed.

**For learning**: Implement DDPG first (simpler), then add TD3 improvements, finally try SAC.

**For research**: SAC is the baseline. Improvements should beat SAC on standard benchmarks.

Congratulations on completing the core RL curriculum! You now have the tools to tackle real-world continuous control problems.
