# Week 13 Resources: TRPO & PPO

## Key Papers (Must Read)

### 1. Schulman et al. 2015: Trust Region Policy Optimization (TRPO)
**Paper**: [ArXiv](https://arxiv.org/abs/1502.05477)

**Essential sections**:
- Section 2: Preliminaries (notation)
- Section 3: Monotonic Improvement Guarantee
- Section 4: Optimization of Parameterized Policies
- Section 6: Experiments

**Why read**: Theoretical foundation for constrained policy optimization. Understanding TRPO makes PPO much clearer.

### 2. Schulman et al. 2017: Proximal Policy Optimization (PPO)
**Paper**: [ArXiv](https://arxiv.org/abs/1707.06347)

**Essential sections**:
- Section 3: Clipped Surrogate Objective
- Section 4: Adaptive KL Penalty (alternative)
- Section 5: Algorithm details
- Section 6: Experiments

**Why read**: The most important RL paper of the last decade. PPO is used everywhere.

### 3. Kakade & Langford 2002: Approximately Optimal Approximate Reinforcement Learning
**Paper**: [ICML 2002](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)

**Contribution**: Conservative Policy Iteration, foundation for monotonic improvement theory.

**Why read**: Original performance bounds that motivated TRPO.

## Primary Lectures

### CS285 (Berkeley) - Lecture 9: Advanced Policy Gradients
**Video**: [YouTube](https://www.youtube.com/watch?v=AKbX1Zvo7r8&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=9)

**Coverage**:
- Why policy gradients can fail
- Natural gradients and trust regions
- TRPO derivation
- PPO as practical approximation

**Why watch**: Best explanation of the motivation and theory behind TRPO/PPO.

### OpenAI Spinning Up: PPO
**Documentation**: [spinningup.openai.com/en/latest/algorithms/ppo.html](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

**Coverage**:
- Algorithm description
- Implementation details
- PyTorch code
- Benchmarks

**Why read**: Practical focus, excellent for implementation.

## Blog Posts and Tutorials

### Jonathan Hui: RL — Proximal Policy Optimization (PPO) Explained
**URL**: [jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12](https://jonathan-hui.medium.com/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12)

**Coverage**:
- Visual explanations of clipping
- TRPO vs PPO comparison
- Implementation tips

**Why read**: Great visuals, intuitive explanations.

### Lilian Weng: Policy Gradient Algorithms
**URL**: [lilianweng.github.io/posts/2018-04-08-policy-gradient/](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**Coverage**:
- TRPO section with math details
- PPO variants
- Natural gradient explanation

**Why read**: Comprehensive reference with all key formulas.

### Daniel Takeshi: Notes on TRPO and PPO
**URL**: [danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/)

**Coverage**:
- Detailed TRPO math
- Conjugate gradient explanation
- Connection to natural gradients

**Why read**: Deep dive into the theory.

### PPO Implementation Details
**Paper**: [Huang et al. 2022 - "The 37 Implementation Details of PPO"](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

**Coverage**:
- 37 specific implementation tricks
- Which ones matter most
- Ablation studies

**Why read**: Critical for reproducing PPO results. Many "implementation details" matter more than algorithm itself!

## Code Repositories

### CleanRL - PPO
**Repo**: [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)

**Files**:
- `cleanrl/ppo.py` - Single-file PPO for simple envs
- `cleanrl/ppo_continuous_action.py` - PPO for continuous control
- `cleanrl/ppo_atari.py` - PPO for Atari games
- `cleanrl/ppo_atari_lstm.py` - PPO with recurrent networks

**Why use**: Best for learning. Single-file, well-commented, modern PyTorch.

**Getting started**:
```bash
pip install cleanrl[mujoco]
python cleanrl/ppo_continuous_action.py --env-id HalfCheetah-v4
```

### Stable-Baselines3 - PPO
**Repo**: [github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

**Why use**: Production-ready, easy API, well-maintained.

**Example**:
```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "LunarLander-v2", verbose=1)
model.learn(total_timesteps=1000000)
model.save("ppo_lunar")
```

### OpenAI Baselines - PPO2
**Repo**: [github.com/openai/baselines](https://github.com/openai/baselines)

**Files**: `baselines/ppo2/`

**Note**: TensorFlow 1.x, somewhat dated, but reference implementation from original authors.

### PPO from Scratch Tutorials

**Phil Tabor**: [YouTube PPO Tutorial](https://www.youtube.com/watch?v=hlv79rcHws0)
- Step-by-step implementation
- Good for beginners

**Machine Learning with Phil**: [GitHub](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO)
- Accompanying code for video tutorial

## Advanced Topics

### Natural Gradients and TRPO

**Paper**: [Kakade 2001 - "A Natural Policy Gradient"](https://papers.nips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)

**Topic**: Natural gradient from information geometry perspective.

**Paper**: [Amari 1998 - "Natural Gradient Works Efficiently in Learning"](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746)

**Topic**: Original natural gradient paper (from neural networks, not RL).

### Conjugate Gradient Method

**Resource**: [Jonathan Shewchuk - "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

**Why read**: Understand TRPO's optimization procedure.

### Fisher Information Matrix

**Tutorial**: [Martens 2014 - "New insights and perspectives on the natural gradient method"](https://arxiv.org/abs/1412.1193)

**Coverage**: Deep dive into natural gradients, K-FAC, and second-order methods.

## PPO in Practice

### RLHF and ChatGPT

**Blog**: [OpenAI - ChatGPT: Optimizing Language Models for Dialogue](https://openai.com/blog/chatgpt/)

**Paper**: [Ouyang et al. 2022 - "Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155)

**Key insight**: PPO used to fine-tune GPT-3 with human preferences. Most impactful RL application to date!

### Robotics Applications

**Paper**: [Andrychowicz et al. 2020 - "Learning Dexterous In-Hand Manipulation"](https://arxiv.org/abs/1808.00177)

**Application**: OpenAI trained robot hand to solve Rubik's Cube using PPO.

**Paper**: [Lee et al. 2020 - "Learning Quadrupedal Locomotion over Challenging Terrain"](https://arxiv.org/abs/2010.11251)

**Application**: PPO for quadruped robot locomotion.

## Debugging and Best Practices

### Debugging RL (John Schulman)
**Blog**: [joschu.net/blog/opinionated-guide-ml-research.html](http://joschu.net/blog/opinionated-guide-ml-research.html)

**From the author of TRPO/PPO**: Invaluable practical advice.

### PPO Debugging Checklist

1. **Monitor KL divergence**: Should be 0.01-0.05
2. **Track clipping fraction**: Should be 0.1-0.3
3. **Check explained variance**: Should be >0.7
4. **Plot approx_kl**: E[r - 1 - log r]
5. **Visualize value predictions vs returns**
6. **Monitor policy entropy** (should gradually decrease)
7. **Log gradient norms** (should be stable)

### Common PPO Bugs

**Blog**: [Costa Huang - Common PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

**Coverage**: 37 details that matter, ablation studies showing impact.

## Hyperparameter Tuning

### PPO Hyperparameters

**Default values** (work for many tasks):
```python
{
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'clip_range_vf': None,  # or 0.2
    'ent_coef': 0.0,  # or 0.01 for exploration
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'target_kl': None,  # or 0.01 for early stopping
}
```

**Tuning guide**:
- If unstable: Lower learning_rate (1e-4), lower n_epochs (4)
- If slow learning: Increase learning_rate (5e-4), increase n_steps (4096)
- If policy collapse: Add entropy bonus (ent_coef=0.01)
- If value function poor: Increase vf_coef (1.0), separate networks

### RL Baselines Zoo - Tuned Hyperparameters
**Repo**: [github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

**Folder**: `hyperparams/ppo.yml`

**What's included**: Tuned hyperparameters for 100+ environments.

## Environments for Practice

### Continuous Control (Recommended)
- **MuJoCo**: HalfCheetah-v4, Ant-v4, Humanoid-v4
- **PyBullet**: HalfCheetahBulletEnv-v0 (free MuJoCo alternative)
- **Gymnasium**: BipedalWalker-v3, LunarLanderContinuous-v2

### Discrete Action Spaces
- **Atari**: Pong, Breakout, BeamRider
- **Classic**: CartPole, LunarLander

### Hard Exploration
- **Montezuma's Revenge**: Sparse rewards, needs curiosity
- **Hard-Exploration Atari**: Venture, PrivateEye

## Advanced Variants

### PPG (PPO + Auxiliary Tasks)
**Paper**: [Cobbe et al. 2021 - "Phasic Policy Gradient"](https://arxiv.org/abs/2009.04416)

**Contribution**: Separate policy and value training phases, auxiliary tasks.

**Impact**: SOTA on Procgen benchmark.

### DAAC (PPO with Decoupled Advantage)
**Paper**: [Raileanu & Fergus 2021 - "Decoupling Value and Policy for Generalization"](https://arxiv.org/abs/2102.10330)

**Contribution**: Separate encoders for policy and value, better generalization.

### APE-X PPO (Distributed)
**Paper**: [Espeholt et al. 2018 - "IMPALA: Scalable Distributed Deep-RL"](https://arxiv.org/abs/1802.01561)

**Contribution**: Distributed PPO at scale (thousands of actors).

## Study Plan

### Week 13 Day 1-2: Foundations
1. Read TRPO paper (Sections 1-4)
2. Watch CS285 Lecture 9
3. Understand monotonic improvement theory

### Week 13 Day 3-4: PPO Deep Dive
4. Read PPO paper thoroughly
5. Read "37 Implementation Details" blog
6. Study CleanRL PPO code

### Week 13 Day 5-6: Implementation
7. Implement PPO from scratch
8. Test on CartPole, then LunarLander
9. Debug using diagnostic metrics

### Week 13 Day 7: Advanced
10. Experiment with continuous control (HalfCheetah)
11. Try hyperparameter variations
12. Read about RLHF applications

## Quick Reference

### Key Equations

**TRPO Objective**:
```
max E[r(θ) * A]
s.t. E[KL(π_old || π_new)] ≤ δ
```

**PPO-Clip Objective**:
```
L = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]
```

**Probability Ratio**:
```
r(θ) = π(a|s; θ) / π(a|s; θ_old)
```

### Typical Hyperparameters

**For continuous control**:
```python
{
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10,
    'learning_rate': 3e-4,
    'clip_range': 0.2,
    'gae_lambda': 0.95,
    'ent_coef': 0.0,
    'vf_coef': 0.5,
}
```

**For Atari**:
```python
{
    'n_steps': 128,
    'batch_size': 256,
    'n_epochs': 4,
    'learning_rate': 2.5e-4,
    'clip_range': 0.1,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
}
```

## Community

### Reddit
- r/reinforcementlearning
- r/MachineLearning

### Twitter
Follow:
- @johnschulman2 (TRPO/PPO author)
- @OpenAI
- @berkeleyai

### Discord
- OpenAI Scholars
- Weights & Biases Community

## Next Week Preview

**Week 14: Continuous Control (DDPG, TD3, SAC)**
- Off-policy actor-critic
- Deterministic policies
- Maximum entropy RL
- State-of-the-art for robotics

**Key difference**: PPO is on-policy, next week covers off-policy methods (much more sample-efficient).

## Final Tips

1. **Start with CleanRL**: Best code for learning
2. **Monitor metrics**: KL, clipping_fraction, explained_variance
3. **Use Stable-Baselines3 as baseline**: Verify your implementation
4. **Read "37 Details" blog**: Implementation matters!
5. **Be patient**: PPO takes millions of steps for hard tasks
6. **Tune carefully**: Small hyperparameter changes = big impact
7. **Understand why it works**: Don't just cargo-cult the code

**Remember**: PPO is simple in concept but requires attention to implementation details. Master it and you can solve most RL problems!
