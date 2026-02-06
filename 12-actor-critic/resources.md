# Week 12 Resources: Actor-Critic Methods

## Primary Textbooks and Lectures

### CS285 (Berkeley Deep RL) - Lectures 5-6
- **Lecture 5: Policy Gradients**
  - Actor-critic introduction
  - Reducing variance with critics
  - On-policy vs off-policy

- **Lecture 6: Actor-Critic Algorithms**
  - Advantage estimation
  - Baseline design
  - Architecture choices

**Videos**: [CS285 Fall 2023 Playlist](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps)

**Slides**: [rail.eecs.berkeley.edu/deeprlcourse](http://rail.eecs.berkeley.edu/deeprlcourse/)

**Why watch**: Best modern treatment of actor-critic. Levine is excellent at connecting theory to practice.

### David Silver's RL Course - Lecture 7
- **Lecture 7: Policy Gradient Methods**
  - Second half covers actor-critic preview
  - Bias-variance trade-off in advantage estimation
  - Critic architectures

**Video**: [YouTube Lecture 7](https://www.youtube.com/watch?v=KHZVXao4qXs)

**Why watch**: Clear mathematical foundations. Good complement to CS285.

### Sutton & Barto - Chapter 13
- **Chapter 13: Policy Gradient Methods**
  - Section 13.5: Actor-Critic Methods
  - Connection to classic actor-critic (1980s)
  - Eligibility traces perspective

**Download**: [Free PDF](http://incompleteideas.net/book/the-book-2nd.html)

**Why read**: Historical perspective, connects to classic RL theory.

## Key Papers

### 1. Mnih et al. 2016: Asynchronous Methods for Deep RL (A3C)

**Paper**: [ArXiv](https://arxiv.org/abs/1602.01783)

**Contribution**:
- Introduced A3C: asynchronous advantage actor-critic
- Showed parallelism can replace experience replay
- Achieved state-of-the-art on Atari (2016)

**Key innovations**:
- Multiple workers collect data asynchronously
- No replay buffer needed (unlike DQN)
- Linear in speed-up with number of workers

**Impact**: Revolutionary at the time, though A2C (synchronous) now preferred.

**Read if**: You want to understand the history of modern RL. Sections 3-4 on the algorithm are most important.

### 2. Schulman et al. 2015: High-Dimensional Continuous Control Using GAE

**Paper**: [ArXiv](https://arxiv.org/abs/1506.02438)

**Contribution**:
- Introduced Generalized Advantage Estimation (GAE)
- Analyzed bias-variance trade-off rigorously
- Showed single λ parameter provides smooth interpolation

**Key equations**:
```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

**Impact**: GAE is now standard in all modern policy gradient methods (PPO, TRPO, SAC). One of the most influential RL papers.

**Read if**: You want deep understanding of advantage estimation. Section 3 on GAE is essential.

### 3. Williams 1992: Simple Statistical Gradient-Following Algorithms

**Paper**: [REINFORCE](https://link.springer.com/article/10.1007/BF00992696)

**Contribution**:
- Original policy gradient algorithm (REINFORCE)
- Introduced baseline for variance reduction
- Actor-critic roots

**Historical importance**: Foundation for all modern policy gradient methods.

**Read if**: You want historical perspective and original proofs.

### 4. Konda & Tsitsiklis 1999: Actor-Critic Algorithms

**Paper**: [MIT Technical Report](https://web.mit.edu/jnt/www/Papers/J094-03-kon-actors.pdf)

**Contribution**:
- Convergence proofs for actor-critic
- Analysis of two-timescale updates
- Theoretical foundations

**Read if**: You want rigorous theory. Heavy on math, but important for understanding guarantees.

## Blog Posts and Tutorials

### Lilian Weng: Policy Gradient Algorithms

**URL**: [lilianweng.github.io/posts/2018-04-08-policy-gradient/](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**Coverage**:
- Actor-critic section covers advantage estimation
- GAE explained with visuals
- Comparison of different advantage estimators

**Why read**: Best single-page reference. Excellent for quick lookup of formulas.

### OpenAI Spinning Up: Intro to Policy Optimization

**URL**: [spinningup.openai.com/en/latest/spinningup/rl_intro3.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

**Coverage**:
- Section on advantage functions
- Practical implementation advice
- Code examples in PyTorch

**Why read**: Practical focus, good for implementation.

### OpenAI Spinning Up: Vanilla Policy Gradient (VPG)

**URL**: [spinningup.openai.com/en/latest/algorithms/vpg.html](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

**Coverage**:
- Complete VPG implementation (REINFORCE + baseline)
- This is essentially a simple actor-critic
- Well-commented PyTorch code

**Why read**: Reference implementation for your own code.

### Jonathan Hui: RL — Policy Gradient Explained

**URL**: [jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146](https://jonathan-hui.medium.com/rl-policy-gradients-explained-9b13b688b146)

**Coverage**:
- Visual explanations of actor-critic
- Advantage function intuition
- Comparison with value-based methods

**Why read**: Good visuals and intuitive explanations.

### Arthur Juliani: Simple Reinforcement Learning with TensorFlow: Part 8 - A3C

**URL**: [medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

**Coverage**:
- Gentle introduction to A3C
- Code walkthrough (TensorFlow, but concepts translate)
- Visualizations of learning process

**Why read**: Beginner-friendly, good for first exposure to actor-critic.

## Code Repositories

### OpenAI Baselines - A2C

**Repo**: [github.com/openai/baselines](https://github.com/openai/baselines)

**Specific files**:
- `baselines/a2c/a2c.py` - Main A2C algorithm
- `baselines/common/policies.py` - Policy networks
- `baselines/common/runners.py` - Rollout collection

**What's good**:
- Production-quality implementation
- Handles many edge cases
- Includes useful utilities

**What's challenging**:
- Complex codebase (many abstractions)
- Hard to read for beginners
- TensorFlow 1.x (somewhat dated)

**How to use**: Reference for implementation details after understanding basics.

### CleanRL - A2C

**Repo**: [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)

**Specific files**:
- `cleanrl/a2c.py` - Single-file A2C
- `cleanrl/a2c_atari.py` - A2C for Atari
- `cleanrl/a2c_continuous_action.py` - Continuous control

**What's good**:
- Single-file implementations (easy to understand)
- Modern PyTorch style
- Extensive benchmarking and documentation
- Active development

**Why use**: Best for learning. Start here for your own implementation.

**Getting started**:
```bash
git clone https://github.com/vwxyzjn/cleanrl
cd cleanrl
pip install -r requirements.txt
python cleanrl/a2c.py --env-id CartPole-v1
```

### Stable-Baselines3 - A2C

**Repo**: [github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

**Docs**: [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/)

**What's good**:
- Clean, well-documented API
- Easy to use for experiments
- Good for baselines and comparisons
- Active maintenance

**Example usage**:
```python
from stable_baselines3 import A2C

model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(total_timesteps=100000)
model.save("a2c_cartpole")
```

**Why use**: Quick experiments, baseline comparisons, production use.

### PyTorch Actor-Critic Example

**Repo**: [github.com/pytorch/examples/tree/main/reinforcement_learning](https://github.com/pytorch/examples/tree/main/reinforcement_learning)

**What's good**:
- Official PyTorch example
- Minimal but complete
- Good starting point

**Why use**: Simple reference implementation for learning.

## Video Tutorials

### DeepMind x UCL RL Lecture Series - Actor-Critic

**Video**: [YouTube](https://www.youtube.com/watch?v=TCCjZe0y4Qc)

**Coverage**:
- Modern perspective from DeepMind researchers
- Connections to recent work (IMPALA, V-trace)
- Advanced topics

**Why watch**: Cutting-edge perspective, complements academic lectures.

### Arxiv Insights: A3C Explained

**Video**: [YouTube](https://www.youtube.com/watch?v=OcIx_TBu90Q)

**Coverage**:
- Visual explanation of A3C
- Comparison with DQN
- Parallel training visualization

**Why watch**: Good visuals, accessible for beginners.

## Practical Guides and Debugging

### Debugging RL Agents

**Blog**: [andyljones.com/posts/rl-debugging.html](https://andyljones.com/posts/rl-debugging.html)

**Coverage**:
- Common bugs in actor-critic implementations
- Diagnostic plots and sanity checks
- How to interpret learning curves

**Why read**: Will save you hours of debugging time. Essential practical guide.

### Tips for Reproducing RL Papers

**Blog**: [amid.fish/reproducing-deep-rl](https://amid.fish/reproducing-deep-rl)

**Coverage**:
- Hyperparameter sensitivity
- Random seed effects
- Evaluation protocols

**Why read**: Understand why your results might differ from papers.

### Nuts and Bolts of Deep RL

**Blog**: [joschu.net/blog/opinionated-guide-ml-research.html](http://joschu.net/blog/opinionated-guide-ml-research.html)

**Coverage** (RL-relevant sections):
- Practical tips for RL research
- Debugging strategies
- Experiment design

**Why read**: From John Schulman (PPO/TRPO author), invaluable practical advice.

## Implementation Guides

### Actor-Critic Implementation Checklist

Key components to implement:

1. **Network Architecture**
   - [ ] Shared encoder or separate networks
   - [ ] Proper initialization (orthogonal for RL)
   - [ ] Separate actor and critic heads

2. **Advantage Estimation**
   - [ ] GAE implementation with proper indexing
   - [ ] Handle episode termination correctly
   - [ ] Advantage normalization

3. **Loss Functions**
   - [ ] Actor: policy gradient with advantages
   - [ ] Critic: value function loss (MSE or Huber)
   - [ ] Entropy bonus for exploration

4. **Training Loop**
   - [ ] Vectorized environment support
   - [ ] Rollout collection
   - [ ] Gradient clipping
   - [ ] Proper logging

5. **Debugging Tools**
   - [ ] Plot value predictions vs. actual returns
   - [ ] Monitor policy entropy
   - [ ] Track gradient norms
   - [ ] Log explained variance of critic

### Common Implementation Mistakes

1. **Wrong GAE computation**
   - Forgetting to iterate backwards
   - Not handling episode boundaries
   - Using wrong next_value for terminal states

2. **Gradient issues**
   - Not detaching advantages when computing actor loss
   - Forgetting gradient clipping
   - Wrong loss signs (maximize vs. minimize)

3. **Architecture problems**
   - Shared network with conflicting gradients
   - Poor initialization
   - Wrong activation functions

4. **Training issues**
   - Not normalizing advantages
   - Wrong learning rates (actor vs. critic)
   - Insufficient entropy bonus

## Environments for Practice

### Easy (Start Here)
- **CartPole-v1**: Simple, fast iteration
- **LunarLander-v2**: Slightly harder, good for A2C

### Medium
- **BipedalWalker-v3**: Continuous control
- **Atari games**: Pong, Breakout (need CNN)

### Advanced
- **MuJoCo**: HalfCheetah, Ant, Humanoid
- **PyBullet**: Free MuJoCo alternatives

## Advanced Topics

### IMPALA: Scalable Distributed Deep-RL

**Paper**: [ArXiv](https://arxiv.org/abs/1802.01561)

**Contribution**:
- V-trace for off-policy correction
- Highly scalable actor-critic
- Separates acting and learning

**Read if**: Interested in distributed training at scale.

### Actor-Critic with Experience Replay (ACER)

**Paper**: [ArXiv](https://arxiv.org/abs/1611.01224)

**Contribution**:
- Off-policy actor-critic
- Importance sampling corrections
- Replay buffer for sample efficiency

**Read if**: Want to understand off-policy actor-critic.

### Reactor: Sample-Efficient Off-Policy Actor-Critic

**Paper**: [ArXiv](https://arxiv.org/abs/1704.04651)

**Contribution**:
- Combines multiple techniques for efficiency
- Retrace operator for off-policy
- Distributed prioritized replay

**Read if**: Interested in sample-efficient actor-critic.

## Research Papers (Advanced)

### Natural Actor-Critic

**Paper**: [Peters & Schaal 2008](https://www.sciencedirect.com/science/article/pii/S0925231208000532)

**Topic**: Natural gradient for actor-critic

**Preview for Week 13**: Foundation for TRPO.

### Deterministic Policy Gradient

**Paper**: [Silver et al. 2014](http://proceedings.mlr.press/v32/silver14.pdf)

**Topic**: Deterministic policies for continuous control

**Preview for Week 14**: Foundation for DDPG.

### Soft Actor-Critic

**Paper**: [Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)

**Topic**: Maximum entropy actor-critic

**Preview for Week 14**: State-of-the-art for continuous control.

## Community and Discussion

### Reddit

- **r/reinforcementlearning**: General RL discussion
- **r/MachineLearning**: Broader ML community, RL paper discussions

### Discord Servers

- **OpenAI Scholars Discord**: RL learning community
- **Weights & Biases Community**: ML engineering and RL

### Twitter

Follow for latest research:
- @OpenAIResearch
- @DeepMindAI
- @berkeleyai
- @josh_tobin_ (debugging RL advice)
- @sferika (Sergey Levine, CS285 instructor)

### Conferences

- **NeurIPS**: Top venue for RL research
- **ICML**: Machine learning, including RL
- **ICLR**: Deep learning and RL
- **CoRL**: Conference on Robot Learning (applied RL)

## Benchmarking and Evaluation

### RL Baselines Zoo

**Repo**: [github.com/DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

**What's included**:
- Tuned hyperparameters for many environments
- Training scripts and configurations
- Benchmark results

**Why use**: Compare your implementation against known-good baselines.

### Gym and Gymnasium

**Documentation**: [gymnasium.farama.org](https://gymnasium.farama.org/)

**Environments**:
- Classic control: CartPole, Pendulum, MountainCar
- Box2D: LunarLander, BipedalWalker, CarRacing
- Atari: Pong, Breakout, Space Invaders

**Why use**: Standard RL benchmarks, easy to get started.

## Study Plan

### Week 12 Day 1-2: Fundamentals
1. CS285 Lecture 5-6 (watch at 1.5x if comfortable)
2. Read GAE paper (Schulman et al. 2015) - focus on Section 3
3. Review Lilian Weng blog for quick reference

### Week 12 Day 3-4: Implementation
4. Read CleanRL A2C code thoroughly
5. Implement A2C from scratch on CartPole
6. Debug using diagnostic plots

### Week 12 Day 5-6: Experiments
7. Apply A2C to LunarLander-v2
8. Experiment with different λ values (0.9, 0.95, 0.99)
9. Compare shared vs. separate architectures

### Week 12 Day 7: Deep Dive
10. Read A3C paper (Mnih et al. 2016)
11. Understand why A2C is preferred over A3C
12. Prepare for Week 13 (TRPO/PPO) by understanding trust regions concept

## Quick Reference

### Key Equations

**Advantage Function**:
```
A(s,a) = Q(s,a) - V(s) ≈ r + γV(s') - V(s)
```

**GAE**:
```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
           = δ_t + γλ A_{t+1}^GAE(λ)
```

**A2C Losses**:
```
L_actor = -E[log π(a|s) · A(s,a)]
L_critic = E[(V(s) - G_t)²]
L_total = L_actor + c_v L_critic - c_e H(π)
```

### Typical Hyperparameters

```python
{
    'num_envs': 8,
    'n_steps': 5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'learning_rate': 3e-4,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
}
```

## Recommended Path Forward

After completing Week 12:

**Week 13: TRPO & PPO**
- Build on actor-critic foundations
- Add trust regions and clipped objectives
- These are the most widely used RL algorithms today

**Week 14: Continuous Control**
- Deterministic policies (DDPG, TD3)
- Maximum entropy (SAC)
- State-of-the-art for robotics

**Beyond**:
- Model-based RL
- Offline RL
- Multi-agent RL
- Hierarchical RL

## Final Tips

1. **Implement from scratch**: Don't just use libraries. Understanding comes from implementation.

2. **Start simple**: Get CartPole working before moving to harder environments.

3. **Debug systematically**: Use diagnostic plots, sanity checks, and gradual complexity.

4. **Compare with baselines**: Use Stable-Baselines3 to verify your implementation.

5. **Tune hyperparameters**: A2C is sensitive. Expect to spend time tuning.

6. **Visualize learning**: Plot value predictions, policy entropy, advantage distributions.

7. **Read papers actively**: Implement key algorithms while reading.

8. **Join community**: Ask questions on Reddit, Discord, Stack Overflow.

9. **Be patient**: RL is hard. Bugs are subtle. Learning curves are noisy.

10. **Have fun**: Actor-critic is where RL gets really powerful. Enjoy the journey!

## Next Week Preview

Week 13 covers TRPO and PPO, which improve upon A2C by:
- Constraining policy updates (trust regions)
- Achieving more stable training
- Becoming the default choice for most RL applications

Key papers to preview:
- Schulman et al. 2015: TRPO
- Schulman et al. 2017: PPO

Start thinking about: "Why can large policy updates be harmful?"
