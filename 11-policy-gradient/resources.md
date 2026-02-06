# Week 11 Resources: Policy Gradient Methods

## Primary Textbooks

### Sutton & Barto: Reinforcement Learning (2nd Edition)
- **Chapter 13: Policy Gradient Methods**
  - Section 13.1: Policy Approximation and its Advantages
  - Section 13.2: The Policy Gradient Theorem
  - Section 13.3: REINFORCE: Monte Carlo Policy Gradient
  - Section 13.4: REINFORCE with Baseline
  - Section 13.5: Actor-Critic Methods (preview for next week)
  - Section 13.6: Policy Gradient for Continuing Problems
  - Section 13.7: Policy Parameterization for Continuous Actions

**Why read this**: Most comprehensive and accessible introduction to policy gradients. Excellent mathematical rigor with clear intuitions.

**Download**: [Free PDF from the authors](http://incompleteideas.net/book/the-book-2nd.html)

### David Silver's RL Course - Lecture 7
- **Lecture 7: Policy Gradient Methods**
  - Policy objective functions
  - Finite difference vs. likelihood ratio policy gradient
  - Score function and log-derivative trick
  - Policy gradient theorem proof
  - REINFORCE algorithm
  - Actor-critic preview

**Video**: [YouTube Lecture 7](https://www.youtube.com/watch?v=KHZVXao4qXs)

**Slides**: [Lecture 7 Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/pg.pdf)

**Why watch this**: Clear mathematical derivations with excellent intuition. Silver's proofs are more detailed than S&B.

### CS285 (Berkeley Deep RL) - Lecture 5
- **Lecture 5: Policy Gradients**
  - Deriving policy gradient estimators
  - Variance reduction techniques in depth
  - Causality and reward-to-go
  - Baselines and advantages
  - Off-policy policy gradients (brief)

**Video**: [CS285 Fall 2023 Lecture 5](https://www.youtube.com/watch?v=AKbX1Zvo7r8&list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps&index=5)

**Slides**: [Lecture 5 Slides](http://rail.eecs.berkeley.edu/deeprlcourse/)

**Why watch this**: Modern perspective with focus on practical implementation. Best treatment of variance reduction.

## Blog Posts and Tutorials

### Lilian Weng: Policy Gradient Algorithms
- **URL**: [lilianweng.github.io/posts/2018-04-08-policy-gradient/](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

**Coverage**:
- Comprehensive overview from REINFORCE to modern methods
- Mathematical derivations with clear notation
- Comparison of different policy gradient variants
- Connection to natural gradients and trust regions

**Why read**: Best single-post overview of the policy gradient landscape. Excellent reference for formulas.

### OpenAI Spinning Up: Intro to Policy Optimization
- **URL**: [spinningup.openai.com/en/latest/spinningup/rl_intro3.html](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

**Coverage**:
- Derivation of policy gradient theorem
- Log-derivative trick explained intuitively
- Practical implementation advice
- Connection to later algorithms (TRPO, PPO)

**Why read**: Very practical focus, good for implementation. Clear code examples.

### OpenAI Spinning Up: Vanilla Policy Gradient (VPG)
- **Documentation**: [spinningup.openai.com/en/latest/algorithms/vpg.html](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

**Coverage**:
- Complete VPG (REINFORCE with baseline) algorithm
- PyTorch implementation with detailed comments
- Hyperparameter choices and tuning advice
- Performance benchmarks on MuJoCo tasks

**Why read**: Reference implementation. Shows how to structure a policy gradient codebase.

### Andrej Karpathy: Deep Reinforcement Learning - Pong from Pixels
- **Blog**: [karpathy.github.io/2016/05/31/rl/](http://karpathy.github.io/2016/05/31/rl/)

**Coverage**:
- Policy gradients explained from first principles
- Minimal 130-line Python implementation
- Training Pong with REINFORCE
- Intuitive explanations and visualizations

**Why read**: Best intuitive introduction. Shows policy gradients can be simple. Great for beginners.

## Papers (Historical and Foundational)

### Williams 1992: Simple Statistical Gradient-Following Algorithms
- **Paper**: [REINFORCE Algorithm](https://link.springer.com/article/10.1007/BF00992696)

**Contribution**: Original REINFORCE algorithm and the policy gradient theorem.

**Why read**: Historical perspective. Surprisingly readable for a 1992 paper. Shows policy gradients aren't new.

### Sutton et al. 1999: Policy Gradient Methods for RL with Function Approximation
- **Paper**: [Policy Gradient Theorem](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

**Contribution**: Formal proof of policy gradient theorem with function approximation. Foundation for all modern work.

**Why read**: Rigorous theoretical foundation. Important for understanding convergence guarantees.

### Peters & Schaal 2008: Natural Actor-Critic
- **Paper**: [Natural Gradients in RL](https://www.sciencedirect.com/science/article/pii/S0925231208000532)

**Contribution**: Natural gradient perspective on policy optimization. Lays groundwork for TRPO/PPO.

**Why read**: Important theoretical bridge to Week 13. Shows why policy gradient can be unstable.

## Code Repositories

### OpenAI Spinning Up - VPG Implementation
- **Repo**: [github.com/openai/spinningup](https://github.com/openai/spinningup)
- **Specific file**: `spinup/algos/pytorch/vpg/vpg.py`

**What's good**:
- Production-quality implementation
- Well-documented and readable
- Includes logging, saving, evaluation
- Performance matches published results

**How to use**: Clone and run experiments, read code for implementation details.

### CleanRL - REINFORCE Implementation
- **Repo**: [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl)
- **Specific file**: `cleanrl/reinforce_continuous_action.py`

**What's good**:
- Single-file implementations (easier to understand)
- Minimal dependencies
- Modern PyTorch style
- Extensive hyperparameter tuning

**How to use**: Start here for learning. Single files are easier to read than OpenAI's modular code.

### Stable-Baselines3 - PPO (includes policy gradient foundations)
- **Repo**: [github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- **Docs**: [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/)

**What's good**:
- Production-ready library
- Easy to use for experiments
- Good for comparing your implementation

**How to use**: Use as baseline to test environments, compare performance.

## Video Tutorials and Lectures

### DeepMind x UCL RL Lecture Series - Policy Gradients
- **Video**: [YouTube DeepMind Lecture](https://www.youtube.com/watch?v=TCCjZe0y4Qc)

**Coverage**: Modern perspective from DeepMind researchers, connections to recent work.

**Why watch**: Complements Silver's lectures with newer insights.

### MIT 6.S091: Introduction to Deep RL and Control
- **Lecture 3**: [Policy Gradients](https://www.youtube.com/watch?v=MQ6pP65o7OM)

**Coverage**: Practical focus, demo of training agents in real-time.

**Why watch**: Engaging presentation, good for motivation.

## Practical Guides

### GitHub: Policy Gradient Implementations Comparison
- Various repositories comparing implementations:
  - [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
  - [rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)

**What's useful**: See different implementation choices, performance comparisons.

### Reddit r/reinforcementlearning
- **URL**: [reddit.com/r/reinforcementlearning](https://www.reddit.com/r/reinforcementlearning/)

**What's useful**: Community help with implementation issues, paper discussions.

## Interactive Resources

### Distill.pub Articles
- While no specific policy gradient article exists, related articles on:
  - Optimization visualization
  - Neural network interpretation

**URL**: [distill.pub](https://distill.pub)

**Why explore**: Excellent visualizations that build intuition.

### Visualizing Policy Gradients
- **OpenAI Gym Leaderboard**: See what's possible
- **Weights & Biases Reports**: Example training runs with visualizations

## Supplementary Topics

### Continuous Action Spaces
- **Normal (Gaussian) Distribution**: For unbounded continuous actions
- **Beta Distribution**: For bounded actions (alternative to squashed Gaussian)
- **Mixture of Gaussians**: For multimodal action distributions

**Resources**:
- PyTorch `torch.distributions` documentation
- OpenAI Spinning Up documentation on action spaces

### Variance Reduction in Monte Carlo Methods
- **General statistics resources**: Understanding importance sampling, control variates
- **Connection to statistics**: Policy gradient as stochastic optimization

## Practice Environments

### Easy (Start Here)
- **CartPole-v1**: Discrete actions, short episodes, fast iteration
- **LunarLander-v2**: Slightly harder, discrete actions
- **Pendulum-v1**: Continuous actions, simple dynamics

### Medium
- **BipedalWalker-v3**: Continuous control, longer episodes
- **MountainCarContinuous-v0**: Sparse rewards, tests exploration

### Hard (After Mastering Basics)
- **MuJoCo environments**: Industry standard (requires license or use new Mujoco)
- **PyBullet environments**: Free alternative to MuJoCo
- **Custom environments**: Build your own

## Common Pitfalls and Debugging

### Blog Post: Debugging RL Agents
- **URL**: [andyljones.com/posts/rl-debugging.html](https://andyljones.com/posts/rl-debugging.html)

**Coverage**: Common bugs, how to diagnose them, sanity checks.

**Why read**: Will save you hours of debugging. Essential practical guide.

### Spinning Up: Running Experiments
- **URL**: [spinningup.openai.com/en/latest/spinningup/spinningup.html](https://spinningup.openai.com/en/latest/spinningup/spinningup.html)

**Coverage**: Hyperparameter tuning, evaluation, plotting results.

**Why read**: Learn to run experiments properly from the start.

## Advanced Topics (Optional)

### Off-Policy Policy Gradients
- Importance sampling corrections
- Connection to actor-critic and Q-learning

**Resources**: CS285 later lectures, research papers

### Natural Gradients and Trust Regions
- Fisher information matrix
- KL divergence constraints
- Preview of TRPO (Week 13)

**Resources**: Peters & Schaal 2008, CS285 Lecture 9

### Pathwise Derivative Policy Gradients
- Reparameterization trick
- Deterministic policy gradients
- Preview of DDPG/SAC (Week 14)

**Resources**: Deterministic Policy Gradient paper (Silver et al. 2014)

## Summary: Recommended Reading Order

### Week 11 Day 1-2: Foundations
1. Sutton & Barto Chapter 13 (Sections 13.1-13.4)
2. Andrej Karpathy blog post
3. David Silver Lecture 7

### Week 11 Day 3-4: Implementation
4. OpenAI Spinning Up VPG documentation
5. CleanRL REINFORCE code walkthrough
6. Implement REINFORCE on CartPole

### Week 11 Day 5-6: Deep Dive
7. CS285 Lecture 5 (variance reduction)
8. Lilian Weng blog post
9. Experiment with baselines and advantage estimation

### Week 11 Day 7: Practice and Extensions
10. Try LunarLander and Pendulum
11. Read policy gradient theorem paper (optional)
12. Prepare for actor-critic (Week 12)

## Quick Reference

### Key Equations
- Policy Gradient Theorem: See S&B Chapter 13.2
- REINFORCE Update: See S&B Chapter 13.3
- Baseline Proof: See CS285 Lecture 5 slides

### Cheat Sheets
- [RL Algorithms Cheat Sheet](https://github.com/udacity/deep-reinforcement-learning)
- Create your own as you learn!

### Community
- **Discord**: Various RL learning communities
- **Twitter**: Follow @OpenAIResearch, @DeepMindAI, @berkeleyai for latest research
- **ArXiv**: Subscribe to cs.LG and cs.AI for new papers

## Next Week Preview

Week 12 will cover Actor-Critic methods, which combine policy gradients with value function bootstrapping for much lower variance. Key papers to preview:
- Mnih et al. 2016: A3C
- Schulman et al. 2015: GAE

Start thinking about how to reduce variance beyond baselines.
