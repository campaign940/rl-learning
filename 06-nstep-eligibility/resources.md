# Week 6 Resources: n-step Methods & Eligibility Traces

## Primary Textbooks

### Sutton & Barto: Reinforcement Learning - An Introduction (2nd Edition)

**Chapter 7: n-step Bootstrapping**
- [Full Book PDF](http://incompleteideas.net/book/RLbook2020.pdf)
- [Chapter 7 Direct Link](http://incompleteideas.net/book/RLbook2020.pdf#page=165)
- Essential sections:
  - 7.1: n-step TD Prediction
  - 7.2: n-step SARSA
  - 7.3: n-step Off-policy Learning
  - 7.4: Per-decision Methods with Control Variates

**Chapter 12: Eligibility Traces**
- [Chapter 12 Direct Link](http://incompleteideas.net/book/RLbook2020.pdf#page=305)
- Must-read sections:
  - 12.1: The λ-return
  - 12.2: TD(λ)
  - 12.3: n-step Truncated λ-return Methods
  - 12.4: Redoing Updates: The Online λ-return Algorithm
  - 12.5: True Online TD(λ)
  - 12.7: SARSA(λ)
  - 12.8: Variable λ and γ
  - 12.10: Implementation Issues

### David Silver's RL Course

- **Lecture 4: Model-Free Prediction (Second Half)**
  - [Lecture Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)
  - [Video Lecture](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
  - Topics: TD(λ), Forward view, Backward view, Eligibility traces
  - Slides 39-57 focus on eligibility traces

## University Courses

### CS234: Reinforcement Learning (Stanford)
- **Lecture 5: Value Function Approximation**
  - Covers n-step methods and traces with function approximation
  - [Course Website](http://web.stanford.edu/class/cs234/)

### CS285: Deep Reinforcement Learning (UC Berkeley)
- **Lecture 8: Advanced Policy Gradients**
  - Discusses n-step returns in policy gradient context
  - [Course Materials](http://rail.eecs.berkeley.edu/deeprlcourse/)

### UCL Course on RL (David Silver)
- **Lecture 4 Video** (~1:30:00 onward for eligibility traces)
  - [YouTube](https://www.youtube.com/watch?v=PnHCvfgC_ZA&t=5400s)
  - Excellent visual explanations

## Research Papers

### Foundational Papers

- **Sutton (1988): Learning to Predict by the Methods of Temporal Differences**
  - [Paper PDF](https://link.springer.com/article/10.1007/BF00115009)
  - Original TD(λ) paper
  - Historical importance

- **Watkins (1989): Learning from Delayed Rewards (PhD Thesis)**
  - [Thesis PDF](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)
  - Chapter on eligibility traces
  - Foundational work

- **Peng & Williams (1996): Incremental Multi-Step Q-Learning**
  - [Paper PDF](https://link.springer.com/article/10.1007/BF00114731)
  - n-step Q-learning algorithms

### Modern Papers

- **van Seijen & Sutton (2014): True Online TD(λ)**
  - [Paper PDF](https://proceedings.mlr.press/v32/seijen14.pdf)
  - Exact equivalence between forward and backward view
  - State-of-the-art TD(λ) algorithm

- **Sutton et al. (2016): A New Approach to Eligibility Traces**
  - [Paper PDF](https://arxiv.org/pdf/1502.07001.pdf)
  - Emphasis-based traces
  - Theoretical improvements

- **Munos et al. (2016): Safe and Efficient Off-Policy Reinforcement Learning (Retrace)**
  - [Paper PDF](https://arxiv.org/pdf/1606.02647.pdf)
  - Off-policy n-step algorithm
  - Used in modern deep RL

### Applications in Deep RL

- **Mnih et al. (2016): Asynchronous Methods for Deep RL (A3C)**
  - [Paper PDF](https://arxiv.org/pdf/1602.01783.pdf)
  - Uses n-step returns extensively
  - Major breakthrough in deep RL

- **Schulman et al. (2017): Proximal Policy Optimization (PPO)**
  - [Paper PDF](https://arxiv.org/pdf/1707.06347.pdf)
  - Uses generalized advantage estimation (related to λ-returns)
  - Most popular deep RL algorithm

- **Hessel et al. (2018): Rainbow DQN**
  - [Paper PDF](https://arxiv.org/pdf/1710.02298.pdf)
  - Combines n-step learning with other improvements
  - State-of-the-art value-based deep RL

## Blog Posts and Tutorials

### Comprehensive Tutorials

- **Lil'Log: Eligibility Traces**
  - [RL Introduction](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
  - Scroll to TD(λ) section
  - Clear mathematical explanations

- **Towards Data Science: Understanding Eligibility Traces**
  - [Article](https://towardsdatascience.com/eligibility-traces-in-reinforcement-learning-a6b458c019ff)
  - Practical Python examples

- **Medium: n-step Methods Explained**
  - [Article](https://medium.com/@jonathan_hui/rl-n-step-bootstrapping-and-eligibility-traces-5d5e9a5aebd)
  - Visual diagrams
  - Code snippets

### Intuitive Explanations

- **RL Weekly: TD(λ) Deep Dive**
  - [Blog Post](https://seungjaeryanlee.github.io/rlweekly/14/)
  - Beginner-friendly introduction

- **The RL Boosted: Eligibility Traces Tutorial**
  - Multiple examples comparing TD(0), MC, and TD(λ)

## Implementation Resources

### Code Repositories

- **Dennybritz RL Repository**
  - [GitHub](https://github.com/dennybritz/reinforcement-learning)
  - Clean implementations:
    - [n-step TD Prediction](https://github.com/dennybritz/reinforcement-learning/tree/master/TD)
    - SARSA with eligibility traces

- **Sutton & Barto Official Code**
  - [GitHub](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
  - Python implementations for all textbook examples:
    - [Chapter 7 Code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter07)
    - [Chapter 12 Code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter12)
  - 19-state random walk implementation
  - Mountain Car with eligibility traces

- **CleanRL: True Online TD(λ)**
  - [GitHub](https://github.com/vwxyzjn/cleanrl)
  - Modern clean implementations

### Example Environments

**19-State Random Walk** (Classic benchmark):
```python
# Available in S&B code repository
# Perfect for comparing n-step methods
```

**Mountain Car** (Sparse rewards):
```python
import gymnasium as gym
env = gym.make('MountainCar-v0')
# Great for testing eligibility traces
```

**Custom Grid Worlds**:
- Build your own with loops and sparse rewards
- Test accumulating vs replacing traces

## Interactive Visualizations

### Online Demos

- **Eligibility Traces Visualization**
  - [Demo by Patrick Coady](https://pat-coady.github.io/rl/2016/09/24/eligibility-traces.html)
  - Interactive trace decay visualization
  - Forward vs backward view comparison

- **n-step TD Interactive**
  - Adjust n slider and see learning curves
  - Compare different n values in real-time

### Notebooks

- **Google Colab: TD(λ) Tutorial**
  - Step-by-step implementation
  - Visualizations of traces and learning curves

- **Jupyter Notebooks in S&B Repository**
  - Run experiments from textbook
  - Modify and experiment with parameters

## Video Resources

### Lecture Series

- **David Silver's Lecture 4**
  - [YouTube](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
  - Timestamp 1:30:00 - 2:00:00 for eligibility traces
  - Best video explanation

- **DeepMind x UCL RL Lecture Series**
  - [Full Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
  - Lecture 4: Model-Free Prediction and Control
  - Modern perspective

- **Stanford CS234 Lecture 5**
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
  - Emma Brunskill's clear explanations

### Tutorial Videos

- **Mutual Information: Eligibility Traces**
  - Short animated explanation
  - Intuitive visualization

- **Arxiv Insights: TD(λ)**
  - Theoretical background
  - Connection to modern deep RL

## Tools and Libraries

### Python Libraries

**Core RL Libraries**:
```bash
pip install gymnasium  # Environments
pip install numpy      # Numerical computing
pip install matplotlib # Visualization
```

**Advanced Libraries**:
```bash
pip install stable-baselines3  # Includes A3C (uses n-step)
pip install ray[rllib]         # Distributed RL with n-step
```

### Visualization Tools

**Learning Curves**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot RMSE vs episodes for different n
plt.plot(episodes, rmse_n1, label='n=1')
plt.plot(episodes, rmse_n4, label='n=4')
plt.plot(episodes, rmse_n8, label='n=8')
plt.legend()
```

**Eligibility Trace Visualization**:
```python
# Plot trace decay over time
time = np.arange(0, 20)
trace = (gamma * lambda_) ** time
plt.plot(time, trace)
plt.title(f'Trace Decay (γλ={gamma*lambda_})')
```

**Value Function Heatmaps**:
```python
import seaborn as sns
sns.heatmap(value_function.reshape(grid_shape))
```

## Key Equations Reference

### n-step Return
```
G_t:t+n = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})
```

### λ-return
```
G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t:t+n
```

### TD(λ) with Eligibility Traces
```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
e_t(s) = γλ e_{t-1}(s) + 𝟙(s = S_t)
V(s) ← V(s) + α δ_t e_t(s)
```

### SARSA(λ)
```
δ_t = R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
e_t(s,a) = γλ e_{t-1}(s,a) + 𝟙(s=S_t, a=A_t)
Q(s,a) ← Q(s,a) + α δ_t e_t(s,a)
```

## Common Mistakes and Debugging

### Implementation Pitfalls

1. **Forgetting to Reset Traces**
```python
# Wrong
def train_episode():
    # ... training ...
    # Traces carry over to next episode!

# Correct
def train_episode():
    e = np.zeros_like(V)  # Reset at episode start
    # ... training ...
```

2. **Wrong Decay Factor**
```python
# Wrong
e = lambda_ * e + 1  # Missing γ

# Correct
e = gamma * lambda_ * e + 1  # Both γ and λ
```

3. **Not Updating All States**
```python
# Wrong (TD(0) style)
V[state] += alpha * delta

# Correct (TD(λ) style)
for s in states:
    V[s] += alpha * delta * e[s]
```

### Debugging Strategies

- **Print trace values**: Check they decay properly
- **Verify weight sum**: (1-λ) Σ λ^{n-1} should equal 1
- **Compare with TD(0)**: λ=0 should match plain TD
- **Check terminal handling**: Traces at terminal should be 0

## Practice Projects

### Project 1: n-step Comparison Study
Implement TD(0), n-step TD for n=[2,4,8,16], and MC on 19-state random walk. Plot RMSE vs episodes.

### Project 2: SARSA(λ) on Mountain Car
Solve Mountain Car with different λ values. Compare episodes to solution.

### Project 3: Trace Type Comparison
Implement accumulating, replacing, and dutch traces. Compare on grid world with revisits.

### Project 4: True Online TD(λ)
Implement the modern True Online TD(λ) algorithm and compare with standard TD(λ).

## Theoretical Exercises

1. **Derive λ-return**: Show (1-λ) Σ λ^{n-1} = 1
2. **Prove equivalence**: Forward and backward view produce same total update
3. **Analyze variance**: Show Var[G_t^λ] increases with λ
4. **Optimal λ**: For what task characteristics is λ=1 optimal?

## Books

### Comprehensive Texts

- **Sutton & Barto (2018): Reinforcement Learning: An Introduction**
  - Chapters 7 and 12 are essential
  - Most comprehensive treatment

- **Csaba Szepesvári (2010): Algorithms for Reinforcement Learning**
  - [Free PDF](https://sites.ualberta.ca/~szepesva/RLBook.html)
  - Section on eligibility traces
  - More mathematical

### Practical Books

- **Phil Winder (2020): Reinforcement Learning**
  - Industrial applications
  - Practical implementation advice

- **Maxim Lapan (2020): Deep Reinforcement Learning Hands-On**
  - PyTorch implementations
  - A3C with n-step returns

## Advanced Topics

### Research Frontiers

- **Emphasis-based traces**: New theoretical framework
- **Variable λ and γ**: State-dependent parameters
- **Off-policy traces**: Combining IS with eligibility traces
- **Truncated λ-returns**: Practical compromises for online learning

### Modern Applications

- **A3C/A2C**: Asynchronous n-step actor-critic
- **PPO**: Generalized advantage estimation (GAE) uses λ-returns
- **Ape-X DQN**: Distributed n-step Q-learning
- **IMPALA**: Importance-weighted n-step learning

## Discussion Forums

- **Reddit: r/reinforcementlearning**
  - [Eligibility Traces Questions](https://www.reddit.com/r/reinforcementlearning/search?q=eligibility+traces)

- **Stack Overflow: [eligibility-traces] tag**
  - Implementation questions

- **RL Discord Servers**
  - Real-time help with code

## Quick Reference Card

```
Method          | Update Target          | Bias | Variance | Speed
----------------|------------------------|------|----------|-------
TD(0) (λ=0)    | R + γV(S')            | High | Low      | Fast
n-step TD       | Σ γ^k R + γ^n V       | Med  | Med      | Med
MC (λ=1)       | Σ γ^k R               | None | High     | Slow
TD(λ)          | (1-λ)Σ λ^{n-1} G_t:t+n| Low  | Medium   | Fast

Choosing λ:
- Sparse rewards: λ = 0.9-0.95
- Dense rewards: λ = 0.3-0.7
- Continuing tasks: λ = 0.8-0.9
- Default: λ = 0.8
```

## Next Steps

After mastering Week 6:

1. **Implement all methods**: n-step TD, n-step SARSA, TD(λ), SARSA(λ)
2. **Understand forward/backward equivalence** deeply
3. **Experiment with λ values** on different tasks
4. **Compare trace types** (accumulating vs replacing)
5. **Move to Week 7**: Planning and Learning (Dyna, MCTS)

### Bridge to Advanced RL

Eligibility traces are used in:
- **Actor-Critic methods**: Advantage estimation
- **Policy Gradient**: Baseline subtraction
- **Deep RL**: A3C, PPO, IMPALA
- **Model-based RL**: Combining with planning

Understanding eligibility traces is **essential** for modern deep RL!

### Recommended Learning Path

1. Read S&B Chapter 7 (n-step methods)
2. Implement n-step TD on 19-state random walk
3. Read S&B Chapter 12 sections 12.1-12.2 (λ-return and TD(λ))
4. Watch David Silver Lecture 4 (eligibility traces part)
5. Implement SARSA(λ) on Mountain Car
6. Read sections 12.4-12.5 (True Online TD(λ))
7. Complete quiz problems
8. Ready for Week 7!
