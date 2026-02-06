# Week 1: Introduction to Reinforcement Learning

## Learning Objectives

- [ ] Understand the reinforcement learning problem formulation
- [ ] Grasp the agent-environment interface and interaction loop
- [ ] Understand the reward hypothesis and its implications
- [ ] Master the k-armed bandit problem
- [ ] Understand exploration vs exploitation tradeoffs

## Key Concepts

### 1. The Reinforcement Learning Problem

**Agent-Environment Interface**
- **Agent**: The learner or decision maker
- **Environment**: Everything outside the agent that it interacts with
- **State (S_t)**: A representation of the current situation
- **Action (A_t)**: Choices available to the agent
- **Reward (R_t)**: Scalar feedback signal indicating how good the action was
- **Trajectory**: A sequence of states, actions, and rewards: S_0, A_0, R_1, S_1, A_1, R_2, ...

The agent's goal is to maximize the cumulative reward over time, not just immediate reward.

**The Reward Hypothesis**: All goals can be described as the maximization of expected cumulative reward.

### 2. Multi-armed Bandits

The k-armed bandit problem is a simplified RL problem where:
- There is only one state (non-associative setting)
- You have k different actions (arms) to choose from
- Each action yields a reward from a probability distribution
- Goal: Maximize total reward over time

**Action-Value Methods**
- **Action value**: q_*(a) = E[R_t | A_t = a]
- We estimate q_*(a) with Q_t(a), the sample average of rewards
- **Greedy action selection**: A_t = argmax_a Q_t(a)

**Epsilon-Greedy**:
- With probability 1-ε: select greedy action (exploitation)
- With probability ε: select random action (exploration)

**Upper Confidence Bound (UCB)**:
- Select actions based on potential to be optimal
- Considers both the value estimate and uncertainty

**Gradient Bandit**:
- Learn numerical preferences H_t(a) for each action
- Convert to probabilities using softmax
- Update preferences based on reward comparison to baseline

### 3. Exploration vs Exploitation

**The Fundamental Tradeoff**:
- **Exploitation**: Choose the action that currently appears best
- **Exploration**: Try other actions to potentially discover better options

**Methods to Balance**:
- **Optimistic Initial Values**: Start with high Q_0(a) values to encourage exploration
- **UCB**: Systematically decrease uncertainty by exploring less-tried actions
- **Decaying ε**: Start with high exploration, gradually exploit more

## Textbook References

- Sutton & Barto Chapter 1: Introduction
- Sutton & Barto Chapter 2: Multi-armed Bandits
- David Silver Lecture 1: Introduction to RL
- CS234 Week 1: Introduction and Course Overview

## Implementation Tasks

Implement a k-armed bandit testbed that includes:

1. **Environment**:
   - k-armed bandit with true action values q_*(a) sampled from N(0,1)
   - Rewards sampled from N(q_*(a), 1)

2. **Agents**:
   - Epsilon-greedy (ε = 0.1, 0.01, 0)
   - UCB with c = 2
   - Gradient bandit with α = 0.1

3. **Experiments**:
   - Run 2000 time steps
   - Average over 2000 independent runs
   - Plot average reward over time
   - Plot % optimal action over time

## Key Equations

**Sample-Average Action-Value Estimate**:
```
Q_t(a) = (sum of rewards when a taken prior to t) / (number of times a taken prior to t)
      = (R_1 + R_2 + ... + R_{n-1}) / (n-1)
```

**Incremental Update Rule**:
```
Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
NewEstimate = OldEstimate + StepSize[Target - OldEstimate]
```

**Upper Confidence Bound (UCB)**:
```
A_t = argmax_a [Q_t(a) + c * sqrt(ln(t) / N_t(a))]
```
where N_t(a) is the number of times action a has been selected prior to time t, and c > 0 controls exploration degree.

**Gradient Bandit Algorithm**:
```
H_{t+1}(a) = H_t(a) + α(R_t - R̄_t)(1{A_t=a} - π_t(a))

π_t(a) = exp(H_t(a)) / sum_b exp(H_t(b))
```
where R̄_t is the average of all rewards up to time t, and 1{A_t=a} is 1 if a=A_t, else 0.

## Review Questions

1. **What is the key difference between reinforcement learning and supervised learning?**
   - In supervised learning, we are given correct labels/actions by an external supervisor
   - In RL, the agent must discover good actions through trial and error
   - RL involves a tradeoff between exploration and exploitation

2. **Why can't we simply always choose the greedy action (pure exploitation)?**
   - Our action-value estimates are imperfect, based on limited samples
   - The currently best action might not be the truly optimal action
   - Without exploration, we might never discover the optimal action
   - Early exploration can lead to higher long-term cumulative reward

3. **How does UCB address the exploration-exploitation tradeoff?**
   - UCB adds an exploration bonus to each action's value estimate
   - The bonus is larger for actions that have been tried less often
   - This systematically reduces uncertainty about all actions
   - As t increases, the bonus shrinks, leading to more exploitation
   - UCB provides a principled way to "explore optimistically"

## Next Steps

After completing this week:
- Move to Week 2: Markov Decision Processes
- Understand how to extend from single-state (bandit) to multi-state problems
- Learn about value functions and Bellman equations
