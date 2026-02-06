# Week 16: Exploration and Exploitation

## Overview

Exploration is one of the fundamental challenges in reinforcement learning. An agent must balance exploiting what it knows to maximize reward with exploring to discover potentially better strategies. In sparse-reward environments, naive exploration strategies like epsilon-greedy often fail completely. This week covers principled exploration methods including count-based exploration, curiosity-driven methods, and random network distillation.

## Learning Objectives

- Understand the exploration-exploitation tradeoff and when it matters
- Learn count-based exploration and the exploration bonus principle
- Study curiosity-driven exploration: ICM and RND
- Implement exploration bonuses in sparse-reward environments
- Analyze the theoretical limits of exploration efficiency

## Key Concepts

### 1. The Exploration Problem

**Why Exploration Matters:**

In dense-reward environments (e.g., CartPole), random exploration works:
```
Episode 1: Random actions -> score 20 -> learn something useful
Episode 2: Slightly better -> score 30 -> learn more
...
```

In sparse-reward environments (e.g., Montezuma's Revenge), random exploration fails:
```
Episode 1: Random actions -> score 0 -> learn nothing
Episode 2: Random actions -> score 0 -> learn nothing
...
Episode 10000: Random actions -> score 0 -> still learn nothing
```

**The Problem:**
- Reward signal is too sparse to guide learning
- Random exploration unlikely to discover reward
- Agent gets stuck in local optima (e.g., staying safe but unrewarded)

### 2. Multi-Armed Bandits and UCB

The simplest exploration problem: multi-armed bandit.

**Setup:**
- K arms (actions)
- Each arm has unknown reward distribution
- Goal: maximize total reward over T steps

**Regret:**
```
Regret = T * mu* - sum_t r_t
```
where mu* is the best arm's mean reward.

**Upper Confidence Bound (UCB):**

Select arm that maximizes:
```
UCB(a) = Q(a) + c * sqrt(log(t) / N(a))
```

where:
- Q(a): estimated value of arm a
- N(a): number of times arm a was selected
- t: total timesteps
- c: exploration constant

**Intuition:**
- Exploit: Choose arm with high Q(a)
- Explore: Choose arm with high uncertainty (low N(a))
- Balance: Exploration bonus decreases as we gather more data

**Theorem (Lai & Robbins):**
UCB achieves logarithmic regret: O(log T)

### 3. Count-Based Exploration

Extend UCB idea to MDPs: bonus for visiting rare states.

**Pseudo-Count (Bellemare et al., 2016):**

Use density model to estimate state visitation:
```
rho(s): probability of state s under current experience
rho'(s): probability after adding s to experience
```

**Pseudo-count N(s):**
```
N(s) = rho(s) * (1 - rho'(s)) / (rho'(s) - rho(s))
```

**Exploration Bonus:**
```
r+(s) = beta / sqrt(N(s))
```

**Total Reward:**
```
r_total = r_extrinsic + r_intrinsic
        = r_env + beta / sqrt(N(s))
```

**Algorithm:**
```
1. Train density model p_theta(s) on visited states
2. For each new state s:
   - Compute pseudo-count N(s) from p_theta
   - Add exploration bonus r+(s)
   - Use r_total for RL updates
3. Update density model with s
```

**Advantage:** Theoretically principled (related to PAC-MDP bounds)

**Disadvantage:** Hard to train good density models in high dimensions

### 4. Curiosity-Driven Exploration: ICM

Intrinsic Curiosity Module (Pathak et al., 2017)

**Key Idea:** Use prediction error as curiosity signal.

**Naive Approach (fails):**
```
Predict next state: s' = f(s, a)
Curiosity = ||s' - s'_predicted||^2
```

Problem: Many aspects of s' are unpredictable but irrelevant (e.g., leaves moving in wind).

**ICM Solution:** Predict in learned feature space that captures only controllable aspects.

**Components:**

1. **Inverse Model:** Predicts action from state transition
   ```
   a_predicted = g(phi(s), phi(s'))
   ```
   Trains feature encoder phi to capture action-relevant information.

2. **Forward Model:** Predicts next state features
   ```
   phi(s')_predicted = f(phi(s), a)
   ```

3. **Curiosity Reward:**
   ```
   r_intrinsic = eta * ||phi(s') - phi(s')_predicted||^2
   ```

**Loss Function:**
```
L = (1-beta) * L_inverse + beta * L_forward

L_inverse = ||a - g(phi(s), phi(s'))||^2
L_forward = ||phi(s') - f(phi(s), a)||^2
```

**Algorithm:**
```
1. Observe transition (s, a, s')
2. Compute forward prediction error -> intrinsic reward
3. Train policy on r_extrinsic + r_intrinsic
4. Update ICM (inverse and forward models)
```

**Advantage:**
- Learns what's predictable and controllable
- Ignores unpredictable distractions

**Disadvantage:**
- Can get distracted by predictable but irrelevant dynamics
- "Noisy TV problem": staring at random noise that's perfectly unpredictable

### 5. Random Network Distillation (RND)

Burda et al., 2018

**Key Insight:** Use prediction error of random features as novelty signal.

**Components:**

1. **Target Network** (fixed random network):
   ```
   f_target(s): S -> R^k
   ```
   Initialized randomly, never trained.

2. **Predictor Network** (trained):
   ```
   f(s; theta): S -> R^k
   ```
   Trained to predict f_target(s).

**Novelty Bonus:**
```
r_intrinsic(s) = ||f(s; theta) - f_target(s)||^2
```

**Why it works:**
- Predictor learns to match target on visited states
- For novel states, prediction error is high
- As state is visited more, prediction error decreases

**Training:**
```
1. Observe state s
2. Compute RND bonus: ||f(s; theta) - f_target(s)||^2
3. Train policy on r_extrinsic + r_intrinsic
4. Update predictor f(s; theta) to minimize prediction error
```

**Advantages:**
- Simple to implement
- No forward dynamics modeling needed
- Naturally handles stochastic environments
- State-of-the-art on Montezuma's Revenge

**Key Trick: Observation Normalization**
```python
# Running statistics of observations
obs_mean = running_mean(observations)
obs_std = running_std(observations)

# Normalize
s_normalized = (s - obs_mean) / (obs_std + epsilon)

# Use normalized obs for RND
bonus = ||f(s_normalized; theta) - f_target(s_normalized)||^2
```

### 6. Go-Explore

Ecoffet et al., 2019

**Motivation:** Hard exploration games require both:
1. Exploring to find promising states
2. Exploiting to return to promising states

**Phase 1: Exploration**
```
1. Maintain archive of interesting states
2. Sample state from archive
3. Return to state (using saved trajectory or imitation)
4. Explore from that state
5. Add new interesting states to archive
```

**Phase 2: Robustification**
```
Train policy to reach high-reward states discovered in Phase 1
```

**State Selection:**
- Prioritize states that are:
  - Rarely visited
  - Lead to high-scoring trajectories
  - Near domain boundaries

**Results:**
- Solved Montezuma's Revenge (first time)
- Achieved superhuman scores

**Limitations:**
- Requires ability to reset to arbitrary states (not always possible)
- Phase 1 uses domain-specific cell representations

### 7. Posterior Sampling (Thompson Sampling)

**Idea:** Maintain distribution over Q-functions, sample for exploration.

**Algorithm:**
```
1. Maintain posterior over Q-functions: p(Q | D)
2. At each episode:
   - Sample Q~ from posterior
   - Act greedily w.r.t. Q~
3. Update posterior with new data
```

**Implementation (Bootstrapped DQN):**
- Train ensemble of Q-networks with different random initializations
- Each episode, randomly select one Q-network to follow
- Each network sees different bootstrap sample of data

**Advantage:**
- Theoretically optimal exploration (in bandits)
- Naturally balances exploration and exploitation

**Disadvantage:**
- Expensive (need to train multiple networks)
- Less effective in deep RL compared to simpler methods

## Implementation: Sparse GridWorld with RND

We'll implement a sparse-reward grid world where RND exploration is critical.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class SparseGridWorld:
    """
    Grid world with sparse reward in corner.
    Without exploration bonus, agent never finds reward.
    """
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        # Start in center
        self.pos = [self.size // 2, self.size // 2]
        return self._get_obs()

    def step(self, action):
        # Actions: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.pos[0] > 0:
            self.pos[0] -= 1
        elif action == 1 and self.pos[0] < self.size - 1:
            self.pos[0] += 1
        elif action == 2 and self.pos[1] > 0:
            self.pos[1] -= 1
        elif action == 3 and self.pos[1] < self.size - 1:
            self.pos[1] += 1

        # Sparse reward only in top-right corner
        reward = 1.0 if self.pos == [0, self.size - 1] else 0.0
        done = (reward > 0)

        return self._get_obs(), reward, done

    def _get_obs(self):
        # One-hot encoding of position
        obs = np.zeros(self.size * self.size)
        obs[self.pos[0] * self.size + self.pos[1]] = 1.0
        return obs

class RandomNetwork(nn.Module):
    """Fixed random target network"""
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # Fix weights (never train)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.net(x)

class PredictorNetwork(nn.Module):
    """Trained to predict random network output"""
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class QNetwork(nn.Module):
    """Q-network for policy"""
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class RNDAgent:
    def __init__(self, obs_dim, action_dim, intrinsic_coef=1.0):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.intrinsic_coef = intrinsic_coef

        # RND networks
        self.target_net = RandomNetwork(obs_dim)
        self.predictor_net = PredictorNetwork(obs_dim)

        # Q-network
        self.q_net = QNetwork(obs_dim, action_dim)

        # Optimizers
        self.predictor_optimizer = optim.Adam(
            self.predictor_net.parameters(),
            lr=1e-3
        )
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        # Replay buffer
        self.buffer = deque(maxlen=10000)

        # Observation normalization
        self.obs_rms_mean = np.zeros(obs_dim)
        self.obs_rms_var = np.ones(obs_dim)
        self.obs_count = 0

    def normalize_obs(self, obs):
        """Normalize observations using running statistics"""
        return (obs - self.obs_rms_mean) / (np.sqrt(self.obs_rms_var) + 1e-8)

    def update_obs_stats(self, obs):
        """Update running mean and variance of observations"""
        self.obs_count += 1
        delta = obs - self.obs_rms_mean
        self.obs_rms_mean += delta / self.obs_count
        delta2 = obs - self.obs_rms_mean
        self.obs_rms_var += (delta * delta2 - self.obs_rms_var) / self.obs_count

    def compute_intrinsic_reward(self, obs):
        """Compute RND-based intrinsic reward"""
        obs_normalized = self.normalize_obs(obs)
        obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0)

        with torch.no_grad():
            target_features = self.target_net(obs_tensor)
        predictor_features = self.predictor_net(obs_tensor)

        # MSE between predictor and target
        intrinsic_reward = torch.mean((predictor_features - target_features) ** 2).item()
        return intrinsic_reward

    def select_action(self, obs, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return q_values.argmax().item()

    def store_transition(self, obs, action, reward, next_obs, done, intrinsic_reward):
        """Store transition in replay buffer"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'intrinsic_reward': intrinsic_reward
        })

    def train_predictor(self, batch_size=32):
        """Train predictor to match target network"""
        if len(self.buffer) < batch_size:
            return 0.0

        batch = random.sample(self.buffer, batch_size)
        obs_batch = np.array([t['obs'] for t in batch])

        # Normalize observations
        obs_normalized = np.array([self.normalize_obs(obs) for obs in obs_batch])
        obs_tensor = torch.FloatTensor(obs_normalized)

        # Predict target features
        with torch.no_grad():
            target_features = self.target_net(obs_tensor)
        predictor_features = self.predictor_net(obs_tensor)

        # MSE loss
        loss = nn.MSELoss()(predictor_features, target_features)

        self.predictor_optimizer.zero_grad()
        loss.backward()
        self.predictor_optimizer.step()

        return loss.item()

    def train_policy(self, batch_size=32):
        """Train Q-network with extrinsic + intrinsic rewards"""
        if len(self.buffer) < batch_size:
            return 0.0

        batch = random.sample(self.buffer, batch_size)

        obs = torch.FloatTensor([t['obs'] for t in batch])
        actions = torch.LongTensor([t['action'] for t in batch])
        rewards = torch.FloatTensor([t['reward'] for t in batch])
        intrinsic_rewards = torch.FloatTensor([t['intrinsic_reward'] for t in batch])
        next_obs = torch.FloatTensor([t['next_obs'] for t in batch])
        dones = torch.FloatTensor([t['done'] for t in batch])

        # Total reward = extrinsic + intrinsic
        total_rewards = rewards + self.intrinsic_coef * intrinsic_rewards

        # Q-learning update
        current_q = self.q_net(obs).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.q_net(next_obs).max(1)[0]
            target_q = total_rewards + 0.99 * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

def train_with_rnd():
    """Train agent with RND exploration bonus"""
    env = SparseGridWorld(size=10)
    agent = RNDAgent(
        obs_dim=env.size * env.size,
        action_dim=4,
        intrinsic_coef=0.1
    )

    num_episodes = 500
    max_steps = 200

    episode_rewards = []
    extrinsic_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_extrinsic = 0
        epsilon = max(0.01, 0.5 - episode / 200)

        for step in range(max_steps):
            # Update observation statistics
            agent.update_obs_stats(obs)

            # Compute intrinsic reward
            intrinsic_reward = agent.compute_intrinsic_reward(obs)

            # Select and execute action
            action = agent.select_action(obs, epsilon)
            next_obs, reward, done = env.step(action)

            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done, intrinsic_reward)

            # Train networks
            if len(agent.buffer) > 32:
                agent.train_predictor(batch_size=32)
                agent.train_policy(batch_size=32)

            episode_reward += reward + agent.intrinsic_coef * intrinsic_reward
            episode_extrinsic += reward
            obs = next_obs

            if done:
                break

        episode_rewards.append(episode_reward)
        extrinsic_rewards.append(episode_extrinsic)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_extrinsic = np.mean(extrinsic_rewards[-50:])
            print(f"Episode {episode}, Avg Total: {avg_reward:.2f}, "
                  f"Avg Extrinsic: {avg_extrinsic:.2f}, Epsilon: {epsilon:.3f}")

    return episode_rewards, extrinsic_rewards

def train_without_rnd():
    """Train agent without exploration bonus (baseline)"""
    env = SparseGridWorld(size=10)
    agent = RNDAgent(
        obs_dim=env.size * env.size,
        action_dim=4,
        intrinsic_coef=0.0  # No intrinsic reward
    )

    num_episodes = 500
    max_steps = 200

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        epsilon = max(0.01, 0.5 - episode / 200)

        for step in range(max_steps):
            action = agent.select_action(obs, epsilon)
            next_obs, reward, done = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done, 0.0)

            if len(agent.buffer) > 32:
                agent.train_policy(batch_size=32)

            episode_reward += reward
            obs = next_obs

            if done:
                break

        episode_rewards.append(episode_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards

if __name__ == "__main__":
    print("Training with RND exploration bonus...")
    rnd_rewards, rnd_extrinsic = train_with_rnd()

    print("\nTraining without exploration bonus...")
    baseline_rewards = train_without_rnd()

    print(f"\nRND Final Extrinsic Reward: {np.mean(rnd_extrinsic[-50:]):.2f}")
    print(f"Baseline Final Reward: {np.mean(baseline_rewards[-50:]):.2f}")
```

## Key Equations Summary

**UCB:**
```
UCB(a) = Q(a) + c * sqrt(log(t) / N(a))
```

**Pseudo-Count:**
```
N(s) = rho(s) * (1 - rho'(s)) / (rho'(s) - rho(s))
Bonus: r+(s) = beta / sqrt(N(s))
```

**ICM:**
```
L_ICM = (1-beta) * ||a - g(phi(s), phi(s'))||^2
        + beta * ||phi(s') - f(phi(s), a)||^2

r_intrinsic = eta * ||phi(s') - f(phi(s), a)||^2
```

**RND:**
```
r_intrinsic(s) = ||f(s; theta) - f_target(s)||^2

where f_target is fixed random, f(s; theta) is trained
```

## Required Readings

1. **Silver Lecture 9:** Exploration and Exploitation
2. **CS285 Lecture 12:** Exploration
3. **Bellemare et al. (2016):** "Unifying Count-Based Exploration and Intrinsic Motivation"
4. **Pathak et al. (2017):** "Curiosity-Driven Exploration by Self-Supervised Prediction"
5. **Burda et al. (2018):** "Exploration by Random Network Distillation"

## Exercises

1. Implement UCB for multi-armed bandit and compare with epsilon-greedy
2. Add ICM to the GridWorld environment and compare with RND
3. Analyze how intrinsic reward coefficient affects exploration
4. Design an exploration strategy for a new sparse-reward environment
5. Implement simple version of Go-Explore for GridWorld

## Discussion Questions

1. Why does epsilon-greedy fail in Montezuma's Revenge?
2. How does RND avoid the "noisy TV problem" that affects ICM?
3. When would count-based exploration be preferable to curiosity-based?
4. What are the fundamental limits to exploration efficiency (PAC-MDP)?
5. How can we detect when an agent has stopped meaningfully exploring?

## Next Week Preview

Week 17 introduces reward modeling and RLHF (Reinforcement Learning from Human Feedback), the technique behind ChatGPT and modern aligned AI systems. We'll learn how to train reward models from preferences and use RL to optimize for human values.
