# Week 15: Model-Based Reinforcement Learning

## Overview

Model-based reinforcement learning (MBRL) learns a model of the environment's dynamics and uses it for planning or generating synthetic experience. This can dramatically improve sample efficiency compared to model-free methods, but introduces challenges around model accuracy and exploitation.

## Learning Objectives

- Understand the core components of model-based RL: dynamics models, reward models, and planning
- Learn about Dyna-style algorithms that combine model learning with model-free RL
- Study modern MBRL methods: MBPO, World Models, Dreamer, MuZero
- Implement a simple model-based agent using learned dynamics
- Analyze the tradeoffs between model-based and model-free approaches

## Key Concepts

### 1. Dynamics Models

A dynamics model predicts the next state (and possibly reward) given current state and action:

```
f_theta: (s, a) -> s'
or
f_theta: (s, a) -> (s', r)
```

**Model Loss:**
```
L = E[(s,a,s') ~ D][||f_theta(s,a) - s'||^2]
```

For stochastic environments, we typically model a distribution:
```
f_theta(s,a) ~ N(mu_theta(s,a), Sigma_theta(s,a))
```

### 2. Planning with Learned Models

Once we have a model, we can use it for:

**a) Model Predictive Control (MPC):**
- At each timestep, plan a sequence of actions
- Execute only the first action
- Replan at the next timestep

**b) Dyna-style Learning:**
- Collect real experience
- Learn model from real data
- Generate synthetic experience using model
- Train policy on mix of real and synthetic data

**Algorithm (Dyna-Q):**
```
1. Interact with environment, store (s,a,r,s') in replay buffer
2. Update Q-function on real data
3. Update dynamics model f_theta on real data
4. For k steps:
   - Sample (s,a) from buffer
   - Generate s' = f_theta(s,a)
   - Update Q-function on (s,a,r,s') with imagined s'
```

### 3. Model-Based Policy Optimization (MBPO)

MBPO uses short model rollouts to augment training data:

**Algorithm:**
```
1. Collect data from environment using current policy
2. Train dynamics model on all real data
3. For each real state s:
   - Perform k-step rollout using model: s -> s_1 -> ... -> s_k
   - Add synthetic transitions to model buffer
4. Train policy (e.g., SAC) on mix of real + model data
5. Repeat
```

**Key insight:** Short rollouts (k=1-5) prevent compounding model errors while still improving sample efficiency.

**Optimal rollout length** depends on model accuracy:
```
k* ~ (epsilon / (1 - gamma))^(1/2)
```
where epsilon is model error.

### 4. World Models (Ha & Schmidhuber 2018)

Learn a generative model of the environment and train policy entirely in the "dream":

**Components:**
- **V (Vision):** VAE encodes observations to latent z
- **M (Memory):** RNN predicts next latent: z_{t+1} = M(z_t, a_t)
- **C (Controller):** Policy trained in latent space

**Training:**
1. Collect random rollouts
2. Train VAE to encode/decode observations
3. Train RNN to predict latent dynamics
4. Train controller using CMA-ES in learned model

**Advantage:** Can train policy quickly in imagined rollouts without environment interaction.

### 5. Dreamer (Hafner et al. 2019)

Extends World Models with:
- Continuous training (not separate stages)
- Actor-critic in latent space
- Better representation learning

**Model:**
```
Representation model: h_t = f(h_{t-1}, a_{t-1}, o_t)
Transition model: h_t = f(h_{t-1}, a_{t-1})
Reward model: r_t = r(h_t)
```

**Policy training:**
- Imagine trajectories from current states
- Compute value targets using imagined rewards
- Train actor and critic on imagined experience

### 6. MuZero (Schrittwieser et al. 2020)

Combines tree search (like AlphaZero) with learned model:

**Key innovation:** Model doesn't predict actual next state, but a latent representation sufficient for planning.

**Components:**
- **Representation function:** s^0 = h(o_1, ..., o_t)
- **Dynamics function:** s^{k+1}, r^k = g(s^k, a^k)
- **Prediction function:** p^k, v^k = f(s^k)

**Planning:**
- Use MCTS with learned dynamics g
- Search in latent space, not actual state space
- Can handle complex observations (images) and long horizons

**Loss:**
```
L = sum_k [l^r(r^k, u^k) + l^v(v^k, z^k) + l^p(p^k, pi^k)]
```
where u^k, z^k, pi^k are observed rewards, values, and policies from MCTS.

## Model Errors and Compounding

**Challenge:** Small prediction errors compound exponentially over long rollouts.

If model error per step is epsilon:
```
Error after k steps ~ k * epsilon (optimistic)
Error after k steps ~ epsilon^k (pessimistic with compounding)
```

**Solutions:**
1. **Short rollouts** (MBPO): Limit rollout length based on model accuracy
2. **Model ensembles:** Train multiple models, use disagreement for uncertainty
3. **Conservative planning:** Penalize uncertain regions
4. **Latent models:** Plan in learned latent space (Dreamer, MuZero)

## When to Use Model-Based RL

**Advantages:**
- Much better sample efficiency (10-100x)
- Can plan and look ahead
- Transfer learned model to new tasks
- Interpretable (can visualize model predictions)

**Disadvantages:**
- Model errors can hurt asymptotic performance
- Harder to implement and tune
- Computationally expensive (model learning + planning)
- Can exploit model errors ("model hacking")

**Use when:**
- Sample efficiency is critical (robotics, expensive simulations)
- Environment has learnable structure
- You have good model architecture for the domain

**Avoid when:**
- Environment is extremely complex or stochastic
- Asymptotic performance matters more than sample efficiency
- You have unlimited samples (games, simple simulations)

## Implementation: CartPole with Learned Model

We'll implement a simple model-based agent:
1. Learn dynamics model from experience
2. Use model to generate synthetic rollouts
3. Train policy on real + synthetic data

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random

class DynamicsModel(nn.Module):
    """Learns s' = f(s, a)"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        # Predict next state (delta from current state)
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)
        return state + delta

class Policy(nn.Module):
    """Simple policy network"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.net(state)

class ModelBasedAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Model and policy
        self.dynamics = DynamicsModel(state_dim, action_dim)
        self.policy = Policy(state_dim, action_dim)

        # Optimizers
        self.model_optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Replay buffer
        self.buffer = deque(maxlen=10000)

    def select_action(self, state, epsilon=0.1):
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def train_model(self, batch_size=64):
        """Train dynamics model on real data"""
        if len(self.buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.buffer, batch_size)
        states, actions, _, next_states, _ = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor([[a] for a in actions])
        next_states = torch.FloatTensor(next_states)

        # One-hot encode actions
        actions_onehot = torch.zeros(batch_size, self.action_dim)
        actions_onehot.scatter_(1, actions.long(), 1)

        # Predict next states
        pred_next_states = self.dynamics(states, actions_onehot)

        # MSE loss
        loss = nn.MSELoss()(pred_next_states, next_states)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        return loss.item()

    def generate_synthetic_rollouts(self, num_rollouts=10, rollout_length=5):
        """Generate synthetic experience using learned model"""
        synthetic_data = []

        if len(self.buffer) < num_rollouts:
            return synthetic_data

        # Start from real states
        start_transitions = random.sample(self.buffer, num_rollouts)

        for state, _, _, _, _ in start_transitions:
            current_state = torch.FloatTensor(state).unsqueeze(0)

            for _ in range(rollout_length):
                # Select action
                with torch.no_grad():
                    action = self.policy(current_state).argmax().item()

                    # One-hot encode action
                    action_onehot = torch.zeros(1, self.action_dim)
                    action_onehot[0, action] = 1

                    # Predict next state
                    next_state = self.dynamics(current_state, action_onehot)

                # Simple reward model (for CartPole, reward is 1 if not done)
                reward = 1.0

                synthetic_data.append({
                    'state': current_state.squeeze().numpy(),
                    'action': action,
                    'reward': reward,
                    'next_state': next_state.squeeze().numpy()
                })

                current_state = next_state

        return synthetic_data

    def train_policy(self, batch_size=64, use_synthetic=True):
        """Train policy on real + synthetic data"""
        if len(self.buffer) < batch_size:
            return 0.0

        # Real data
        real_batch = random.sample(self.buffer, batch_size // 2)

        # Synthetic data
        if use_synthetic:
            synthetic_data = self.generate_synthetic_rollouts(
                num_rollouts=10,
                rollout_length=5
            )
            if len(synthetic_data) >= batch_size // 2:
                synthetic_batch = random.sample(synthetic_data, batch_size // 2)
                # Convert synthetic data to same format as real data
                synthetic_batch = [
                    (d['state'], d['action'], d['reward'], d['next_state'], False)
                    for d in synthetic_batch
                ]
                batch = real_batch + synthetic_batch
            else:
                batch = real_batch
        else:
            batch = real_batch

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q-learning update
        current_q = self.policy(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q = self.policy(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

        return loss.item()

def train_model_based():
    env = gym.make('CartPole-v1')
    agent = ModelBasedAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    num_episodes = 200
    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        epsilon = max(0.01, 0.5 - episode / 100)

        done = False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)

            # Train model and policy
            if len(agent.buffer) > 64:
                agent.train_model(batch_size=64)
                agent.train_policy(batch_size=64, use_synthetic=True)

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()
    return episode_rewards

if __name__ == "__main__":
    rewards = train_model_based()
    print(f"\nFinal average reward: {np.mean(rewards[-10:]):.2f}")
```

## Key Equations Summary

**Dynamics Model Loss:**
```
L_model = E[(s,a,s')~D][||f_theta(s,a) - s'||^2]
```

**MBPO Rollout Length:**
```
k* ~ O(sqrt(epsilon / (1 - gamma)))
```
where epsilon is model error per step.

**MuZero Loss:**
```
L = sum_{k=0}^K [l^r(r^k, u^k) + l^v(v^k, z^k) + l^p(p^k, pi^k)]
```

**Model Ensemble Uncertainty:**
```
Var[f(s,a)] = (1/N) sum_i [f_i(s,a) - mean(f(s,a))]^2
```

## Required Readings

1. **Silver Lecture 8:** Model-Based RL
2. **CS285 Lecture 11:** Model-Based RL
3. **Ha & Schmidhuber (2018):** "World Models"
4. **Janner et al. (2019):** "When to Trust Your Model: Model-Based Policy Optimization"
5. **Schrittwieser et al. (2020):** "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"

## Exercises

1. Implement MBPO-style short rollouts in the CartPole agent
2. Add model ensemble for uncertainty estimation
3. Compare sample efficiency of model-based vs model-free on CartPole
4. Implement a simple version of World Models on a simple environment
5. Analyze when the model makes accurate vs inaccurate predictions

## Discussion Questions

1. Why does MBPO use short rollouts instead of full episodes?
2. How does MuZero avoid the need to predict actual observations?
3. What are the failure modes of model-based RL?
4. When would you choose model-based over model-free RL?
5. How can we detect and mitigate model exploitation?

## Next Week Preview

Week 16 covers exploration strategies, including count-based exploration, curiosity-driven methods (ICM), and random network distillation (RND). We'll see how exploration bonuses can help in sparse-reward environments.
