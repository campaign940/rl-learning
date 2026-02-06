# Week 12: Actor-Critic Methods

## Learning Objectives

- [ ] Understand the advantage function A(s,a) and why it reduces variance
- [ ] Implement Advantage Actor-Critic (A2C) algorithm
- [ ] Understand asynchronous parallel training (A3C) and its trade-offs
- [ ] Master Generalized Advantage Estimation (GAE) and the bias-variance trade-off
- [ ] Design effective network architectures with shared vs. separate networks

## Key Concepts

### 1. Advantage Function A(s,a) = Q(s,a) - V(s)

**Definition**: The advantage function measures how much better action a is compared to the average action in state s.

**Equation**:
```
A^π(s,a) = Q^π(s,a) - V^π(s)
```

**Intuition**:
- Q(s,a): absolute value of taking action a
- V(s): average value of being in state s
- A(s,a): relative advantage of action a over the average

**Why it's important**:
- Reduces variance: relative comparisons are more stable than absolute values
- Centers values around 0: better for gradient-based learning
- No bias added: E_π[A(s,a)] = 0

**Estimation methods**:

1. **One-step TD** (low variance, high bias):
```
A(s,a) ≈ r + γV(s') - V(s) = δ  (TD error)
```

2. **Monte Carlo** (high variance, no bias):
```
A(s,a) ≈ G_t - V(s)
```

3. **n-step** (intermediate):
```
A(s,a) ≈ Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) - V(s_t)
```

4. **GAE** (best, see section 4):
```
A^GAE(λ) = exponentially-weighted average of k-step advantages
```

### 2. Advantage Actor-Critic (A2C)

**Definition**: Policy gradient method that uses a learned critic V(s) to estimate advantages and reduce variance.

**Architecture**:
- **Actor**: Policy network π(a|s; θ) that selects actions
- **Critic**: Value network V(s; w) that evaluates states

**Algorithm**:
```
1. Initialize actor π(a|s; θ) and critic V(s; w)
2. For each episode:
   a. Collect batch of transitions using π
   b. For each transition (s, a, r, s'):
      - Compute advantage: A = r + γV(s'; w) - V(s; w)
      - Update critic: w ← w + β_v · A · ∇_w V(s; w)
      - Update actor: θ ← θ + β_π · A · ∇_θ log π(a|s; θ)
```

**Loss functions**:

**Actor loss** (policy gradient):
```
L_π(θ) = -E[log π(a|s; θ) · A(s,a)]
```

**Critic loss** (value function):
```
L_V(w) = E[(r + γV(s'; w) - V(s; w))^2]
```

**Total loss** (often combined):
```
L = L_π + c_1 · L_V - c_2 · H(π)
where H(π) = -E[log π(a|s)] is entropy bonus for exploration
```

**Key hyperparameters**:
- c_1 = 0.5: value loss coefficient
- c_2 = 0.01: entropy coefficient
- γ = 0.99: discount factor
- β_π = 3e-4: actor learning rate
- β_v = 1e-3: critic learning rate (often higher than actor)

**Advantages over REINFORCE**:
- Lower variance (bootstrapping with V(s'))
- Faster learning (can use smaller batches)
- Can learn online (no need for complete episodes)

**Disadvantages**:
- Biased (critic approximation errors)
- Requires tuning two learning rates
- More complex implementation

### 3. Asynchronous Advantage Actor-Critic (A3C)

**Definition**: Parallel training with multiple actors collecting data asynchronously.

**Architecture**:
```
Global Network (shared)
    ↓ copy parameters
Worker 1, Worker 2, ..., Worker N (parallel)
    ↓ accumulate gradients
Global Network (updated asynchronously)
```

**Algorithm**:
```
Global: Initialize global network parameters θ, w

Each worker (in parallel):
  1. Copy global parameters: θ_local ← θ, w_local ← w
  2. Collect trajectory of length t_max
  3. Compute advantages using local critic
  4. Compute gradients ∇θ L_π and ∇w L_V
  5. Asynchronously update global parameters:
     θ ← θ + ∇θ L_π
     w ← w + ∇w L_V
  6. Repeat
```

**Key innovations**:
- **Asynchronous updates**: No waiting for other workers (fast)
- **Parallel exploration**: Different workers explore different parts of state space
- **Decorrelates data**: Different workers at different states reduces correlation
- **No replay buffer**: On-policy, uses parallelism instead

**Hyperparameters**:
- Number of workers: 16-32 typical (more is not always better)
- t_max: 20-40 steps before update
- Learning rates: same as A2C

**A3C vs. A2C (Synchronous)**:

| Aspect | A3C (Asynchronous) | A2C (Synchronous) |
|--------|-------------------|-------------------|
| Workers | Update independently | Wait for all to finish batch |
| Gradient staleness | Yes (older parameters) | No (always current) |
| Implementation | Complex (threading) | Simpler (vectorized) |
| Speed | Faster wall-clock time | Slower but more efficient |
| Stability | Less stable | More stable |
| Modern usage | Rarely used | Preferred |

**Why A2C won over A3C**:
- Synchronous updates more stable
- Modern GPUs handle vectorization efficiently
- Easier to debug and implement
- Similar or better sample efficiency

### 4. Generalized Advantage Estimation (GAE)

**Definition**: Exponentially-weighted average of k-step advantages, parameterized by λ ∈ [0, 1].

**Motivation**: Trade-off between bias and variance in advantage estimation.

**Equation**:
```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t) is the TD error
```

**Recursive form** (easier to compute):
```
A_t^GAE(λ) = δ_t + γλ A_{t+1}^GAE(λ)
```

**Special cases**:
- **λ = 0**: A_t = δ_t (1-step TD, high bias, low variance)
- **λ = 1**: A_t = Σ_k γ^k δ_{t+k} = G_t - V(s_t) (Monte Carlo, no bias, high variance)
- **λ = 0.95** (typical): Good balance

**Intuition**:
- λ controls how much we trust future TD errors
- Small λ: trust nearby estimates (low variance, but biased if V is wrong)
- Large λ: look further ahead (less biased, but more variance)

**Implementation**:
```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: [T] array of rewards
    values: [T+1] array of value estimates (includes V(s_T))
    """
    advantages = []
    gae = 0

    # Iterate backwards through trajectory
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return advantages
```

**Connection to n-step returns**:
```
A^GAE(λ) is equivalent to exponentially-weighted average of:
- 1-step: r + γV(s') - V(s)
- 2-step: r + γr' + γ²V(s'') - V(s)
- 3-step: r + γr' + γ²r'' + γ³V(s''') - V(s)
- ...
with weights: (1-λ), (1-λ)λ, (1-λ)λ², ...
```

**Effect of λ**:

| λ | Bias | Variance | When to use |
|---|------|----------|-------------|
| 0.0 | High | Low | Stable environments, good critic |
| 0.5 | Medium | Medium | Balanced |
| 0.95 | Low | Medium | Most common default |
| 0.99 | Very low | High | Long-horizon tasks |
| 1.0 | None | Very high | Short episodes only |

**Why GAE is powerful**:
- Smooth interpolation between TD and Monte Carlo
- Single hyperparameter λ controls trade-off
- Works well across diverse tasks
- Standard in modern algorithms (PPO, TRPO)

### 5. Network Architecture: Shared vs. Separate

**Design choice**: Should actor and critic share parameters?

**Option A: Separate Networks**
```python
class SeparateActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
```

**Pros**:
- Independent learning rates for actor and critic
- No interference between actor and critic gradients
- Easier to debug (clear separation)

**Cons**:
- More parameters (2x networks)
- Slower forward passes
- No shared representations

**Option B: Shared Encoder**
```python
class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # Separate heads
        self.actor_head = nn.Linear(256, act_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, obs):
        features = self.encoder(obs)
        action_logits = self.actor_head(features)
        value = self.critic_head(features)
        return action_logits, value
```

**Pros**:
- Parameter efficient (shared features)
- Faster forward passes (single encoder)
- Shared representations can help both tasks

**Cons**:
- Gradient conflicts between actor and critic
- Single learning rate (or need careful tuning)
- Can hurt performance if tasks are very different

**Practical recommendations**:
- **Start with shared encoder**: Usually works well, more efficient
- **Try separate networks**: If performance plateaus or gradients conflict
- **Use gradient clipping**: Prevents exploding gradients in either path
- **Different learning rates**: Use separate optimizers or gradient scaling

**Typical architecture choices by environment type**:

| Environment | Recommendation | Reasoning |
|-------------|----------------|-----------|
| Low-dim state (< 20) | Shared | Simple features benefit both |
| Images | Shared CNN encoder | Both need visual features |
| Complex observations | Separate | Actor and critic may need different features |
| Continuous control | Shared | Works well in practice |
| Atari games | Shared CNN | Standard, works well |

## Textbook References

- **CS285 (Berkeley)**: Lectures 5-6
  - Lecture 5: Actor-critic algorithms
  - Lecture 6: Value functions and critics

## Key Papers

### Mnih et al. 2016: Asynchronous Methods for Deep RL (A3C)
- **Paper**: [ArXiv Link](https://arxiv.org/abs/1602.01783)
- **Contribution**: Introduced A3C and parallel training for RL
- **Impact**: Showed that parallelism can replace experience replay

### Schulman et al. 2015: High-Dimensional Continuous Control Using GAE
- **Paper**: [ArXiv Link](https://arxiv.org/abs/1506.02438)
- **Contribution**: Introduced GAE, analyzed bias-variance trade-off
- **Impact**: GAE is now standard in PPO, TRPO, and most policy gradient methods

## Implementation Task

### LunarLander-v2 with A2C

**Environment**:
- **Observation**: 8-dimensional (position, velocity, angle, angular velocity, leg contact)
- **Action**: 4 discrete actions (do nothing, fire left, fire main, fire right)
- **Reward**: +100 for landing, penalties for crashes and fuel usage
- **Success**: Average score > 200 over 100 episodes

**Implementation requirements**:

1. **Networks**:
   - Try both separate and shared architectures
   - Hidden layers: [256, 256] with ReLU
   - Actor output: 4 action logits with softmax
   - Critic output: single value estimate

2. **Algorithm**:
   - Collect batches of N=5 parallel environments (vectorized)
   - Update every t=5 steps
   - Use GAE with λ=0.95
   - Entropy bonus c_2=0.01
   - Gradient clipping: max_norm=0.5

3. **Training**:
   - 2000-3000 episodes typically needed
   - Learning rates: actor=3e-4, critic=1e-3
   - Discount γ=0.99
   - Log rewards, value estimates, policy entropy

4. **Experiments**:
   - **Experiment 1**: Compare λ ∈ {0.0, 0.5, 0.8, 0.95, 1.0}
   - **Experiment 2**: Separate vs. shared network architecture
   - **Experiment 3**: Effect of entropy bonus
   - **Experiment 4**: Effect of number of parallel environments

5. **Evaluation**:
   - Plot learning curves (smoothed with window=100)
   - Compare sample efficiency (episodes to reach 200)
   - Analyze variance of gradient estimates
   - Visualize learned policy (optional: render episodes)

**Expected results**:
- Solves in 1500-2500 episodes with good hyperparameters
- GAE λ=0.95 typically best
- Shared architecture usually faster convergence

**Code structure**:
```python
# a2c.py
class A2CAgent:
    def __init__(self, ...):
        self.actor_critic = ActorCriticNetwork(...)
        self.optimizer = Adam(self.actor_critic.parameters())

    def select_action(self, state):
        # Return action and log_prob for training
        pass

    def compute_returns_and_advantages(self, rewards, values, dones):
        # Implement GAE
        pass

    def update(self, states, actions, returns, advantages):
        # Actor-critic update
        pass

    def train(self, env, num_episodes):
        # Main training loop
        pass
```

## Key Equations Summary

### Advantage Function
```
A^π(s,a) = Q^π(s,a) - V^π(s)
         ≈ r + γV(s') - V(s)  (one-step)
         ≈ G_t - V(s)  (Monte Carlo)
```

### A2C Update Rules
```
Actor: θ ← θ + α_π · A(s,a) · ∇_θ log π(a|s; θ)
Critic: w ← w + α_v · (G_t - V(s; w)) · ∇_w V(s; w)
```

### Generalized Advantage Estimation
```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
           = δ_t + γλ A_{t+1}^GAE(λ)

where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### Combined Loss (A2C with Entropy)
```
L = L_policy + c_1 · L_value - c_2 · H

L_policy = -E[log π(a|s) · A]
L_value = E[(V(s) - G_t)^2]
H = -E[log π(a|s)]  (entropy)
```

## Common Pitfalls

1. **Gradient conflicts in shared networks**: Use gradient clipping, different loss weights
2. **Critic underfitting**: Increase critic learning rate or capacity
3. **Policy collapse**: Add entropy bonus, check gradient norms
4. **High variance still**: Lower λ in GAE, increase batch size
5. **Unstable training**: Lower learning rates, normalize advantages
6. **Value function diverging**: Clip value targets, use Huber loss
7. **Wrong advantage computation**: Ensure you compute GAE backward through trajectory

## Extensions and Variations

### Parallel Synchronized A2C
```python
# Modern approach: use vectorized environments
envs = gym.vector.make('LunarLander-v2', num_envs=8)
# Collect data from all environments simultaneously
# More stable than asynchronous A3C
```

### V-Trace (IMPALA)
- Off-policy correction for actor-critic
- Allows greater asynchrony without bias
- Used in large-scale distributed training

### Multi-step Returns
- Use n-step returns instead of one-step
- Intermediate between TD and Monte Carlo
- Often combined with GAE

### Recurrent Actor-Critic
- Use LSTM or GRU for partially observable environments
- Maintain hidden state across timesteps
- Necessary for memory-dependent tasks

## Debugging Tips

1. **Critic learning**: Plot predicted values vs. actual returns (should match)
2. **Advantage distribution**: Should be roughly centered at 0
3. **Policy entropy**: Should decrease but not collapse to 0
4. **Gradient norms**: Track both actor and critic (should be stable)
5. **Explained variance**: (1 - Var[returns - V(s)] / Var[returns]) should be high (> 0.7)
6. **Sanity check**: Run vanilla policy gradient (no critic) as baseline

## Next Steps

After mastering A2C:
- **Week 13**: TRPO & PPO (stabilize policy updates with trust regions)
- **Week 14**: Continuous control with deterministic policies (DDPG, TD3, SAC)
- **Advanced**: Distributed training (IMPALA, Ape-X), curiosity-driven exploration

## Key Takeaways

1. **Actor-critic combines best of both**: Policy optimization (actor) + value-based (critic)
2. **Advantage reduces variance**: Use A(s,a) instead of Q(s,a) or returns
3. **GAE is powerful**: Single λ parameter controls bias-variance trade-off
4. **A2C > A3C**: Synchronous parallel training is more stable and preferred today
5. **Architecture matters**: Start with shared encoder, experiment if needed
6. **Hyperparameters critical**: Learning rates, λ, entropy coefficient need tuning

Actor-critic is the foundation for modern RL algorithms. Understanding it deeply is essential before moving to PPO and SAC.
