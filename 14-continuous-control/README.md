# Week 14: Continuous Control (DDPG, TD3, SAC)

## Learning Objectives

- [ ] Understand deterministic policy gradients and why they're efficient for continuous control
- [ ] Implement DDPG (Deep Deterministic Policy Gradient) algorithm
- [ ] Understand TD3 improvements: twin Q-networks, delayed policy updates, target policy smoothing
- [ ] Master SAC (Soft Actor-Critic) and maximum entropy framework
- [ ] Understand the reparameterization trick for stochastic policies
- [ ] Apply off-policy continuous control to robotics tasks

## Key Concepts

### 1. Deterministic Policy Gradient (DPG)

**Motivation**: Stochastic policies in continuous action spaces require integrating over actions.

**Stochastic policy gradient** (REINFORCE, PPO):
```
∇_θ J(θ) = E_s,a [∇_θ log π(a|s; θ) Q(s,a)]
         = ∫∫ ρ(s) π(a|s) ∇_θ log π(a|s) Q(s,a) ds da
```
Requires sampling actions from π to estimate integral.

**Deterministic policy**: μ(s) → a (single action per state)
```
∇_θ J(θ) = E_s [∇_θ Q(s, μ(s; θ))]
         = E_s [∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s; θ)]
```
Much simpler! No integral over actions.

**Key insight** (Silver et al. 2014): Deterministic policy gradient is limit of stochastic policy gradient as variance → 0.

**Advantages**:
- More efficient (no action sampling needed)
- Works well in high-dimensional action spaces
- Natural for continuous control

**Disadvantages**:
- No exploration from policy (need separate exploration strategy)
- Can get stuck in local optima

### 2. DDPG (Deep Deterministic Policy Gradient)

**Algorithm**: Lillicrap et al. 2015

**Core idea**: Actor-critic with deterministic policy + experience replay + target networks

**Components**:

1. **Actor** (deterministic policy): μ(s; θ)
2. **Critic** (Q-function): Q(s,a; w)
3. **Target actor**: μ'(s; θ')
4. **Target critic**: Q'(s,a; w')
5. **Replay buffer**: Store (s, a, r, s', done)
6. **Exploration noise**: Add noise to actions during training

**Training loop**:
```python
# Initialize networks and replay buffer
actor = Actor(θ)
critic = Critic(w)
target_actor = copy(actor)
target_critic = copy(critic)
replay_buffer = ReplayBuffer()

for episode in episodes:
    state = env.reset()
    for t in steps:
        # 1. Select action with exploration noise
        action = actor(state) + noise()
        next_state, reward, done = env.step(action)

        # 2. Store transition
        replay_buffer.add(state, action, reward, next_state, done)

        # 3. Sample minibatch from replay
        batch = replay_buffer.sample(batch_size)

        # 4. Update critic (minimize TD error)
        target_Q = reward + γ * target_critic(next_state, target_actor(next_state))
        critic_loss = (critic(state, action) - target_Q)²
        update critic using gradient descent

        # 5. Update actor (maximize Q)
        actor_loss = -critic(state, actor(state))
        update actor using gradient ascent

        # 6. Soft update target networks
        target_actor ← τ * actor + (1-τ) * target_actor
        target_critic ← τ * critic + (1-τ) * target_critic
```

**Key innovations**:

**Target networks** (from DQN):
```python
# Slow-moving targets for stability
target_Q = r + γ * Q'(s', μ'(s'; θ'); w')
```

**Soft updates**:
```python
# Slowly blend current into target (τ=0.005 typical)
θ' ← τθ + (1-τ)θ'
w' ← τw + (1-τ)w'
```

**Replay buffer** (off-policy learning):
- Store past experiences
- Break correlation in data
- Sample efficiency

**Ornstein-Uhlenbeck noise** (temporal correlation):
```python
dx_t = θ(μ - x_t)dt + σdW_t
# Correlated noise for smoother exploration
```

**Advantages**:
- Off-policy: sample efficient
- Works on continuous action spaces
- Stable with target networks

**Disadvantages**:
- Sensitive to hyperparameters
- Can overestimate Q-values
- Requires careful tuning

### 3. TD3 (Twin Delayed DDPG)

**Algorithm**: Fujimoto et al. 2018

**Motivation**: DDPG suffers from Q-value overestimation, leading to poor performance.

**Three key improvements**:

#### 1. Twin Q-Networks (Clipped Double Q-Learning)

**Problem**: Single critic overestimates Q-values (positive bias).

**Solution**: Train two critics Q_1, Q_2, use minimum for targets.
```python
# Two critics
Q1_target = r + γ * Q1'(s', μ'(s'))
Q2_target = r + γ * Q2'(s', μ'(s'))

# Use minimum (pessimistic)
target = r + γ * min(Q1'(s', μ'(s')), Q2'(s', μ'(s')))

# Update both critics
loss1 = (Q1(s,a) - target)²
loss2 = (Q2(s,a) - target)²
```

**Why it works**: Taking min reduces positive bias from function approximation errors.

#### 2. Delayed Policy Updates

**Problem**: Policy update frequency affects stability.

**Solution**: Update actor less frequently than critic (e.g., every 2 critic updates).
```python
for step in steps:
    # Always update critics
    update_critics()

    # Update actor only every d steps
    if step % d == 0:
        update_actor()
        soft_update_targets()
```

**Why it works**: Gives critic time to stabilize before policy changes.

#### 3. Target Policy Smoothing

**Problem**: Deterministic policies can exploit Q-function errors.

**Solution**: Add noise to target actions to smooth Q-values.
```python
# Regular target (DDPG)
target_action = μ'(s')

# TD3 target with noise
target_action = μ'(s') + clip(ε, -c, c)
where ε ~ N(0, σ)

# Then compute target Q
target_Q = r + γ * Q'(s', target_action)
```

**Why it works**: Smooths Q-function, makes it harder for policy to exploit errors.

**TD3 vs DDPG**:
| Feature | DDPG | TD3 |
|---------|------|-----|
| Critics | 1 | 2 (twin) |
| Target for actor | Q'(s', μ'(s')) | min(Q1', Q2')(s', μ'(s')+noise) |
| Policy update freq | Every step | Every d steps |
| Performance | Good | Better (more stable) |

### 4. SAC (Soft Actor-Critic)

**Algorithm**: Haarnoja et al. 2018

**Motivation**: Maximum entropy reinforcement learning.

**Objective**: Maximize expected return AND entropy:
```
J(π) = E[Σ_t r_t + α H(π(·|s_t))]

where H(π(·|s)) = -E_a~π[log π(a|s)] is entropy
```

**Key idea**: Encourage exploration by maximizing entropy while maximizing reward.

**Benefits**:
- Better exploration (high entropy = diverse actions)
- More robust to reward function misspecification
- Naturally handles multi-modal action distributions

**Components**:

1. **Stochastic policy** (Gaussian): π(a|s) = N(μ(s), Σ(s))
2. **Twin Q-functions**: Q_1, Q_2 (like TD3)
3. **Target Q-networks**: Q_1', Q_2'
4. **Automatic temperature tuning**: α (entropy coefficient)

**SAC training**:

**Critic update**:
```python
# Sample action from current policy
a' ~ π(·|s')
log_prob = log π(a'|s')

# Compute target (includes entropy term)
target_Q = r + γ (min(Q1', Q2')(s', a') - α * log_prob)

# Update both critics
loss1 = (Q1(s,a) - target_Q)²
loss2 = (Q2(s,a) - target_Q)²
```

**Actor update** (maximize Q + entropy):
```python
# Sample action from current policy
a ~ π(·|s; θ)
log_prob = log π(a|s; θ)

# Actor loss (negative because we maximize)
actor_loss = -E[min(Q1, Q2)(s, a) - α * log_prob]

# Update actor
∇_θ actor_loss
```

**Temperature update** (automatic tuning):
```python
# Target entropy (usually -dim(action_space))
target_entropy = -dim(A)

# Temperature loss
α_loss = -α * (log_prob + target_entropy)

# Update α
∇_α α_loss
```

### 5. Reparameterization Trick

**Problem**: How to backpropagate through stochastic action sampling?

**Naive approach** (doesn't work):
```python
a ~ N(μ(s), σ(s))  # Stochastic, can't backprop through sampling!
loss = -Q(s, a)
```

**Reparameterization trick**:
```python
# Separate deterministic and stochastic parts
ε ~ N(0, I)  # Standard normal (no parameters)
a = μ(s) + σ(s) * ε  # Deterministic transformation

# Now can backprop through μ, σ
loss = -Q(s, a)
∇_θ loss = ∇_θ Q(s, μ(s) + σ(s)*ε) = ∇_a Q · (∇_θ μ + ∇_θ σ * ε)
```

**Why it works**: Moves randomness outside the parameters, allowing gradients to flow.

**For SAC with tanh squashing**:
```python
# Sample from Gaussian
μ, log_σ = actor(s)
ε ~ N(0, I)
a_raw = μ + exp(log_σ) * ε

# Squash to bounded range
a = tanh(a_raw)

# Adjust log_prob for change of variables
log_prob = log π(a_raw) - Σ log(1 - tanh²(a_raw))
```

## Algorithm Comparison

| Algorithm | Policy | Off-Policy | Q-Networks | Exploration | Sample Efficiency | Stability |
|-----------|--------|------------|------------|-------------|-------------------|-----------|
| DDPG | Deterministic | Yes | 1 + target | Noise | High | Medium |
| TD3 | Deterministic | Yes | 2 + targets | Noise | High | High |
| SAC | Stochastic | Yes | 2 + targets | Entropy | Very High | Very High |
| PPO | Stochastic | No | 0 (value net) | Entropy | Medium | High |

**When to use each**:
- **DDPG**: Baseline, educational, simple continuous control
- **TD3**: When you want off-policy + deterministic + stability
- **SAC**: Default choice for continuous control (best overall)
- **PPO**: When on-policy is acceptable, simpler implementation

## Implementation Task

### Pendulum-v1 with DDPG, BipedalWalker with SAC

#### Task 1: Pendulum-v1 with DDPG
**Environment**:
- Observation: 3D (cos(θ), sin(θ), angular velocity)
- Action: 1D continuous torque in [-2, 2]
- Reward: -(θ² + 0.1*θ_dot² + 0.001*action²)
- Goal: Swing up and balance

**Implementation**:
```python
hyperparams_ddpg = {
    'buffer_size': 100000,
    'learning_rate_actor': 1e-4,
    'learning_rate_critic': 1e-3,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 100,
    'exploration_noise': 0.1,
    'hidden_dim': [256, 256],
}
```

#### Task 2: BipedalWalker-v3 with SAC
**Environment**:
- Observation: 24D (joint angles, velocities, ground contact, etc.)
- Action: 4D continuous (hip and knee motors)
- Reward: Forward progress - energy cost
- Goal: Walk forward smoothly

**Implementation**:
```python
hyperparams_sac = {
    'buffer_size': 1000000,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'tau': 0.005,
    'batch_size': 256,
    'target_entropy': -4,  # -dim(action_space)
    'hidden_dim': [256, 256],
    'auto_tune_alpha': True,
}
```

## Key Equations Summary

### Deterministic Policy Gradient
```
∇_θ J(θ) = E_s [∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s; θ)]
```

### DDPG Updates
```
Critic: w ← w + α_critic ∇_w (Q(s,a; w) - y)²
        where y = r + γ Q'(s', μ'(s'; θ'); w')

Actor:  θ ← θ + α_actor ∇_θ Q(s, μ(s; θ); w)
```

### TD3 Target
```
y = r + γ min_i Q_i'(s', μ'(s') + ε)
where ε ~ clip(N(0,σ), -c, c)
```

### SAC Objective
```
J(π) = E[Σ_t r_t + α H(π(·|s_t))]
     = E[Q(s,a) - α log π(a|s)]
```

## Common Pitfalls

1. **Forgetting target networks**: Leads to instability
2. **Wrong soft update**: Use τ=0.005, not 0.001 or 0.01
3. **Insufficient exploration**: Need noise or entropy
4. **Too small replay buffer**: Need 10^6 for complex tasks
5. **Not using gradient clipping**: Can diverge
6. **Wrong reward scaling**: Normalize rewards
7. **Updating actor too frequently**: Use delayed updates (TD3)

## Extensions

- **Distributional RL**: Learn full Q-distribution, not just mean
- **Prioritized Experience Replay**: Sample important transitions more
- **Hindsight Experience Replay**: Learn from failures in goal-conditioned tasks
- **Multi-task/Meta-RL**: Single policy for multiple tasks

## Next Steps

After Week 14, you've mastered core RL algorithms! Next directions:
- **Model-based RL**: Learn environment models, plan
- **Offline RL**: Learn from fixed datasets
- **Multi-agent RL**: Coordination, competition
- **Hierarchical RL**: Temporal abstraction
- **Real-world applications**: Robotics, autonomous vehicles, finance

## Key Takeaways

1. **Deterministic policies efficient for continuous control**: No action sampling needed
2. **Off-policy + replay buffer = sample efficiency**: Reuse past data
3. **TD3 fixes DDPG**: Twin Q-networks, delayed updates, target smoothing
4. **SAC is SOTA**: Maximum entropy, automatic tuning, robust
5. **Reparameterization trick**: Enables backprop through stochastic policies

**Practical advice**: Start with SAC for continuous control. It's robust, sample-efficient, and widely used in robotics.
