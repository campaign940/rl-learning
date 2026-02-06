# Week 12 Quiz: Actor-Critic Methods

## Question 1: Conceptual Understanding

**How does actor-critic combine the strengths of value-based and policy-based methods? What weaknesses does it inherit from each?**

<details>
<summary>Click to reveal answer</summary>

### Answer

Actor-critic methods create a powerful hybrid by combining two complementary approaches:

## Strengths Inherited

### From Policy-Based Methods (Actor)

**1. Natural handling of continuous/large action spaces**
```python
# Actor directly outputs action distribution
mu, sigma = actor_network(state)
action = Normal(mu, sigma).sample()
# No need for argmax over infinite actions
```

**2. Stochastic policies**
- Can represent mixed strategies (important for game theory)
- Natural exploration through policy entropy
- Handles partial observability better

**3. Guaranteed convergence**
- Following policy gradient → local optimum convergence
- Smooth policy changes (no sudden jumps)

### From Value-Based Methods (Critic)

**1. Low variance estimation**
```python
# Instead of Monte Carlo return (high variance):
G_t = sum([gamma**k * r[t+k] for k in range(T-t)])  # Sum T-t random variables

# Use bootstrapped estimate (lower variance):
advantage = r + gamma * V(s_next) - V(s)  # Only 1 random variable (r)
```

**2. Sample efficiency**
- Bootstrapping allows learning from incomplete episodes
- Can update after every step (not just episode end)
- Reuses value estimates across multiple policy updates

**3. Credit assignment**
- Critic provides better signal about which actions were actually good
- Advantages A(s,a) center around 0 → clearer gradient direction

## How They Synergize

**Variance Reduction**:
```
Pure Policy Gradient: ∇J = E[∇log π · G_t]  (high variance from G_t)
Actor-Critic: ∇J = E[∇log π · A(s,a)]      (low variance from bootstrapped A)
```

**Bias-Variance Trade-off**:
- Policy gradient: unbiased but high variance
- Bootstrapping: biased but low variance
- Actor-critic: slightly biased, much lower variance (net win!)

**Faster Learning**:
- Critic provides dense learning signal (every step)
- Actor gets more informative gradients (advantages vs. raw returns)
- Typical speedup: 5-10x fewer samples vs. pure policy gradient

## Weaknesses Inherited

### From Policy-Based Methods

**1. Local optima**
```
Policy gradient methods converge to LOCAL optimum only
- Can get stuck in suboptimal policies
- Sensitive to initialization
- May need multiple random seeds
```

**2. On-policy learning** (for standard actor-critic)
- Must generate new data with current policy
- Cannot efficiently reuse old data
- Lower sample efficiency than off-policy methods (DQN, SAC)

**3. Sensitive to hyperparameters**
- Learning rates for actor and critic must be balanced
- Entropy coefficient, GAE λ, value loss weight all matter
- More knobs to tune than pure value-based methods

### From Value-Based Methods

**1. Biased gradient estimates**
```python
# True advantage: A(s,a) = Q(s,a) - V(s)
# Estimated: A(s,a) ≈ r + γV(s') - V(s)

# If V is inaccurate, we get biased policy gradients!
# This can slow learning or lead to suboptimal policies
```

**2. Critic approximation errors compound**
- Critic errors → biased advantages → wrong policy updates → worse data → worse critic
- Can create a vicious cycle
- Requires careful value function training

**3. Instability from bootstrapping**
- Deadly triad issues (function approximation + bootstrapping + off-policy)
- Value function can diverge if not careful
- Requires techniques like target networks, gradient clipping

### New Weaknesses from Combination

**1. Gradient conflicts in shared architecture**
```python
# Actor wants to maximize: E[log π · A]
# Critic wants to minimize: E[(V - G)²]

# Gradients flow through shared encoder can conflict
# Actor: "Make this feature useful for policy"
# Critic: "Make this feature useful for value"
```

**2. Two learning rates to balance**
```python
optimizer_actor = Adam(actor.parameters(), lr=3e-4)
optimizer_critic = Adam(critic.parameters(), lr=1e-3)  # Usually higher

# If critic LR too low: biased advantages, slow learning
# If critic LR too high: unstable values, actor confused
# If actor LR too high: policy changes too fast, critic can't keep up
```

**3. More complex implementation**
- Must maintain two networks (or carefully share parameters)
- GAE computation requires careful indexing
- Advantage normalization, entropy bonuses, gradient clipping all needed

## Practical Comparison

| Method | Sample Efficiency | Stability | Implementation | Continuous Actions |
|--------|------------------|-----------|----------------|-------------------|
| DQN (value) | High (off-policy) | Medium | Medium | Poor |
| REINFORCE (policy) | Low | High | Easy | Excellent |
| A2C/A3C (actor-critic) | Medium | Medium | Hard | Excellent |
| PPO (advanced actor-critic) | Medium-High | High | Hard | Excellent |
| SAC (off-policy actor-critic) | Very High | High | Very Hard | Excellent |

## Key Insights

**Why actor-critic became dominant**:
1. Gets most of the benefits from both paradigms
2. Weaknesses are manageable with modern techniques (GAE, PPO clipping, etc.)
3. Scales to high-dimensional continuous control
4. Foundation for state-of-the-art algorithms (PPO, SAC)

**When to use alternatives**:
- **Pure value-based (DQN)**: Discrete actions, off-policy data, don't need stochastic policy
- **Pure policy-based (REINFORCE)**: Simple baseline, educational, don't mind sample inefficiency
- **Actor-critic (A2C/PPO)**: Default choice for most RL problems today

**The evolution**:
```
Value-based (1980s-2010s) → Policy-based (2010s) → Actor-Critic (2015+) → Modern (PPO/SAC, 2017+)
                                                                         ↓
                                                        Dominant paradigm today
```

## Conclusion

Actor-critic is a "best of both worlds" approach that:
- **Inherits**: Continuous action handling + convergence (from policy-based); low variance + sample efficiency (from value-based)
- **Sacrifices**: Unbiased gradients (now biased from bootstrapping); simplicity (more complex than either alone)
- **Net result**: Much better than pure policy gradient, competitive with value-based, scales to modern problems

The biases introduced are usually worth the massive variance reduction. With techniques like GAE, PPO clipping, and careful architecture design, actor-critic methods are the foundation of modern RL.

</details>

---

## Question 2: Mathematical Derivation

**Derive GAE as a weighted average of k-step advantage estimates A^(1), A^(2), ..., A^(∞). Show how λ controls the weights.**

<details>
<summary>Click to reveal answer</summary>

### Answer

## Generalized Advantage Estimation (GAE) Derivation

We'll build GAE from first principles, starting with k-step advantage estimates.

### Step 1: Define k-Step Advantage Estimates

**1-step advantage** (TD, high bias, low variance):
```
A_t^(1) = r_t + γV(s_{t+1}) - V(s_t)
        = δ_t  (TD error)
```

**2-step advantage**:
```
A_t^(2) = r_t + γr_{t+1} + γ²V(s_{t+2}) - V(s_t)
        = r_t + γ(r_{t+1} + γV(s_{t+2})) - V(s_t)
        = r_t + γV(s_{t+1}) - V(s_t) + γ(r_{t+1} + γV(s_{t+2}) - V(s_{t+1}))
        = δ_t + γδ_{t+1}
```

**3-step advantage**:
```
A_t^(3) = r_t + γr_{t+1} + γ²r_{t+2} + γ³V(s_{t+3}) - V(s_t)
        = δ_t + γδ_{t+1} + γ²δ_{t+2}
```

**k-step advantage** (general):
```
A_t^(k) = Σ_{l=0}^{k-1} γ^l r_{t+l} + γ^k V(s_{t+k}) - V(s_t)
        = Σ_{l=0}^{k-1} γ^l δ_{t+l}
```

**∞-step advantage** (Monte Carlo, no bias, high variance):
```
A_t^(∞) = Σ_{l=0}^∞ γ^l r_{t+l} - V(s_t)
        = G_t - V(s_t)  (return minus baseline)
        = Σ_{l=0}^∞ γ^l δ_{t+l}
```

### Step 2: Bias-Variance Trade-off

Each k-step estimator has different properties:

| k | Bias | Variance | Formula |
|---|------|----------|---------|
| 1 | High | Low | δ_t |
| 2 | Medium | Medium | δ_t + γδ_{t+1} |
| k | Lower | Higher | Σ_{l=0}^{k-1} γ^l δ_{t+l} |
| ∞ | None | Very High | Σ_{l=0}^∞ γ^l δ_{t+l} |

**Key insight**: We want to combine these estimates to get a good bias-variance trade-off!

### Step 3: GAE as Weighted Average

**Idea**: Take a weighted average of all k-step advantages:
```
A_t^GAE = Σ_{k=1}^∞ w_k · A_t^(k)
```

**What weights w_k should we use?**

Requirements:
1. Weights should sum to 1: Σ_{k=1}^∞ w_k = 1
2. Should exponentially decay (trust nearby estimates more)
3. Single hyperparameter to control the trade-off

**Solution**: Use geometric series with parameter λ ∈ [0,1]:
```
w_k = (1 - λ) λ^{k-1}  for k = 1, 2, 3, ...
```

**Verify weights sum to 1**:
```
Σ_{k=1}^∞ w_k = (1-λ) Σ_{k=1}^∞ λ^{k-1}
               = (1-λ) · 1/(1-λ)  (geometric series)
               = 1 ✓
```

### Step 4: Full GAE Formula

**GAE with weighted k-step advantages**:
```
A_t^GAE(λ) = Σ_{k=1}^∞ (1-λ) λ^{k-1} A_t^(k)
           = Σ_{k=1}^∞ (1-λ) λ^{k-1} · Σ_{l=0}^{k-1} γ^l δ_{t+l}
```

**Simplify by reordering summations**:
```
A_t^GAE(λ) = (1-λ) [A_t^(1) + λA_t^(2) + λ²A_t^(3) + ...]
           = (1-λ) [δ_t + λ(δ_t + γδ_{t+1}) + λ²(δ_t + γδ_{t+1} + γ²δ_{t+2}) + ...]
           = (1-λ) [δ_t(1 + λ + λ² + ...) + γδ_{t+1}(λ + λ² + ...) + γ²δ_{t+2}(λ² + ...) + ...]
           = (1-λ) [δ_t/(1-λ) + γδ_{t+1}·λ/(1-λ) + γ²δ_{t+2}·λ²/(1-λ) + ...]
           = δ_t + γλδ_{t+1} + γ²λ²δ_{t+2} + ...
           = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

**Final GAE formula**:
```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### Step 5: Recursive Form (Computational)

**For implementation, use recursive form**:
```
A_t^GAE(λ) = δ_t + γλ A_{t+1}^GAE(λ)
```

**Proof**:
```
A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
    = δ_t + Σ_{l=1}^∞ (γλ)^l δ_{t+l}
    = δ_t + γλ Σ_{l=0}^∞ (γλ)^l δ_{t+1+l}
    = δ_t + γλ A_{t+1}  ✓
```

**Implementation**:
```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute GAE advantages.

    Args:
        rewards: [T] rewards
        values: [T+1] value estimates (includes bootstrap value)
        dones: [T] episode termination flags
        gamma: discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: [T] GAE advantages
    """
    T = len(rewards)
    advantages = np.zeros(T)
    last_gae = 0

    # Backward iteration (from T-1 to 0)
    for t in reversed(range(T)):
        # TD error
        if t == T - 1:
            next_value = values[t+1] if not dones[t] else 0
        else:
            next_value = values[t+1] * (1 - dones[t])

        delta = rewards[t] + gamma * next_value - values[t]

        # GAE recursion
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    return advantages
```

### Step 6: Special Cases of GAE

**λ = 0** (1-step TD):
```
A_t^GAE(0) = Σ_{l=0}^∞ (γ·0)^l δ_{t+l}
           = δ_t + 0 + 0 + ...
           = δ_t
           = r_t + γV(s_{t+1}) - V(s_t)
```
High bias (trusts V fully), low variance.

**λ = 1** (Monte Carlo):
```
A_t^GAE(1) = Σ_{l=0}^∞ γ^l δ_{t+l}
           = Σ_{l=0}^∞ γ^l (r_{t+l} + γV(s_{t+l+1}) - V(s_{t+l}))
```

Telescope the sum:
```
= r_t + γV(s_{t+1}) - V(s_t)
  + γr_{t+1} + γ²V(s_{t+2}) - γV(s_{t+1})
  + γ²r_{t+2} + γ³V(s_{t+3}) - γ²V(s_{t+2})
  + ...
= r_t + γr_{t+1} + γ²r_{t+2} + ... - V(s_t)
= G_t - V(s_t)
```

No bias (doesn't trust V at all), high variance.

**λ = 0.95** (typical):
Exponential decay with effective horizon ≈ 20 steps.
```
Weights: 1, 0.95, 0.90, 0.86, 0.81, ...
         (95%, 90%, 86%, 81%, ...)
```

### Step 7: Intuition for λ

**Interpretation 1: Time horizon**
- λ controls how far into the future we look
- Small λ: short-sighted (trust immediate V estimates)
- Large λ: far-sighted (look many steps ahead)

**Interpretation 2: Trust in value function**
- λ = 0: full trust in V(s) (use 1-step bootstrapping)
- λ = 1: no trust in V(s) (use actual returns)
- λ = 0.95: moderate trust (blend both)

**Interpretation 3: Effective horizon**
```
Effective horizon ≈ 1 / (1 - γλ)

For γ=0.99, λ=0.95: horizon ≈ 1/(1-0.9405) ≈ 17 steps
For γ=0.99, λ=0.99: horizon ≈ 1/(1-0.9801) ≈ 50 steps
For γ=0.99, λ=0.0:  horizon = 1 step
```

### Step 8: Visualizing the Weights

**How much does each k-step estimate contribute?**

```
w_k = (1-λ) λ^{k-1}

For λ=0.95:
k=1: w_1 = 0.05 (5%)
k=2: w_2 = 0.0475 (4.75%)
k=3: w_3 = 0.0451 (4.51%)
...
k=20: w_20 ≈ 0.018 (1.8%)
k=50: w_50 ≈ 0.004 (0.4%)

# Decays exponentially, but all contribute
```

For λ=0.0 (all weight on k=1):
```
k=1: w_1 = 1.0 (100%)
k≥2: w_k = 0
```

For λ=1.0 (equal weight to all):
```
This doesn't work! Weights don't sum to 1 unless we use Monte Carlo formula
```

### Summary: Key Formulas

**Main formula**:
```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
```

**Recursive form**:
```
A_t^GAE(λ) = δ_t + γλ A_{t+1}^GAE(λ)
```

**Weighted k-step form**:
```
A_t^GAE(λ) = Σ_{k=1}^∞ (1-λ)λ^{k-1} A_t^(k)
```

**Special cases**:
```
λ=0: A_t = δ_t (TD)
λ=1: A_t = G_t - V(s_t) (MC)
```

## Conclusion

GAE elegantly solves the bias-variance trade-off by:
1. Taking weighted average of all k-step advantages
2. Using exponential weights controlled by single parameter λ
3. Providing smooth interpolation from TD (λ=0) to MC (λ=1)
4. Enabling efficient recursive computation

In practice, λ=0.95 works remarkably well across diverse tasks, making GAE a cornerstone of modern actor-critic algorithms like PPO and TRPO.

</details>

---

## Question 3: Algorithm Comparison

**Compare A2C vs A3C: parallelism strategy, gradient staleness, modern relevance. Why did A2C become the preferred method despite A3C being published first?**

<details>
<summary>Click to reveal answer</summary>

### Answer

## A3C vs A2C: The Great Actor-Critic Showdown

### Historical Context

**2016**: Mnih et al. publish A3C (Asynchronous Advantage Actor-Critic)
- Revolutionary: showed parallelism can replace experience replay
- Achieved state-of-the-art on Atari
- Asynchronous updates from multiple workers

**2016-2017**: Community discovers synchronous version works better
- OpenAI Baselines implements A2C (synchronous variant)
- Finds A2C more stable and efficient
- A3C rarely used in practice today

### Core Difference: Parallelism Strategy

**A3C (Asynchronous)**:
```python
# Pseudocode for A3C

# Global shared network
global_network = ActorCritic()

def worker(worker_id):
    local_network = ActorCritic()

    while training:
        # 1. Copy global parameters (may be stale!)
        local_network.load(global_network.parameters())

        # 2. Collect trajectory
        for t in range(t_max):
            action = local_network.act(state)
            next_state, reward = env.step(action)
            # ... store transition

        # 3. Compute gradients locally
        advantages = compute_gae(trajectory)
        grads = compute_gradients(local_network, advantages)

        # 4. ASYNCHRONOUSLY update global network
        with global_lock:  # Brief lock for update
            global_network.apply_gradients(grads)

# Run multiple workers in parallel threads/processes
for i in range(num_workers):
    Thread(target=worker, args=(i,)).start()
```

**A2C (Synchronous)**:
```python
# Pseudocode for A2C

network = ActorCritic()

# Vectorized environments (all run simultaneously)
envs = VectorEnv(num_envs=16)

while training:
    # 1. All workers use SAME current parameters
    states = envs.get_states()  # [num_envs, obs_dim]

    # 2. Collect batch of trajectories in parallel
    for t in range(t_max):
        actions = network.act(states)  # Vectorized
        next_states, rewards, dones = envs.step(actions)
        # ... store transitions

    # 3. Compute advantages for ALL trajectories
    advantages = compute_gae_batch(all_trajectories)

    # 4. Single SYNCHRONOUS update with averaged gradients
    grads = compute_gradients(network, advantages)
    network.apply_gradients(grads)
    # All workers automatically use new parameters next iteration
```

### Detailed Comparison

#### 1. Gradient Staleness

**A3C Problem**: Workers compute gradients using outdated parameters.

```python
# A3C Timeline for Worker 1:
t=0:  Copy params (version 100)
t=1:  Collect data using params v100
t=2:  More data with v100
t=20: Compute gradients based on v100
t=21: Lock and update global network
      # But global is now at version 140! (other workers updated it)
t=22: Apply gradients computed from v100 to v140 (stale!)
```

**Staleness effects**:
- Gradients computed for old policy applied to new policy
- Can point in wrong direction (policy has changed)
- Reduces effective learning rate
- Can cause instability

**How stale can it get?**
```
Max staleness ≈ num_workers * t_max updates

Example: 16 workers, t_max=20 steps
→ Could be 320 updates behind!
```

**A2C Solution**: No staleness. All workers use current parameters.
```python
# A2C: Everyone waits, then everyone updates together
# All workers ALWAYS have same parameters
```

#### 2. Implementation Complexity

**A3C Challenges**:
```python
# Must handle:
# 1. Thread-safe parameter sharing
global_network = ActorCritic().share_memory()  # PyTorch

# 2. Locks for updates (contention!)
with global_lock:
    global_network.apply_gradients(grads)

# 3. Managing multiple processes/threads
import torch.multiprocessing as mp
processes = [mp.Process(target=worker) for _ in range(num_workers)]

# 4. Potential deadlocks, race conditions
# 5. Harder to debug (non-deterministic)
```

**A2C Simplicity**:
```python
# Just vectorize!
envs = gym.vector.make('CartPole-v1', num_envs=16)
states = envs.reset()  # [16, obs_dim]
actions = policy(states)  # [16, action_dim]
next_states, rewards, dones, _ = envs.step(actions)

# No threads, no locks, fully deterministic
```

**Lines of code**:
- A3C implementation: ~800-1000 lines
- A2C implementation: ~400-600 lines

#### 3. Computational Efficiency

**A3C (Asynchronous)**:
```
Wall-clock time: FAST (no waiting for slow workers)
GPU utilization: POOR (one worker at a time on GPU)
CPU utilization: GOOD (all workers busy)

Example (16 workers):
- Worker 1: 100ms data collection, 50ms gradient
- Worker 2: 120ms data collection, 50ms gradient
- ...
- Workers don't wait for each other
- But GPU update is serial (only one worker updates at a time)
```

**A2C (Synchronous)**:
```
Wall-clock time: Slower (wait for slowest worker)
GPU utilization: EXCELLENT (batch update)
CPU utilization: GOOD (all workers collect data simultaneously)

Example (16 workers vectorized):
- All 16 workers collect data: max(worker_times) ≈ 120ms
- Single batched gradient update: 50ms (for all 16!)
- GPU processes batch of 16 efficiently
```

**Modern GPUs favor A2C**:
```python
# A3C: 16 separate forward passes
for i in range(16):
    action = network(states[i])  # Small batch size=1

# A2C: 1 batched forward pass
actions = network(states)  # Large batch size=16

# GPU throughput much higher with batching!
# A2C can be faster wall-clock time on modern hardware
```

#### 4. Training Stability

**A3C Instability Sources**:

1. **Stale gradients**: Can point in wrong direction
2. **Inconsistent updates**: Different workers see different policies
3. **Effective learning rate chaos**:
```python
# Sometimes many workers update at once → big jump
# Sometimes sparse updates → slow learning
# Hard to reason about effective learning rate
```

**A2C Stability**:

1. **Consistent gradients**: All from current policy
2. **Predictable updates**: Fixed batch size each iteration
3. **Stable effective learning rate**:
```python
# Always average of exactly num_envs trajectories
# Stable variance in gradient estimates
# Easier to tune learning rate
```

**Empirical results**:
```
Across 57 Atari games (OpenAI Baselines experiments):
- A2C: More stable learning curves
- A3C: Higher variance, occasional divergence
- A2C: Easier hyperparameter tuning
```

#### 5. Sample Efficiency

**Theoretical**:
- Both collect same amount of data per wall-clock time
- Both on-policy (can't reuse old data)

**Practical A2C wins**:
```
A2C collects less redundant data:
- Workers are synchronized (explore different states at same time)
- Better coverage of state space

A3C can have redundancy:
- Multiple workers might explore similar states
- Stale policies lead to revisiting same regions
```

**Typical results**:
```
To reach same performance:
A2C: X samples
A3C: 1.2-1.5X samples (20-50% more)
```

#### 6. Hyperparameter Sensitivity

**A3C has more hyperparameters**:
```python
# A2C hyperparameters
num_envs = 16
t_max = 5
learning_rate = 7e-4

# A3C additional hyperparameters
num_workers = 16
t_max = 20  (usually higher for async)
learning_rate = 7e-4
# Plus: depends heavily on lock contention, CPU speed, etc.
```

**A3C more sensitive**:
- Too many workers → too much staleness
- Too few workers → not enough parallelism
- t_max interacts with staleness

**A2C more robust**:
- num_envs just affects batch size (fairly robust)
- Standard batch size principles apply

### Why A2C Won

**5 Key Reasons**:

1. **Modern hardware favors batching**
   - GPUs excellent at batch processing
   - A2C exploits this; A3C doesn't
   - Trend: GPUs getting better at batching

2. **Stability matters more than speed**
   - RL already high variance
   - A3C adds noise from staleness
   - A2C more reliable, easier to debug

3. **Simplicity = better research**
   - Easier to implement and modify
   - Deterministic (reproducible)
   - Easier to attribute results to algorithm vs. implementation

4. **Staleness was a false friend**
   - Initially thought: staleness aids exploration
   - Reality: staleness just adds noise
   - Proper exploration better addressed elsewhere (entropy bonus, curiosity)

5. **Ecosystem support**
   - Major libraries (Stable-Baselines3, RLlib) default to A2C-style
   - More examples and tutorials
   - Easier to get help

### When A3C Might Still Win

**Rare scenarios where A3C preferred**:

1. **CPU-only training**
   - No GPU batching advantage
   - Async can fully utilize all CPU cores
   - Each core runs independent worker

2. **Extremely large number of workers** (>100)
   - Synchronization overhead becomes significant
   - Waiting for slowest worker hurts
   - Async allows faster workers to keep going

3. **Heterogeneous hardware**
   - Workers on different machines with different speeds
   - Async doesn't penalize fast workers
   - Distributed training scenarios

4. **Historical codebases**
   - Legacy systems built on A3C
   - Not worth porting if already working

### Modern Landscape (2025)

**What people actually use**:

```
Synchronous parallel actor-critic (A2C-style): 90%
├─ PPO (most common)
├─ IMPALA (distributed)
└─ SAC (off-policy, but still synchronous)

Asynchronous (A3C-style): <5%
└─ Mostly legacy systems

Single-agent (no parallelism): 5%
└─ Debugging, research, toy examples
```

**Quote from John Schulman** (OpenAI, PPO author):
> "We found that synchronous A2C was more stable and easier to work with than A3C, despite the latter's theoretical appeal. Modern RL algorithms like PPO build on the A2C foundation."

### Practical Recommendations

**Starting a new project?**
→ Use A2C-style (synchronous parallel)

**Using a library?**
```python
# Stable-Baselines3: PPO (A2C-style with clipping)
from stable_baselines3 import PPO
model = PPO("MlpPolicy", "CartPole-v1", n_steps=128)

# RLlib: PPO (also A2C-style)
from ray.rllib.algorithms.ppo import PPO
trainer = PPO(config)
```

**Implementing from scratch?**
→ Start with single-agent actor-critic
→ Then add vectorized environments (A2C)
→ Don't bother with async (A3C) unless specific reason

**Debugging RL algorithm?**
→ Disable parallelism first (single env)
→ Get that working, then add vectorization

### Summary Table

| Aspect | A3C (Async) | A2C (Sync) | Winner |
|--------|-------------|------------|--------|
| Gradient staleness | High (up to 100s of updates) | None | A2C |
| Implementation | Complex (threads/locks) | Simple (vectorization) | A2C |
| GPU efficiency | Poor (serial updates) | Excellent (batched) | A2C |
| Stability | Moderate (noisy updates) | High (consistent) | A2C |
| Hyperparameter sensitivity | High | Moderate | A2C |
| Sample efficiency | Moderate | Better | A2C |
| Wall-clock speed (GPU) | Slower | Faster | A2C |
| Wall-clock speed (CPU-only) | Faster | Slower | A3C |
| Reproducibility | Poor (non-deterministic) | Perfect (deterministic) | A2C |
| Modern usage | Rare | Standard | A2C |

## Conclusion

A3C was a important historical contribution that showed parallelism could replace experience replay. However, A2C's synchronous approach proved superior in practice due to:
- Better GPU utilization
- No gradient staleness
- Simpler implementation
- More stable training

Today, virtually all modern on-policy algorithms (PPO, IMPALA) use A2C-style synchronous parallelism. A3C remains important for historical understanding but is rarely used in practice.

The lesson: **Simple and stable beats clever and complex** in RL.

</details>

---

## Question 4: Application Design

**Design an A2C architecture for LunarLander-v2. Specify network architecture, loss function for both actor and critic, and provide training hyperparameters. Explain each choice.**

<details>
<summary>Click to reveal answer</summary>

### Answer

## Complete A2C Solution for LunarLander-v2

I'll design a production-ready A2C agent with detailed justification for every choice.

### Environment Analysis

**LunarLander-v2**:
```python
Observation space: Box(8,)
- x, y position (continuous)
- x, y velocity (continuous)
- angle, angular velocity (continuous)
- left leg contact, right leg contact (binary)

Action space: Discrete(4)
- 0: do nothing
- 1: fire left engine
- 2: fire main engine
- 3: fire right engine

Reward:
- Landing: +100 to +140 (depends on softness)
- Crashing: -100
- Leg contact: +10 each
- Fuel usage: -0.3 per frame for main engine, -0.03 for side

Episode termination:
- Lands successfully
- Crashes
- Reaches 1000 steps (timeout)

Solved: average reward > 200 over 100 consecutive episodes
```

### Network Architecture

#### Option 1: Shared Encoder (Recommended)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedActorCritic(nn.Module):
    """
    Shared encoder with separate heads for actor and critic.
    Memory efficient and works well for LunarLander.
    """
    def __init__(
        self,
        obs_dim=8,
        action_dim=4,
        hidden_dim=256,
        activation='relu'
    ):
        super().__init__()

        # Shared feature extractor
        # Why 256? Large enough to learn complex features,
        # small enough to train quickly (LunarLander not too complex)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy)
        # Output: unnormalized log probabilities (logits)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        # Output: single scalar value estimate
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Orthogonal initialization (better for RL)
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Orthogonal initialization with specific gains.
        Why orthogonal? Maintains gradient flow, works well for RL.
        """
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Actor: small gain (0.01) for stable initial policy
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0)

        # Critic: gain=1 for value estimation
        nn.init.orthogonal_(self.critic_head.weight, gain=1)
        nn.init.constant_(self.critic_head.bias, 0)

    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim] observations

        Returns:
            logits: [batch_size, action_dim] action logits
            value: [batch_size, 1] state value estimate
        """
        features = self.encoder(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action_and_value(self, obs, action=None):
        """
        Sample action and compute log prob + value.
        Used during both training and rollout collection.
        """
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value
```

**Why shared encoder?**
- **Parameter efficiency**: ~65K params vs ~120K for separate networks
- **Shared features**: Position, velocity useful for both policy and value
- **Faster forward passes**: Single encoder call
- **Works well in practice**: Standard for environments like LunarLander

**Why separate heads?**
- **Flexibility**: Different output dimensions (4 vs 1)
- **Independent scaling**: Actor and critic need different output ranges
- **Easier debugging**: Can analyze each head separately

**Alternative: Separate Networks** (if shared has gradient conflicts):
```python
class SeparateActorCritic(nn.Module):
    def __init__(self, obs_dim=8, action_dim=4, hidden_dim=256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

Use separate networks if:
- Shared architecture shows unstable value function
- Actor and critic learning rates very different
- Debugging reveals gradient conflicts

### Loss Functions

#### Actor Loss (Policy Gradient)

```python
def compute_actor_loss(log_probs, advantages, entropy, entropy_coef=0.01):
    """
    Policy gradient loss with entropy regularization.

    Args:
        log_probs: [batch_size] log π(a|s)
        advantages: [batch_size] advantage estimates
        entropy: [batch_size] policy entropy per sample
        entropy_coef: weight for entropy bonus

    Returns:
        actor_loss: scalar loss for policy
    """
    # Policy gradient: maximize E[log π(a|s) * A(s,a)]
    # Negative because we minimize loss (do gradient descent)
    policy_loss = -(log_probs * advantages).mean()

    # Entropy bonus: encourage exploration
    # Negative because we want to MAXIMIZE entropy
    entropy_loss = -entropy.mean()

    # Combined actor loss
    actor_loss = policy_loss + entropy_coef * entropy_loss

    return actor_loss
```

**Why negative log prob times advantage?**
- Gradient: ∇_θ log π(a|s) points towards increasing π(a|s)
- Advantage > 0: action better than average → increase probability
- Advantage < 0: action worse than average → decrease probability

**Why entropy bonus?**
- Prevents premature convergence to deterministic policy
- Maintains exploration throughout training
- Especially important early in training

**Why entropy_coef = 0.01?**
- 0.01: Typical value for discrete action spaces
- Too high (>0.1): Policy stays random, doesn't learn
- Too low (<0.001): Policy collapses early, poor exploration
- Auto-tune: Can decay over training

#### Critic Loss (Value Function)

```python
def compute_critic_loss(values, returns, use_huber=True):
    """
    Value function loss.

    Args:
        values: [batch_size] predicted V(s)
        returns: [batch_size] actual returns (targets)
        use_huber: use Huber loss instead of MSE

    Returns:
        critic_loss: scalar loss for value function
    """
    if use_huber:
        # Huber loss: MSE for small errors, MAE for large
        # More robust to outliers than MSE
        critic_loss = F.huber_loss(values, returns)
    else:
        # Standard MSE loss
        critic_loss = F.mse_loss(values, returns)

    return critic_loss
```

**Why MSE/Huber loss?**
- Regressing values → standard supervised learning
- Target: returns G_t or advantages + values
- Huber more robust to noisy returns (common in RL)

**MSE vs Huber**:
```
MSE: L = (V - G)²
- Standard, easy to optimize
- Sensitive to outliers (large errors get squared)

Huber: L = {  0.5(V - G)²     if |V - G| <= δ
            { δ|V - G| - 0.5δ² otherwise
- Robust to outliers (linear for large errors)
- Preferred for RL (returns can be noisy)
```

#### Combined Loss

```python
def compute_total_loss(
    log_probs,
    values,
    advantages,
    returns,
    entropy,
    value_coef=0.5,
    entropy_coef=0.01
):
    """
    Combined A2C loss.

    Args:
        value_coef: weight for critic loss
        entropy_coef: weight for entropy bonus

    Returns:
        total_loss: scalar loss for joint update
        loss_dict: dictionary of individual losses (for logging)
    """
    # Actor loss
    policy_loss = -(log_probs * advantages.detach()).mean()
    entropy_loss = -entropy.mean()
    actor_loss = policy_loss + entropy_coef * entropy_loss

    # Critic loss
    critic_loss = F.huber_loss(values, returns)

    # Total loss
    total_loss = actor_loss + value_coef * critic_loss

    # For logging
    loss_dict = {
        'loss/total': total_loss.item(),
        'loss/actor': actor_loss.item(),
        'loss/critic': critic_loss.item(),
        'loss/entropy': entropy.mean().item()
    }

    return total_loss, loss_dict
```

**Why advantages.detach()?**
- Advantages computed using critic
- Don't want actor gradients to flow through critic
- Detach treats advantages as constants for actor update

**Why value_coef = 0.5?**
- Balances critic and actor learning
- 0.5: Critic slightly less weighted than actor
- Typical range: 0.25-1.0
- Tune if value function not learning or actor unstable

### Training Hyperparameters

```python
hyperparameters = {
    # Environment
    'env_name': 'LunarLander-v2',
    'num_envs': 8,  # Vectorized parallel environments

    # Network
    'hidden_dim': 256,
    'activation': 'relu',
    'shared_encoder': True,

    # Training
    'total_timesteps': 1_000_000,  # 1M steps usually enough
    'n_steps': 5,  # Rollout length (collect 5 steps before update)
    'batch_size': 8 * 5,  # num_envs * n_steps = 40

    # Optimizer
    'learning_rate': 3e-4,  # Adam learning rate
    'eps': 1e-5,  # Adam epsilon (for stability)
    'max_grad_norm': 0.5,  # Gradient clipping

    # GAE
    'gamma': 0.99,  # Discount factor
    'gae_lambda': 0.95,  # GAE lambda

    # Loss coefficients
    'value_coef': 0.5,  # Critic loss weight
    'entropy_coef': 0.01,  # Entropy bonus weight

    # Evaluation
    'eval_freq': 10_000,  # Evaluate every 10k steps
    'eval_episodes': 10,  # Average over 10 episodes
}
```

### Justification for Each Hyperparameter

**num_envs = 8**:
- More envs → more stable gradients (lower variance)
- 8 is sweet spot: good parallelism, not too much overhead
- Could try 4 (faster) or 16 (more stable)

**n_steps = 5**:
- Short rollouts for LunarLander (episodes ~200-400 steps)
- 5-step bootstrapping: low variance, moderate bias
- Could try 10-20 for longer-term dependencies

**learning_rate = 3e-4**:
- Standard for Adam in RL
- Higher (1e-3): faster but less stable
- Lower (1e-4): more stable but slower

**max_grad_norm = 0.5**:
- Prevents exploding gradients (common in RL)
- Clips gradient norm to max 0.5
- Essential for stability with shared networks

**gamma = 0.99**:
- Standard discount factor
- 0.99: values 100 steps ahead worth 37% of immediate
- LunarLander episodes ~300 steps → 0.99 appropriate

**gae_lambda = 0.95**:
- Balances bias-variance
- Effective horizon ≈ 20 steps (reasonable for LunarLander)
- Try 0.9 (lower variance) or 0.99 (lower bias) if needed

**value_coef = 0.5**:
- Weights critic loss relative to actor
- 0.5: Slight preference for actor learning
- Increase if critic underfitting, decrease if overfitting

**entropy_coef = 0.01**:
- Maintains exploration
- Decays naturally as policy improves
- Could anneal: 0.01 → 0.001 over training

### Complete Training Loop

```python
import gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def train_a2c(hyperparams):
    # Create vectorized environments
    envs = gym.vector.make(hyperparams['env_name'],
                           num_envs=hyperparams['num_envs'])

    # Create agent
    agent = SharedActorCritic(
        obs_dim=envs.single_observation_space.shape[0],
        action_dim=envs.single_action_space.n,
        hidden_dim=hyperparams['hidden_dim']
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=hyperparams['learning_rate'],
        eps=hyperparams['eps']
    )

    # Logging
    writer = SummaryWriter()
    global_step = 0

    # Storage for rollouts
    obs = torch.FloatTensor(envs.reset())

    # Training loop
    num_updates = hyperparams['total_timesteps'] // hyperparams['batch_size']

    for update in range(num_updates):
        # Storage for this rollout
        rollout_obs = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        # Collect n_steps of experience
        for step in range(hyperparams['n_steps']):
            rollout_obs.append(obs)

            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(obs)

            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_values.append(value)

            # Step environments
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rollout_rewards.append(torch.FloatTensor(reward))
            rollout_dones.append(torch.FloatTensor(done))

            obs = torch.FloatTensor(next_obs)
            global_step += hyperparams['num_envs']

        # Bootstrap value for last state
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(obs)

        # Compute returns and advantages using GAE
        returns, advantages = compute_gae(
            rollout_rewards,
            rollout_values,
            rollout_dones,
            next_value,
            hyperparams['gamma'],
            hyperparams['gae_lambda']
        )

        # Flatten batch
        b_obs = torch.cat(rollout_obs)
        b_actions = torch.cat(rollout_actions)
        b_log_probs = torch.cat(rollout_log_probs)
        b_returns = torch.cat(returns)
        b_advantages = torch.cat(advantages)

        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # Forward pass
        _, new_log_probs, entropy, values = agent.get_action_and_value(b_obs, b_actions)

        # Compute loss
        total_loss, loss_dict = compute_total_loss(
            new_log_probs,
            values.squeeze(),
            b_advantages,
            b_returns,
            entropy,
            hyperparams['value_coef'],
            hyperparams['entropy_coef']
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            agent.parameters(),
            hyperparams['max_grad_norm']
        )
        optimizer.step()

        # Logging
        if update % 10 == 0:
            for key, value in loss_dict.items():
                writer.add_scalar(key, value, global_step)

    return agent

def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """Compute GAE advantages."""
    advantages = []
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value_t = next_value
        else:
            next_value_t = values[t + 1]

        delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages.insert(0, last_gae)

    advantages = torch.stack(advantages)
    returns = advantages + torch.stack(values)

    return returns, advantages
```

### Expected Performance

**Training metrics**:
- **Episodes to solve**: 1500-2500 (average reward > 200)
- **Wall-clock time**: ~15-30 minutes on modern CPU
- **Sample efficiency**: ~300K-500K steps
- **Final performance**: 220-250 average reward

**Learning curve**:
```
Steps        Avg Reward    Notes
0-100K       -200 to -50   Learning basics
100K-300K    -50 to 100    Occasional landings
300K-500K    100 to 200    Consistent landings, solving soon
500K+        200+          Solved! Can fine-tune further
```

### Debugging Checklist

1. **Critic learning**: Plot predicted vs. actual returns (should correlate)
2. **Policy entropy**: Should decrease from ~1.4 (uniform) to ~0.3-0.5 (peaked)
3. **Advantage distribution**: Check mean ≈ 0, std ≈ 1 after normalization
4. **Gradient norms**: Should be stable, <1.0 after clipping
5. **Episode lengths**: Should increase (longer = agent surviving longer)

### Extensions to Try

1. **Learning rate annealing**: Decay LR from 3e-4 to 1e-5
2. **Value clipping**: Clip value updates (PPO-style)
3. **Advantage normalization**: Per-minibatch instead of per-batch
4. **Recurrence**: Add LSTM for memory (probably not needed for LunarLander)
5. **Reward normalization**: Normalize rewards (can help stability)

## Conclusion

This A2C design for LunarLander-v2 should achieve reliable solving in 1500-2500 episodes. Key design choices:
- **Shared encoder**: Parameter efficient, works well for this environment
- **GAE λ=0.95**: Good bias-variance trade-off
- **Entropy bonus**: Maintains exploration
- **Gradient clipping**: Essential for stability
- **Reasonable defaults**: Based on empirical RL research

Stick to these hyperparameters as a starting point, then tune based on observed learning curves and debugging metrics.

</details>

---

## Question 5: Critical Analysis

**How does the choice of GAE lambda affect the bias-variance trade-off? What lambda would you use for (a) a task with short episodes and dense rewards, (b) a task with long episodes and sparse rewards? Justify your choices.**

<details>
<summary>Click to reveal answer</summary>

### Answer

## GAE Lambda: Bias-Variance Trade-off Deep Dive

### Theoretical Foundation

**GAE formula**:
```
A_t^GAE(λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t) is TD error
```

**Bias-variance spectrum**:
```
λ=0: A_t = δ_t                    (1-step, high bias, low variance)
λ=1: A_t = G_t - V(s_t)          (MC, no bias, high variance)
λ∈(0,1): Weighted average         (intermediate)
```

### How Lambda Affects Bias and Variance

#### Bias

**Definition**: Difference between expected estimate and true value.

**λ=0 (Maximum Bias)**:
```
A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)

Bias comes from V(s_{t+1}):
- If V is perfect: no bias
- If V is learned and imperfect: biased towards V's errors
- Early in training, V is very wrong → high bias
```

**λ=1 (No Bias)**:
```
A_t = G_t - V(s_t) = (Σ γ^k r_k) - V(s_t)

No V(s') terms → no bias from value function!
Only uses actual rewards → unbiased estimate
But: high variance from summing many random rewards
```

**λ=0.95 (Low Bias)**:
```
A_t = δ_t + 0.95γδ_{t+1} + (0.95γ)²δ_{t+2} + ...

Exponentially-weighted:
- Short-term: mostly actual rewards (low bias)
- Long-term: relies on V (some bias, but far away so smaller impact)
- Effective horizon ~20 steps
```

**Quantifying bias**:
```
Bias(λ) ∝ (1 - λ) · || V_learned - V_true ||

λ=0: Maximum dependence on V
λ=0.5: Moderate dependence
λ=0.95: Weak dependence
λ=1: No dependence (zero bias)
```

#### Variance

**Definition**: Variability of the estimate across different samples.

**λ=1 (Maximum Variance)**:
```
A_t = Σ_{k=0}^T γ^k r_{t+k} - V(s_t)

Variance = Var[Σ_{k=0}^T γ^k r_{t+k}]
         = Σ_{k=0}^T γ^{2k} Var[r_k]  (assuming independence)
         ≈ O(T)  (grows with episode length)

Long episodes → very high variance!
```

**λ=0 (Minimum Variance)**:
```
A_t = r_t + γV(s_{t+1}) - V(s_t)

Variance = Var[r_t + γV(s_{t+1})]
         ≈ Var[r_t]  (if V is learned and relatively stable)
         = O(1)  (single reward, doesn't depend on episode length)

Much lower variance!
```

**λ=0.95 (Intermediate Variance)**:
```
Variance grows with effective horizon:

Effective horizon = 1 / (1 - γλ)

For γ=0.99, λ=0.95:
h_eff = 1 / (1 - 0.9405) ≈ 17 steps

Variance ∝ h_eff
Variance[λ=0.95] ≈ 17 × Variance[λ=0]
```

**Quantifying variance**:
```
Var[A_t^GAE(λ)] ≈ Var[r] · Σ_{l=0}^∞ (γλ)^{2l}
                = Var[r] / (1 - γ²λ²)

λ=0: Var ≈ Var[r] / (1 - 0.99²) ≈ 50 · Var[r]
λ=0.95: Var ≈ Var[r] / (1 - 0.9405²) ≈ 900 · Var[r]
λ=1: Var → ∞ (depends on full episode)
```

### Visualizing the Trade-off

```
Bias-Variance Trade-off for GAE(λ)

High ↑
     │
     │     Variance
     │        ╱
     │       ╱
     │      ╱
     │     ╱
     │    ╱
     │   ╱ Bias
     │  ╱
     │ ╱
     │╱________
Low  ├─────────────────→ λ
     0   0.5  0.95  1.0

Total Error = Bias² + Variance
Optimal λ minimizes total error (usually around 0.9-0.97)
```

### Scenario A: Short Episodes, Dense Rewards

**Example environments**:
- CartPole (200 steps max, reward every step)
- Mountain Car (200 steps, reward=-1 per step)
- Simple gridworld (20-50 steps, reward at every move)

**Characteristics**:
```
- Episode length: T < 500
- Rewards: Every step or most steps
- Noise: Lower (averaging over many rewards reduces variance)
```

**Optimal λ: HIGH (0.95-0.99)**

**Reasoning**:

1. **Short episodes → variance is manageable**
```
Even λ=1 (MC) only sums ~200 rewards
Variance is T × Var[r] ≈ 200 × Var[r]
Tolerable for most learning rates
```

2. **Dense rewards → good signal without bootstrapping**
```
G_t = Σ_{k=0}^T r_k  includes many informative rewards
Don't need to rely on V(s) as much
Can afford to use more actual rewards (higher λ)
```

3. **Lower bias more important**
```
With short episodes and dense rewards:
- Value function might not be accurate early in training
- Using actual rewards (high λ) avoids biased value estimates
- Bias from poor V can mislead policy more than variance hurts
```

4. **Empirical validation**
```
CartPole experiments (from literature):
λ=0.9:  Solves in ~2000 episodes
λ=0.95: Solves in ~1200 episodes  ← Best
λ=0.99: Solves in ~1500 episodes
λ=1.0:  Solves in ~2000 episodes (high variance slows learning)

Sweet spot: 0.95-0.97
```

**Practical recommendation**:
```python
# Short episodes, dense rewards
hyperparams = {
    'gae_lambda': 0.95,  # or 0.97
    'n_steps': 128,      # Can use longer rollouts
    'gamma': 0.99,
}

# Why it works:
# - High λ: trusts actual rewards (low bias)
# - Short episodes: variance still manageable
# - Dense rewards: good learning signal without excessive variance
```

### Scenario B: Long Episodes, Sparse Rewards

**Example environments**:
- Montezuma's Revenge (10000+ steps, very sparse rewards)
- Robot navigation with goal-reaching (1000+ steps, reward only at goal)
- MuJoCo Humanoid (long episodes, mostly 0 rewards except for progress)

**Characteristics**:
```
- Episode length: T > 1000
- Rewards: Few non-zero rewards (< 1% of timesteps)
- Noise: Very high (long episodes amplify variance)
```

**Optimal λ: LOW-MEDIUM (0.85-0.95)**

**Reasoning**:

1. **Long episodes → variance is catastrophic with high λ**
```
λ=1 (MC): Var ∝ T × Var[r] ≈ 10000 × Var[r]
Even λ=0.95: h_eff ≈ 20 steps → still high variance

With 10000 steps:
- λ=0.95: gradient estimates have huge variance
- Learning is extremely slow or unstable
- Need to reduce effective horizon
```

2. **Sparse rewards → even worse variance**
```
Most rewards are 0, then occasionally get +1 or +100
Variance is dominated by rare events
MC estimates (λ=1) have extreme outliers:
- Most episodes: A_t ≈ -V(s_t)  (no rewards)
- Rare episodes: A_t ≈ 100 - V(s_t)  (got reward!)

This high variance prevents learning
```

3. **Lower λ reduces variance via bootstrapping**
```
λ=0.85: h_eff = 1/(1 - 0.99×0.85) ≈ 7 steps

Advantages only sum ~7 TD errors:
- Much lower variance than summing 10000 rewards
- Can actually learn despite sparse rewards
```

4. **Accept some bias to get much lower variance**
```
With sparse rewards, we must trust the critic:
- V(s) learns to predict "will I eventually reach goal?"
- Even if V is imperfect, it reduces variance enough to learn
- Bias from V is acceptable trade-off for massive variance reduction
```

5. **Empirical validation**
```
Sparse reward experiments (from literature):
λ=0.99: No learning (variance too high)
λ=0.95: Slow learning (still high variance)
λ=0.90: Moderate learning  ← Good balance
λ=0.85: Faster learning    ← Often best
λ=0.80: Slower (too much bias, especially early when V is poor)

Sweet spot: 0.85-0.92
```

**Practical recommendation**:
```python
# Long episodes, sparse rewards
hyperparams = {
    'gae_lambda': 0.90,  # Lower than standard 0.95
    'n_steps': 128,      # Can't go too high (memory issues)
    'gamma': 0.99,       # Or even 0.995-0.999 for very long-term credit

    # Other helpful techniques:
    'normalize_advantage': True,  # Essential with sparse rewards
    'reward_scaling': True,        # Scale sparse rewards to reasonable range
    'value_loss_coef': 1.0,        # Train critic well (critical for low λ)
}

# Why it works:
# - Lower λ: reduces variance dramatically
# - Relies on critic (so must train critic well!)
# - Sacrifice some bias to get tractable variance
```

### Additional Considerations

#### 1. Quality of Value Function

**Early in training** (V is poor):
- Higher λ better (don't trust bad V)
- Even in long episodes, might start with λ=0.95
- Anneal down as V improves

**Late in training** (V is good):
- Can use lower λ (trust accurate V)
- Reduces variance without much bias

**Adaptive strategy**:
```python
# Start high, anneal down
lambda_start = 0.95
lambda_end = 0.85
lambda_current = lambda_start - (lambda_start - lambda_end) * progress
```

#### 2. Reward Noise

**Low noise rewards** (deterministic):
- Can use higher λ (variance from rewards is low)
- Example: Grid world with deterministic rewards

**High noise rewards** (stochastic):
- Need lower λ (variance from rewards is high)
- Example: Noisy robot simulation

#### 3. Discount Factor Interaction

**GAE effective horizon**:
```
h_eff = 1 / (1 - γλ)

For γ=0.99, λ=0.95: h_eff ≈ 17 steps
For γ=0.995, λ=0.95: h_eff ≈ 34 steps
For γ=0.9, λ=0.95: h_eff ≈ 7 steps

Low γ → already short effective horizon → can use higher λ
High γ → long effective horizon → need lower λ to compensate
```

#### 4. Environment Horizon vs Episode Length

**Fixed horizon** (episode terminates at T):
- Episode length predictable
- Can tune λ based on T

**Variable horizon** (episode ends based on state):
- Episode length varies widely
- Need λ that works for both short and long episodes
- Usually favor lower λ (handles worst case)

### Summary Table

| Scenario | Optimal λ | Effective Horizon | Reasoning |
|----------|-----------|-------------------|-----------|
| Short episodes (<500), dense rewards | 0.95-0.99 | 17-50 steps | Variance manageable, bias matters more |
| Medium episodes (500-2000), moderate rewards | 0.92-0.95 | 12-17 steps | Balance bias and variance |
| Long episodes (>2000), sparse rewards | 0.85-0.92 | 7-12 steps | Variance reduction critical |
| Very long (>5000), very sparse | 0.80-0.90 | 5-10 steps | Must bootstrap heavily |

### Practical Heuristics

1. **Default**: Start with λ=0.95 (works for many tasks)
2. **If learning is noisy/unstable**: Lower λ (0.90 or 0.85)
3. **If learning plateaus early**: Raise λ (0.97 or 0.99)
4. **If critic is struggling**: Lower λ (and increase critic LR)
5. **Monitor**: Track explained variance of V → if low, lower λ

### Advanced: Adaptive Lambda

**Per-timestep adaptive**:
```python
# Use high λ for high-confidence V, low λ for uncertain V
lambda_t = 0.85 + 0.10 * confidence(V(s_t))
```

**Per-episode adaptive**:
```python
# Use high λ for short episodes, low λ for long
lambda = 0.95 if episode_length < 500 else 0.90
```

**Uncertainty-based**:
```python
# Use ensemble of critics, adapt λ based on disagreement
uncertainty = std([V_1(s), V_2(s), ..., V_N(s)])
lambda_t = max(0.85, 0.95 - uncertainty)
```

## Final Recommendations

### Scenario A: Short Episodes, Dense Rewards (e.g., CartPole)
```python
gae_lambda = 0.95  # High: trust actual rewards
gamma = 0.99
n_steps = 128      # Can use longer rollouts

# Justification:
# - 200-step episodes → variance is manageable even with λ=0.95
# - Dense rewards → good signal without bootstrapping
# - Bias from poor value function worse than variance
```

### Scenario B: Long Episodes, Sparse Rewards (e.g., Montezuma's Revenge)
```python
gae_lambda = 0.88  # Low: reduce variance
gamma = 0.99
n_steps = 128

# Additional essential techniques:
normalize_advantages = True    # Essential with high variance
value_loss_coef = 1.0          # Train critic well
entropy_coef = 0.01            # Maintain exploration
reward_scaling = 0.01          # Scale sparse +100 rewards down

# Justification:
# - 10000-step episodes → variance catastrophic without low λ
# - Sparse rewards → rely heavily on value function bootstrapping
# - Accept bias from V to get tractable variance
# - Monitor critic quality (explained variance should be >0.7)
```

## Conclusion

GAE λ is a powerful hyperparameter that controls the bias-variance trade-off:
- **High λ (0.95-0.99)**: Low bias, high variance → use for short episodes with dense rewards
- **Low λ (0.85-0.92)**: Higher bias, low variance → use for long episodes with sparse rewards
- **Default 0.95**: Works well across many tasks, but don't be afraid to tune it

The key insight: **variance scales with effective horizon**, so match λ to your episode structure and reward density. Always monitor critic quality and gradient stability when tuning λ.

</details>

