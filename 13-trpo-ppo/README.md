# Week 13: TRPO & PPO (Trust Region Policy Optimization & Proximal Policy Optimization)

## Learning Objectives

- [ ] Understand why large policy updates can be catastrophic
- [ ] Master monotonic improvement theory and performance bounds
- [ ] Understand trust regions and KL divergence constraints
- [ ] Grasp natural gradient and Fisher information matrix intuition
- [ ] Implement PPO with clipped objective
- [ ] Understand PPO implementation details (value clipping, normalization, learning rate annealing)

## Key Concepts

### 1. Monotonic Improvement Theory

**The Problem**: Policy gradient doesn't guarantee improvement.

**Why?**
```
Policy gradient: θ_new = θ_old + α∇J(θ_old)

Problems:
1. Step size α is arbitrary
2. Large α → large policy change → could get much worse
3. No guarantee that J(θ_new) > J(θ_old)
```

**Example of catastrophic forgetting**:
```
Episode 1000: Policy achieving reward 200 (near-optimal)
Large gradient update with α=0.01
Episode 1001: Policy achieving reward -500 (catastrophic)
Never recovers (stuck in bad region of policy space)
```

**Kakade & Langford 2002: Conservative Policy Iteration**

**Key insight**: If policy change is small enough, we can guarantee improvement.

**Performance bound**:
```
J(π_new) ≥ J(π_old) + E_s~π_new [E_a~π_new[A^π_old(s,a)]]
                     - (2εγ)/(1-γ)² max_s |E_a~π_new[A^π_old(s,a)]|

where ε = max_s KL(π_new(·|s) || π_old(·|s))
```

**Intuition**:
- First term: expected advantage (how much better is new policy on average)
- Second term: penalty for large KL divergence (how different is new policy)

**Key takeaway**: If KL divergence is small (ε is small), improvement is guaranteed.

### 2. Trust Regions

**Definition**: Region of policy space where our local approximation is valid.

**Idea**: Take the largest step possible within the trust region.

**Trust region constraint**:
```
max θ  E_s,a~π_old [π_new(a|s)/π_old(a|s) · A^π_old(s,a)]
s.t.   E_s~π_old [KL(π_old(·|s) || π_new(·|s))] ≤ δ
```

**Why KL divergence?**
- Measures "distance" between probability distributions
- Invariant to parameterization (better than L2 distance in parameter space)
- Prevents policy from changing too much in any state

**Typical values**: δ = 0.01 to 0.05

**Visualization**:
```
Policy Space

                  Good region
                    ____
                 .-'    '-.
     Bad       /  Trust    \     Bad
     region   |   Region    |   region
               \  (δ=0.01) /
                 '-._  _.-'
                     ��'
                  π_old

Trust region = "safe" policies near π_old
Outside = uncertain, could be much worse
```

### 3. TRPO (Trust Region Policy Optimization)

**Full algorithm**: Schulman et al. 2015

**Objective**:
```
max θ  E_s,a~π_old [π(a|s; θ)/π(a|s; θ_old) · A^π_old(s,a)]
s.t.   E_s~π_old [KL(π(·|s; θ_old) || π(·|s; θ))] ≤ δ
```

**Importance sampling ratio**: r(θ) = π(a|s; θ) / π(a|s; θ_old)

**Solving the constrained optimization**:

Use **conjugate gradient** to approximate natural gradient:
```
1. Compute gradient: g = ∇θ L_surrogate
2. Compute natural gradient: F^{-1}g
   where F = Fisher information matrix (Hessian of KL)
3. Line search to satisfy KL constraint
```

**TRPO pseudocode**:
```
for iteration in training_iterations:
    1. Collect trajectories using π_old
    2. Compute advantages using GAE
    3. Compute surrogate loss gradient: g
    4. Solve Fx = g using conjugate gradient (get search direction)
    5. Compute step size using line search (backtracking)
    6. Update policy: θ ← θ + step_size * search_direction
```

**Advantages**:
- Monotonic improvement (in theory)
- Large effective step sizes
- Robust across different tasks

**Disadvantages**:
- Complex to implement (conjugate gradient, line search)
- Computationally expensive (2nd order method)
- Requires many hyperparameters for CG
- Rarely used in practice (PPO replaced it)

### 4. Natural Gradient

**Gradient descent in parameter space**:
```
θ_new = θ_old + α∇θ J(θ)
```

**Problem**: Parameterization-dependent!

**Example**:
```
Two parameterizations of same policy:
1. θ = [μ, log σ] for Gaussian
2. θ' = [μ, σ²] for Gaussian

Same policy, different gradients!
∇θ J ≠ ∇θ' J (even though they represent same policy)
```

**Natural gradient**: Uses geometry of policy space (not parameter space).

**Formula**:
```
∇_natural J(θ) = F^{-1} ∇θ J(θ)

where F = Fisher information matrix
F_ij = E[∂log π/∂θ_i · ∂log π/∂θ_j]
```

**Why it's better**:
- Invariant to parameterization
- Takes larger steps in "flat" directions (high curvature constraints)
- Connects to trust regions (constraint on KL ⟹ natural gradient)

**Intuition**:
- Regular gradient: "Which direction increases J most?"
- Natural gradient: "Which direction increases J most per unit KL divergence?"

**Computational cost**: O(n³) for matrix inversion (n = # parameters).

**Approximations**:
- Conjugate gradient (TRPO): Avoids explicit inversion
- Diagonal Fisher (K-FAC): Approximate F as diagonal
- Empirical Fisher: Estimate F from samples

### 5. PPO (Proximal Policy Optimization)

**Motivation**: TRPO works well but is complex. Can we simplify?

**Key insight**: Instead of hard constraint, use clipped objective.

#### PPO-Clip (Most Common)

**Clipped surrogate objective**:
```
L^CLIP(θ) = E_t [min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

where:
- r_t(θ) = π(a_t|s_t; θ) / π(a_t|s_t; θ_old)  (importance ratio)
- A_t = advantage at time t
- ε = clipping threshold (typically 0.2)
```

**How clipping works**:

```python
if A_t > 0:  # Good action, want to increase probability
    # Clip r_t to [1, 1+ε]
    # Prevents increasing π(a|s) too much
    objective = min(r_t * A_t, (1+ε) * A_t)

if A_t < 0:  # Bad action, want to decrease probability
    # Clip r_t to [1-ε, 1]
    # Prevents decreasing π(a|s) too much
    objective = min(r_t * A_t, (1-ε) * A_t)
```

**Visualization**:
```
Objective as function of r_t (for A_t > 0)

L^CLIP
  │    Clipped region
  │      ↓
  │     ╱────── (flat, no gradient)
  │    ╱
  │   ╱  Allowed region
  │  ╱   (has gradient)
  │ ╱
  │╱
  └────────────────→ r_t
  0    1   1+ε

When r_t > 1+ε: clipping prevents further increase
Policy can't change too much in one update
```

**Why it works**:
- Simple: Just clip the ratio!
- Effective: Achieves similar performance to TRPO
- Efficient: First-order method, no conjugate gradient
- Robust: Works across many tasks with same hyperparameters

#### PPO-Penalty (Less Common)

**Alternative**: Add KL penalty to objective.

```
L^PENALTY(θ) = E_t [r_t(θ)A_t] - β·KL(π_old, π_new)
```

Adaptively adjust β to target desired KL.

**Rarely used**: Clipping works better in practice.

### 6. PPO Implementation Details

**These matter a LOT for performance!**

#### Value Function Clipping

```python
# Without clipping
value_loss = (V(s) - return)²

# With clipping (prevents large value updates)
V_clipped = V_old + clip(V - V_old, -ε, ε)
value_loss = max((V - return)², (V_clipped - return)²)
```

**Why**: Prevents value function from changing too much (like policy clipping).

#### Advantage Normalization

```python
# Per-minibatch normalization
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Why**: Makes algorithm robust to reward scale, stabilizes training.

#### Multiple Epochs per Batch

```python
# Unlike A2C (1 update per batch), PPO does multiple
for epoch in range(K):  # K=4 typical
    for minibatch in shuffle(batch):
        update_policy_and_value(minibatch)
```

**Why**: Reuse data efficiently (but not too much → stay on-policy).

#### Learning Rate Annealing

```python
# Linear decay
lr = lr_init * (1 - progress)  # progress = 0 to 1

# Or exponential decay
lr = lr_init * decay_rate ** epoch
```

**Why**: Large LR early (fast learning), small LR late (fine-tuning).

#### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**Why**: Prevents exploding gradients, especially with multiple epochs.

#### Entropy Bonus

```python
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

# Often decay entropy_coef over training
entropy_coef = entropy_coef_init * (1 - progress)
```

**Why**: Maintains exploration, especially early in training.

#### GAE for Advantage

```python
# Use GAE with λ=0.95 (standard for PPO)
advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
```

**Why**: Lower variance than Monte Carlo, lower bias than 1-step TD.

### Complete PPO Loss Function

```python
def ppo_loss(
    old_log_probs,
    new_log_probs,
    advantages,
    values,
    old_values,
    returns,
    entropy,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    value_clip=True
):
    # Policy loss (clipped)
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss (optionally clipped)
    if value_clip:
        values_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        value_loss1 = (values - returns).pow(2)
        value_loss2 = (values_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
    else:
        value_loss = 0.5 * (values - returns).pow(2).mean()

    # Entropy bonus
    entropy_loss = entropy.mean()

    # Total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

    return loss, {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'entropy': entropy_loss.item()
    }
```

## Textbook References

- **Schulman et al. 2015**: Trust Region Policy Optimization (TRPO)
- **Schulman et al. 2017**: Proximal Policy Optimization Algorithms (PPO)
- **CS285 Lecture 9**: Advanced Policy Gradients

## Key Papers

### Schulman et al. 2015: Trust Region Policy Optimization (TRPO)

**Paper**: [ArXiv](https://arxiv.org/abs/1502.05477)

**Contribution**:
- Monotonic improvement theory
- Trust region with KL constraint
- Conjugate gradient solution

**Impact**: Showed constrained optimization works for policy gradients.

### Schulman et al. 2017: Proximal Policy Optimization (PPO)

**Paper**: [ArXiv](https://arxiv.org/abs/1707.06347)

**Contribution**:
- Clipped surrogate objective
- Simpler than TRPO, similar performance
- Extensive empirical evaluation

**Impact**: Became most widely used RL algorithm (2017-present).

## Implementation Task

### MuJoCo HalfCheetah or Gymnasium Humanoid with PPO

**Environment**: HalfCheetah-v4 (or Humanoid-v4)

**Observation**: 17D (HalfCheetah) or 376D (Humanoid)
**Action**: 6D continuous (HalfCheetah) or 17D (Humanoid)
**Reward**: Forward velocity minus control cost

**Implementation requirements**:

1. **PPO Agent**:
   - Clipped objective with ε=0.2
   - Multiple epochs (K=10) per batch
   - Minibatch training (batch_size=2048, minibatch=64)
   - GAE with λ=0.95

2. **Network Architecture**:
   - Shared encoder or separate actor/critic
   - Hidden layers: [256, 256] with Tanh activation
   - Gaussian policy: μ(s), learned log_std

3. **Hyperparameters**:
```python
{
    'num_envs': 8,
    'n_steps': 2048,
    'batch_size': 2048,
    'minibatch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_eps': 0.2,
    'learning_rate': 3e-4,
    'value_coef': 0.5,
    'entropy_coef': 0.0,  # Often 0 for continuous control
    'max_grad_norm': 0.5,
    'value_clip': True,
    'normalize_advantages': True,
    'lr_anneal': True,
}
```

4. **Training**:
   - 2M-10M timesteps (depends on task)
   - Log metrics: policy loss, value loss, KL divergence, entropy
   - Monitor clipping fraction (should be 0.1-0.3)

5. **Evaluation**:
   - Plot learning curves
   - Compare with/without value clipping
   - Compare with/without learning rate annealing
   - Analyze KL divergence over training

**Expected results**:
- HalfCheetah: 2000-3000 reward after 1M steps
- Humanoid: 5000-6000 reward after 10M steps

## Key Equations Summary

### TRPO Objective
```
max θ  E[π_new(a|s)/π_old(a|s) · A(s,a)]
s.t.   E[KL(π_old(·|s) || π_new(·|s))] ≤ δ
```

### PPO Clipped Objective
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

where r(θ) = π(a|s; θ) / π(a|s; θ_old)
```

### Natural Gradient
```
∇_natural J = F^{-1} ∇J
where F = Fisher information matrix
```

### Value Function Clipping
```
V_clipped = V_old + clip(V - V_old, -ε, ε)
L_V = max((V - G)², (V_clipped - G)²)
```

## Common Pitfalls

1. **Too many epochs**: Reusing data too much → off-policy issues
2. **Too large ε**: Clipping doesn't work → large policy changes
3. **Forgetting advantage normalization**: Unstable training
4. **No learning rate annealing**: Performance plateaus
5. **Wrong KL calculation**: Use KL(π_old || π_new), not reversed
6. **Not monitoring KL**: Should be ~0.01-0.05 per update
7. **Skipping value clipping**: Less stable value function
8. **Wrong minibatch shuffling**: Don't shuffle across episodes

## Extensions and Variations

### PPO with Recurrent Networks (PPO-LSTM)
- LSTM in policy/value networks
- For partially observable environments
- Handles memory and temporal dependencies

### PPO with Curiosity
- Intrinsic reward for exploration
- Random Network Distillation (RND)
- Important for sparse reward environments

### Distributed PPO (Ape-X, IMPALA-style)
- Many parallel workers
- Centralized learner
- Scales to 1000s of CPUs

### Multi-task PPO
- Single policy for multiple tasks
- Task embedding as input
- Transfer learning across tasks

## Debugging Tips

1. **Monitor KL divergence**: Should be small (0.01-0.05)
2. **Check clipping fraction**: 0.1-0.3 is good range
3. **Plot approx_kl**: Monitor E[r - 1 - log r] ≈ KL
4. **Track explained variance**: Should be >0.7
5. **Visualize policy entropy**: Should gradually decrease
6. **Log gradient norms**: Should be stable after clipping
7. **Compare with Stable-Baselines3**: Sanity check your implementation

## Next Steps

After mastering PPO:
- **Week 14**: Continuous control with off-policy methods (DDPG, TD3, SAC)
- **Advanced**: Model-based RL, offline RL, multi-agent RL
- **Applications**: Robotics, game AI, LLM fine-tuning (RLHF)

## Key Takeaways

1. **Trust regions prevent catastrophic updates**: Small policy changes → guaranteed improvement
2. **TRPO is theoretically sound but complex**: Conjugate gradient, line search
3. **PPO simplifies TRPO with clipping**: First-order, easy to implement, works great
4. **Implementation details matter**: Value clipping, normalization, annealing, multiple epochs
5. **PPO is the default choice**: Most widely used RL algorithm (2017-2025)
6. **Works across diverse tasks**: Robotics, games, NLP (RLHF for ChatGPT)

PPO is the "sweet spot" of RL algorithms: theoretically motivated, practically effective, easy to implement. Master it, and you can solve most RL problems.
