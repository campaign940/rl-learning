# Week 11: Policy Gradient Methods

## Learning Objectives

- [ ] Understand the policy gradient theorem and its derivation
- [ ] Implement REINFORCE algorithm from scratch
- [ ] Understand baseline variance reduction techniques
- [ ] Master the score function estimator and log-derivative trick
- [ ] Apply policy gradient methods to continuous action spaces

## Key Concepts

### 1. Why Policy-Based Methods?

**Definition**: Policy-based methods directly parameterize the policy π(a|s; θ) and optimize it by gradient ascent on expected return.

**Intuition**: Instead of learning a value function and deriving a policy from it (value-based), we directly learn the policy itself. This is particularly powerful when:
- The action space is continuous or high-dimensional
- We want stochastic policies (for exploration or game-theoretic reasons)
- The optimal policy is simpler than the optimal value function

**Advantages**:
- Better convergence properties (guaranteed to converge to local optimum)
- Effective in high-dimensional or continuous action spaces
- Can learn stochastic policies
- Follows the gradient of performance directly

**Disadvantages**:
- High variance in gradient estimates
- Sample inefficient
- Converge to local optima (not global)

### 2. Policy Gradient Theorem

**Equation**:
```
∇_θ J(θ) = E_π [∇_θ log π(a|s; θ) Q^π(s,a)]
         = E_π [∇_θ log π(a|s; θ) G_t]  (Monte Carlo estimate)
```

**Intuition**: The gradient of expected return is the expected gradient of log probability, weighted by how good the action was (Q-value or return). Actions with higher returns get "reinforced" (probability increased).

**Key insight**: We don't need to differentiate through the environment dynamics or reward function, only through the policy itself.

### 3. REINFORCE Algorithm

**Definition**: Monte Carlo policy gradient algorithm using complete episode returns.

**Algorithm**:
```
1. Initialize policy parameters θ
2. For each episode:
   a. Generate trajectory τ = {s_0, a_0, r_1, ..., s_T} using π(·|·; θ)
   b. For each timestep t in trajectory:
      - Compute return G_t = Σ_{k=t}^T γ^{k-t} r_k
      - Update: θ ← θ + α G_t ∇_θ log π(a_t|s_t; θ)
```

**Equation**:
```
θ_{t+1} = θ_t + α G_t ∇_θ log π(a_t|s_t; θ_t)
```

**Properties**:
- Unbiased gradient estimate
- High variance (entire episode return used)
- On-policy (must use current policy to generate data)

### 4. Baseline for Variance Reduction

**Definition**: Subtract a state-dependent function b(s) from the return to reduce variance without introducing bias.

**Equation**:
```
∇_θ J(θ) = E_π [∇_θ log π(a|s; θ) (G_t - b(s))]
```

**Why it works**: The expectation E[∇_θ log π(a|s; θ)] = 0, so subtracting any state-dependent baseline doesn't change the expected gradient.

**Common baselines**:
- Constant: b(s) = mean of historical returns
- State value: b(s) = V^π(s) (optimal choice)
- Learned baseline: Neural network approximating V^π(s)

**REINFORCE with baseline**:
```
θ ← θ + α (G_t - V(s_t; w)) ∇_θ log π(a_t|s_t; θ)
w ← w + β (G_t - V(s_t; w)) ∇_w V(s_t; w)
```

### 5. Score Function Estimator / Log-Derivative Trick

**The trick**:
```
∇_θ π(a|s; θ) = π(a|s; θ) ∇_θ log π(a|s; θ)
```

Therefore:
```
∇_θ E_π[R] = ∇_θ Σ_a π(a|s; θ) R(s,a)
           = Σ_a ∇_θ π(a|s; θ) R(s,a)
           = Σ_a π(a|s; θ) ∇_θ log π(a|s; θ) R(s,a)
           = E_π [∇_θ log π(a|s; θ) R(s,a)]
```

**Key insight**: This allows us to push the gradient inside the expectation and estimate it via sampling, even when R(s,a) is not differentiable.

## Textbook References

- **Sutton & Barto**: Chapter 13 (Policy Gradient Methods)
  - Section 13.1: Policy Approximation
  - Section 13.2: Policy Gradient Theorem
  - Section 13.3: REINFORCE Algorithm
  - Section 13.4: REINFORCE with Baseline

- **David Silver**: Lecture 7 (Policy Gradient Methods)
  - Finite difference vs. score function
  - Policy gradient theorem proof
  - Actor-critic preview

- **CS285 (Berkeley)**: Lecture 5 (Policy Gradients)
  - Derivation details
  - Variance reduction techniques

## Implementation Tasks

### CartPole-v1 with REINFORCE

Implement a complete REINFORCE agent with:

1. **Policy Network**:
   - Input: 4-dimensional state (cart position, velocity, pole angle, angular velocity)
   - Hidden layers: 2 layers with 128 units, ReLU activation
   - Output: 2 action logits (left/right)
   - Softmax for action probabilities

2. **Baseline Network**:
   - Same input as policy
   - Hidden layers: 2 layers with 128 units
   - Output: Single value estimate V(s)

3. **Training Loop**:
   - Generate complete episodes
   - Compute returns G_t for each timestep
   - Update policy using: θ ← θ + α (G_t - V(s_t)) ∇_θ log π(a_t|s_t; θ)
   - Update baseline using MSE loss: (G_t - V(s_t))^2

4. **Experiments**:
   - Compare REINFORCE vs. REINFORCE with baseline (plot variance)
   - Try different learning rates: 1e-2, 1e-3, 1e-4
   - Plot learning curves (episode return vs. episodes)
   - Analyze gradient variance over training

**Expected Results**: CartPole-v1 should be solved (average reward > 195) within 500-1000 episodes with baseline, 1500-2500 without.

## Key Equations Summary

### Policy Gradient Theorem
```
∇_θ J(θ) = E_π [Σ_t ∇_θ log π(a_t|s_t; θ) G_t]
```

### REINFORCE Update
```
θ ← θ + α G_t ∇_θ log π(a_t|s_t; θ)
```

### REINFORCE with Baseline
```
θ ← θ + α (G_t - b(s_t)) ∇_θ log π(a_t|s_t; θ)
```

### Softmax Policy (Discrete Actions)
```
π(a|s; θ) = exp(h(s,a; θ)) / Σ_b exp(h(s,b; θ))
∇_θ log π(a|s; θ) = ∇_θ h(s,a; θ) - E_π[∇_θ h(s,·; θ)]
```

### Gaussian Policy (Continuous Actions)
```
π(a|s; θ) = N(μ(s; θ), σ^2)
∇_θ log π(a|s; θ) = (a - μ(s; θ))/σ^2 · ∇_θ μ(s; θ)
```

## Common Pitfalls

1. **Forgetting to clear gradients**: Always zero gradients before backward pass
2. **Not normalizing returns**: Can cause numerical instability
3. **Using on-policy data**: REINFORCE requires fresh data from current policy
4. **Ignoring variance**: High variance can prevent learning entirely
5. **Wrong gradient direction**: Maximize, not minimize (use negative loss or gradient ascent)

## Extensions and Variations

- **Reward-to-go**: Use Σ_{k=t}^T γ^{k-t} r_k instead of Σ_{k=0}^T γ^k r_k (reduces variance)
- **Advantage estimation**: Use A(s,a) = Q(s,a) - V(s) instead of Q(s,a)
- **Natural gradients**: Precondition with Fisher information matrix
- **Trust regions**: Constrain update size (leads to TRPO/PPO)

## Debugging Tips

1. **Check gradient signs**: Policy gradient should increase probability of good actions
2. **Verify baseline reduces variance**: Plot gradient variance with/without baseline
3. **Monitor entropy**: Policy should maintain exploration (entropy shouldn't collapse to 0)
4. **Check for numerical issues**: Log probabilities can underflow; use log-sum-exp trick
5. **Visualize policy**: For simple environments, plot action probabilities across states

## Next Steps

After mastering REINFORCE, you're ready for:
- **Week 12**: Actor-Critic methods (reduce variance with bootstrapping)
- **Week 13**: TRPO & PPO (trust region methods for stable updates)
- **Week 14**: Continuous control with deterministic policies (DDPG, TD3, SAC)
