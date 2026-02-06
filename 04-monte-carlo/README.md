# Week 4: Monte Carlo Methods

## Learning Objectives

By the end of this week, you should be able to:

1. Understand and implement **first-visit** and **every-visit** Monte Carlo prediction
2. Apply **MC control with exploring starts** to find optimal policies
3. Implement **off-policy Monte Carlo** methods using importance sampling
4. Understand the difference between **ordinary** and **weighted importance sampling**
5. Recognize when Monte Carlo methods are preferable to dynamic programming

## Key Concepts

### 1. Monte Carlo Prediction

Monte Carlo (MC) methods learn value functions directly from experience without requiring a model of the environment. They estimate values by averaging complete returns from episodes.

**First-Visit MC**: Estimate v_π(s) as the average of returns following the first visit to state s in each episode.

**Every-Visit MC**: Estimate v_π(s) as the average of returns following all visits to state s across all episodes.

Key properties:
- Only works for episodic tasks (must terminate)
- Estimates are independent for each state (no bootstrapping)
- Unbiased estimates that converge to true values as number of visits → ∞
- Higher variance than TD methods but zero bias

### 2. Monte Carlo Control

MC control finds optimal policies by alternating between policy evaluation and policy improvement without requiring environment dynamics.

**Exploring Starts**: Ensure all state-action pairs are visited by starting episodes from random (s, a) pairs. This guarantees exploration but is often impractical.

**ε-soft Policies**: Maintain exploration by assigning non-zero probability (at least ε/|A|) to all actions. This approach is more practical and works without exploring starts.

**Policy Improvement Theorem**: Making the policy greedy with respect to the current value function is guaranteed to improve it (or leave it optimal).

### 3. Off-Policy Monte Carlo

Off-policy methods learn about target policy π while following behavior policy b. This separation enables:
- Learning optimal policy while exploring
- Learning from human demonstrations
- Reusing data from old policies

**Importance Sampling Ratio**:
```
ρ_t:T-1 = ∏_{k=t}^{T-1} π(A_k|S_k) / b(A_k|S_k)
```

This ratio corrects for the difference in probabilities between the two policies.

**Ordinary Importance Sampling**:
- Unbiased but can have unbounded variance
- V(s) = E[ρ_t:T-1 G_t | S_t = s]

**Weighted Importance Sampling**:
- Biased (converges to unbiased) but lower variance
- Preferred in practice for stability

### 4. Incremental Implementation

Instead of storing all returns and recomputing averages, use incremental updates:

```
V_n = V_{n-1} + α[G_n - V_{n-1}]
```

Or for weighted importance sampling:
```
V_n = V_{n-1} + (W_n/C_n)[G_n - V_{n-1}]
```

where C_n accumulates the sum of weights.

## Key Equations

### First-Visit MC Estimate

```
V(s) ← average(Returns(s))

where Returns(s) contains all returns following first visits to s
```

### Importance Sampling Ratio

```
ρ_t:T-1 = ∏_{k=t}^{T-1} π(A_k|S_k) / b(A_k|S_k)
```

### Weighted Importance Sampling

```
V(s) = Σ_i (ρ_i G_i) / Σ_i ρ_i

where i indexes episodes where s was visited
```

### Incremental MC Update

```
Q(S_t, A_t) ← Q(S_t, A_t) + α[G_t - Q(S_t, A_t)]
```

### MC Control with ε-greedy Policy

```
π(a|s) = {
  1 - ε + ε/|A(s)|  if a = argmax_a Q(s,a)
  ε/|A(s)|           otherwise
}
```

## Textbook References

- **Sutton & Barto**: Chapter 5 - Monte Carlo Methods
  - Section 5.1: Monte Carlo Prediction
  - Section 5.2: Monte Carlo Estimation of Action Values
  - Section 5.3: Monte Carlo Control
  - Section 5.4: Monte Carlo Control without Exploring Starts
  - Section 5.5: Off-policy Prediction via Importance Sampling
  - Section 5.6: Incremental Implementation
  - Section 5.7: Off-policy Monte Carlo Control

- **David Silver's RL Course**: Lecture 4 - Model-Free Prediction
  - Monte Carlo Learning
  - Temporal-Difference Learning (comparison with MC)

- **CS234 (Stanford)**: Week 4 - Model-Free Control
  - Monte Carlo Control
  - Importance Sampling

## Implementation Tasks

### Task 1: Blackjack with Monte Carlo

Implement a Blackjack agent using Gymnasium's Blackjack-v1 environment:

1. **MC Prediction**: Estimate state values under a fixed policy (e.g., stick on 20 or 21, hit otherwise)
2. **MC Control with Exploring Starts**: Learn optimal policy
3. **MC Control with ε-greedy**: Learn optimal policy without exploring starts
4. **Compare**: Visualize value functions and policies learned by different methods

**Expected Observations**:
- Policy should learn to hit on low sums, stick on high sums
- Usable ace changes optimal strategy
- Dealer's showing card significantly affects decisions

### Task 2: Off-Policy MC Prediction

Implement off-policy MC for Blackjack:

1. Use a random behavior policy (uniform over actions)
2. Estimate values for a target policy (same as Task 1)
3. Compare ordinary vs weighted importance sampling
4. Analyze variance and convergence rates

**Key Insights**:
- Weighted IS should show smoother convergence
- Ordinary IS may have high variance with rare trajectories
- Requires more episodes than on-policy methods

## Comparison: MC vs DP vs TD

| Property | Monte Carlo | Dynamic Programming | Temporal Difference |
|----------|-------------|---------------------|---------------------|
| Model Required | No | Yes | No |
| Bootstrapping | No | Yes | Yes |
| Episodic Only | Yes | No | No |
| Bias | None | None | Yes (initially) |
| Variance | High | None | Medium |
| Converges to | v_π | v_π | v_π |
| Computational | O(1) per step | O(\|S\|²) sweep | O(1) per step |

## Advantages of Monte Carlo Methods

1. **Model-Free**: Learn from experience without environment dynamics
2. **Simple**: No bootstrapping, just average returns
3. **Unbiased**: Converge to true values with infinite data
4. **State Independence**: Estimating value of one state doesn't require others
5. **Works with Simulation**: Can learn from simulated or sample episodes

## Disadvantages of Monte Carlo Methods

1. **Episodic Only**: Requires tasks that terminate
2. **High Variance**: Individual returns can vary significantly
3. **Slow Convergence**: Must wait until episode end to update
4. **No Online Learning**: Cannot update during an episode
5. **Exploration Problem**: May miss important states without exploring starts

## Practical Tips

1. **Initialization**: Initialize Q(s,a) optimistically to encourage exploration
2. **Decay ε**: Gradually reduce ε in ε-greedy policies as learning progresses
3. **Discount Factor**: Use γ < 1 to reduce variance in long episodes
4. **Incremental Implementation**: Essential for computational efficiency
5. **Weighted IS**: Prefer over ordinary IS for off-policy learning

## Questions to Consider

1. When would you prefer MC over TD methods?
2. How does episode length affect MC performance?
3. Why is exploring starts often impractical in real applications?
4. How does importance sampling ratio grow with episode length?
5. Can MC methods be applied to continuing tasks?

## Next Steps

After mastering Monte Carlo methods, you'll be ready for:
- **Week 5**: Temporal-Difference learning (TD(0), SARSA, Q-Learning)
- Understanding the bias-variance tradeoff between MC and TD
- Combining the best of both worlds with n-step methods
