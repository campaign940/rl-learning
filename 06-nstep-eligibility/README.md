# Week 6: n-step Methods & Eligibility Traces

## Learning Objectives

By the end of this week, you should be able to:

1. Understand **n-step returns** as a bridge between Monte Carlo and TD methods
2. Implement **n-step TD** for prediction and **n-step SARSA** for control
3. Understand the **λ-return** and **TD(λ)** algorithm
4. Distinguish between **forward view** and **backward view** of TD(λ)
5. Implement **eligibility traces** for efficient credit assignment
6. Compare **accumulating**, **replacing**, and **dutch** traces
7. Apply eligibility traces to improve learning speed

## Key Concepts

### 1. n-step Returns - The TD-MC Spectrum

**The Unifying Idea**: We can interpolate between TD (n=1) and Monte Carlo (n=∞) by looking n steps ahead.

**n-step Return Definition**:
```
G_t:t+n = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
        = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})
```

**Special Cases**:
- **n=1** (TD(0)): G_t:t+1 = R_{t+1} + γV(S_{t+1})
- **n=∞** (Monte Carlo): G_t:t+∞ = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = G_t

**n-step TD Update**:
```
V(S_t) ← V(S_t) + α[G_t:t+n - V(S_t)]
```

**Properties**:
- Bias decreases as n increases (more actual rewards, less bootstrapping)
- Variance increases as n increases (more random rewards in return)
- Optimal n depends on task: typically n ∈ [3, 10] works well

### 2. n-step SARSA - Control with n-step Returns

Extend n-step returns to action values for control.

**n-step Q-return**:
```
G_t:t+n = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})
```

**n-step SARSA Update**:
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[G_t:t+n - Q(S_t, A_t)]
```

**Algorithm Requirements**:
- Must store last n state-action pairs: (S_t, A_t), ..., (S_{t+n-1}, A_{t+n-1})
- Updates are delayed by n steps
- At episode end, must handle truncated returns

**n-step Off-Policy Variants**:
- **n-step Q-Learning**: Use max_a Q(S_{t+n}, a) instead of Q(S_{t+n}, A_{t+n})
- **n-step Tree Backup**: Incorporate importance sampling for off-policy learning

### 3. TD(λ) - The λ-Return

Instead of choosing a single n, **average over all n-step returns** weighted by λ.

**λ-return Definition**:
```
G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t:t+n
```

This is a **weighted average** of all n-step returns:
- More weight on shorter returns (small n) when λ is small
- More weight on longer returns (large n) when λ is large

**Special Cases**:
- **λ=0**: G_t^λ = G_t:t+1 (TD(0), one-step return)
- **λ=1**: G_t^λ = G_t (Monte Carlo, complete return)

**Expansion**:
```
G_t^λ = (1-λ)[G_t:t+1 + λG_t:t+2 + λ²G_t:t+3 + ... ] + λ^{T-t-1}G_t
      = (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_t:t+n + λ^{T-t-1} G_t
```

**TD(λ) Update (Forward View)**:
```
V(S_t) ← V(S_t) + α[G_t^λ - V(S_t)]
```

**Problem**: This is not computable online! Need the entire episode to compute G_t^λ.

**Solution**: Backward view with eligibility traces (equivalent but online).

### 4. Forward View vs Backward View

**Forward View** (conceptual):
- Look forward from current state
- Weight future n-step returns by λ
- Requires complete trajectory (offline)
- Useful for understanding

**Backward View** (implementable):
- Look backward from current TD error
- Credit assignment through eligibility traces
- Fully online (update at each step)
- Practical algorithm

**Theorem (Equivalence)**:
Under certain conditions, forward and backward views produce identical updates when summed over an episode.

### 5. Eligibility Traces - Efficient Credit Assignment

**The Problem**: When reward is received, which past states/actions deserve credit?

**Solution**: Maintain an **eligibility trace** e_t(s) for each state s.

**Trace Update (Accumulating)**:
```
e_t(s) = {
  γλ e_{t-1}(s) + 1   if s = S_t
  γλ e_{t-1}(s)       otherwise
}
```

**Interpretation**:
- e_t(s) tracks how "eligible" state s is for receiving credit
- Increases by 1 when state is visited
- Decays by γλ at each step

**TD(λ) Update with Traces (Backward View)**:
```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)   (TD error)

For all s ∈ S:
    e_t(s) ← γλ e_{t-1}(s) + 𝟙(s = S_t)
    V(s) ← V(s) + α δ_t e_t(s)
```

**Key Insight**: All states are updated at each step, weighted by their eligibility trace!

**Three Types of Traces**:

1. **Accumulating Traces**:
   ```
   e_t(s) ← γλ e_{t-1}(s) + 𝟙(s = S_t)
   ```
   - Traces accumulate with repeated visits
   - Standard choice

2. **Replacing Traces**:
   ```
   e_t(s) ← max(γλ e_{t-1}(s), 𝟙(s = S_t))
   ```
   - Reset to 1 on visit (don't accumulate)
   - Often works better in practice
   - Particularly good with function approximation

3. **Dutch Traces**:
   ```
   e_t(s) ← (1 - α) γλ e_{t-1}(s) + 𝟙(s = S_t)
   ```
   - Accounts for learning rate in trace decay
   - Theoretical advantages
   - Less common in practice

### 6. SARSA(λ) - Control with Eligibility Traces

Extend TD(λ) to action values for control.

**SARSA(λ) Algorithm**:
```
Initialize Q(s,a) arbitrarily, e(s,a) = 0 for all s,a

For each episode:
    Initialize S, choose A using policy from Q (ε-greedy)
    e(s,a) = 0 for all s,a

    For each step:
        Take action A, observe R, S'
        Choose A' from S' using policy from Q
        δ ← R + γQ(S',A') - Q(S,A)

        e(S,A) ← e(S,A) + 1  (or replace: e(S,A) ← 1)

        For all s,a:
            Q(s,a) ← Q(s,a) + α δ e(s,a)
            e(s,a) ← γλ e(s,a)

        S ← S', A ← A'
    Until S is terminal
```

**Advantages**:
- Faster learning than SARSA (credit spreads backward)
- Single parameter λ controls credit assignment
- Particularly effective for sparse rewards

## Key Equations

### n-step Return
```
G_t:t+n = Σ_{k=0}^{n-1} γ^k R_{t+k+1} + γ^n V(S_{t+n})

n-step TD: V(S_t) ← V(S_t) + α[G_t:t+n - V(S_t)]
```

### λ-return
```
G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t:t+n

Special cases:
  λ=0: G_t^λ = G_t:t+1  (TD(0))
  λ=1: G_t^λ = G_t      (MC)
```

### TD(λ) with Eligibility Traces (Tabular)
```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

e_t(s) = γλ e_{t-1}(s) + 𝟙(s = S_t)

V(s) ← V(s) + α δ_t e_t(s)  for all s
```

### SARSA(λ)
```
δ_t = R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)

e_t(s,a) = γλ e_{t-1}(s,a) + 𝟙(s = S_t, a = A_t)

Q(s,a) ← Q(s,a) + α δ_t e_t(s,a)  for all s,a
```

## Textbook References

- **Sutton & Barto**:
  - Chapter 7: n-step Bootstrapping
    - Section 7.1: n-step TD Prediction
    - Section 7.2: n-step SARSA
    - Section 7.3: n-step Off-policy Learning
  - Chapter 12: Eligibility Traces
    - Section 12.1: The λ-return
    - Section 12.2: TD(λ)
    - Section 12.3: n-step Truncated λ-return Methods
    - Section 12.4: Redoing Updates: Online λ-return Algorithm
    - Section 12.5: True Online TD(λ)
    - Section 12.7: SARSA(λ)
    - Section 12.10: Implementation Issues

- **David Silver's RL Course**:
  - Lecture 4: Model-Free Prediction (second half on eligibility traces)
  - [Lecture Slides](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)

- **CS234 Supplementary Material**:
  - Week 6: Multi-step TD and Eligibility Traces

## Implementation Tasks

### Task 1: 19-State Random Walk with n-step TD

The classic environment for comparing n-step methods (S&B Example 7.1).

**Environment**:
- States: 1, 2, 3, ..., 19 (start at 10)
- Terminal states: 0 (left, reward 0), 20 (right, reward 1)
- Random walk: equal probability left/right each step
- Discount: γ = 1

**True Values**: v_π(i) = i/20 for i = 1, ..., 19

**Implementation**:
1. Implement n-step TD for n = 1, 2, 4, 8, 16, 32
2. Measure RMSE vs episodes for each n
3. Compare learning curves
4. Find optimal n

**Expected Observations**:
- n=1 (TD(0)): Slower convergence, smooth learning
- n=4 to 8: Typically best performance
- Large n (>16): Approaches MC, higher variance
- Optimal n depends on α

### Task 2: Mountain Car with SARSA(λ)

Apply eligibility traces to the challenging Mountain Car task.

**Environment**:
- Continuous state (position, velocity)
- Must discretize or use tile coding
- Sparse reward: -1 per step, 0 at goal
- Challenge: Must build momentum

**Implementation**:
1. Implement tile coding or discretization
2. Implement SARSA(λ) with λ = 0, 0.5, 0.9, 0.95
3. Compare learning speed (episodes to goal)
4. Visualize value function and policy

**Expected Observations**:
- λ=0 (SARSA): Very slow, struggles with sparse reward
- λ=0.9: Much faster, credit propagates backward
- λ close to 1: Best performance for this sparse reward task

### Task 3: GridWorld with Replacing vs Accumulating Traces

Compare different trace types on a simple grid world.

**Environment**:
- 10×10 grid
- Start: bottom-left, Goal: top-right
- Obstacles scattered throughout
- Reward: -1 per step, 0 at goal

**Implementation**:
1. Implement SARSA(λ) with accumulating traces
2. Implement SARSA(λ) with replacing traces
3. Compare on grids with/without revisiting states
4. Measure episodes to convergence

**Expected Observations**:
- Similar performance on most tasks
- Replacing traces may converge faster with revisiting
- Accumulating traces more sensitive to λ

### Task 4: Comparing n-step SARSA Variants

Implement and compare different n-step control methods.

**Methods to Compare**:
1. SARSA (n=1)
2. n-step SARSA (n=5)
3. n-step Expected SARSA
4. SARSA(λ) with equivalent λ

**Environments**: CliffWalking, Taxi, FrozenLake

**Analysis**:
- Learning curves (cumulative reward vs episodes)
- Sample efficiency
- Computational cost per step
- Final policy quality

## Comparison Tables

### n-step Methods Spectrum

| n | Method | Bias | Variance | Update Delay | Equivalent λ |
|---|--------|------|----------|--------------|--------------|
| 1 | TD(0) | High | Low | 1 step | 0 |
| 2-10 | n-step TD | Medium | Medium | n steps | Varies |
| ∞ | Monte Carlo | None | High | Episode end | 1 |

### Eligibility Traces Types

| Type | Update Rule | Behavior | Use Case |
|------|-------------|----------|----------|
| Accumulating | e ← γλe + 1 | Increases with visits | Standard choice |
| Replacing | e ← max(γλe, 1) | Resets to 1 | Function approximation |
| Dutch | e ← (1-α)γλe + 1 | Learning-rate aware | Theoretical work |

### TD(λ) Parameter Settings

| λ | Behavior | Bias | Variance | Speed | Best For |
|---|----------|------|----------|-------|----------|
| 0 | TD(0) | High | Low | Fast | Short-term credit |
| 0.3-0.5 | Light traces | Medium | Low | Fast | General purpose |
| 0.8-0.9 | Medium traces | Low | Medium | Medium | Moderate delay |
| 0.95-0.99 | Heavy traces | Very low | High | Slow | Long delays, sparse rewards |
| 1.0 | MC | None | High | Slow | Episodic with accurate returns |

## Advantages of n-step and Eligibility Traces

**n-step Methods**:
1. **Tunable bias-variance**: Choose n to match task characteristics
2. **Intermediate convergence**: Often faster than both TD and MC
3. **Flexible**: Easy to understand and implement
4. **Effective**: n ∈ [3, 10] works well for many tasks

**Eligibility Traces**:
1. **Efficient credit assignment**: Update all visited states, not just recent ones
2. **Single parameter λ**: Easier to tune than choosing n
3. **Online learning**: No delay, update at every step
4. **Memory efficient**: Only store traces, not state history
5. **Fast learning**: Particularly for sparse rewards
6. **Bridges TD and MC**: Smoothly interpolates between extremes

## Practical Considerations

### Choosing n for n-step Methods

**Guidelines**:
- **Short episodes**: Use larger n (or MC)
- **Long episodes**: Use small n (3-10)
- **High variance**: Use smaller n
- **High bias**: Use larger n
- **Sparse rewards**: Use larger n or eligibility traces

**Tuning Strategy**:
1. Start with n=1 (TD) as baseline
2. Try n=4, 8, 16
3. Measure RMSE or learning speed
4. Choose best n for your task

### Choosing λ for Eligibility Traces

**Guidelines**:
- **Dense rewards**: λ = 0.3 to 0.7
- **Sparse rewards**: λ = 0.9 to 0.99
- **Short-term dependencies**: λ < 0.5
- **Long-term dependencies**: λ > 0.8
- **Function approximation**: Often λ = 0.9 works well

**Tuning Strategy**:
1. Start with λ=0.5
2. If learning is slow, increase λ (more credit to past states)
3. If learning is noisy, decrease λ (less credit propagation)
4. Grid search over {0, 0.3, 0.5, 0.7, 0.9, 0.95}

### Implementation Tips

**1. Trace Initialization**:
```python
# Reset traces at episode start (episodic tasks)
e = np.zeros_like(Q)

# Or decay traces across episodes (continuing tasks)
e = gamma * lambda * e
```

**2. Efficient Trace Storage**:
```python
# Sparse traces (only store non-zero)
e = {}  # dict: (s,a) -> trace value

# Prune small traces
e = {k: v for k, v in e.items() if v > threshold}
```

**3. Learning Rate with Traces**:
- Eligibility traces amplify updates
- May need smaller α than without traces
- Try α = 0.05 to 0.2 instead of 0.5

**4. Replacing vs Accumulating**:
- Start with accumulating (standard)
- Switch to replacing if states are revisited often
- Replacing is more stable with function approximation

## Common Pitfalls

1. **Forgetting to reset traces**: Must reset e at episode start (or decay for continuing)
2. **Wrong decay**: Use γλ, not λ alone
3. **Trace explosion**: Traces can become very large; consider capping
4. **Memory issues**: For large state spaces, use sparse trace storage
5. **Update order**: Update traces before using them in Q update
6. **Terminal states**: Properly handle traces at episode termination

## Connection to Modern Deep RL

n-step methods and eligibility traces are foundational for:

**n-step Methods**:
- **n-step DQN**: Multi-step TD for deep Q-learning
- **Retrace**: Off-policy n-step with importance sampling
- **IMPALA**: Distributed n-step learning

**Eligibility Traces**:
- **A3C**: Uses n-step or eligibility traces for advantage estimation
- **PPO**: Multi-step returns for policy optimization
- **SAC**: n-step backups in soft actor-critic
- **Rainbow DQN**: Combines n-step with other improvements

Understanding these methods deeply is essential for modern RL research and applications.

## Questions to Consider

1. Why is there often an optimal n between 1 and ∞?
2. How does λ in TD(λ) relate to n in n-step TD?
3. Why are eligibility traces more memory efficient than storing n-step history?
4. When would you prefer accumulating vs replacing traces?
5. How do eligibility traces help with sparse rewards?
6. Can you derive the TD(λ) backward view from the forward view?

## Next Steps

After mastering n-step methods and eligibility traces:
- **Week 7**: Planning and Learning (Dyna, MCTS)
- Understanding model-based vs model-free integration
- Combining learning from real experience with simulated experience
- Preparation for advanced topics like policy gradients and function approximation

This week bridges tabular TD methods to more sophisticated algorithms that balance bias and variance for efficient learning!
