# Week 9 Quiz: Deep Q-Networks (DQN)

## Question 1: Conceptual Understanding

**Why is naive Q-learning with neural networks unstable? Explain at least three specific reasons and describe how each contributes to instability. Provide concrete examples of what can go wrong.**

<details>
<summary>Answer</summary>

Naive Q-learning with neural networks faces three fundamental instability issues:

## 1. Correlated Sequential Samples

**Problem**:
- Online RL generates highly correlated sequences: (s_t, a_t, r_t, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}), ...
- Neural networks assume independent and identically distributed (i.i.d.) training data
- Consecutive states are similar (e.g., in Pong, frames differ by small pixel changes)

**What goes wrong**:
```
Episode 1: Agent explores left side of state space
→ Network overfits to left-side patterns
→ Forgets how to handle right-side states

Episode 2: Agent explores right side
→ Network overfits to right-side patterns
→ Catastrophic forgetting of left-side knowledge
```

**Concrete example (CartPole)**:
- Agent happens to keep pole tilted right for 100 steps
- Network trains on 100 consecutive "right-tilted" samples
- Learns: "always push right when pole is right"
- Fails completely when pole tilts left (overfitting)

**Result**: Oscillating performance, catastrophic forgetting, inability to maintain diverse knowledge.

---

## 2. Non-Stationary Targets

**Problem**:
- TD target: y = r + γ · max_a' Q(s', a'; θ)
- Target depends on current parameters θ
- Every update to θ changes both Q(s,a) and the target
- "Chasing a moving target"

**What goes wrong**:
```
Step 1: Q(s,a) = 5, Target = 10
        → Update increases Q(s,a) toward 10

Step 2: Q(s,a) = 7, but Target is now 12 (because network changed)
        → Target moved away!

Step 3: Q(s,a) = 9, Target = 8 (target decreased!)
        → Oscillation
```

**Concrete example (GridWorld)**:
```
Suppose we're updating Q(s_1, right):

Iteration 1:
  Q(s_1, right) = 0
  Q(s_2, up) = 5  ← Best action from s_2
  Target = r + γ·Q(s_2, up) = 1 + 0.9·5 = 5.5
  Update: Q(s_1, right) → 2.75

Iteration 2 (immediately after):
  Q(s_1, right) = 2.75
  Q(s_2, up) = 7  ← Increased due to other updates!
  Target = 1 + 0.9·7 = 7.3  ← Target moved!
  Update: Q(s_1, right) → 5.0

Iteration 3:
  Q(s_1, right) = 5.0
  Q(s_2, up) = 4  ← Decreased (instability)
  Target = 1 + 0.9·4 = 4.6  ← Target moved down!
  Update: Q(s_1, right) → 4.8

Result: Oscillation, never converges
```

**Mathematical insight**: Unlike supervised learning where labels are fixed, RL targets are functions of the parameters being optimized. This creates a feedback loop.

**Result**: Oscillations, divergence, failure to converge.

---

## 3. Deadly Triad

**Problem**: Combination of three elements:
1. **Function approximation** (neural network) — affects multiple states per update
2. **Bootstrapping** (TD learning) — estimates based on estimates
3. **Off-policy** (Q-learning) — learns different policy than it follows

**Why this combination is toxic**:

**Function approximation**:
- Update to one state affects many others through shared weights
- Small error in one region propagates to others
- Amplifies instabilities

**Bootstrapping**:
- Using Q(s') to update Q(s)
- Errors in Q(s') directly corrupt Q(s)
- Error propagation through value function

**Off-policy**:
- Training distribution ≠ target distribution
- Q-learning learns optimal Q* while acting ε-greedily
- Distribution mismatch breaks theoretical guarantees
- Updates have wrong statistical properties

**Concrete example (Baird's counterexample)**:
```
Simple MDP with linear function approximation:
- 7 states, 2 actions
- Off-policy updates (learn optimal while following uniform policy)
- Weights diverge to infinity!
- Proven counterexample showing danger of deadly triad
```

**Modern manifestation (Pre-DQN deep Q-learning)**:
```
Researcher's experience circa 2012:
1. Train DQN on Pong
2. Initially learns: Q-values increase
3. Suddenly: Q-values explode to +1000
4. Then: Q-values collapse to 0
5. Learning fails completely

Root cause: Deadly triad without stabilization
```

---

## How Bad Can It Get?

**Empirical observations without DQN stabilizations**:

1. **Complete divergence**: Q-values → ±∞
2. **Catastrophic forgetting**: Performance drops to zero suddenly
3. **Oscillation**: Q-values oscillate wildly, never stable
4. **Inability to learn**: No progress even after millions of steps

**Visualizing instability**:
```
With stabilization (DQN):
Reward │     ┌─────────
       │    ╱
       │   ╱
       │  ╱
       │ ╱
       └─────────────── Time

Without stabilization:
Reward │  ╱╲    ╱╲
       │ ╱  ╲  ╱  ╲╱╲
       │╱    ╲╱      ╲  ╱╲
       │              ╲╱  ╲
       └─────────────────── Time
(Never converges, high variance, frequent crashes)
```

---

## Why Tabular Q-Learning Doesn't Have These Problems

**Tabular methods avoid these issues**:

1. **No generalization**: Each Q(s,a) is independent
   - Updating Q(s_1, a_1) doesn't affect Q(s_2, a_2)
   - No propagation of errors through function approximation

2. **Proven convergence**: Q-learning is proven convergent for tabular case
   - Robbins-Monro conditions satisfied
   - Contraction mapping guarantees convergence

3. **No catastrophic forgetting**: Each state has its own table entry
   - Learning about one state doesn't overwrite others

**The challenge**: Neural networks are necessary for large/continuous state spaces (like Atari with 10^67 possible states), but they break the assumptions that make tabular Q-learning work.

**DQN's achievement**: Made neural Q-learning work by addressing these three instabilities with experience replay and target networks.

---

## Summary Table

| Issue | Cause | Symptom | DQN Solution |
|-------|-------|---------|--------------|
| Correlated samples | Sequential trajectory | Overfitting, forgetting | Experience replay |
| Non-stationary targets | Target depends on θ | Oscillation, divergence | Target network |
| Deadly triad | FA + bootstrap + off-policy | Amplified instability | Both innovations |

</details>

---

## Question 2: Mathematical Derivation

**How does experience replay break correlations in the training data? Derive or explain the statistical properties of sampling from a replay buffer versus sampling from sequential trajectories. Show mathematically why this helps learning.**

<details>
<summary>Answer</summary>

## Sequential Trajectory Sampling (Naive Q-Learning)

**Data generation**:
```
Follow policy π to generate trajectory:
τ = (s_0, a_0, r_0, s_1, a_1, r_1, s_2, a_2, r_2, ...)

Training samples: {(s_t, a_t, r_t, s_{t+1})}_{t=0}^{T}
```

**Statistical properties**:

### 1. High Temporal Correlation

**Autocorrelation** between consecutive samples:
```
Corr(s_t, s_{t+1}) ≈ 1  (very high)

For example, in Atari:
- s_t = [frame_{t-3}, frame_{t-2}, frame_{t-1}, frame_t]
- s_{t+1} = [frame_{t-2}, frame_{t-1}, frame_t, frame_{t+1}]

Overlap: 3 out of 4 frames are identical!
Correlation ≈ 0.75 or higher
```

**Problem for gradient descent**:
```
Gradient at step t: ∇_θ L(s_t, a_t; θ)
Gradient at step t+1: ∇_θ L(s_{t+1}, a_{t+1}; θ)

These gradients point in similar directions
→ Redundant information
→ Inefficient learning
→ Overfitting to local trajectory
```

### 2. Non-Stationary Distribution

The state distribution changes over time:
```
Early training: ρ_π_early(s) — random exploration
Late training: ρ_π_late(s) — near-optimal policy

ρ_π_early ≠ ρ_π_late

Network must adapt to shifting distribution
```

### 3. Batch Gradients Are Correlated

Minibatch of size B from consecutive samples:
```
Batch = {(s_t, a_t, r_t, s_{t+1}), ..., (s_{t+B-1}, a_{t+B-1}, r_{t+B-1}, s_{t+B})}

Covariance matrix of gradients:
Cov[∇_θ L_i, ∇_θ L_j] ≠ 0  for i ≈ j

High covariance → redundant information
```

---

## Experience Replay Sampling

**Data structure**:
```
Replay buffer D = {e_1, e_2, ..., e_N}
where e_i = (s_i, a_i, r_i, s'_i, done_i)

Collected from many episodes over time
```

**Sampling process**:
```
Sample minibatch B uniformly at random:
B = {e_{i_1}, e_{i_2}, ..., e_{i_k}} where i_j ~ Uniform(1, N)
```

**Statistical properties**:

### 1. Broken Temporal Correlation

**Autocorrelation** between samples in a batch:
```
For i ≠ j uniformly sampled:
E[Corr(s_i, s_j)] ≈ 0

Expected number of consecutive samples in batch of size 64 from buffer of 10,000:
≈ 64/10,000 × 64 ≈ 0.4  (less than 1 pair!)

Practically zero correlation
```

**Mathematical derivation**:

For large buffer N and small batch B:
```
P(i and j are consecutive | i, j sampled uniformly) = O(B/N) → 0 as N → ∞

Cov[s_i, s_j] → 0 for i ≠ j
```

### 2. Approximates Stationary Distribution

Over many episodes, replay buffer accumulates data from:
```
D ≈ Sample from stationary distribution μ(s)

where μ(s) = lim_{t→∞} ρ_π_t(s)
```

**Why this helps**:
- Neural network sees diverse states
- Not biased toward recent trajectory
- More representative of true state distribution

**Mathematical justification**:

Under ergodicity assumptions:
```
(1/N) Σ_{i=1}^{N} f(s_i) → E_{s~μ}[f(s)]  as N → ∞

Buffer empirically approximates expectation over stationary distribution
```

### 3. Gradient Variance Reduction

**Sequential sampling variance**:
```
Var[∇_θ L_sequential] = Var[∇_θ L_i] + 2·Σ_{i<j} Cov[∇_θ L_i, ∇_θ L_j]
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        Positive covariance terms inflate variance
```

**Experience replay variance**:
```
Var[∇_θ L_replay] = Var[∇_θ L_i]  (covariance terms ≈ 0)

Lower variance → more stable gradient estimates
```

---

## Formal Analysis

### Sequential Sampling

**Objective**: Minimize expected TD error
```
J(θ) = E_{(s,a)~ρ_π, s'~P(·|s,a)} [(r + γ max_a' Q(s',a';θ) - Q(s,a;θ))²]
```

**Stochastic gradient** (sequential):
```
g_t = ∇_θ [(r_t + γ max_a' Q(s_{t+1},a';θ) - Q(s_t,a_t;θ))²]

Problem: g_t and g_{t+1} are correlated!
```

**Effective sample size**:
```
n_eff = n / (1 + 2·Σ_k ρ_k)

where ρ_k is autocorrelation at lag k

High autocorrelation → n_eff << n
100 correlated samples ≈ 10-20 independent samples
```

### Experience Replay Sampling

**Stochastic gradient** (replay):
```
g = (1/B) Σ_{i=1}^{B} ∇_θ [(r_i + γ max_a' Q(s'_i,a';θ) - Q(s_i,a_i;θ))²]

where (s_i, a_i, r_i, s'_i) sampled uniformly from D
```

**Independence assumption**:
```
For i ≠ j: Cov[g_i, g_j] ≈ 0

Variance:
Var[g] = (1/B) · Var[g_i]  (standard B^{-1} scaling)

Sequential would have Var[g] ≈ Var[g_i] (no benefit from B samples!)
```

**Convergence rate**:
```
Experience replay: O(1/√B) convergence
Sequential: O(1/√n_eff) where n_eff << B

Experience replay converges faster!
```

---

## Empirical Demonstration

**Experiment**: Train on 1000 transitions

**Sequential sampling**:
```
Episode 1: 200 transitions (explore left)
Episode 2: 200 transitions (explore center)
Episode 3: 200 transitions (explore right)
Episode 4: 200 transitions (explore left again)
Episode 5: 200 transitions (random)

Gradient correlation matrix (5 batches of 200):
     1     2     3     4     5
1  [1.0   0.8   0.2   0.7   0.3]
2  [0.8   1.0   0.4   0.3   0.2]
3  [0.2   0.4   1.0   0.1   0.3]
4  [0.7   0.3   0.1   1.0   0.4]
5  [0.3   0.2   0.3   0.4   1.0]

High off-diagonal values → correlated
```

**Experience replay**:
```
All 1000 transitions in buffer
Sample 5 random batches of 200

Gradient correlation matrix:
     1     2     3     4     5
1  [1.0   0.1  -0.1   0.0   0.1]
2  [0.1   1.0   0.0   0.1  -0.1]
3  [-0.1  0.0   1.0   0.1   0.0]
4  [0.0   0.1   0.1   1.0   0.1]
5  [0.1  -0.1   0.0   0.1   1.0]

Off-diagonal ≈ 0 → uncorrelated
```

---

## Why This Matters for Neural Networks

**Neural networks are particularly sensitive to correlated data**:

1. **Overfitting**: Correlated samples → network memorizes recent patterns
2. **Catastrophic forgetting**: New correlated batch overwrites old knowledge
3. **Poor generalization**: Network doesn't see diverse examples

**Experience replay provides i.i.d.-like properties**:
- Each minibatch is diverse
- Network learns from varied experiences
- Better generalization across state space

---

## Quantitative Improvement

**Sample efficiency** (empirical results):
```
Naive Q-learning: 10M steps to solve Atari game
DQN with replay: 1M steps to solve same game

10x improvement!
```

**Why such a large gain?**:
- Each transition used ~10 times (sampled multiple epochs)
- Decorrelation provides effective 2-5x more information per sample
- Combined: 10 × (2-5) = 20-50x effective sample efficiency

---

## Mathematical Summary

| Property | Sequential | Experience Replay |
|----------|------------|-------------------|
| Sample correlation | ρ ≈ 0.5-0.9 | ρ ≈ 0 |
| Effective samples | n_eff ≈ 0.1n | n_eff ≈ n |
| Gradient variance | High | Low |
| Convergence rate | O(1/√n_eff) | O(1/√n) |
| Distribution | Non-stationary | ≈ Stationary |

**Conclusion**: Experience replay transforms highly correlated sequential data into approximately i.i.d. samples, dramatically improving learning stability and efficiency.

</details>

---

## Question 3: Comparison

**Compare DQN with and without target network. Explain the target network mechanism, why it stabilizes learning, and what trade-offs it introduces. Include a concrete numerical example showing how targets evolve differently.**

<details>
<summary>Answer</summary>

## DQN Without Target Network (Naive Approach)

**Update rule**:
```
Target: y = r + γ · max_a' Q(s', a'; θ)
Loss: L(θ) = [y - Q(s, a; θ)]²
Update: θ ← θ - α · ∇_θ L(θ)
```

**Problem**: Target depends on θ, which we're updating!

---

## DQN With Target Network

**Two networks**:
1. **Online network**: Q(s, a; θ) — updated every step
2. **Target network**: Q(s, a; θ^-) — updated every C steps

**Update rule**:
```
Target: y = r + γ · max_a' Q(s', a'; θ^-)  ← Uses θ^-, not θ!
Loss: L(θ) = [y - Q(s, a; θ)]²
Update: θ ← θ - α · ∇_θ L(θ)

Every C steps: θ^- ← θ  (hard update)
```

---

## Concrete Numerical Example

**Setup**: Simple GridWorld
- 3 states: s_1 → s_2 → s_3 (deterministic)
- One action per state
- Rewards: r(s_1) = 0, r(s_2) = 0, r(s_3) = 10
- γ = 0.9

**Initial Q-values**: Q(s_1) = Q(s_2) = Q(s_3) = 0

**Learning rate**: α = 0.5 (high for illustration)

### Without Target Network

**Step 1**: Update Q(s_3)
```
Experience: (s_3, r=10, terminal)
Target: y = 10 (terminal state)
Current: Q(s_3; θ) = 0
Loss: (10 - 0)² = 100
Update: Q(s_3) ← 0 + 0.5·(10 - 0) = 5

Result: Q(s_1)=0, Q(s_2)=0, Q(s_3)=5
```

**Step 2**: Update Q(s_2)
```
Experience: (s_2, r=0, s_3)
Target: y = 0 + 0.9·Q(s_3; θ) = 0 + 0.9·5 = 4.5
Current: Q(s_2; θ) = 0
Update: Q(s_2) ← 0 + 0.5·(4.5 - 0) = 2.25

Result: Q(s_1)=0, Q(s_2)=2.25, Q(s_3)=5
```

**Step 3**: Update Q(s_3) again
```
Experience: (s_3, r=10, terminal)
Target: y = 10
Current: Q(s_3; θ) = 5
Update: Q(s_3) ← 5 + 0.5·(10 - 5) = 7.5

Result: Q(s_1)=0, Q(s_2)=2.25, Q(s_3)=7.5
← Q(s_3) changed!
```

**Step 4**: Update Q(s_2) again
```
Experience: (s_2, r=0, s_3)
Target: y = 0 + 0.9·Q(s_3; θ) = 0 + 0.9·7.5 = 6.75
← Target increased because Q(s_3) increased!

Current: Q(s_2; θ) = 2.25
Update: Q(s_2) ← 2.25 + 0.5·(6.75 - 2.25) = 4.5

Result: Q(s_1)=0, Q(s_2)=4.5, Q(s_3)=7.5
```

**Step 5**: Update Q(s_3) again
```
Target: y = 10
Current: Q(s_3; θ) = 7.5
Update: Q(s_3) ← 7.5 + 0.5·(10 - 7.5) = 8.75

Result: Q(s_1)=0, Q(s_2)=4.5, Q(s_3)=8.75
```

**Step 6**: Update Q(s_2) again
```
Target: y = 0 + 0.9·Q(s_3; θ) = 0 + 0.9·8.75 = 7.875
← Target keeps increasing!

Current: Q(s_2; θ) = 4.5
Update: Q(s_2) ← 4.5 + 0.5·(7.875 - 4.5) = 6.1875

Result: Q(s_1)=0, Q(s_2)=6.1875, Q(s_3)=8.75
```

**Observation**: Target for s_2 keeps changing: 4.5 → 6.75 → 7.875 → ...
- Chasing a moving target!
- Eventually converges, but with oscillations

### With Target Network (C = 3 steps)

**Steps 1-3**: Same as above, use target network θ^-

**Step 1**: Update Q(s_3)
```
Target: y = 10 (terminal)
Q(s_3; θ) ← 5

Online: Q(s_1)=0, Q(s_2)=0, Q(s_3)=5
Target: Q(s_1)=0, Q(s_2)=0, Q(s_3)=0  ← Not updated yet!
```

**Step 2**: Update Q(s_2)
```
Target: y = 0 + 0.9·Q(s_3; θ^-) = 0 + 0.9·0 = 0
← Uses old target network value!

Q(s_2; θ) ← 0 + 0.5·(0 - 0) = 0  (no change)

Online: Q(s_1)=0, Q(s_2)=0, Q(s_3)=5
Target: Q(s_1)=0, Q(s_2)=0, Q(s_3)=0
```

**Step 3**: Update Q(s_3) again
```
Target: y = 10
Q(s_3; θ) ← 5 + 0.5·(10 - 5) = 7.5

Online: Q(s_1)=0, Q(s_2)=0, Q(s_3)=7.5
Target: Q(s_1)=0, Q(s_2)=0, Q(s_3)=0
```

**Step 4**: Target network update! θ^- ← θ
```
Online: Q(s_1)=0, Q(s_2)=0, Q(s_3)=7.5
Target: Q(s_1)=0, Q(s_2)=0, Q(s_3)=7.5  ← Synchronized!
```

**Step 5**: Update Q(s_2)
```
Target: y = 0 + 0.9·Q(s_3; θ^-) = 0 + 0.9·7.5 = 6.75
← Target is stable (won't change for 3 steps)

Q(s_2; θ) ← 0 + 0.5·(6.75 - 0) = 3.375

Online: Q(s_1)=0, Q(s_2)=3.375, Q(s_3)=7.5
Target: Q(s_1)=0, Q(s_2)=0, Q(s_3)=7.5
```

**Steps 6-7**: Continue with stable target = 6.75

**Key difference**: Target stays 6.75 for 3 steps, even though Q(s_3) changes in online network!

---

## Visualization of Target Evolution

**Without target network**:
```
Step │ Q(s_2) │ Q(s_3) │ Target for s_2
─────┼────────┼────────┼───────────────
  1  │  0.00  │  5.00  │     —
  2  │  2.25  │  5.00  │   4.50
  3  │  2.25  │  7.50  │     —
  4  │  4.50  │  7.50  │   6.75  ← Changed!
  5  │  4.50  │  8.75  │     —
  6  │  6.19  │  8.75  │   7.875 ← Changed again!
  7  │  6.19  │  9.38  │     —
  8  │  7.34  │  9.38  │   8.438 ← Keep changing!

Target is non-stationary, moving target
```

**With target network (C=3)**:
```
Step │ Q(s_2; θ) │ Q(s_3; θ) │ Q(s_3; θ^-) │ Target
─────┼───────────┼───────────┼──────────────┼────────
  1  │   0.00    │   5.00    │    0.00      │   0.00
  2  │   0.00    │   5.00    │    0.00      │   0.00
  3  │   0.00    │   7.50    │    0.00      │   0.00
  4* │   0.00    │   7.50    │    7.50      │   —    ← Update target net
  5  │   3.38    │   8.75    │    7.50      │   6.75
  6  │   5.04    │   9.06    │    7.50      │   6.75
  7  │   5.87    │   9.53    │    7.50      │   6.75
  8* │   5.87    │   9.53    │    9.53      │   —    ← Update target net
  9  │   6.84    │   9.76    │    9.53      │   8.58
 10  │   7.36    │   9.88    │    9.53      │   8.58

Target is piecewise constant, stable periods
```

---

## Why Target Network Stabilizes Learning

### 1. Fixed Targets for C Steps

**Without target network**: Every update to Q(s') changes target for Q(s)
- Feedback loop: Q → target → Q → target → ...
- Targets shift constantly
- Hard to make progress

**With target network**: Target fixed for C steps
- Can optimize toward stable target
- Like supervised learning (labels don't change during epoch)
- Makes progress before target moves

### 2. Breaks Feedback Loop

**Feedback loop** without target network:
```
Increase Q(s,a) → Increases target for Q(s_prev, a_prev)
                → Increases Q(s_prev, a_prev)
                → Increases target for Q(s_prev_prev, a_prev_prev)
                → ...

Errors propagate backward through value function rapidly
Potential for divergence
```

**With target network**:
```
Increase Q(s,a; θ) → Target for Q(s_prev) unchanged (uses θ^-)
                   → Q(s_prev) updates toward old target
                   → Feedback delayed by C steps

Errors propagate slowly, giving network time to stabilize
```

### 3. Reduces Oscillations

**Without target network** (from example):
```
Q(s_2) trajectory: 0 → 2.25 → 4.5 → 6.19 → 7.34 → ...
Targets: 4.5, 6.75, 7.875, 8.438, ...

Large swings, overshooting
```

**With target network**:
```
Q(s_2) trajectory: 0 → 0 → 0 → 3.38 → 5.04 → 5.87 → 6.84 → 7.36 → ...
Targets: 0 (held for 3 steps), then 6.75 (held for 3 steps), then 8.58, ...

Smoother updates, less overshooting
```

---

## Trade-offs of Target Network

### Advantages

✅ **Stability**: Much more stable learning
✅ **Convergence**: Reduces oscillations and divergence
✅ **Simpler optimization**: Fixed targets like supervised learning
✅ **Empirical success**: Critical for DQN to work on Atari

### Disadvantages

❌ **Lag**: Target network is outdated
- Learning slower because using old Q-values
- Can't immediately incorporate improvements

❌ **Memory**: Must store two copies of network
- 2x memory consumption
- Not significant for modern hardware

❌ **Hyperparameter**: Must tune C
- C too small: Less stable (approaches no target network)
- C too large: Too slow to adapt
- Typical: C = 1,000 - 10,000 steps

❌ **Not true gradient**: Optimizing toward moving target
- Even with target network, target still moves (every C steps)
- Just moves more slowly

---

## Choosing Update Frequency C

**Small C** (e.g., C = 100):
- More responsive to improvements
- Less stable
- Better for fast-changing environments

**Large C** (e.g., C = 10,000):
- More stable
- Slower to incorporate improvements
- Better for complex, unstable problems

**Typical choice**: C = 1,000 - 10,000
- Atari DQN: 10,000 steps
- CartPole: 100-500 steps

**Rule of thumb**: Set C so target updates every few episodes

---

## Alternative: Soft Target Updates (Polyak Averaging)

**Instead of hard updates** (θ^- ← θ every C steps):

**Soft updates** every step:
```
θ^- ← τ·θ + (1-τ)·θ^-

where τ << 1 (e.g., τ = 0.001)
```

**Advantages**:
- Smoother target evolution
- No sudden jumps
- One less hyperparameter (no C)

**Used in**: DDPG, TD3, SAC (later weeks)

---

## Empirical Comparison

**Atari Pong results**:

```
Without target network:
- Fails to learn
- Q-values diverge to ±1000
- Performance: random

With target network (C=10,000):
- Learns successfully
- Q-values stable
- Performance: near-optimal after 2M frames
```

**CartPole results**:

```
Without target network:
- High variance
- Occasional divergence
- Average episodes to solve: 1000 (when it works)

With target network (C=100):
- Low variance
- Never diverges
- Average episodes to solve: 200
```

---

## Summary Table

| Aspect | Without Target Net | With Target Net (C=1000) |
|--------|-------------------|--------------------------|
| Target stability | Changes every step | Fixed for 1000 steps |
| Convergence | Oscillations | Smooth |
| Sample efficiency | Variable | Consistent |
| Computational cost | 1x | 1x (minimal overhead) |
| Memory | 1x | 2x |
| Hyperparameters | Fewer | One more (C) |
| Empirical success | Poor | Excellent |

**Conclusion**: Target network is essential for DQN's success. The trade-off (slightly slower learning, extra memory) is worth the massive stability improvement.

</details>

---

## Question 4: Application

**Design a DQN architecture for CartPole. Specify the network structure (layers, activations, sizes), hyperparameters (learning rate, buffer size, target update frequency, epsilon schedule), and justify your choices. Then describe how you would diagnose and fix common problems.**

<details>
<summary>Answer</summary>

## DQN Architecture for CartPole

### Environment Specification

**CartPole-v1**:
- **State space**: 4-dimensional continuous
  - Position: x ∈ [-4.8, 4.8]
  - Velocity: ẋ ∈ [-∞, ∞]
  - Angle: θ ∈ [-24°, 24°]
  - Angular velocity: θ̇ ∈ [-∞, ∞]
- **Action space**: 2 discrete actions
  - 0: Push cart left
  - 1: Push cart right
- **Reward**: +1 for each timestep alive
- **Termination**: Angle > 12°, position > 2.4, or 500 steps
- **Solved criterion**: Average reward ≥ 195 over 100 episodes

---

## Network Architecture

### Design Choice: Fully Connected Network

**Justification**:
- Low-dimensional input (4 values) → no need for CNNs
- Fully connected networks sufficient for small state spaces
- Simple, fast, easy to train

### Architecture Specification

```python
import torch
import torch.nn as nn

class CartPoleDQN(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=128):
        super(CartPoleDQN, self).__init__()

        self.network = nn.Sequential(
            # Input layer
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            # Hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            # Output layer
            nn.Linear(hidden_dim, action_dim)
            # No activation (linear Q-values)
        )

    def forward(self, state):
        return self.network(state)
```

### Layer-by-Layer Justification

**Layer 1**: Linear(4 → 128) + ReLU
- **Input size**: 4 (state dimensions)
- **Output size**: 128 (hidden units)
- **Justification**:
  - 128 units provides enough capacity for CartPole (simple problem)
  - Not too large (would slow training, risk overfitting)
  - Not too small (might lack capacity)
- **Activation**: ReLU (standard, non-saturating, computationally efficient)

**Layer 2**: Linear(128 → 128) + ReLU
- **Why second hidden layer?**:
  - Increases representational power
  - CartPole value function has non-linear structure
  - Two layers enough for simple problems (universal approximation)
- **Same size**: Keeps capacity constant, avoids bottleneck

**Layer 3**: Linear(128 → 2)
- **Output size**: 2 (one Q-value per action)
- **No activation**: Q-values can be any real number
  - Not probabilities (no softmax)
  - Not bounded (no sigmoid/tanh)
- **Linear output**: Standard for DQN

### Alternative Architectures Considered

**Shallow network** (1 hidden layer):
```python
nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)
```
- Faster training
- Might lack capacity
- Try if overfitting occurs

**Deeper network** (3+ hidden layers):
```python
nn.Linear(4, 128), ReLU(),
nn.Linear(128, 128), ReLU(),
nn.Linear(128, 64), ReLU(),
nn.Linear(64, 2)
```
- More capacity
- Unnecessary for CartPole (overkill)
- Slower training, potential overfitting

**Wider network** (more hidden units):
```python
nn.Linear(4, 256), ReLU(), nn.Linear(256, 256), ReLU(), nn.Linear(256, 2)
```
- More parameters
- Slower training
- Unlikely to help for simple problem

**Chosen architecture (2 × 128) is the sweet spot for CartPole.**

---

## Hyperparameters

### Learning Rate

**Choice**: α = 0.001 (1e-3)

**Justification**:
- Standard starting point for Adam optimizer
- Not too large (would cause instability)
- Not too small (would slow learning)

**Tuning guide**:
- Too large symptoms: Q-values explode, high variance, no learning
- Too small symptoms: Very slow learning, may not solve within reasonable time
- Try: [0.0001, 0.001, 0.01] and pick best

### Replay Buffer Size

**Choice**: N = 10,000 transitions

**Justification**:
- CartPole episodes are ~200-500 steps
- 10,000 transitions ≈ 20-50 episodes
- Enough diversity to decorrelate samples
- Not too large (memory efficient, faster sampling)

**Alternatives**:
- Smaller (5,000): Less memory, but less diversity
- Larger (50,000): More diversity, but unnecessary for simple problem

### Minibatch Size

**Choice**: B = 64

**Justification**:
- Standard batch size for neural networks
- Good balance: stable gradients, computational efficiency
- Powers of 2 are GPU-friendly

**Alternatives**:
- B = 32: Faster updates, higher variance
- B = 128: More stable, slower updates

### Target Network Update Frequency

**Choice**: C = 100 steps

**Justification**:
- CartPole episodes are ~200-500 steps
- Update target every ~0.2-0.5 episodes
- Fast enough to incorporate improvements
- Slow enough for stability

**Alternatives**:
- C = 10: Too frequent, less stable
- C = 1000: Too infrequent, slower learning

**Rule**: Set C ≈ 0.2-1 episode length

### Discount Factor

**Choice**: γ = 0.99

**Justification**:
- Standard value for episodic tasks
- Horizon: 1/(1-γ) = 100 steps (reasonable for CartPole)
- Values future rewards highly (important for delayed rewards)

**Alternatives**:
- γ = 0.95: Shorter horizon, faster learning, might undervalue stability
- γ = 0.999: Longer horizon, might be unnecessary

### Epsilon Schedule (Exploration)

**Choice**: Epsilon-greedy with decay

**Schedule**:
```python
epsilon_start = 1.0      # Pure exploration initially
epsilon_end = 0.01       # 1% exploration at end
epsilon_decay = 0.995    # Decay rate per episode

epsilon = max(epsilon_end, epsilon * epsilon_decay)  # After each episode
```

**Justification**:
- Start with full exploration (discover all states)
- Gradually exploit learned policy
- Maintain small exploration (prevent premature convergence)

**Alternative**: Linear decay
```python
epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / total_episodes)
```

**Decay rate tuning**:
- Too fast (0.99): Premature exploitation, might miss good strategies
- Too slow (0.999): Too much exploration, slow learning
- 0.995 balances exploration and exploitation over ~500 episodes

### Training Duration

**Choice**: 500 episodes (typically solves in 200-300)

**Why**:
- CartPole is simple, solves quickly
- 500 episodes = safety margin
- Training time: ~5-10 minutes on CPU

---

## Complete Hyperparameter Table

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Hidden layers | 2 | Sufficient capacity, not overkill |
| Hidden units | 128 | Sweet spot for CartPole |
| Learning rate | 0.001 | Standard for Adam |
| Optimizer | Adam | Adaptive, works well |
| Buffer size | 10,000 | ~20-50 episodes worth |
| Batch size | 64 | Standard, GPU-friendly |
| Target update | 100 steps | ~0.2-0.5 episodes |
| Discount γ | 0.99 | Standard for episodic |
| ε start | 1.0 | Full exploration initially |
| ε end | 0.01 | Maintain 1% exploration |
| ε decay | 0.995 | Decay over ~500 episodes |
| Episodes | 500 | Enough to solve + margin |

---

## Common Problems and Diagnostics

### Problem 1: No Learning (Random Performance)

**Symptoms**:
- Reward stuck at ~20-30 (random policy level)
- Q-values don't increase
- No improvement after many episodes

**Diagnosis**:
```python
# Check if network is updating
print(f"Q-values: {q_network(state).detach().numpy()}")
# Should change over time, not stay at initialization values

# Check if agent is learning
print(f"Loss: {loss.item()}")
# Should be decreasing over time
```

**Possible causes and fixes**:

1. **Learning rate too small**
   - Fix: Increase α to 0.01
   - Check: Loss decreasing?

2. **Not training enough**
   - Fix: Check `if len(buffer) > batch_size` before training
   - Check: Are we actually calling `agent.update()`?

3. **Epsilon too high**
   - Fix: Check epsilon value, should decay
   - Debug: `print(f"Epsilon: {agent.epsilon}")`

4. **Gradient not flowing**
   - Fix: Check for `requires_grad=False` accidentally set
   - Check: `print(q_network.fc1.weight.grad)` after backward pass

### Problem 2: Unstable Learning (High Variance)

**Symptoms**:
- Reward increases, then crashes
- Q-values oscillate wildly
- Occasional episodes with high reward, but not consistent

**Diagnosis**:
```python
# Plot Q-values over time
plt.plot(q_values_history)
plt.show()
# Should be smooth, not spiky

# Check TD errors
plt.plot(td_errors)
# Should decrease and stabilize
```

**Possible causes and fixes**:

1. **Learning rate too high**
   - Fix: Reduce α to 0.0001
   - Symptom: Q-values explode to ±1000

2. **Target network updating too frequently**
   - Fix: Increase C to 500
   - Symptom: Non-stationary targets

3. **Batch size too small**
   - Fix: Increase to 128
   - Symptom: High gradient variance

4. **Insufficient exploration**
   - Fix: Slower epsilon decay (0.999 instead of 0.995)
   - Symptom: Gets stuck in local optimum

### Problem 3: Slow Learning

**Symptoms**:
- Eventually solves, but takes 1000+ episodes
- Steady but very slow improvement

**Diagnosis**:
```python
# Check learning curve
plt.plot(episode_rewards)
# Should reach 195 by episode 300-400
```

**Possible causes and fixes**:

1. **Learning rate too small**
   - Fix: Increase α to 0.005
   - Safe to be aggressive for simple problems

2. **Buffer size too small**
   - Fix: Increase to 50,000
   - More diverse samples

3. **Target updates too infrequent**
   - Fix: Decrease C to 50
   - Faster incorporation of improvements

4. **Epsilon decay too slow**
   - Fix: Faster decay (0.99 instead of 0.995)
   - Exploit earlier

### Problem 4: Overfitting/Catastrophic Forgetting

**Symptoms**:
- Solves initially, then performance degrades
- Learns one strategy, forgets others
- Unstable after "solving"

**Diagnosis**:
```python
# Check buffer diversity
plt.hist(buffer.states[:, 0])  # Position distribution
# Should cover range [-2, 2]
```

**Possible causes and fixes**:

1. **Buffer too small**
   - Fix: Increase to 20,000
   - Retain more diverse experiences

2. **Not enough exploration after learning**
   - Fix: Keep ε ≥ 0.05 (instead of 0.01)
   - Continue exploring even after solving

3. **Network too large**
   - Fix: Reduce to 64 hidden units
   - Less capacity to overfit

### Problem 5: Q-Value Overestimation

**Symptoms**:
- Q-values become unrealistically high (>500)
- CartPole maximum return is 500, but Q-values >> 500
- Performance still okay, but Q-values inflated

**Diagnosis**:
```python
print(f"Max Q-value: {q_network(state).max().item()}")
# Should be ≤ 500 for CartPole
```

**Possible causes and fixes**:

1. **Inherent to Q-learning (max bias)**
   - Expected: Q-values slightly overestimated
   - Fix: Use Double DQN (Week 10)

2. **Accumulation of errors**
   - Fix: Huber loss instead of MSE
   - Reduces impact of large errors

---

## Debugging Checklist

Before asking for help, check:

- [ ] Network architecture correct (no softmax on output!)
- [ ] Loss function: MSE or Huber loss
- [ ] Optimizer: Adam with reasonable learning rate
- [ ] Target network: Updated periodically, not every step
- [ ] Experience replay: Sampling random minibatches
- [ ] Epsilon-greedy: Decaying over time
- [ ] Gradients: Flowing (not None, not NaN)
- [ ] Rewards: Being recorded correctly
- [ ] Shapes: State (batch, 4), Q-values (batch, 2)

---

## Expected Learning Curve

```
Episodes 0-50:    Reward ~20-50 (exploring, mostly random)
Episodes 50-100:  Reward ~50-100 (starting to learn)
Episodes 100-200: Reward ~100-195 (rapid improvement)
Episodes 200+:    Reward ~195-500 (solved, maintaining)
```

If your curve looks different, refer to diagnostics above!

---

## Full Implementation Tips

1. **Start simple**: Get basic DQN working before optimizations
2. **Log everything**: Q-values, losses, epsilon, rewards
3. **Visualize**: Plot learning curves in real-time
4. **Reproducibility**: Set random seeds
5. **Checkpointing**: Save best model, not just final model
6. **Test deterministically**: Evaluate with ε=0 every N episodes

**Solve CartPole before moving to Atari!** CartPole is the unit test for DQN.

</details>

---

## Question 5: Critical Thinking

**What are the remaining limitations of DQN even with experience replay and target networks? Discuss at least three fundamental limitations and suggest how they might be addressed (hint: some will be covered in Week 10 DQN Extensions).**

<details>
<summary>Answer</summary>

Despite its groundbreaking success, DQN has several fundamental limitations:

---

## Limitation 1: Overestimation Bias

### The Problem

**Q-learning inherently overestimates action values** due to using max in the target:

```
Target: y = r + γ · max_a' Q(s', a'; θ^-)
```

**Why overestimation occurs**:

**Mathematical explanation**:
```
E[max(X, Y)] ≥ max(E[X], E[Y])

Applied to Q-learning:
E[max_a' Q(s', a')] ≥ max_a' E[Q(s', a')]

Taking max of noisy estimates → biased toward overestimation
```

**Concrete example**:
```
True Q-values: Q*(s', a1) = 5, Q*(s', a2) = 5
Estimates with noise: Q(s', a1) = 6, Q(s', a2) = 4

max_a' Q(s', a') = 6 > 5 = Q*(s')

We select the action with positive noise!
```

**Accumulation over learning**:
```
Overestimated Q(s') → Overestimated target for Q(s) → Overestimated Q(s)
→ Overestimated target for Q(s_prev) → ...

Bias propagates through value function
```

### Impact

- **Suboptimal policies**: Agent prefers overestimated bad actions
- **Instability**: Positive feedback loop amplifies overestimation
- **Slower learning**: Must correct overestimations over time
- **Poor performance**: Empirically shown to hurt performance

### Evidence in Practice

**Atari games**:
- DQN Q-values often 2-3x higher than true values
- Games with stochastic dynamics show larger overestimation
- Correlation between overestimation and poor performance

### Solution: Double DQN (Week 10)

**Key idea**: Decouple action selection from action evaluation

```
DQN: y = r + γ · Q(s', argmax_a' Q(s', a'; θ^-); θ^-)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      Use same network to select and evaluate

Double DQN: y = r + γ · Q(s', argmax_a' Q(s', a'; θ); θ^-)
                              ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^
                              Select with online        Evaluate with target
```

**Why it works**: Reduces positive bias because selection and evaluation are independent

**Improvement**: Typically 5-20% performance gain on Atari

---

## Limitation 2: Inefficient Representation (Dueling Architecture)

### The Problem

**Single stream architecture wastes representational capacity**:

Standard DQN:
```
State → Hidden layers → Q(s, a1), Q(s, a2), ..., Q(s, a|A|)
```

**Issue**: For many states, Q-values for all actions are similar
- State is good/bad regardless of action
- Don't need to compute Q separately for each action
- Redundant computation

**Example (GridWorld)**:
```
State near goal: Q(s, left)=8, Q(s, right)=9, Q(s, up)=8.5, Q(s, down)=7
All actions have similar high value (state is good)

State in wall: Q(s, left)=0, Q(s, right)=0, Q(s, up)=0, Q(s, down)=0
All actions have same low value (state is bad)
```

**Key insight**: Q(s, a) = V(s) + A(s, a)
- V(s): Value of being in state s
- A(s, a): Advantage of action a over average

For many states, A(s, a) ≈ 0 for all actions (state value dominates)

### Impact

- **Sample inefficiency**: Must learn V(s) separately for each action
- **Slower learning**: Redundant learning across actions
- **Poor generalization**: Doesn't exploit structure in Q-function

### Solution: Dueling DQN (Week 10)

**Architecture**:
```
State → Shared layers → Split into two streams:
                        ├── Value stream: V(s)
                        └── Advantage stream: A(s, a) for all a

Combine: Q(s, a) = V(s) + [A(s, a) - mean_a' A(s, a')]
```

**Why it works**:
- Explicitly separates state value from action advantages
- V(s) learned once, shared across all actions
- A(s, a) focuses on relative differences (easier to learn)

**Improvement**: 20-50% better sample efficiency on many Atari games

---

## Limitation 3: Uniform Experience Replay (Importance Sampling)

### The Problem

**All transitions are treated equally** in replay buffer:

```
Sample minibatch uniformly at random from buffer
All transitions have equal probability 1/|D|
```

**Issue**: Not all transitions are equally useful
- High TD-error transitions: Network hasn't learned them well yet
- Low TD-error transitions: Already learned, less informative
- Rare transitions: Important but sampled infrequently

**Analogy**: Studying for an exam
- Bad: Spend equal time on topics you know and don't know
- Good: Focus on topics you don't know (high error)

**Example**:
```
Transition 1: Easy state, TD error = 0.1 (already learned well)
Transition 2: Rare state, TD error = 5.0 (poorly learned)

Uniform sampling: 50% chance to sample each
Better: Sample transition 2 more often (higher error → more to learn)
```

### Impact

- **Sample inefficiency**: Wasting updates on already-learned transitions
- **Slower learning**: Not focusing on hard cases
- **Rare events**: Important rare transitions sampled too infrequently

### Solution: Prioritized Experience Replay (PER) (Week 10)

**Key idea**: Sample transitions proportional to TD error

```
Priority: p_i = |δ_i| + ε
where δ_i = r + γ · max_a' Q(s', a'; θ^-) - Q(s, a; θ)

Probability: P(i) = p_i^α / Σ_j p_j^α
```

**Parameters**:
- α: Controls how much to prioritize (α=0: uniform, α=1: proportional to error)
- ε: Small constant to ensure all transitions have nonzero probability

**Importance sampling correction**:
```
Weight: w_i = (N · P(i))^{-β}

Update: θ ← θ - α · w_i · ∇_θ L_i(θ)
```

**Why it works**: Focuses learning on transitions where network makes large errors

**Improvement**: 30-50% faster learning on Atari, huge gains on sparse reward tasks

---

## Limitation 4: Discrete Actions Only

### The Problem

**DQN fundamentally requires discrete actions**:

```
Target: y = r + γ · max_a' Q(s', a'; θ^-)
                    ^^^^^^^
                    Must enumerate all actions
```

**Why continuous actions don't work**:
- Can't compute max over infinite action space
- Can't represent Q(s, a) for all a ∈ ℝ^d

**Workaround**: Discretize action space
```
Continuous action: a ∈ [-1, 1]
Discretize: a ∈ {-1, -0.5, 0, 0.5, 1}

Problems:
- Loses precision
- Curse of dimensionality (d-dimensional action → |A|^d discrete actions)
- Example: 10 actions per dimension, 3D action space → 1000 actions
```

### Impact

- **Cannot handle continuous control**: Robot manipulation, vehicle control, etc.
- **Poor performance on discretized continuous actions**: Loses precision
- **Not applicable to many real-world problems**

### Solution: Actor-Critic Methods (Weeks 11-14)

**Different approach**: Policy gradient methods (REINFORCE, A2C, PPO)
- Learn policy π(a|s) directly
- No need for max_a Q(s, a)
- Naturally handles continuous actions

**Deterministic policy gradient** (DDPG, TD3):
- Learns deterministic policy μ(s)
- Uses gradient ∇_a Q(s, a) to optimize policy
- Extends Q-learning to continuous actions

---

## Limitation 5: Sample Inefficiency

### The Problem

**DQN requires millions of samples** even for relatively simple games:

- **Atari**: 10-50 million frames to reach human performance
- **Simple games**: 1-2 million frames
- **Humans**: Can learn Pong in minutes (~10,000 frames)

**Why so sample inefficient?**:
1. **Off-policy**: Can't fully trust off-policy data (distribution mismatch)
2. **Bootstrapping**: Slow propagation of value information
3. **No model**: Doesn't learn model of environment (can't plan)
4. **No transfer**: Learns from scratch for each game

### Impact

- **Expensive**: Long training times (days on GPUs)
- **Impractical for real robots**: Can't run millions of trials
- **Sample complexity**: Theoretical sample complexity unknown

### Partial Solutions

**Model-based RL**:
- Learn model of environment
- Use model for planning and data augmentation
- 10-100x more sample efficient

**Transfer learning**:
- Pre-train on related tasks
- Fine-tune on target task
- Reduces sample complexity

**Imitation learning**:
- Learn from expert demonstrations
- Bootstrap learning with good initial policy

---

## Limitation 6: Exploration

### The Problem

**ε-greedy is naive exploration**:

```
With probability ε: random action
Otherwise: argmax_a Q(s, a)
```

**Issues**:
1. **Random exploration**: No directed exploration toward uncertain states
2. **No curiosity**: Doesn't seek out informative states
3. **Fails on hard exploration problems**: Sparse rewards, deep exploration

**Example failure (Montezuma's Revenge)**:
- Atari game requiring long sequence of specific actions
- Random exploration almost never finds reward
- DQN fails completely (score ≈ 0)

### Impact

- **Fails on hard exploration problems**: Montezuma, Pitfall, Private Eye
- **Suboptimal on easy problems**: Could explore more efficiently
- **No systematic exploration**: Pure luck-based

### Solutions

**Exploration bonuses**:
- Add intrinsic reward for visiting novel states
- Examples: Curiosity-driven exploration, count-based exploration

**Noisy networks**:
- Add learned noise to network weights
- State-dependent exploration

**Distributional RL**:
- Learn distribution over returns, not just expectation
- Implicit exploration bonus

---

## Summary Table of Limitations and Solutions

| Limitation | Impact | Solution | Improvement |
|------------|--------|----------|-------------|
| Overestimation bias | Suboptimal policy | Double DQN | 5-20% |
| Inefficient representation | Slow learning | Dueling DQN | 20-50% |
| Uniform replay | Sample inefficiency | PER | 30-50% |
| Discrete actions only | Can't do continuous control | DDPG, SAC | N/A |
| Sample inefficiency | Expensive training | Model-based RL | 10-100x |
| Poor exploration | Fails on hard problems | Exploration bonuses | Qualitative |

---

## The Rainbow Solution (Week 10)

**Key insight**: Combining improvements is better than any individual improvement

**Rainbow DQN** combines:
1. Double DQN (overestimation)
2. Dueling DQN (representation)
3. PER (sampling)
4. Multi-step returns (credit assignment)
5. Distributional RL (value distribution)
6. Noisy nets (exploration)

**Result**: State-of-the-art on Atari, addresses most limitations

**But still doesn't solve**:
- Continuous actions (need different algorithms)
- Extreme sample inefficiency (need model-based methods)
- Hard exploration (need better exploration strategies)

---

## Philosophical Perspective

**DQN was revolutionary, not perfect**:
- Proved deep RL is possible
- Established experience replay and target networks as standard
- Opened door to modern deep RL

**But it's just the beginning**:
- Many fundamental challenges remain
- Active research area with rapid progress
- No single "best" algorithm—different methods for different problems

**The lesson**: Every algorithm has limitations. Understanding them is key to choosing the right tool for your problem.

</details>

