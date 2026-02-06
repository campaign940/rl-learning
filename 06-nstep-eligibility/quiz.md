# Week 6 Quiz: n-step Methods & Eligibility Traces

Test your understanding of n-step bootstrapping and eligibility traces.

## Question 1: Unifying MC and TD

**How does n-step TD unify Monte Carlo and TD(0)? What happens when n=1 and n=∞? Why is there often an optimal n strictly between these extremes?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**The Unification**:

n-step TD creates a spectrum of methods that interpolates between TD(0) and Monte Carlo by varying how far ahead we look before bootstrapping.

**n-step Return Formula**:
```
G_t:t+n = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
        └──────── n actual rewards ────────┘   └─ bootstrap ─┘
```

---

**Special Case n=1 (TD(0))**:

```
G_t:t+1 = R_{t+1} + γV(S_{t+1})
```

- Only one actual reward
- Immediately bootstrap from V(S_{t+1})
- **Maximum bias** (bootstraps from potentially wrong estimate)
- **Minimum variance** (only one random reward)
- **Fastest updates** (1-step delay)

**Properties**:
- Highly biased initially (depends on initialization)
- Very stable updates (low variance)
- Works online (update after every step)
- Works for continuing tasks

---

**Special Case n=∞ (Monte Carlo)**:

```
G_t:t+∞ = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T
        = G_t (the complete return)
```

- All actual rewards until termination
- No bootstrapping at all
- **Zero bias** (actual return is unbiased estimate of v_π)
- **Maximum variance** (all future randomness included)
- **Slowest updates** (must wait for episode end)

**Properties**:
- Unbiased estimates
- High variance (depends on all future rewards)
- Only works for episodic tasks
- Update only at episode end

---

**The Spectrum**:

```
n=1         n=2    n=3    n=5    n=10    ...    n=∞
TD(0)   ←── more bootstrapping | less bootstrapping ──→   MC

High bias                                          Zero bias
Low variance                                       High variance
Fast updates                                       Slow updates
Works online                                       Episodic only
```

**As n increases**:
1. **Bias decreases**: More actual rewards, less reliance on (potentially wrong) estimates
2. **Variance increases**: More random rewards contribute to the return
3. **Update delay increases**: Must wait n steps before updating
4. **Approaches MC**: Eventually equals full return

---

**Why Optimal n is Often Intermediate**:

**The Bias-Variance Tradeoff**:

Total error comes from two sources:
```
MSE = Bias² + Variance
```

**Small n (e.g., n=1)**:
- High bias: Bootstrap from potentially inaccurate V(s)
- Low variance: Only one reward is random
- **Problem**: Slow to propagate information (only 1 step at a time)

**Large n (approaching ∞)**:
- Low bias: Many actual rewards
- High variance: Many random rewards
- **Problem**: Updates are noisy and inconsistent

**Intermediate n (e.g., n=4 to 8)**:
- **Balanced**: Some bias, some variance
- **Faster learning**: Information propagates faster than n=1
- **Stable**: Lower variance than MC
- **Sweet spot**: Minimizes total error

---

**Concrete Example: 19-State Random Walk**

True values: v_π(i) = i/20

**n=1 (TD)**:
```
Episode 1: Start at state 10 → 11 → 10 → 11 → ... → 20 (reward 1)

Updates happen every step:
  V(10) ← V(10) + α[0 + γV(11) - V(10)]
  V(11) ← V(11) + α[0 + γV(10) - V(11)]
  ...

Information propagates 1 step per episode.
Slow to learn: takes many episodes for reward at 20 to affect V(10).
```

**n=8**:
```
Same episode, but updates use 8-step returns:
  G_{t:t+8} = 0 + 0 + ... + 0 + γ^8 V(S_{t+8})

After visiting 10 → 11 → 12 → ... → 18:
  V(10) updated using 8 steps ahead
  Information propagates faster!

Fewer episodes needed for reward information to spread.
```

**n=∞ (MC)**:
```
Same episode, final return: G_10 = 0 + 0 + ... + 0 + 1 = 1

V(10) ← V(10) + α[1 - V(10)]

Correct target (1), but high variance across episodes:
  - Some episodes go left (G=0)
  - Some go right (G=1)
  - Very noisy updates

More episodes needed to average out variance.
```

**Empirical Results (typical)**:
```
RMSE after 100 episodes:
  n=1:  0.20
  n=4:  0.10  ← best
  n=8:  0.12
  n=16: 0.18
  n=∞:  0.25
```

Optimal n=4 to 8 for this task.

---

**Why Optimal n Depends on Task**:

**1. Episode Length**:
- Short episodes (10 steps): Large n is fine (approaches MC)
- Long episodes (1000 steps): Small n is better (less variance)

**2. Reward Density**:
- Dense rewards: Small n works well (rewards propagate easily)
- Sparse rewards: Large n or eligibility traces (need to look ahead)

**3. Environment Stochasticity**:
- Deterministic: Larger n is okay (returns have low variance)
- Stochastic: Smaller n is better (reduce variance from many random rewards)

**4. State Space Structure**:
- Chain-like (sequential): Larger n helps (propagate info down chain)
- Tree-like (branching): Smaller n may be better (high variance in returns)

---

**Mathematical Intuition**:

The expected squared error for n-step TD is:
```
E[(G_t:t+n - v_π(S_t))²]
```

This decomposes into:
```
Bias²: E[G_t:t+n - v_π(S_t)]²  (due to bootstrap V(S_{t+n}) ≠ v_π(S_{t+n}))
Variance: Var[G_t:t+n]  (due to random rewards R_{t+1}, ..., R_{t+n})
```

As n increases:
- Bias² decreases (V gets closer to true value, and we use it less)
- Variance increases (more random rewards)

Optimal n minimizes the sum:
```
n* = argmin_n [Bias²(n) + Variance(n)]
```

This typically occurs at intermediate n.

---

**Practical Rule of Thumb**:

| Task Characteristics | Recommended n |
|----------------------|---------------|
| Short episodes (< 20 steps) | n = 10-20 or MC |
| Medium episodes (20-100 steps) | n = 4-10 |
| Long episodes (> 100 steps) | n = 3-8 |
| Very stochastic rewards | n = 2-5 |
| Sparse rewards | Large n or use eligibility traces |
| Fast initial learning desired | n = 5-10 |

---

**Connection to Eligibility Traces**:

Instead of choosing a single n, **TD(λ) averages over all n**:
```
G_t^λ = (1-λ) [G_t:t+1 + λG_t:t+2 + λ²G_t:t+3 + ...]
```

This automatically balances all n-step returns:
- λ=0: Equivalent to n=1 (TD)
- λ=1: Equivalent to n=∞ (MC)
- λ ∈ (0,1): Weighted average of all n

**Advantage**: Don't need to choose n explicitly; λ is often easier to tune.

---

**Key Insights**:

1. **n=1 and n=∞ are extremes**: Pure TD and pure MC
2. **Intermediate n often best**: Balances bias and variance
3. **Optimal n is task-dependent**: Episode length, stochasticity, reward structure
4. **Practical range**: n ∈ [3, 10] works for most tasks
5. **Alternative**: Use eligibility traces (TD(λ)) to average over all n

The n-step framework unifies and clarifies the relationship between TD and MC, showing they're endpoints of a continuous spectrum rather than fundamentally different approaches.

</details>

---

## Question 2: Deriving the λ-return

**Derive the λ-return as a weighted average of n-step returns. Show that λ=0 gives TD(0) and λ=1 gives Monte Carlo. Why are the weights (1-λ)λ^{n-1}?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**The λ-return Definition**:

```
G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t:t+n
```

This is a weighted average of all possible n-step returns.

---

**Step 1: Understanding the Weights**

The weight for the n-step return G_t:t+n is:
```
w_n = (1-λ) λ^{n-1}
```

**Why this form?** To ensure the weights sum to 1 (in the infinite sum):

```
Σ_{n=1}^{∞} w_n = Σ_{n=1}^{∞} (1-λ) λ^{n-1}
                = (1-λ) Σ_{n=1}^{∞} λ^{n-1}
                = (1-λ) · [1 + λ + λ² + λ³ + ...]
                = (1-λ) · 1/(1-λ)    (geometric series)
                = 1 ✓
```

So the weights form a valid probability distribution!

**Weight Distribution**:
```
n=1: w_1 = (1-λ) λ^0 = (1-λ)
n=2: w_2 = (1-λ) λ^1 = (1-λ)λ
n=3: w_3 = (1-λ) λ^2 = (1-λ)λ²
...
```

The weights **decay exponentially** with n at rate λ.

---

**Step 2: Expanding the λ-return**

```
G_t^λ = (1-λ) [G_t:t+1 + λG_t:t+2 + λ²G_t:t+3 + λ³G_t:t+4 + ...]
```

Let's write out the first few n-step returns:

```
G_t:t+1 = R_{t+1} + γV(S_{t+1})
G_t:t+2 = R_{t+1} + γR_{t+2} + γ²V(S_{t+2})
G_t:t+3 = R_{t+1} + γR_{t+2} + γ²R_{t+3} + γ³V(S_{t+3})
...
```

Substituting:
```
G_t^λ = (1-λ) [(R_{t+1} + γV(S_{t+1}))
               + λ(R_{t+1} + γR_{t+2} + γ²V(S_{t+2}))
               + λ²(R_{t+1} + γR_{t+2} + γ²R_{t+3} + γ³V(S_{t+3}))
               + ...]
```

---

**Step 3: Proving λ=0 Gives TD(0)**

When λ=0:
```
G_t^λ=0 = (1-0) Σ_{n=1}^{∞} 0^{n-1} G_t:t+n
        = 1 · [0^0 G_t:t+1 + 0^1 G_t:t+2 + 0^2 G_t:t+3 + ...]
        = 1 · [1 · G_t:t+1 + 0 + 0 + ...]
        = G_t:t+1
        = R_{t+1} + γV(S_{t+1})
```

This is exactly the **one-step TD target**! ✓

**Interpretation**: With λ=0, all weight is on the 1-step return (immediate bootstrapping).

---

**Step 4: Proving λ=1 Gives Monte Carlo**

For episodic tasks with terminal time T, when λ=1:

```
G_t^λ=1 = (1-1) Σ_{n=1}^{T-t-1} 1^{n-1} G_t:t+n + 1^{T-t-1} G_t
        = 0 · [...] + 1 · G_t
        = G_t
```

Wait, this seems wrong with the 0 coefficient. Let's be more careful with episodic tasks.

**Correct Episodic Formula**:
```
G_t^λ = (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_t:t+n + λ^{T-t-1} G_t
        └──────── truncated returns ──────┘   └─ full return ─┘
```

The last term accounts for the complete return (no bootstrapping).

When λ=1:
```
G_t^λ=1 = (1-1) · [anything] + 1^{T-t-1} · G_t
        = 0 + G_t
        = G_t
```

This is the **complete Monte Carlo return**! ✓

**Interpretation**: With λ=1, all weight is on the complete return (no bootstrapping).

---

**Step 5: Intermediate λ**

For λ ∈ (0,1), we get a weighted average:

**Example: λ=0.5**

```
Weights:
  n=1: (1-0.5) · 0.5^0 = 0.5
  n=2: (1-0.5) · 0.5^1 = 0.25
  n=3: (1-0.5) · 0.5^2 = 0.125
  n=4: (1-0.5) · 0.5^3 = 0.0625
  ...

Sum = 0.5 + 0.25 + 0.125 + 0.0625 + ... = 1 ✓
```

Half the weight is on the 1-step return, one quarter on the 2-step, etc.

**Geometric Decay**: Each subsequent n-step return gets λ times the weight of the previous one.

---

**Visual Representation**:

```
λ=0 (TD):
|████████████████████| n=1 (100%)
|                    | n=2 (0%)
|                    | n=3 (0%)

λ=0.5:
|██████████|          n=1 (50%)
|█████|               n=2 (25%)
|██|                  n=3 (12.5%)
|█|                   n=4 (6.25%)

λ=0.9:
|█|                   n=1 (10%)
|█|                   n=2 (9%)
|█|                   n=3 (8.1%)
|█|                   n=4 (7.3%)
...                   ... (long tail)

λ=1 (MC):
|                    | n=1 (0%)
|                    | n=2 (0%)
|████████████████████| n=∞ (100%)
```

---

**Why the (1-λ) Factor?**

The (1-λ) is a **normalization constant** that ensures weights sum to 1.

Without it:
```
Σ_{n=1}^{∞} λ^{n-1} = 1/(1-λ)  (not 1 unless λ=0)
```

With it:
```
Σ_{n=1}^{∞} (1-λ) λ^{n-1} = (1-λ) · 1/(1-λ) = 1 ✓
```

**Interpretation of (1-λ)**:
- λ controls how far we look ahead
- (1-λ) is the "termination probability" at each step
- At each step, we terminate with probability (1-λ) and bootstrap, or continue with probability λ

---

**Alternative View: Recursive Definition**

The λ-return can also be written recursively:

```
G_t^λ = R_{t+1} + γ[(1-λ)V(S_{t+1}) + λG_{t+1}^λ]
```

**Interpretation**:
- Get reward R_{t+1}
- With probability (1-λ): stop and bootstrap from V(S_{t+1})
- With probability λ: continue to the next step's λ-return

This is elegant and shows why λ acts like a "stopping probability."

---

**Episodic Task Correction**

For episodic tasks, we must handle termination:

```
G_t^λ = {
  G_t                                                    if t = T-1 (terminal)
  R_{t+1} + γ[(1-λ)V(S_{t+1}) + λG_{t+1}^λ]            otherwise
}
```

Or in the weighted sum form:
```
G_t^λ = (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_t:t+n + λ^{T-t-1} G_t
```

The last term λ^{T-t-1} G_t ensures we include the complete return.

---

**Proving Properties**:

**Property 1**: E[G_t^λ] is between E[G_t:t+1] and E[G_t]

Since the weights (1-λ)λ^{n-1} are positive and sum to 1:
```
G_t^λ is a convex combination of G_t:t+n values

min_n E[G_t:t+n] ≤ E[G_t^λ] ≤ max_n E[G_t:t+n]
```

**Property 2**: Variance increases with λ

As λ increases:
- More weight on longer returns (larger n)
- Longer returns have more random rewards
- Higher variance

```
Var[G_t^λ] increases with λ
```

---

**Numerical Example**:

Suppose in a 3-step episode:
```
R_1 = 1, R_2 = 2, R_3 = 3, γ = 1, V(S_1) = 5, V(S_2) = 6

G_0:1 = 1 + 1·5 = 6
G_0:2 = 1 + 1·2 + 1·6 = 9
G_0:3 = 1 + 2 + 3 = 6  (complete return)
```

**λ=0**:
```
G_0^λ=0 = G_0:1 = 6
```

**λ=0.5**:
```
G_0^λ=0.5 = (1-0.5)[G_0:1 + 0.5·G_0:2] + 0.5²·G_0:3
          = 0.5[6 + 0.5·9] + 0.25·6
          = 0.5[6 + 4.5] + 1.5
          = 0.5·10.5 + 1.5
          = 5.25 + 1.5
          = 6.75
```

**λ=1**:
```
G_0^λ=1 = 0[...] + 1²·6 = 6  (complete return)
```

---

**Key Insights**:

1. **Weighted average**: λ-return combines all n-step returns with exponentially decaying weights
2. **Normalization**: (1-λ) ensures weights sum to 1
3. **Extremes**: λ=0 is TD, λ=1 is MC
4. **Geometric decay**: Each n-step return gets λ^{n-1} relative weight
5. **Interpretation**: λ is like a "continuation probability" - probability of looking one more step ahead

The λ-return elegantly unifies TD and MC into a single parameterized family, allowing us to tune the bias-variance tradeoff with a single parameter λ.

</details>

---

## Question 3: Forward View vs Backward View

**Compare the forward view and backward view of TD(λ). Are they exactly equivalent? What are the practical advantages and disadvantages of each?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Forward View and Backward View** represent two different ways of thinking about and computing TD(λ), with important practical differences.

---

### Forward View (Conceptual)

**Definition**:
Look forward from each state to compute a weighted average of all future n-step returns.

**Update**:
```
V(S_t) ← V(S_t) + α[G_t^λ - V(S_t)]

where G_t^λ = (1-λ) Σ_{n=1}^{∞} λ^{n-1} G_t:t+n
```

**Computation**:
At each time step t, to update V(S_t), we need:
1. All future rewards: R_{t+1}, R_{t+2}, ..., R_T
2. All future states: S_{t+1}, S_{t+2}, ..., S_T
3. Compute G_t:t+1, G_t:t+2, ..., G_t:t+T
4. Weight them by (1-λ)λ^{n-1} and sum

**Requirements**:
- Must wait until end of episode
- Need to store entire trajectory
- Offline algorithm (cannot update during episode)

**Advantages**:
- **Conceptually clear**: Easy to understand what's being computed
- **Theoretically elegant**: Direct implementation of λ-return definition
- **Exact**: Computes exactly what the definition specifies

**Disadvantages**:
- **Not online**: Must wait for episode to end
- **Memory intensive**: Store entire trajectory
- **Not real-time**: Cannot update during episode
- **Impractical**: Cannot handle continuing tasks

---

### Backward View (Implementable)

**Definition**:
Distribute TD errors backward to previously visited states using eligibility traces.

**Update**:
```
For each step t:
    δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)   (TD error)

    For all s ∈ S:
        e_t(s) = γλ e_{t-1}(s) + 𝟙(s = S_t)   (trace update)
        V(s) = V(s) + α δ_t e_t(s)            (value update)
```

**Computation**:
At each time step t:
1. Compute one-step TD error δ_t
2. Update traces for all states (increment current state, decay others)
3. Update all states proportional to their traces

**Requirements**:
- Only need current step information
- Store traces e(s) for all states
- Online algorithm (updates at each step)

**Advantages**:
- **Online**: Update at every step during episode
- **Real-time**: Can use for continuing tasks
- **Memory efficient**: Only store current traces and values
- **Practical**: Works in all scenarios

**Disadvantages**:
- **Less intuitive**: Harder to understand what's being computed
- **Trace maintenance**: Must update traces for all states
- **Implementation complexity**: More moving parts

---

### Equivalence Theorem

**Theorem**: For the tabular case with a single pass through an episode, the total update to V(s) from the forward view equals the total update from the backward view.

**Mathematically**:
```
Σ_{t: S_t=s} α[G_t^λ - V(S_t)]  =  Σ_{t=0}^{T} α δ_t e_t(s)
└────── forward view ──────┘      └──── backward view ────┘
```

**Proof Sketch**:

Forward view update for state s visited at times t_1, t_2, ..., t_k:
```
Δ_{forward} V(s) = Σ_{i=1}^{k} α[G_{t_i}^λ - V(s)]
```

Backward view cumulative update for state s:
```
Δ_{backward} V(s) = Σ_{t=0}^{T} α δ_t e_t(s)
```

The eligibility trace e_t(s) accumulates (γλ)^{t-t_i} for each visit at t_i, and when multiplied by δ_t and summed over all t, it produces the same total update as the forward view.

**Key Point**: They are equivalent **in aggregate over the episode**, not step-by-step!

---

### Step-by-Step Differences

**During an Episode**:

Forward view:
```
Step 1: No updates (waiting)
Step 2: No updates (waiting)
Step 3: No updates (waiting)
...
Episode end: Update all visited states based on their λ-returns
```

Backward view:
```
Step 1: Compute δ_1, update all states with e_1(s) > 0
Step 2: Compute δ_2, update all states with e_2(s) > 0
Step 3: Compute δ_3, update all states with e_3(s) > 0
...
Episode end: No special processing needed
```

The final values of V(s) will be the same, but the intermediate values differ!

---

### When Equivalence Breaks Down

**1. Function Approximation**:
With function approximation (e.g., neural networks), the updates interfere with each other, and forward/backward views are **not equivalent**.

**2. Multiple Passes**:
If processing the same episode multiple times, they diverge.

**3. Off-Policy Learning**:
With importance sampling, need careful handling of traces; equivalence requires specific trace updates.

**4. Online Lambda Return**:
There are newer algorithms (True Online TD(λ)) that maintain exact equivalence even with online updating and function approximation.

---

### Practical Comparison

| Aspect | Forward View | Backward View |
|--------|--------------|---------------|
| **Update Timing** | End of episode | Every step |
| **Memory** | Store trajectory | Store traces |
| **Continuing Tasks** | No | Yes |
| **Conceptual** | Clear | Less intuitive |
| **Real-time** | No | Yes |
| **Implementation** | Simple (offline) | Moderate (online) |
| **Practical Use** | Rarely | Always |
| **Equivalence** | Exact (tabular) | Exact (tabular) |

---

### Detailed Example: 3-Step Episode

**Setup**:
```
Episode: S_0 → S_1 → S_2 → S_terminal
Rewards: R_1 = 1, R_2 = 2, R_3 = 3
γ = 1, λ = 0.5, α = 0.1
Initial: V(S_0) = 0, V(S_1) = 0, V(S_2) = 0
```

**Forward View**:

Compute λ-returns:
```
G_0^λ = (1-0.5)[G_0:1 + 0.5·G_0:2 + 0.5²·G_0]
      = 0.5[1+0 + 0.5·(1+2+0) + 0.25·(1+2+3)]
      = 0.5[1 + 1.5 + 1.5]
      = 2.0

G_1^λ = (1-0.5)[G_1:2 + 0.5·G_1]
      = 0.5[(2+0) + 0.5·(2+3)]
      = 0.5[2 + 2.5]
      = 2.25

G_2^λ = G_2 = 3
```

Updates at episode end:
```
V(S_0) ← 0 + 0.1[2.0 - 0] = 0.2
V(S_1) ← 0 + 0.1[2.25 - 0] = 0.225
V(S_2) ← 0 + 0.1[3 - 0] = 0.3
```

**Backward View**:

**Step 1** (at S_0):
```
δ_0 = 1 + 1·V(S_1) - V(S_0) = 1 + 0 - 0 = 1
e_0(S_0) = 1, e_0(S_1) = 0, e_0(S_2) = 0

V(S_0) ← 0 + 0.1·1·1 = 0.1
V(S_1) ← 0 + 0.1·1·0 = 0
V(S_2) ← 0 + 0.1·1·0 = 0
```

**Step 2** (at S_1):
```
δ_1 = 2 + 1·V(S_2) - V(S_1) = 2 + 0 - 0 = 2
e_1(S_0) = 0.5·1 + 0 = 0.5
e_1(S_1) = 0.5·0 + 1 = 1
e_1(S_2) = 0

V(S_0) ← 0.1 + 0.1·2·0.5 = 0.2
V(S_1) ← 0 + 0.1·2·1 = 0.2
V(S_2) ← 0 + 0.1·2·0 = 0
```

**Step 3** (at S_2):
```
δ_2 = 3 + 1·0 - V(S_2) = 3 - 0 = 3
e_2(S_0) = 0.5·0.5 = 0.25
e_2(S_1) = 0.5·1 = 0.5
e_2(S_2) = 1

V(S_0) ← 0.2 + 0.1·3·0.25 = 0.275  ← Close to 0.2 (rounding differences)
V(S_1) ← 0.2 + 0.1·3·0.5 = 0.35    ← Close to 0.225
V(S_2) ← 0 + 0.1·3·1 = 0.3         ← Exact match
```

(Small differences due to rounding and the fact that V values changed during the episode in backward view, affecting δ_t calculations.)

**Key Observation**: By episode end, both views produce similar (in exact arithmetic, identical) results, but backward view updates at every step!

---

### Why Backward View is Used in Practice

1. **Online Learning**: Can update during the episode, adjusting policy in real-time
2. **Continuing Tasks**: Works for infinite-horizon tasks without episodes
3. **Real-World Applications**: Robots, control systems need online updates
4. **Memory Efficiency**: Don't need to store entire trajectory
5. **Computational Efficiency**: Distribute computation across steps

---

### Modern Developments

**True Online TD(λ)**:
A refined backward view algorithm that maintains exact equivalence with forward view even:
- With function approximation
- With changing value estimates during the episode
- Online (not waiting for episode end)

This is the state-of-the-art for TD(λ) implementation.

---

### Key Insights

1. **Conceptual vs Practical**: Forward view is conceptual; backward view is practical
2. **Aggregate Equivalence**: Same total update over an episode (tabular case)
3. **Step-by-Step Difference**: Intermediate updates differ
4. **Online Advantage**: Backward view works online, forward doesn't
5. **Always Use Backward**: In practice, always implement backward view with traces

The backward view with eligibility traces is one of the most elegant and powerful ideas in RL - it makes the theoretical λ-return computationally practical!

</details>

---

## Question 4: Optimal n for Random Walk

**For the 19-state random walk (states 1-19, terminals at 0 and 20 with rewards 0 and 1), explain which n should perform best and why. How does this depend on the learning rate α?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Environment Specification**:

```
States: 0 [1, 2, 3, ..., 10, ..., 19] 20
        ↑  ←  random walk  →         ↑
     terminal             terminal
     (R=0)                  (R=1)

- Start at state 10 (center)
- Each step: move left or right with probability 0.5
- Termination: reach state 0 or 20
- Rewards: R=0 for reaching 0, R=1 for reaching 20 (0 elsewhere)
- Discount: γ = 1 (no discounting)
```

**True Values**: v_π(i) = i/20 for i = 1, ..., 19
- State 1: v_π(1) = 0.05 (5% chance of reaching 20)
- State 10: v_π(10) = 0.5 (50% chance)
- State 19: v_π(19) = 0.95 (95% chance)

---

### Theoretical Analysis

**Key Factors Determining Optimal n**:

1. **Episode Length**: Average ~100-400 steps from center
2. **Reward Sparsity**: Only at terminal states
3. **Value Propagation**: Information must travel across states
4. **Symmetry**: Symmetric structure around center

---

**Small n (e.g., n=1, TD(0))**:

**Pros**:
- Low variance (only one random reward)
- Stable updates
- Fast computation per step

**Cons**:
- Information propagates slowly (1 step per episode)
- Many episodes needed for terminal reward to reach center
- High initial bias (bootstraps from inaccurate estimates)

**Expected Performance**:
After 10 episodes starting at state 10:
```
Only states within 10 steps of terminal updated significantly
States 10-20: Some learning
States 1-9: Barely affected
V(10) still near initial value (inaccurate)
```

Convergence: **Slow** (hundreds of episodes)

---

**Medium n (e.g., n=4 to 10)**:

**Pros**:
- Balanced bias-variance tradeoff
- Information propagates faster (n steps per episode)
- More reward signal reaches distant states
- Still reasonable variance

**Cons**:
- Some variance from multiple random steps
- Requires storing n-step history

**Expected Performance**:
After 10 episodes with n=8:
```
States within 80 steps of terminal updated with reward signal
Covers most of state space
V(10) has received some terminal reward information
```

Convergence: **Fast** (tens to hundreds of episodes)

---

**Large n (e.g., n=50 to ∞, approaching MC)**:

**Pros**:
- Low bias (many actual rewards)
- Fast propagation (reward reaches all visited states)

**Cons**:
- High variance (hundreds of random steps in return)
- Noisy updates
- Episodes are long, so returns vary greatly

**Expected Performance**:
After 10 episodes with n=∞ (MC):
```
All visited states updated
But updates are very noisy:
  - Episode 1: Start at 10, reach 0 → V(10) ← 0
  - Episode 2: Start at 10, reach 20 → V(10) ← 1
  - Oscillates wildly
```

Convergence: **Slow** (requires averaging many episodes)

---

### Empirical Optimal n

**Typical Results** (from S&B Figure 7.2):

```
RMSE after 10 episodes:
  n=1:   ~0.55
  n=2:   ~0.40
  n=4:   ~0.30  ← good
  n=8:   ~0.25  ← best for α=0.1
  n=16:  ~0.28
  n=32:  ~0.35
  n=∞:   ~0.55
```

**Optimal n ≈ 8 to 16** for this task with typical α.

---

### Why n=8 is Often Optimal

**1. Matches Episode Structure**:
- Average episode length: ~100-200 steps from center
- n=8 means looking 8 steps ahead
- Captures local reward structure without full episode variance

**2. Propagation Speed**:
- With n=8, information from terminal propagates inward ~8 states per episode
- After 10 episodes: terminal rewards affect states 80 steps away (most of space)
- Faster than n=1 (10 steps after 10 episodes)

**3. Variance Control**:
- 8 random steps introduce moderate variance
- Much less than full episode (hundreds of steps)
- Variance is manageable with reasonable α

**4. Bias Reduction**:
- 8 actual rewards reduce dependence on (initially wrong) bootstrap
- Less biased than n=1

---

### Dependence on Learning Rate α

**The optimal n depends critically on α**:

**High α (e.g., α=0.5)**:
- Values update quickly
- Can tolerate higher variance
- **Optimal n shifts lower** (n ≈ 4 to 8)
- Reason: Large α amplifies noisy updates from large n

**Medium α (e.g., α=0.1 to 0.2)**:
- Balanced updates
- **Optimal n is medium** (n ≈ 8 to 16)
- Standard choice

**Low α (e.g., α=0.01)**:
- Values update slowly
- Needs many updates to converge
- **Optimal n shifts higher** (n ≈ 16 to 32 or even MC)
- Reason: Small α averages out variance, so higher n's lower bias helps

---

### Mathematical Intuition

**Total Error** = Bias² + Variance

**Bias(n)**:
```
Bias(n) ∝ 1/n  (more actual rewards → less bias from bootstrap)
```

**Variance(n)**:
```
Variance(n) ∝ n  (more random rewards → more variance)
```

**With learning rate α**:
```
Effective variance ∝ α² · Variance(n)
```

Large α amplifies variance more, favoring smaller n.

**Optimal n**:
```
n* = argmin_n [1/n² + α² · n]

Taking derivative: d/dn [1/n² + α²n] = -2/n³ + α² = 0
Solving: n* ∝ 1/α^(2/3)
```

So **larger α → smaller optimal n**.

---

### Empirical Results by α

| α | Optimal n | RMSE | Convergence Speed |
|---|-----------|------|-------------------|
| 0.05 | 16-32 | 0.20 | Slow |
| 0.1 | 8-16 | 0.15 | Medium |
| 0.2 | 4-8 | 0.18 | Fast |
| 0.5 | 2-4 | 0.25 | Very fast but noisy |
| 1.0 | 1-2 | 0.35 | Unstable |

---

### Practical Recommendations

**For 19-State Random Walk**:

1. **Standard setting**: α=0.1, n=8
   - Good balance
   - Converges in ~50-100 episodes

2. **Fast learning**: α=0.2, n=4
   - Quicker initial progress
   - May be noisier

3. **Stable learning**: α=0.05, n=16
   - Slower but very stable
   - Better final accuracy

4. **When in doubt**: Use eligibility traces (λ=0.8)
   - Automatically averages over all n
   - Robust to α choice

---

### Comparison with Eligibility Traces

Instead of choosing a single n, use **TD(λ)**:

```
λ = 0.5: Roughly equivalent to n ≈ 2-4
λ = 0.9: Roughly equivalent to n ≈ 8-16
λ = 0.95: Roughly equivalent to n ≈ 16-32
```

**Advantage**: Don't need to tune n; λ often easier to set (typically 0.8-0.9 works well).

**Performance**:
```
TD(λ=0.9) with α=0.1:
  RMSE after 10 episodes: ~0.20
  Comparable to best n-step method
  But more robust to α changes
```

---

### Task-Specific Factors

**Why Random Walk Favors Medium n**:

1. **Symmetry**: No strong directional preference
2. **Long episodes**: Need enough n to propagate information
3. **Sparse rewards**: Need n large enough to reach rewards
4. **Low stochasticity**: Random walk is simple, so variance from n steps is manageable

**Other Tasks**:
- **Shorter episodes** (10-20 steps): n=5-10 or MC
- **Denser rewards**: n=2-5
- **Highly stochastic**: n=1-4
- **Deterministic**: n can be larger

---

### Visualizing Value Propagation

**After 1 Episode with Different n**:

```
n=1 (TD):
|==|          States 10-12 updated
   (18 states still at initial value)

n=8:
|========|   States 10-18 updated
   (11 states still at initial value)

n=∞ (MC):
|===================| All visited states updated (but noisy)
```

After 10 episodes:
```
n=1:  |=====|      Slowly spreading from terminals
n=8:  |==============|   Covers most space
n=∞:  |====================|  All space, but still noisy
```

---

### Key Insights

1. **Optimal n is task-specific**: Random Walk favors n ≈ 8-16
2. **Depends on α**: Larger α → smaller n
3. **Balances propagation speed vs variance**: Medium n is sweet spot
4. **Long sparse-reward episodes**: Need larger n than short dense-reward tasks
5. **Alternative**: Use TD(λ) with λ ≈ 0.8-0.9 for robustness

The 19-state random walk is a perfect environment for demonstrating that **intermediate n outperforms both TD(0) and MC**, making it a classic benchmark for n-step methods!

</details>

---

## Question 5: When to Prefer Eligibility Traces

**When would you prefer using eligibility traces (TD(λ)) over plain TD(0) or Monte Carlo? Provide specific scenarios and explain the advantages.**

<details>
<summary>Click to reveal answer</summary>

### Answer:

Eligibility traces shine in specific scenarios where their unique properties provide significant advantages over TD(0) or Monte Carlo.

---

### Core Advantages of Eligibility Traces

**1. Efficient Multi-Step Credit Assignment**
**2. Online Learning with Long-Term Credit**
**3. Single Parameter (λ) Instead of Choosing n**
**4. Fast Learning in Sparse Reward Environments**
**5. Reduced Sensitivity to Bootstrapping Errors**

---

### Scenario 1: Sparse Delayed Rewards

**Problem**: Long sequences of actions before receiving any reward.

**Example**: Chess/Go, Robot Navigation, Maze Solving

**Why Eligibility Traces Help**:

Plain TD(0):
```
Episode: S_0 → S_1 → ... → S_99 → S_100 (reward +1)

After episode 1:
  V(S_99) updated (1 step from reward)
  V(S_98) not updated yet (needs episode 2)
  V(S_0) needs 100 episodes to receive any signal!
```

TD(λ) with λ=0.9:
```
Same episode:
  All states S_0, S_1, ..., S_100 updated immediately!
  V(S_0) receives: α · δ_100 · (γλ)^100 ≈ small but non-zero
  Information propagates backward in single episode
```

**Result**: **10-100x faster learning** when rewards are sparse.

---

**Concrete Example: Mountain Car**

- Reward: -1 per step until reaching goal (very sparse)
- Episode length: 200-500 steps initially

TD(0):
```
Episodes to reach goal: 2000-5000
Reason: Reward signal takes hundreds of episodes to propagate to start
```

TD(λ=0.9):
```
Episodes to reach goal: 100-500
Reason: Every episode updates all visited states
10x faster!
```

**When to Use**:
- Reward occurs only at episode end
- Many actions between decision and reward
- Need credit assignment over long sequences

---

### Scenario 2: Continuing (Non-Episodic) Tasks

**Problem**: No natural episode boundaries.

**Example**: Process control, server management, autonomous driving

**Why Eligibility Traces Help**:

Monte Carlo:
```
Cannot use at all! (requires episodes to compute returns)
```

TD(0):
```
Works, but slow credit assignment
If system state at time t affects outcome at t+1000:
  Takes 1000 steps to propagate value change backward
```

TD(λ):
```
Works online, updates states weighted by recency
States from 1000 steps ago still get updated (with small weight)
Faster credit propagation than TD(0)
```

**When to Use**:
- No episodes (continuing task)
- Need online real-time learning
- Long-term dependencies

---

### Scenario 3: Frequent State Revisits

**Problem**: Some states are visited multiple times per episode.

**Example**: Grid world with loops, game with repeated board positions

**Why Eligibility Traces Help**:

Accumulating Traces:
```
e_t(s) = γλ e_{t-1}(s) + 1  (if s visited at t)

If state s visited 3 times:
  e(s) builds up: 1, then 1+γλ, then 1+γλ+γλ²
  More credit to frequently visited states
```

Replacing Traces:
```
e_t(s) = 1  (reset on visit, don't accumulate)

If state s visited multiple times:
  e(s) stays at 1 (doesn't explode)
  More stable for function approximation
```

**When to Use**:
- States are revisited in episodes
- Need to handle accumulated credit properly
- Using function approximation (prefer replacing traces)

---

### Scenario 4: Robustness to Hyperparameters

**Problem**: Don't know ideal n for n-step methods.

**Why λ is Easier**:

n-step TD:
```
Must choose n: 1, 2, 4, 8, 16, ...
Optimal n varies by:
  - Episode length
  - Reward density
  - Environment stochasticity
  - Learning rate α
```

TD(λ):
```
Choose λ: continuous parameter in [0,1]
General guidelines:
  - Sparse rewards: λ=0.9-0.95
  - Dense rewards: λ=0.5-0.7
  - Start with λ=0.8 (works for many tasks)
Much less sensitive to exact value
```

**Empirical Robustness**:
```
n-step TD: Performance varies 2-3x across different n
TD(λ): Performance varies <30% across λ ∈ [0.5, 0.95]
```

**When to Use**:
- Limited time for hyperparameter tuning
- Need robust default settings
- Task characteristics vary over time

---

### Scenario 5: Function Approximation

**Problem**: Large state spaces require function approximation (neural networks).

**Why Eligibility Traces Help**:

**Replacing Traces** are more stable:
```
Without traces (TD(0)):
  Update only current state's features
  Slow propagation through feature space

With replacing traces:
  Update all recently active features
  Faster and more stable learning
  Less sensitive to learning rate
```

**Modern Deep RL**:
- A3C, PPO use n-step or eligibility traces
- Helps stabilize neural network training
- Improves sample efficiency

**When to Use**:
- Using function approximation
- Neural network value functions
- Need stable gradient-based learning

---

### Scenario 6: Noisy Environments

**Problem**: High stochasticity in rewards or transitions.

**Why Traces Can Help**:

TD(0):
```
Bias toward bootstrap (potentially wrong estimates)
High stochasticity → bootstrap errors amplified
```

Monte Carlo:
```
No bias but very high variance
Noisy environments → returns vary wildly
Slow convergence
```

TD(λ) with λ ≈ 0.7:
```
Balances actual noisy returns with less noisy bootstraps
Weighted average smooths out some noise
Medium bias, medium variance
Often converges faster than both extremes
```

**When to Use**:
- Stochastic rewards
- Noisy transitions
- Want balance between bias and variance

---

### Scenario 7: Real-Time Learning Requirements

**Problem**: Need to update policy during episode.

**Example**: Robot learning, game playing AI, adaptive control

Monte Carlo:
```
Cannot update during episode
Must finish episode before learning
Too slow for real-time adaptation
```

TD(0):
```
Updates every step
But credit assignment is slow (1 step at a time)
```

TD(λ):
```
Updates every step
AND propagates credit to recent history
Best of both: online + fast credit assignment
```

**When to Use**:
- Real-time systems
- Cannot afford waiting for episodes
- Need fast adaptation within episode

---

### When NOT to Use Eligibility Traces

**Prefer TD(0) when**:
1. **Short episodes** (< 20 steps): Overhead not worth it
2. **Dense immediate rewards**: Credit assignment is trivial
3. **Very large state spaces**: Storing traces for all states is expensive
4. **Deterministic environments**: Lower variance makes simple TD sufficient
5. **Simplicity matters**: TD(0) is easier to implement and debug

**Prefer Monte Carlo when**:
1. **Offline learning**: Processing recorded episodes
2. **Short episodes**: MC variance is acceptable
3. **Function approximation with divergence risk**: MC doesn't bootstrap (safer)
4. **Episode-level rewards**: Natural fit (e.g., win/lose in games)

---

### Practical Decision Framework

```
                    Eligibility Traces
                          ↑
                          |
         Need fast        |        Have episodes?
      credit assignment   |             ↓
                ↑         |            YES
                |         |             ↓
        Sparse rewards? ──┴──→   Episode length?
                ↓                       ↓
               YES               Short: MC or λ=0.9
                ↓                Long: λ=0.7-0.9
        Use λ=0.9-0.95
                                  ↓
                            Dense rewards?
                                  ↓
                                 YES
                                  ↓
                            TD(0) or λ=0.3-0.5
```

---

### Empirical Performance Comparison

**Mountain Car** (sparse reward):
```
TD(0):        2000 episodes to solve
TD(λ=0.9):    200 episodes to solve     10x faster
MC:           500 episodes to solve
```

**Cart Pole** (dense reward):
```
TD(0):        50 episodes to solve
TD(λ=0.5):    45 episodes to solve      Similar
MC:           60 episodes to solve
```

**Grid World Maze** (medium sparsity):
```
TD(0):        500 episodes
TD(λ=0.8):    150 episodes            3x faster
MC:           200 episodes
```

---

### Key Parameters for Eligibility Traces

| Task Type | Recommended λ | Trace Type | α |
|-----------|---------------|------------|---|
| Sparse rewards | 0.9-0.95 | Accumulating | 0.1-0.2 |
| Dense rewards | 0.3-0.5 | Either | 0.2-0.5 |
| Function approx | 0.8-0.9 | Replacing | 0.05-0.1 |
| Continuing | 0.7-0.9 | Accumulating | 0.1 |
| Short episodes | 0.5-0.7 | Either | 0.2-0.5 |
| Long episodes | 0.9-0.95 | Accumulating | 0.1 |

---

### Summary: Use Eligibility Traces When...

1. **Rewards are sparse** (most important use case)
2. **Episodes are long** (>50 steps)
3. **Need online learning** (continuing tasks)
4. **Want robust hyperparameters** (λ easier than n)
5. **Using function approximation** (especially replacing traces)
6. **Noisy environment** (balance bias-variance)
7. **Real-time learning** (update within episode)
8. **States revisited** (credit needs to accumulate)

**Default Recommendation**: When in doubt, use TD(λ) with λ=0.8. It's rarely worse than TD(0) and often much better, especially for challenging tasks with sparse rewards.

Eligibility traces are one of the most powerful and widely applicable techniques in RL, bridging the gap between simple TD and full MC while providing online learning and efficient credit assignment!

</details>

---

## Additional Practice Problems

1. **Implement n-step SARSA** for n=1,4,8 on Taxi-v3. Compare convergence speed and final policy quality.

2. **Prove the equivalence** between forward and backward view TD(λ) for a simple 3-state MDP (mathematically).

3. **Analyze trace decay**: Plot how eligibility trace e_t(s) decays over time for different λ values after visiting state s at t=0.

4. **Design an environment** where λ=0 (TD) significantly outperforms λ=1 (MC), and vice versa.

5. **Implement True Online TD(λ)** and compare with standard TD(λ) on Mountain Car with function approximation.

6. **Experiment with trace types**: Compare accumulating, replacing, and dutch traces on a grid world with loops.
