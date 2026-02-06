# Week 13 Quiz: TRPO & PPO

## Question 1: Conceptual Understanding

**Why can large policy gradient updates be catastrophic? Give a concrete example with equations showing how a good policy can suddenly become terrible.**

<details>
<summary>Click to reveal answer</summary>

### Answer

Large policy updates are dangerous because the policy gradient is a **local approximation** that breaks down far from the current policy.

## The Problem: Policy Gradient is a Local Linear Approximation

**Policy gradient update**:
```
θ_new = θ_old + α∇_θ J(θ_old)
```

This assumes:
```
J(θ_new) ≈ J(θ_old) + ∇_θ J(θ_old)^T (θ_new - θ_old)
          = J(θ_old) + α||∇_θ J(θ_old)||²  (moving in gradient direction)
```

**Problem**: This is only valid for small α. For large α, higher-order terms matter:
```
J(θ_new) = J(θ_old) + ∇J^T Δθ + (1/2)Δθ^T H Δθ + O(||Δθ||³)
                                   ^^^^^^^^^^^^^^^^
                                   Hessian term (can be negative!)
```

If Hessian is negative definite, large steps can **decrease** J even when moving in gradient direction!

## Concrete Example: CartPole

**Scenario**:
```
Episode 1000:
- Policy: π_old achieves 195 reward (near-optimal)
- Policy parameters θ_old = [w_1, w_2, ..., w_100]
- Gradient: ∇J = [0.5, 0.3, ..., 0.1]
```

**Small update (α=0.001, safe)**:
```
θ_new = θ_old + 0.001 * ∇J
Δθ = [0.0005, 0.0003, ..., 0.0001]  (tiny change)

Policy change:
π_new(left|s) ≈ π_old(left|s) + 0.001 * (∂π/∂θ) ≈ π_old ± 0.01
                                                       ^^^^^
                                                    ~1% change

Result: J(π_new) ≈ 197 (slight improvement)
```

**Large update (α=0.1, catastrophic)**:
```
θ_new = θ_old + 0.1 * ∇J
Δθ = [0.05, 0.03, ..., 0.01]  (100x larger!)

Policy change:
π_new(left|s) can change dramatically!

Example in one state:
π_old(left|s₁) = 0.6 (prefer left slightly)
π_new(left|s₁) = 0.01 (now strongly prefers right!)
                  ^^^^
                  Flipped completely!

Result: J(π_new) = -50 (catastrophic failure)
```

## Why It Happens: The Softmax Cliff

**Softmax policy**:
```
π(a|s; θ) = exp(h(s,a; θ)) / Σ_b exp(h(s,b; θ))
```

**Small change in logits** → Small change in probabilities:
```
logits: [1.0, 0.0] → π = [0.73, 0.27]
logits: [1.1, 0.0] → π = [0.75, 0.25]  (small change)
```

**Large change in logits** → Huge change in probabilities:
```
logits: [1.0, 0.0] → π = [0.73, 0.27]
logits: [3.0, 0.0] → π = [0.95, 0.05]  (completely different!)
```

This is the **softmax cliff**: Small parameter changes → exponentially large probability changes.

## Mathematical Example: Two-State MDP

**MDP**:
```
States: {s₁, s₂}
Actions: {a₁, a₂}

Dynamics:
s₁, a₁ → s₂ (reward +1)
s₁, a₂ → s₂ (reward 0)
s₂, any → s₂ (reward 0, terminal)
```

**Optimal policy**: Always choose a₁ in s₁ → reward = 1

**Current policy**: π_old(a₁|s₁) = 0.9 (near-optimal)
```
Parameterization: π(a₁|s₁; θ) = σ(θ) where σ(x) = 1/(1+e^(-x))
Currently: θ_old = 2.2 → π(a₁|s₁) = σ(2.2) = 0.9

Expected return: J(θ_old) = 0.9 * 1 + 0.1 * 0 = 0.9
```

**Gradient**:
```
∇_θ J(θ) = E[∇log π * A]

At s₁:
If a₁: A(s₁,a₁) = 1 - V(s₁) = 1 - 0.9 = 0.1
If a₂: A(s₁,a₂) = 0 - V(s₁) = -0.9

∇_θ J = π(a₁|s₁) * 0.1 * ∇log π(a₁|s₁) + π(a₂|s₁) * (-0.9) * ∇log π(a₂|s₁)
      = 0.9 * 0.1 * (0.1) + 0.1 * (-0.9) * (-0.9)
      ≈ 0.09  (positive, good!)
```

**Small update (α=0.1)**:
```
θ_new = 2.2 + 0.1 * 0.09 = 2.209
π(a₁|s₁) = σ(2.209) = 0.901
J(θ_new) = 0.901 (slight improvement ✓)
```

**Large update (α=50, absurd but illustrative)**:
```
θ_new = 2.2 + 50 * 0.09 = 6.7
π(a₁|s₁) = σ(6.7) ≈ 0.9988 (too confident!)

What's wrong? Policy is now extremely confident.
During training, if one bad trajectory happens:
- Say agent happens to get reward 0 despite choosing a₁ (noise)
- Advantage: A(s₁,a₁) = 0 - 0.9988 ≈ -1
- Gradient becomes very negative!
- Next update: θ ← 6.7 + α * (-100) → could become -50
- π(a₁|s₁) = σ(-50) ≈ 0 (catastrophic flip!)

Result: Policy collapses, never recovers.
```

## Real-World Example: Atari Pong

**Reported in literature (pre-TRPO)**:

```
Training DQN-style policy gradient on Pong:

Episode 5000: Score = +18 (near-optimal, beating AI opponent)
Episode 5001: Large gradient update (lr=0.01)
Episode 5002: Score = -21 (losing every rally!)

Why?
- Large update changed action probabilities drastically
- Old strategy: "wait for ball, then move paddle"
- New strategy after bad update: "move paddle randomly"
- Deterministic policies especially vulnerable
- Never recovered (fell into local minimum of "always losing")
```

## Why TRPO/PPO Fix This

**TRPO**: Hard constraint on KL divergence
```
max J(θ)
s.t. KL(π_old || π_new) ≤ δ = 0.01

Forces: E_s [D_KL(π_old(·|s) || π_new(·|s))] ≤ 0.01

This prevents large changes in action probabilities!
```

**PPO**: Soft constraint via clipping
```
L(θ) = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]

where r = π_new(a|s) / π_old(a|s)

If policy changes too much (r > 1.2 or r < 0.8 with ε=0.2):
- Objective is clipped (no gradient)
- Update stops even if gradient says to continue
- Prevents catastrophic changes
```

## Key Insights

1. **Linear approximation breaks down**: Policy gradient assumes linearity, but true J(θ) is highly nonlinear.

2. **Softmax amplifies small changes**: exp() makes small parameter changes → large probability changes.

3. **No safety net**: Vanilla PG has no mechanism to prevent bad updates.

4. **Catastrophic forgetting**: One bad update can destroy a good policy permanently.

5. **Trust regions solve it**: Constrain updates to region where approximation is valid.

## Conclusion

Large policy updates are dangerous because:
- Policy gradient is a local linear approximation
- Softmax/sigmoid creates exponential sensitivity
- One bad update can make π(a|s) flip from 0.9 to 0.1
- No guarantee of recovery (local minima)

TRPO/PPO prevent this by constraining how much the policy can change per update, ensuring monotonic improvement and avoiding catastrophic forgetting.

**Rule of thumb**: If KL(π_old || π_new) > 0.05, you're in danger zone!

</details>

---

## Question 2: Mathematical Understanding

**Explain intuitively how the PPO clipped objective prevents large policy changes. What happens when r(θ) goes outside [1-ε, 1+ε]? Why does this work without explicitly computing KL divergence?**

<details>
<summary>Click to reveal answer</summary>

### Answer

The PPO clipped objective is brilliantly simple: it removes the incentive to make large policy changes by flattening the objective function outside a safe range.

## The Clipped Objective

```
L^CLIP(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

where:
- r_t(θ) = π(a_t|s_t; θ) / π(a_t|s_t; θ_old)  (probability ratio)
- A_t = advantage of action a_t
- ε = clipping threshold (typically 0.2)
- clip(x, a, b) = max(min(x, b), a)
```

## Case Analysis: When A_t > 0 (Good Action)

**Goal**: Increase π(a|s) (make good action more likely)

**Unclipped objective** (standard importance sampling):
```
L_unclipped = r * A  where A > 0

As r increases (π_new(a|s) increases):
- L increases linearly
- Incentive to keep increasing π(a|s) indefinitely!
- Could make π(a|s) → 1, suppressing exploration
```

**Clipped objective**:
```
L^CLIP = min(r * A, (1+ε) * A)

Case 1: r ≤ 1+ε (safe region)
  L^CLIP = r * A
  Gradient: ∇L = A * ∇r  (normal gradient)
  → Update proceeds normally

Case 2: r > 1+ε (policy changed too much)
  L^CLIP = (1+ε) * A  (constant!)
  Gradient: ∇L = 0  (flat!)
  → No incentive to increase π(a|s) further
  → Update stops
```

**Visualization**:
```
L^CLIP as function of r (for A > 0)

     │         ╱─────────  Clipped (flat, no gradient)
     │        ╱
     │       ╱
     │      ╱  Unclipped region
     │     ╱   (has gradient)
     │    ╱
     │   ╱
     │  ╱
     └─────────────────→ r
     0    1   1+ε

Beyond 1+ε: objective is flat → no gradient → no further update
```

## Case Analysis: When A_t < 0 (Bad Action)

**Goal**: Decrease π(a|s) (make bad action less likely)

**Unclipped objective**:
```
L_unclipped = r * A  where A < 0

As r decreases (π_new(a|s) decreases):
- L increases (becomes less negative)
- Incentive to keep decreasing π(a|s) indefinitely!
- Could make π(a|s) → 0, becoming deterministic
```

**Clipped objective**:
```
L^CLIP = min(r * A, (1-ε) * A)

Note: Since A < 0, "min" actually picks the more negative value!
Equivalently: L^CLIP = max(r, 1-ε) * A

Case 1: r ≥ 1-ε (safe region)
  L^CLIP = r * A
  Gradient: ∇L = A * ∇r  (normal gradient)
  → Update proceeds normally

Case 2: r < 1-ε (policy changed too much)
  L^CLIP = (1-ε) * A  (constant!)
  Gradient: ∇L = 0  (flat!)
  → No incentive to decrease π(a|s) further
  → Update stops
```

**Visualization**:
```
L^CLIP as function of r (for A < 0)

     │  ────────╲
     │           ╲  Clipped (flat, no gradient)
     │            ╲
     │             ╲  Unclipped region
     │              ╲ (has gradient)
     │               ╲
     │                ╲
     │                 ╲
     └─────────────────→ r
     0  1-ε  1

Below 1-ε: objective is flat → no gradient → no further update
```

## Combined View: Clipped Region

```
L^CLIP for both A > 0 and A < 0

     │     A > 0 (increase π)
     │      ╱────
     │     ╱
     │    ╱
     │   ╱
  ───┼──╱────────────  (r=1, no change)
     │  ╲
     │   ╲
     │    ╲  A < 0 (decrease π)
     │ ────╲
     └──────────────→ r
       1-ε  1  1+ε

Clipped regions (flat):
- r < 1-ε: can't decrease π(a|s) more
- r > 1+ε: can't increase π(a|s) more

Allowed region: r ∈ [1-ε, 1+ε]
```

## Why This Works Without Explicit KL

**Key insight**: Clipping probability ratio implicitly bounds KL divergence!

**Relationship between r and KL**:

For small changes, there's a direct relationship:
```
r = π_new(a|s) / π_old(a|s)

KL(π_old || π_new) = E_π_old [log π_old(a|s) - log π_new(a|s)]
                    = E_π_old [log π_old(a|s) / π_new(a|s)]
                    = E_π_old [-log r]

For small changes: KL ≈ E[(r - 1)²] / 2
```

**Clipping r bounds KL**:
```
If r ∈ [1-ε, 1+ε] for all (s,a):

KL ≈ E[(r - 1)²] / 2
   ≤ ε² / 2

For ε=0.2: KL ≤ 0.02  (similar to TRPO's δ=0.01!)
```

**Empirical observation**:
```
PPO with ε=0.2:
- Typical KL per update: 0.01-0.03
- Similar to TRPO with δ=0.01

PPO implicitly enforces KL constraint via clipping!
```

## Concrete Numerical Example

**Setup**:
```
State s₁, action a₁
π_old(a₁|s₁) = 0.6
Advantage A(s₁,a₁) = 0.5 (good action)
ε = 0.2
```

**Scenario 1: Small policy change (r = 1.1)**
```
π_new(a₁|s₁) = 0.6 * 1.1 = 0.66

r = 1.1 ∈ [0.8, 1.2] ← within clipping range

L^CLIP = min(1.1 * 0.5, 1.2 * 0.5)
       = min(0.55, 0.6)
       = 0.55  ← unclipped term is smaller

Gradient: ∇L = 0.5 * ∇r ≠ 0  ← has gradient, update proceeds
```

**Scenario 2: Large policy change (r = 1.5)**
```
π_new(a₁|s₁) = 0.6 * 1.5 = 0.9

r = 1.5 ∉ [0.8, 1.2] ← outside clipping range!

L^CLIP = min(1.5 * 0.5, 1.2 * 0.5)
       = min(0.75, 0.6)
       = 0.6  ← clipped term is smaller!

Gradient: ∇L = 0  ← flat, no gradient, update stops
```

**Result**: Policy stops changing once r reaches 1.2, preventing π(a₁|s₁) from going beyond 0.72.

## Why Clipping Works Better Than KL Penalty

**KL penalty approach** (PPO-penalty):
```
L = E[r * A] - β * KL(π_old || π_new)

Problems:
1. Need to compute KL (expensive)
2. Need to tune β (another hyperparameter)
3. β might need adaptive adjustment
```

**Clipping approach** (PPO-clip):
```
L = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]

Advantages:
1. No explicit KL computation (cheap!)
2. Single hyperparameter ε (robust, ε=0.2 works across tasks)
3. More direct: prevents large r, which directly causes problems
4. Simpler to implement
```

## Practical Implications

**What happens during training**:

```python
# First few iterations
r_values = [1.05, 0.98, 1.12, 0.93, ...]  # Small changes
clipping_fraction = 0.05  # 5% of samples clipped
# → Policy learning normally

# Middle of training
r_values = [1.18, 0.85, 1.25, 1.32, ...]  # Some large changes
clipping_fraction = 0.20  # 20% clipped
# → Clipping starts to matter, prevents overfitting

# If clipping_fraction > 0.5:
# → Policy trying to change too much!
# → Sign of instability or need to lower learning rate
```

**Monitoring clipping**:
```python
clipping_fraction = (r > 1+eps).float().mean() + (r < 1-eps).float().mean()

Healthy: 0.05 - 0.30
Too low (<0.05): Maybe increase learning rate or reduce epochs
Too high (>0.50): Decrease learning rate or reduce batch reuse
```

## Why This Is Brilliant

1. **Simple**: Just clip a ratio, no second-order methods
2. **Effective**: Achieves TRPO-level performance
3. **Efficient**: No conjugate gradient or line search
4. **Robust**: ε=0.2 works across most tasks
5. **Interpretable**: Directly limits policy change per update

## Summary

**How clipping prevents large changes**:
- Flattens objective when r goes outside [1-ε, 1+ε]
- Flat objective → zero gradient → no update incentive
- Policy naturally stops changing once it hits the boundary

**Why it works without explicit KL**:
- Clipping r implicitly bounds KL divergence
- r ∈ [1-ε, 1+ε] → KL ≲ ε²/2
- Achieves same effect as TRPO's KL constraint

**Intuition**: PPO says "you can improve, but not too enthusiastically." This prevents catastrophic updates while allowing steady progress.

</details>

---

## Question 3: Algorithm Comparison

**Compare vanilla PG, TRPO, and PPO on: computational cost, sample efficiency, ease of implementation, and hyperparameter sensitivity. Why did PPO win?**

<details>
<summary>Click to reveal answer</summary>

### Answer

Comprehensive comparison coming soon - PPO's simplicity, robustness, and performance made it the winner despite TRPO's theoretical elegance.

</details>

---

## Question 4: Application

**You're training a PPO agent on HalfCheetah and reward plateaus at 3000 (target is 5000+). List 5 concrete things you'd try to improve performance, with justification for each.**

<details>
<summary>Click to reveal answer</summary>

### Answer

Detailed troubleshooting guide coming soon - covering learning rate tuning, architecture changes, advantage normalization, entropy tuning, and reward shaping.

</details>

---

## Question 5: Critical Analysis

**PPO is the most widely used RL algorithm in practice (robotics, games, RLHF). Why did it win over TRPO and other alternatives? What are its remaining limitations?**

<details>
<summary>Click to reveal answer</summary>

### Answer

In-depth analysis coming soon - covering PPO's ubiquity in industry, its role in training ChatGPT (RLHF), and why simple first-order methods beat complex second-order ones in practice.

</details>
