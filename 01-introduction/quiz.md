# Week 1 Quiz: Introduction to Reinforcement Learning

## Question 1: Conceptual Understanding

**What distinguishes reinforcement learning from supervised learning and unsupervised learning? Explain the key characteristics that make RL unique.**

<details>
<summary>Answer</summary>

Reinforcement learning is distinguished by three key characteristics:

**1. Trial-and-Error Learning**:
- In supervised learning, the agent is directly told the correct output (label) for each input
- In RL, there is no instructor telling the agent which action is correct
- The agent must discover good actions through trial and error by trying them and observing outcomes

**2. Delayed Consequences**:
- Supervised learning provides immediate feedback on whether predictions are correct
- In RL, actions may have long-term consequences that are not immediately apparent
- The reward for an action may come much later, making credit assignment challenging
- Example: In chess, a move might seem good immediately but lead to defeat many moves later

**3. Exploration-Exploitation Tradeoff**:
- Supervised/unsupervised learning use a fixed dataset
- In RL, the agent must balance:
  - **Exploitation**: Using current knowledge to maximize reward
  - **Exploration**: Trying new actions to discover potentially better options
- The agent's actions affect the data it experiences in the future

**Difference from Unsupervised Learning**:
- Unsupervised learning finds hidden structure in unlabeled data
- RL is about learning to take actions to maximize a reward signal
- RL has a clear objective (maximize cumulative reward), while unsupervised learning goals are less defined

**The RL Problem Structure**:
- Agent interacts with environment in a loop: state → action → reward → new state
- Goal is to maximize expected cumulative reward, not just immediate reward
- Learning happens through experience, not from labeled examples

</details>

---

## Question 2: Mathematical Derivation

**Derive the incremental update rule for action-value estimates. Starting from the sample-average definition:**

```
Q_n = (R_1 + R_2 + ... + R_{n-1}) / (n-1)
```

**Show that this can be rewritten as:**

```
Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
```

<details>
<summary>Answer</summary>

**Step-by-step derivation**:

Starting with the sample-average definition for Q_{n+1}:

```
Q_{n+1} = (R_1 + R_2 + ... + R_n) / n
```

We can rewrite the numerator:

```
Q_{n+1} = (R_1 + R_2 + ... + R_{n-1} + R_n) / n
```

Notice that the sum R_1 + R_2 + ... + R_{n-1} equals (n-1)Q_n:

```
Q_{n+1} = [(n-1)Q_n + R_n] / n
```

Distribute the division:

```
Q_{n+1} = (n-1)Q_n/n + R_n/n
```

Rewrite (n-1)/n as 1 - 1/n:

```
Q_{n+1} = (1 - 1/n)Q_n + R_n/n
```

Expand:

```
Q_{n+1} = Q_n - Q_n/n + R_n/n
```

Factor out 1/n:

```
Q_{n+1} = Q_n + (1/n)[R_n - Q_n]
```

**General Form**:
```
NewEstimate = OldEstimate + StepSize[Target - OldEstimate]
```

**Interpretation**:
- The update is proportional to the error (R_n - Q_n)
- If R_n > Q_n, we increase the estimate
- If R_n < Q_n, we decrease the estimate
- The step size 1/n decreases over time (averaging all past rewards equally)
- This can be generalized to constant step size α for non-stationary problems

**Benefits of Incremental Form**:
- Only requires storing Q_n and the count n
- Constant memory (O(1)) instead of storing all rewards (O(n))
- Computational efficiency: O(1) per update instead of O(n)

</details>

---

## Question 3: Algorithm Comparison

**Compare and contrast epsilon-greedy, UCB (Upper Confidence Bound), and gradient bandit algorithms. When does each method excel? What are their strengths and weaknesses?**

<details>
<summary>Answer</summary>

## Epsilon-Greedy

**Algorithm**:
- With probability 1-ε: select A_t = argmax_a Q_t(a) (greedy)
- With probability ε: select random action (explore)

**Strengths**:
- Simple to implement and understand
- Guaranteed to try all actions infinitely often (ε > 0)
- Works well in practice for many problems

**Weaknesses**:
- Explores randomly without preference for promising actions
- Explores equally among all non-greedy actions
- Requires tuning ε parameter
- May waste exploration on clearly bad actions

**When it excels**:
- When you need a simple, robust baseline
- When all actions have similar uncertainty
- When computational resources are limited

## Upper Confidence Bound (UCB)

**Algorithm**:
```
A_t = argmax_a [Q_t(a) + c * sqrt(ln(t) / N_t(a))]
```

**Strengths**:
- Deterministic exploration (no randomness in action selection)
- Explores intelligently based on uncertainty
- Favors actions that are either high-value or uncertain
- Theoretical guarantees (logarithmic regret bound)
- Systematically reduces uncertainty about all actions

**Weaknesses**:
- Harder to extend to non-stationary problems
- Requires counting action selections N_t(a)
- The exploration term can be aggressive early on
- Requires tuning c parameter
- Difficult to extend to large/continuous action spaces

**When it excels**:
- Stationary bandit problems
- When you want principled exploration
- When you can afford to try all actions at least once initially
- Problems where systematic uncertainty reduction is valuable

## Gradient Bandit

**Algorithm**:
```
H_{t+1}(a) = H_t(a) + α(R_t - R̄_t)(1{A_t=a} - π_t(a))
π_t(a) = exp(H_t(a)) / Σ_b exp(H_t(b))
```

**Strengths**:
- Doesn't require knowing reward distributions
- Works with relative preferences rather than absolute values
- Natural probability distribution over actions
- Invariant to adding constant to all rewards
- Based on stochastic gradient ascent (principled optimization)

**Weaknesses**:
- Requires baseline R̄_t for good performance
- More parameters to tune (α and initial H values)
- Can be sensitive to step size α
- Computational overhead of softmax

**When it excels**:
- When rewards are shifted by unknown constant
- When you want soft exploration (probabilistic action selection)
- Non-stationary problems (with appropriate α)
- When relative action preferences matter more than absolute values

## Summary Comparison

| Feature | Epsilon-Greedy | UCB | Gradient Bandit |
|---------|---------------|-----|-----------------|
| Exploration | Random | Systematic | Probabilistic |
| Complexity | Low | Medium | Medium-High |
| Theory | None | Strong | Medium |
| Tuning | ε | c | α, baseline |
| Reward Invariance | No | No | Yes (to shifts) |

**Practical Recommendation**:
- Start with epsilon-greedy (ε=0.1) as baseline
- Use UCB when you need principled exploration
- Use gradient bandit when rewards have unknown baseline shifts
- Always compare multiple methods on your specific problem

</details>

---

## Question 4: Application and Analysis

**Consider a 10-armed bandit problem where the true action values q_*(a) are {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, but the environment shifts all rewards by adding a constant +1000 to every reward. How would this affect:**
1. **Epsilon-greedy with sample-average action-value estimates?**
2. **UCB?**
3. **Gradient bandit (with and without baseline)?**

**Explain which method is most affected and why.**

<details>
<summary>Answer</summary>

## Analysis of Each Method

### 1. Epsilon-Greedy (Sample-Average)

**Effect**: Minimal to no impact

**Explanation**:
- Sample-average estimates: Q_t(a) = average of rewards received for action a
- Original rewards: R ~ N(q_*(a), σ²)
- Shifted rewards: R' = R + 1000 ~ N(q_*(a) + 1000, σ²)
- Estimates become: Q_t(a) ≈ q_*(a) + 1000

**Key Point**: The **relative ordering** of action values is preserved!
- Action 10 still has highest Q value: 10 + 1000 = 1010
- Action 1 still has lowest Q value: 1 + 1000 = 1001
- argmax_a Q_t(a) selects the same action as before

**Conclusion**: Performance unchanged. The shift affects all actions equally, preserving the greedy action.

### 2. Upper Confidence Bound

**Effect**: Minimal to no impact

**Explanation**:
```
A_t = argmax_a [Q_t(a) + c * sqrt(ln(t) / N_t(a))]
```

- The Q_t(a) estimates are shifted by +1000
- The exploration bonus sqrt(ln(t) / N_t(a)) is unchanged
- The argmax still selects based on relative values

**Edge Case**: The exploration bonus becomes relatively smaller compared to Q values
- Before: bonus might be ~2 when Q values are 1-10
- After: bonus is ~2 when Q values are 1001-1010
- This makes UCB slightly more exploitative (relatively less exploration)

**Conclusion**: Mostly unchanged, but exploration bonus becomes relatively less significant. In practice, minimal impact.

### 3. Gradient Bandit WITHOUT Baseline

**Effect**: SEVERELY DEGRADED PERFORMANCE

**Explanation**:
```
H_{t+1}(a) = H_t(a) + α(R_t - R̄_t)(1{A_t=a} - π_t(a))
```

Without baseline (R̄_t = 0):
```
H_{t+1}(a) = H_t(a) + αR_t(1{A_t=a} - π_t(a))
```

**Problem**: All rewards are now ~1000, all positive and large!

For selected action (A_t = a):
```
H_{t+1}(a) = H_t(a) + α(1000)(1 - π_t(a))
```

This is always a large positive update, regardless of whether it was a good action!

For non-selected actions:
```
H_{t+1}(a) = H_t(a) + α(1000)(-π_t(a))
```

This is a large negative update.

**Result**: The algorithm reinforces ALL tried actions strongly, losing the ability to discriminate between good and bad actions. Differences in preferences (5 vs 6 vs 7...) are swamped by the large baseline value of 1000.

### 4. Gradient Bandit WITH Baseline

**Effect**: NO IMPACT

**Explanation**:
With baseline R̄_t = average of all rewards so far:
```
H_{t+1}(a) = H_t(a) + α(R_t - R̄_t)(1{A_t=a} - π_t(a))
```

- Original: R_t ~ N(q_*(a), σ²), R̄_t ≈ average of q_* values
- Shifted: R_t ~ N(q_*(a) + 1000, σ²), R̄_t ≈ average of q_* values + 1000

**Key**: (R_t - R̄_t) is invariant to constant shifts!
- Before: R_t - R̄_t = [q_*(a) + noise] - [avg q_*]
- After: R_t - R̄_t = [q_*(a) + 1000 + noise] - [avg q_* + 1000] = [q_*(a) + noise] - [avg q_*]

**Conclusion**: Complete invariance! This is why the baseline is critical for gradient bandit.

## Summary

| Method | Impact | Reason |
|--------|--------|--------|
| Epsilon-Greedy | None | Relative ordering preserved |
| UCB | Minimal | Relative values preserved, bonus relatively smaller |
| Gradient (no baseline) | **SEVERE** | Cannot discriminate, all rewards reinforced |
| Gradient (with baseline) | None | (R_t - R̄_t) invariant to shifts |

**Most Affected**: Gradient bandit **without baseline**

**Key Insight**: This demonstrates why a baseline is essential for gradient methods. The baseline makes the algorithm invariant to shifting the reward scale, which is a critical property for robust performance.

**Practical Lesson**: Always use a baseline with gradient bandit algorithms! In practice, this applies to policy gradient methods in deep RL (REINFORCE, actor-critic, etc.).

</details>

---

## Question 5: Critical Thinking

**Why is the exploration-exploitation tradeoff considered fundamental and unavoidable in reinforcement learning? Could we design an algorithm that completely solves this tradeoff? Why or why not?**

<details>
<summary>Answer</summary>

## Why the Tradeoff is Fundamental

### 1. **Incomplete Information**

The core issue is **epistemic uncertainty** - we don't know the true value of actions:

- We only have estimates Q_t(a) based on limited samples
- The true value q_*(a) = E[R_t | A_t = a] is unknown
- Our estimates are imperfect and could be wrong
- We can only reduce uncertainty by trying actions

**Implication**: To be certain we've found the optimal action, we'd need infinite samples of all actions - impractical!

### 2. **Opportunity Cost**

Every decision involves a tradeoff:

- **Time spent exploring** = time not exploiting the current best action
- **Time spent exploiting** = time not discovering potentially better actions

**Example**: Imagine a 1000-armed bandit with 1000 time steps
- If we try each arm once, we spend all time exploring, no time exploiting
- If we try arm 1 three times and then exploit, we might miss the optimal arm
- There's no free lunch: exploring has a cost (foregone reward)

### 3. **The Impossibility of Perfect Knowledge**

Even with the best algorithm:
- We can never be 100% certain we've found the optimal action (unless we try all actions infinitely)
- Early in learning, we have high uncertainty
- We must make decisions under uncertainty

**The tradeoff emerges from**:
- Finite time horizon
- Uncertainty about action values
- The need to take actions to learn about them

## Could We Design a Perfect Algorithm?

**Short answer: No, but we can do better than random.**

### What Would "Solving" the Tradeoff Mean?

An algorithm that:
1. Always maximizes expected cumulative reward
2. Never wastes time on suboptimal actions
3. Always finds the optimal action quickly

### Why This is Impossible

**Theoretical Barriers**:

1. **No Free Lunch Theorems**:
   - No algorithm can be optimal for all possible problem instances
   - Any algorithm that performs well on some problems must perform poorly on others
   - The best we can do is make assumptions about the problem structure

2. **Information-Theoretic Lower Bounds**:
   - To distinguish between two similar-valued actions, we need a certain minimum number of samples
   - This is determined by probability theory, not the algorithm
   - Example: If two actions have q_*(a₁)=0.50, q_*(a₂)=0.51, rewards ~ N(q_*, 1)
     - We need many samples to reliably distinguish them
     - Any algorithm faces this fundamental sampling requirement

3. **The Gittins Index (Bayesian Optimal Solution)**:
   - For specific problem settings (discounted infinite horizon, known reward distributions), optimal solution exists
   - But it requires:
     - Known prior distributions over action values
     - Computational intractability for large problems
     - Specific problem assumptions that rarely hold
   - Even then, it doesn't "eliminate" the tradeoff - it optimally balances it

### What We CAN Do

**Principled Approaches**:

1. **Bayesian Methods**:
   - Thompson Sampling: sample from posterior distributions
   - Maintains uncertainty estimates
   - Provably near-optimal for many problems
   - But still explores, just optimally given assumptions

2. **Regret Minimization**:
   - UCB has logarithmic regret: O(log t)
   - This is near-optimal (lower bound exists)
   - But regret still grows - we still make mistakes

3. **Problem-Specific Knowledge**:
   - If we know rewards are smooth over action space → use function approximation
   - If we know problem structure → exploit it
   - But this requires additional assumptions

### The Deeper Truth

The exploration-exploitation tradeoff reflects a fundamental aspect of learning under uncertainty:

**Epistemic Circle**:
- To know which action is best, we need to try actions
- To try actions efficiently, we need to know which are promising
- We can't have both perfect knowledge and zero cost

**It's Not a Bug, It's a Feature**:
- The tradeoff makes learning interesting and non-trivial
- It's what distinguishes RL from planning (where we know the model)
- It's unavoidable in any real-world learning scenario

## Practical Implications

1. **Accept the tradeoff**: No algorithm is perfect
2. **Choose algorithms based on problem properties**:
   - Stationary? Use UCB
   - Non-stationary? Use epsilon-greedy with constant α
   - Large action space? Use function approximation
3. **Use prior knowledge**: Any information about the problem helps
4. **Measure regret**: Compare algorithms by cumulative regret, not perfection

## Conclusion

The exploration-exploitation tradeoff cannot be "solved" in the sense of eliminating it. It is:
- Information-theoretically fundamental
- Unavoidable given finite time and uncertainty
- The central challenge that makes RL interesting

What we can do:
- Design algorithms that balance it intelligently (UCB, Thompson Sampling)
- Minimize regret asymptotically (logarithmic regret is near-optimal)
- Use problem structure and prior knowledge when available

**The tradeoff is not a limitation of our algorithms - it's a fundamental property of learning under uncertainty.**

</details>

