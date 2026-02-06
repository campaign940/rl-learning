# Week 3: Dynamic Programming

## Learning Objectives

- [ ] Understand and implement policy evaluation algorithm
- [ ] Master the policy improvement theorem
- [ ] Implement policy iteration algorithm
- [ ] Implement value iteration algorithm
- [ ] Understand Generalized Policy Iteration (GPI) framework

## Key Concepts

### 1. Policy Evaluation

**Problem**: Given a policy π, compute the state-value function v_π

**Iterative Policy Evaluation Algorithm**:

Starting with arbitrary v_0(s) for all s ∈ S (except terminal states where v_0 = 0), iterate:

```
v_{k+1}(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv_k(s')]
```

for all s ∈ S.

**Properties**:
- Sequence {v_k} converges to v_π as k → ∞
- Guaranteed convergence for any initial v_0 (with terminal states having value 0)
- Each iteration is a "full backup" - updates every state based on all successors
- Can be implemented with two arrays (old and new values) or in-place with one array

**Stopping Criterion**:
```
max_s |v_{k+1}(s) - v_k(s)| < θ
```
where θ is a small positive threshold.

**Computational Complexity**:
- Per iteration: O(|S|² |A|) for dense transition matrices
- Number of iterations: Depends on discount factor γ and desired accuracy

### 2. Policy Improvement

**Policy Improvement Theorem**:

Let π and π' be any pair of deterministic policies such that for all s ∈ S:
```
q_π(s, π'(s)) ≥ v_π(s)
```

Then π' is as good as or better than π:
```
v_π'(s) ≥ v_π(s) for all s ∈ S
```

**Greedy Policy Improvement**:

Given v_π, construct improved policy π':
```
π'(s) = argmax_a q_π(s,a)
       = argmax_a Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
```

This greedy policy π' satisfies the policy improvement theorem.

**Key Insight**:
- Acting greedily with respect to v_π gives a policy at least as good as π
- If π' = π, then π is optimal (satisfies Bellman optimality equation)

### 3. Policy Iteration

**Algorithm**: Alternate between policy evaluation and policy improvement

```
1. Initialize:
   - π_0 arbitrarily for all s ∈ S

2. Policy Evaluation:
   - Compute v_π_k (solve Bellman expectation equation)

3. Policy Improvement:
   - π_{k+1}(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γv_π_k(s')]

4. If π_{k+1} = π_k, stop; else go to step 2
```

**Properties**:
- Sequence of policies: π_0 → π_1 → π_2 → ... → π_*
- Each policy is strictly better than the previous (or optimal)
- Converges in finite number of iterations for finite MDPs
- Guaranteed to find optimal policy π_* and v_*

**Why It Works**:
- Monotonic improvement: v_π_{k+1} ≥ v_π_k
- Finite number of possible (deterministic) policies
- Must eventually reach optimum

### 4. Value Iteration

**Idea**: Combine policy evaluation and improvement into one update

**Algorithm**:

Starting with arbitrary v_0, iterate:
```
v_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv_k(s')]
```

until convergence.

**Extract optimal policy**:
```
π_*(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γv_*(s')]
```

**Relationship to Policy Iteration**:
- Value iteration = policy iteration with just one evaluation sweep
- More efficient when evaluation is expensive
- Avoids explicit policy representation during iteration

**Properties**:
- Converges to v_* as k → ∞
- Convergence rate depends on γ (faster for smaller γ)
- After convergence, one greedy policy improvement yields π_*

**Stopping Criterion**:
```
max_s |v_{k+1}(s) - v_k(s)| < θ
```

The error in v_k propagates to error in policy:
```
||v_π_k - v_*|| ≤ (2γ / (1-γ)) ||v_{k+1} - v_k||
```

### 5. Generalized Policy Iteration (GPI)

**Framework**: Almost all RL methods can be described as GPI

**Two Processes**:
1. **Policy Evaluation**: Make value function consistent with current policy
   - v → v_π
2. **Policy Improvement**: Make policy greedy with respect to current value function
   - π → greedy(v)

**Interaction**:
```
     evaluation
π_0 ---------> v_π_0
 |              |
 | improvement  | evaluation
 ↓              ↓
π_1 ---------> v_π_1
 |              |
 | improvement  | evaluation
 ↓              ↓
π_* ---------> v_*
```

**Key Insights**:
- Evaluation and improvement compete and cooperate
- Evaluation tries to make v consistent with π
- Improvement tries to make π greedy with respect to v
- Together they converge to optimal π_* and v_*

**Variations**:
- **Policy Iteration**: Complete evaluation before improvement
- **Value Iteration**: One evaluation sweep, then improvement
- **Asynchronous DP**: Update states in any order, not sweeping all states
- **Real-time DP**: Focus on states actually visited

**Geometric Interpretation**:
- Value function and policy are two complementary ways to characterize solution
- They interact to move toward optimality
- Like two lines approaching their intersection point

## Textbook References

- Sutton & Barto Chapter 4: Dynamic Programming
  - 4.1: Policy Evaluation (Prediction)
  - 4.2: Policy Improvement
  - 4.3: Policy Iteration
  - 4.4: Value Iteration
  - 4.5: Asynchronous Dynamic Programming
  - 4.6: Generalized Policy Iteration
  - 4.7: Efficiency of Dynamic Programming
- David Silver Lecture 3: Planning by Dynamic Programming
- CS234 Week 3: Model-Free Policy Evaluation (prep for next week)

## Implementation Tasks

### 1. Gambler's Problem (Sutton & Barto Example 4.3)

**Problem Setup**:
- Gambler has capital $s (s ∈ {1, 2, ..., 99})
- Each bet, stakes $a (a ∈ {0, 1, ..., min(s, 100-s)})
- Probability p_h of winning the bet
- If wins: capital becomes s + a
- If loses: capital becomes s - a
- Goal: Reach $100 (reward +1, terminal state)
- Going to $0 is also terminal (reward 0)

**Tasks**:
1. Implement value iteration to find v_*
2. Plot v_* as a function of capital
3. Extract and plot optimal policy π_*
4. Experiment with different p_h values (e.g., 0.25, 0.4, 0.55)
5. Analyze the structure of optimal policy

**Expected Observations**:
- For p_h < 0.5: Aggressive betting (bet to reach goal or bust)
- For p_h = 0.5: Multiple optimal policies
- For p_h > 0.5: More conservative betting

### 2. GridWorld Policy and Value Iteration

**Environment** (from Week 2):
- 4x4 grid with terminal states at (0,0) and (3,3)
- Actions: {UP, DOWN, LEFT, RIGHT}
- Reward: -1 per step
- Deterministic transitions

**Tasks**:

**Part A: Policy Iteration**
1. Implement iterative policy evaluation
2. Implement policy improvement
3. Implement full policy iteration
4. Visualize policy and value function at each iteration
5. Count number of iterations to convergence

**Part B: Value Iteration**
1. Implement value iteration
2. Compare convergence speed with policy iteration
3. Visualize value function evolution
4. Extract final optimal policy

**Part C: Analysis**
1. Compare policy iteration vs. value iteration:
   - Number of iterations
   - Computational time
   - Final policies (should be identical)
2. Test different convergence thresholds θ
3. Analyze effect of discount factor γ

## Key Equations

**Policy Evaluation (Bellman Expectation)**:
```
v_{k+1}(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv_k(s')]
```

**Policy Improvement (Greedy)**:
```
π'(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
      = argmax_a q_π(s,a)
```

**Value Iteration (Bellman Optimality)**:
```
v_{k+1}(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv_k(s')]
```

**Action-Value Function from State-Value**:
```
q_π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
```

**Optimal Policy Extraction**:
```
π_*(s) = argmax_a q_*(s,a)
       = argmax_a Σ_{s',r} p(s',r|s,a)[r + γv_*(s')]
```

**Convergence Criterion**:
```
Δ = max_s |v_{k+1}(s) - v_k(s)| < θ
```

## Algorithm Pseudocode

### Iterative Policy Evaluation

```
Input: policy π, small threshold θ > 0
Initialize: V(s) = 0 for all s ∈ S⁺
Repeat:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ
Output: V ≈ v_π
```

### Policy Iteration

```
1. Initialize:
   V(s) ∈ ℝ arbitrarily for all s ∈ S
   π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation:
   Repeat:
       Δ ← 0
       For each s ∈ S:
           v ← V(s)
           V(s) ← Σ_{s',r} p(s',r|s,π(s))[r + γV(s')]
           Δ ← max(Δ, |v - V(s)|)
   Until Δ < θ

3. Policy Improvement:
   policy-stable ← true
   For each s ∈ S:
       old-action ← π(s)
       π(s) ← argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
       If old-action ≠ π(s), then policy-stable ← false

   If policy-stable, then stop; else go to 2
```

### Value Iteration

```
Initialize: V(s) = 0 for all s ∈ S⁺
Parameters: small threshold θ > 0

Repeat:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ

Output: deterministic policy π ≈ π_*
    π(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
```

## Review Questions

1. **Why is dynamic programming called "planning"?**
   - Requires complete knowledge of MDP (model-based)
   - Computes optimal policy without interaction with environment
   - Uses the model (transition probabilities) to simulate outcomes
   - Contrasts with "learning" where agent learns from experience

2. **What is the difference between policy iteration and value iteration?**
   - **Policy Iteration**: Explicit policy at each step; complete evaluation → improvement
   - **Value Iteration**: Implicit policy; one evaluation sweep combined with improvement
   - Policy iteration: Fewer iterations but more computation per iteration
   - Value iteration: More iterations but simpler per iteration
   - Both converge to optimal policy

3. **Can dynamic programming solve real-world problems?**
   - **Limitations**:
     - Requires complete MDP model (rarely available)
     - Computational cost: O(|S|²|A|) per iteration
     - Curse of dimensionality: State space grows exponentially
   - **Where DP works**:
     - Small state spaces (e.g., tic-tac-toe, small board games)
     - When model is available (e.g., game rules)
     - As subroutine in model-based RL
   - **Modern relevance**:
     - Foundation for RL algorithms
     - Theoretical framework for understanding RL
     - Used in approximate DP with function approximation

4. **What is Generalized Policy Iteration?**
   - Framework describing interaction between evaluation and improvement
   - Two processes: making value consistent with policy, and making policy greedy with respect to value
   - These processes compete (pulling in different directions) but cooperate (toward optimality)
   - Almost all RL algorithms are instances of GPI
   - Can vary: how much evaluation? How much improvement? Which states to update?

5. **Why does policy iteration converge in finite steps for finite MDPs?**
   - Each policy improvement step strictly improves the policy (or finds optimal)
   - Finite number of deterministic policies: at most |A|^|S|
   - Monotonic improvement: cannot cycle
   - Must reach a policy that cannot be improved → optimal
   - Note: Value iteration converges asymptotically (infinite steps in theory)

## Comparison: Policy Iteration vs Value Iteration

| Aspect | Policy Iteration | Value Iteration |
|--------|------------------|-----------------|
| **Update** | Bellman expectation | Bellman optimality |
| **Policy** | Explicit | Implicit (extracted at end) |
| **Evaluation** | Complete (to convergence) | One sweep |
| **Iterations** | Fewer (typically) | More |
| **Per iteration** | More expensive | Less expensive |
| **Convergence** | Finite steps | Asymptotic |
| **Total time** | Problem-dependent | Problem-dependent |

**Rule of thumb**:
- Small action space → Policy iteration often faster
- Large action space → Value iteration may be faster
- Both converge to same optimal policy and value function

## Connection to Previous and Next Weeks

**From Week 2 (MDPs)**:
- DP algorithms operationalize Bellman equations
- Policy evaluation solves Bellman expectation equation
- Value iteration solves Bellman optimality equation

**To Week 4 (Monte Carlo Methods)**:
- DP requires model; MC does not (model-free)
- DP uses bootstrapping (updates from other estimates)
- MC uses full returns (sample-based)
- Both are instances of GPI

## Next Steps

After completing this week:
- Move to Week 4: Monte Carlo Methods
- Learn model-free policy evaluation and control
- Understand how to learn from experience without a model
- See GPI in action with sample-based updates
