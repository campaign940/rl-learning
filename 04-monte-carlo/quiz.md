# Week 4 Quiz: Monte Carlo Methods

Test your understanding of Monte Carlo methods with these questions. Try to answer them before revealing the solutions.

## Question 1: Conceptual Understanding

**Why do MC methods not require a model of the environment? What are the tradeoffs compared to model-based methods?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

MC methods don't require a model because they learn directly from complete sample episodes rather than computing expectations over all possible transitions.

**Why No Model Needed**:
- MC estimates values by averaging actual returns from experience
- Instead of computing E[R + γV(S')] using p(s',r|s,a), it samples trajectories and uses actual returns G_t
- The return G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... implicitly accounts for all future dynamics without knowing transition probabilities

**Tradeoffs**:

*Advantages of being model-free*:
1. **Applicable anywhere**: Works in complex environments where modeling is difficult (games, robotics, real-world systems)
2. **No modeling errors**: Doesn't suffer from inaccurate models that can mislead learning
3. **Sample-based**: Can learn from simulation or real experience equally well
4. **Computation**: O(1) per step vs O(|S|²) for DP value iteration sweeps

*Disadvantages*:
1. **Sample efficiency**: Requires many episodes to average out variance, while model-based methods can plan using the model
2. **Episodic only**: MC requires complete episodes; model-based methods work for continuing tasks
3. **High variance**: Individual returns vary; models provide exact expectations
4. **No lookahead**: Cannot simulate future trajectories for planning; model-based methods can evaluate actions before taking them
5. **Slow propagation**: Values propagate backward one episode at a time; DP propagates through the entire state space in one sweep

**When to prefer MC**:
- Environment is complex and hard to model accurately
- Simulation or sample trajectories are cheap to obtain
- Episodic tasks with reasonable episode lengths
- Model errors would be costly

**When to prefer model-based**:
- Accurate model is available or easy to learn
- Sample efficiency is critical (expensive real-world interaction)
- Need planning or lookahead for decision making
- Continuing tasks without natural episodes

</details>

---

## Question 2: Mathematical Derivation

**Show how importance sampling corrects for the difference between target policy π and behavior policy b. Derive why the importance sampling ratio makes the estimate unbiased.**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Setup**:
- Target policy π: the policy we want to learn about
- Behavior policy b: the policy we're actually following
- Goal: estimate E_π[G_t | S_t = s] using trajectories from b

**The Problem**:
If we simply average returns from behavior policy b, we get:
```
E_b[G_t | S_t = s] ≠ E_π[G_t | S_t = s]
```

**The Solution - Importance Sampling**:

For a trajectory τ = (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, ..., S_T), the probability under each policy is:

```
P_π(τ) = ∏_{k=t}^{T-1} π(A_k|S_k) p(S_{k+1}|S_k, A_k)

P_b(τ) = ∏_{k=t}^{T-1} b(A_k|S_k) p(S_{k+1}|S_k, A_k)
```

The importance sampling ratio is:
```
ρ_t:T-1 = P_π(τ) / P_b(τ)
        = [∏_{k=t}^{T-1} π(A_k|S_k) p(S_{k+1}|S_k, A_k)] / [∏_{k=t}^{T-1} b(A_k|S_k) p(S_{k+1}|S_k, A_k)]
        = ∏_{k=t}^{T-1} π(A_k|S_k) / b(A_k|S_k)
```

Note: The transition probabilities p(s'|s,a) cancel out! This is why we don't need a model.

**Proof of Unbiasedness (Ordinary Importance Sampling)**:

We want to show: E_b[ρ_t:T-1 G_t | S_t = s] = E_π[G_t | S_t = s]

```
E_b[ρ_t:T-1 G_t | S_t = s]
= Σ_τ P_b(τ|S_t=s) · ρ_t:T-1 · G_t(τ)
= Σ_τ P_b(τ|S_t=s) · [P_π(τ) / P_b(τ)] · G_t(τ)
= Σ_τ P_π(τ|S_t=s) · G_t(τ)
= E_π[G_t | S_t = s]
```

**Intuition**:
- If π takes an action more often than b, we weight that trajectory higher (ρ > 1)
- If π takes an action less often than b, we weight that trajectory lower (ρ < 1)
- This reweighting exactly corrects for the sampling distribution difference

**Weighted Importance Sampling**:

Instead of simple average, use weighted average:
```
V(s) = Σ_i (ρ_i G_i) / Σ_i ρ_i
```

This is biased (especially early on when denominators are small) but has lower variance because extreme weights in numerator and denominator partially cancel.

**Key Requirements**:
1. **Coverage**: b(a|s) > 0 whenever π(a|s) > 0 (behavior policy must explore all actions that target policy might take)
2. **Known policies**: Must be able to compute π(a|s) and b(a|s)

**Practical Issue - Variance**:
The ratio ρ_t:T-1 is a product of T-t terms. If any term is large, the product explodes:
- If π(a|s) = 0.9 and b(a|s) = 0.1, then π/b = 9
- Over 10 steps: 9^10 ≈ 3.5 billion!

This is why importance sampling can have unbounded variance and why weighted IS is preferred despite its bias.

</details>

---

## Question 3: Method Comparison

**Compare first-visit MC, every-visit MC, and TD(0) in terms of bias and variance. Which converges faster in practice and why?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

| Method | Bias | Variance | Update Timing | Bootstrapping |
|--------|------|----------|---------------|---------------|
| First-visit MC | Unbiased | High | End of episode | No |
| Every-visit MC | Unbiased | High | End of episode | No |
| TD(0) | Biased (initially) | Lower | Each step | Yes |

**First-Visit MC**:
- **Bias**: Zero. Averages actual returns, which are unbiased samples of true value
- **Variance**: High. Different episodes starting from same state can have very different returns due to stochasticity
- **Convergence**: Each state visit is independent; converges to v_π(s) by Law of Large Numbers
- **Sample estimate**: One data point per episode (even if state visited multiple times)

**Every-Visit MC**:
- **Bias**: Zero (asymptotically). Each return is an unbiased estimate of v_π(s)
- **Variance**: High, but can be lower than first-visit due to more samples per episode
- **Convergence**: Slightly faster than first-visit in practice (more updates per episode)
- **Sample estimate**: Multiple data points per episode if state visited multiple times
- **Note**: Updates are correlated within an episode, but this doesn't affect asymptotic unbiasedness

**TD(0)**:
- **Bias**: Initially biased. Bootstraps from current estimate V(S_{t+1}), which is initially wrong
- **Variance**: Lower. Only depends on one-step reward R_{t+1} and current estimate, not entire future trajectory
- **Convergence**: Often faster in practice despite initial bias
- **Update**: V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
- **Bootstrapping**: Uses estimate V(S_{t+1}) instead of actual return

**Bias-Variance Tradeoff**:

Monte Carlo methods:
```
Target: G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... (actual return)
Variance sources: All future rewards R_{t+1}, R_{t+2}, ..., R_T
Bias: None - G_t is unbiased sample of E[G_t]
```

TD(0):
```
Target: R_{t+1} + γV(S_{t+1}) (one reward + bootstrapped estimate)
Variance sources: Only R_{t+1}
Bias: V(S_{t+1}) is not true value (especially early in learning)
```

**Which Converges Faster?**

**In practice, TD(0) usually converges faster** for several reasons:

1. **Lower variance**: Updates are much more consistent because they depend only on one random reward, not entire return

2. **Online learning**: TD updates after every step; MC must wait until episode end
   - Long episodes: TD gets T updates while MC gets 1
   - Can learn during episode, adjusting behavior online

3. **Continuing tasks**: TD works for non-episodic tasks; MC doesn't

4. **Bias reduces quickly**: Initial bias disappears as estimates improve, and lower variance accelerates this process

**When MC is better**:
- Function approximation: MC's lack of bootstrapping avoids divergence issues
- Few, long episodes: Episode completion isn't a bottleneck
- Highly stochastic returns: If one-step dynamics are unreliable, bootstrapping hurts

**Empirical Example** (Random Walk):
- States: 1-2-3-4-5 (start at 3, terminal at 0 and 6)
- True values: [1/6, 2/6, 3/6, 4/6, 5/6]
- TD(0) typically reaches RMSE < 0.1 in ~10 episodes
- MC typically needs 50-100 episodes for same accuracy
- TD's advantage grows with episode length

**Convergence Guarantees**:
- All three methods converge to v_π with probability 1 given:
  - All states visited infinitely often
  - Step size satisfies Robbins-Monro: Σα_t = ∞ and Σα_t² < ∞
- TD may converge faster in "steps" but both eventually reach true value

**Key Insight**: The bias-variance tradeoff heavily favors lower variance in RL because:
- High variance slows learning more than small bias
- Bias reduces as estimates improve
- Sample efficiency (number of environment interactions) is often the bottleneck

</details>

---

## Question 4: Practical Application

**Design a Monte Carlo control algorithm for a simple card game where you're dealt two cards (values 1-10) and can either "hit" (draw another card) or "stand" (keep current hand). You bust if total exceeds 21. Dealer hits until reaching 17+. Explain your approach including exploration strategy.**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Game Specification**:
- State: (player_sum, dealer_showing, num_cards_held)
- Actions: {hit, stand}
- Reward: +1 (win), -1 (lose), 0 (draw)
- Episode: Terminates when both player and dealer stand or someone busts

**MC Control Algorithm Design**:

```python
def monte_carlo_control_card_game(num_episodes, epsilon=0.1, epsilon_decay=0.9999):
    """
    Monte Carlo Control with epsilon-greedy exploration
    """
    # Initialize action-value function
    Q = defaultdict(lambda: defaultdict(float))  # Q[state][action]
    returns = defaultdict(lambda: defaultdict(list))  # For averaging

    # Initialize policy (start epsilon-greedy)
    def get_action(state, epsilon):
        if random.random() < epsilon:
            return random.choice(['hit', 'stand'])
        else:
            # Greedy action
            if Q[state]['hit'] > Q[state]['stand']:
                return 'hit'
            elif Q[state]['hit'] < Q[state]['stand']:
                return 'stand'
            else:
                return random.choice(['hit', 'stand'])  # Break ties randomly

    for episode_num in range(num_episodes):
        # Generate episode following current policy
        episode = generate_episode(get_action, epsilon)
        # episode = [(S_0, A_0, R_1), (S_1, A_1, R_2), ..., (S_T-1, A_T-1, R_T)]

        # Track visited state-action pairs
        visited = set()
        G = 0  # Return

        # Process episode backwards (from T-1 to 0)
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = reward + gamma * G  # Update return

            # First-visit MC: only update if (s,a) not seen earlier in episode
            if (state, action) not in visited:
                visited.add((state, action))
                returns[state][action].append(G)

                # Update Q value (average of returns)
                Q[state][action] = np.mean(returns[state][action])

                # Implicit policy improvement (epsilon-greedy uses updated Q)

        # Decay exploration
        epsilon *= epsilon_decay
        epsilon = max(0.01, epsilon)  # Minimum epsilon

    return Q

def generate_episode(policy_fn, epsilon):
    """Generate one episode following the given policy"""
    episode = []
    state = initial_state()  # Deal initial cards

    while not is_terminal(state):
        action = policy_fn(state, epsilon)
        next_state, reward = take_action(state, action)
        episode.append((state, action, reward))
        state = next_state

    return episode
```

**Key Design Decisions**:

1. **State Representation**:
   - `player_sum`: Total value of player's cards (1-30+)
   - `dealer_showing`: Dealer's visible card (1-10)
   - `num_cards_held`: Number of cards player has (2+)
   - This captures all decision-relevant information

2. **Exploration Strategy: ε-greedy with decay**
   - Start with ε = 0.1 (10% random actions)
   - Decay by 0.9999 per episode
   - Minimum ε = 0.01 (maintain slight exploration)
   - **Why**: Balances exploration early (find good strategies) with exploitation late (fine-tune optimal policy)

3. **Alternative Exploration Strategies**:

   a. **Exploring Starts** (impractical here):
   ```python
   # Start each episode from random (state, action)
   # Problem: Can't control initial card deal in practice
   ```

   b. **Optimistic Initialization**:
   ```python
   Q = defaultdict(lambda: defaultdict(lambda: 5.0))  # Optimistic
   # Encourages trying all actions initially
   ```

   c. **UCB (Upper Confidence Bound)**:
   ```python
   def ucb_action(state, c=2.0):
       N_state = sum(action_counts[state].values())
       ucb_values = {}
       for action in ['hit', 'stand']:
           N_action = action_counts[state][action]
           if N_action == 0:
               return action  # Try unexplored actions first
           ucb = Q[state][action] + c * sqrt(log(N_state) / N_action)
           ucb_values[action] = ucb
       return max(ucb_values, key=ucb_values.get)
   ```

4. **Why First-Visit MC**:
   - Simpler implementation
   - State revisits unlikely in this game (sum usually increases)
   - Each episode visit is roughly independent

5. **Incremental Update (More Efficient)**:
```python
# Instead of storing all returns
Q[state][action] = mean(returns[state][action])

# Use incremental update
N[state][action] += 1
alpha = 1.0 / N[state][action]
Q[state][action] += alpha * (G - Q[state][action])
```

**Expected Policy Learned**:
- **Low sums (< 12)**: Always hit (can't bust)
- **Medium sums (12-16)**: Depends on dealer's card
  - Dealer showing 2-6: Stand (dealer likely to bust)
  - Dealer showing 7-10: Hit (need higher total)
- **High sums (17-21)**: Always stand
- **Dealer showing Ace**: More aggressive hitting (dealer has advantage)

**Convergence Monitoring**:
```python
# Track policy changes
policy_stable_count = 0
prev_policy = None

for episode in range(num_episodes):
    # ... MC update ...

    # Extract current policy
    current_policy = {s: max(Q[s], key=Q[s].get) for s in Q}

    if current_policy == prev_policy:
        policy_stable_count += 1
        if policy_stable_count > 1000:
            print(f"Policy converged at episode {episode}")
            break
    else:
        policy_stable_count = 0

    prev_policy = current_policy
```

**Advantages of MC for This Game**:
1. **Model-free**: Don't need to know deck probabilities
2. **Natural episodes**: Games have clear start and end
3. **Short episodes**: Fast return computation
4. **Sparse rewards**: Only at game end (MC handles this well)

**Limitations**:
1. Must wait until game ends to update (vs TD learning each step)
2. High variance if game has luck elements (card draw randomness)
3. Requires many episodes for rare state-action pairs

**Practical Implementation Tips**:
- Initialize Q values optimistically (e.g., Q=0 for all, assuming draws)
- Use larger α (learning rate) early, decrease over time
- Track state visit counts to identify under-explored regions
- Consider off-policy learning if you have human expert games (data reuse)

</details>

---

## Question 5: Critical Analysis

**Why does ordinary importance sampling have unbounded variance? Provide a concrete example. How does weighted importance sampling address this problem, and what does it sacrifice in return?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Ordinary Importance Sampling**:
```
V(s) = E_b[ρ_{t:T-1} G_t | S_t = s]
     = (1/n) Σ_{i=1}^n ρ_i G_i
```

where ρ_i = ∏_{k=t}^{T-1} π(A_k|S_k) / b(A_k|S_k)

**Why Unbounded Variance?**

The variance of the estimate is:
```
Var[V(s)] = Var[ρ G] / n
          = E[(ρG)²] - (E[ρG])²
```

The problem: **ρ can be arbitrarily large**, even if G is bounded.

**Concrete Example**:

Consider a simple 2-state MDP:
- States: S = {A, B} (start at A, B is terminal)
- Actions: {left, right}
- From A: 'right' → B (reward +1), 'left' → B (reward +1)
- Episode length: 1 step

Target policy π:
```
π(right|A) = 1.0
π(left|A) = 0.0
```

Behavior policy b:
```
b(right|A) = 0.1
b(left|A) = 0.9
```

**Importance sampling ratio**:
- For trajectory (A, right, B): ρ = π(right|A)/b(right|A) = 1.0/0.1 = 10
- For trajectory (A, left, B): ρ = π(left|A)/b(left|A) = 0.0/0.9 = 0

**Ordinary IS Estimate**:
90% of episodes take 'left' → ρ = 0, contributes 0
10% of episodes take 'right' → ρ = 10, G = 1, contributes 10

```
V(A) = (1/n) Σ ρ_i G_i
     ≈ (1/10) × 10 × 1 = 1.0  (correct!)
```

But variance:
```
E[ρG] = 1.0
E[(ρG)²] = 0.9 × 0² + 0.1 × 10² = 10
Var[ρG] = 10 - 1² = 9
```

**Extreme Example** (unbounded):

Now make behavior policy explore more:
```
π(right|A) = 1.0
b(right|A) = 0.01  (very unlikely to take optimal action)
```

ρ = 1.0/0.01 = 100

```
Var[ρG] = 0.99 × 0² + 0.01 × 100² = 100
```

As b(right|A) → 0, the variance → ∞!

**Multi-Step Episode Example**:

Consider 10-step episode:
```
π(a*|s) = 0.5 for optimal action
b(a*|s) = 0.1 for optimal action

ρ = (0.5/0.1)^10 = 5^10 ≈ 10 million
```

Even if G = 1, one rare trajectory contributes 10 million to the average. This dominates the estimate and creates massive variance.

**Why This Happens**:
1. **Rare events**: Trajectories with low probability under b but high under π
2. **Exponential growth**: ρ is a product; each step multiplies the ratio
3. **Long episodes**: More steps → larger potential ρ values
4. **Policy mismatch**: Greater difference between π and b → higher ratios

---

**Weighted Importance Sampling Solution**:

```
V(s) = Σ_{i=1}^n (ρ_i G_i) / Σ_{i=1}^n ρ_i
```

**Why Lower Variance?**

Using the same example:
```
π(right|A) = 1.0, b(right|A) = 0.1
```

After 10 episodes (9 left, 1 right):
```
Numerator: 0×9 + 10×1 = 10
Denominator: 0×9 + 10×1 = 10
V(A) = 10/10 = 1.0
```

But now consider variance effect:
- Denominator also has the large weight (10)
- When ρ is large, both numerator and denominator are large
- The ratio ρG/Σρ is more stable

**Mathematical Intuition**:

Ordinary IS: One large ρ_i dominates sum
```
(ρ_1 G_1 + ρ_2 G_2 + ... + ρ_huge G_huge + ...) / n
```

Weighted IS: Large ρ_i normalized by total weight
```
(ρ_huge G_huge) / (ρ_1 + ρ_2 + ... + ρ_huge + ...)
```

The large weight ρ_huge appears in both numerator and denominator, reducing its influence.

**Formal Variance Bound**:

For weighted IS, the estimate is bounded:
```
|V_weighted(s)| ≤ max_i |G_i|
```

The estimate is a weighted average of returns, so it can't exceed the maximum return. Ordinary IS has no such bound because the weights (ρ values) can be arbitrarily large.

---

**What Weighted IS Sacrifices: Bias**

Ordinary IS is **unbiased**: E[V_ordinary] = v_π(s) for all n

Weighted IS is **biased**, especially for small n:
```
E[Σ(ρ_i G_i) / Σρ_i] ≠ v_π(s)
```

**Why biased?**
The denominator Σρ_i is random. Taking the ratio of two random variables (even if numerator is unbiased) introduces bias.

**Example of Bias**:

Suppose true value v_π(s) = 1.
With 2 episodes:
- Episode 1: ρ_1 = 0.1, G_1 = 1 → contributes 0.1
- Episode 2: ρ_2 = 10, G_2 = 1 → contributes 10

Ordinary IS: (0.1 + 10)/2 = 5.05 (way off but unbiased on average)
Weighted IS: (0.1×1 + 10×1)/(0.1 + 10) = 10.1/10.1 ≈ 1.0 (close!)

But weighted IS estimate depends on which episodes sampled:
- If only episode 1: 0.1/0.1 = 1.0
- If only episode 2: 10/10 = 1.0
- If both: 10.1/10.1 ≈ 1.0

This seems unbiased, but with different G values, bias appears:
- Episode 1: ρ_1 = 0.1, G_1 = 0
- Episode 2: ρ_2 = 10, G_2 = 1

True value: 0.1×0 + 0.9×1 ≈ 0.9 (under b)
Under π: should be 1.0

Weighted IS: (0.1×0 + 10×1)/(0.1 + 10) = 10/10.1 ≈ 0.99

The bias diminishes as n → ∞, making it **asymptotically unbiased**.

---

**Summary Comparison**:

| Property | Ordinary IS | Weighted IS |
|----------|-------------|-------------|
| Bias | Unbiased | Biased (small n) |
| Variance | Unbounded | Bounded |
| Asymptotic | Unbiased | Unbiased |
| Stability | Poor | Good |
| Practical | Rarely used | Preferred |

**When to Use Each**:

**Ordinary IS**:
- Theoretical analysis requiring unbiased estimates
- Very short episodes (ρ stays small)
- Policies are similar (π ≈ b)
- Infinite data available

**Weighted IS** (almost always in practice):
- Real applications
- Long episodes
- Significant policy mismatch
- Limited data
- Need stable learning

**Key Insight**: The bias-variance tradeoff again favors lower variance. Weighted IS's small bias (which vanishes with more data) is vastly preferable to ordinary IS's potentially unbounded variance that can prevent convergence entirely.

**Practical Mitigation Strategies**:
1. Use weighted IS instead of ordinary
2. Keep b and π similar (don't make b explore too much)
3. Truncate importance ratios: ρ_truncated = min(ρ, c) for some constant c
4. Use per-decision IS instead of per-trajectory
5. Consider doubly robust methods that combine IS with model learning

</details>

---

## Additional Practice Problems

1. Implement both first-visit and every-visit MC for a grid world. Compare their convergence rates.

2. Prove that MC control with exploring starts converges to the optimal policy.

3. Design an off-policy learning scenario where importance sampling is essential.

4. Calculate the importance sampling ratio for a 5-step trajectory with given policies.

5. Explain why MC methods have difficulty with continuing tasks and propose a solution.
