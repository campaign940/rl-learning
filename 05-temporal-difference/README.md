# Week 5: Temporal-Difference Learning

## Learning Objectives

By the end of this week, you should be able to:

1. Understand **bootstrapping** and why TD methods use it
2. Implement **TD(0) prediction** for policy evaluation
3. Apply **SARSA** (on-policy TD control) to find optimal policies
4. Implement **Q-Learning** (off-policy TD control)
5. Understand **Expected SARSA** and its variance reduction properties
6. Recognize and address **maximization bias** using Double Q-Learning
7. Compare TD, MC, and DP methods across multiple dimensions

## Key Concepts

### 1. TD Prediction - Bootstrapping from Estimates

Temporal-Difference learning combines ideas from Monte Carlo and Dynamic Programming:
- Like MC: learns from experience without a model
- Like DP: updates estimates based on other estimates (bootstrapping)

**TD(0) Update Rule**:
```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

**TD Error (δ_t)**:
```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

This error measures the difference between the estimated value and the better estimate (one step later).

**Key Properties**:
- Updates occur after each step (online learning)
- Works for non-episodic (continuing) tasks
- Lower variance than MC (only depends on one reward + estimate)
- Initial bias that decreases as estimates improve
- Converges to v_π under appropriate conditions

**Why Bootstrapping?**
Instead of waiting for the actual return G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ..., TD uses:
```
TD target: R_{t+1} + γV(S_{t+1})
```

This substitute target is available immediately and has lower variance (but is initially biased).

### 2. SARSA - On-Policy TD Control

SARSA (State-Action-Reward-State-Action) learns action values Q(s,a) while following a single policy (typically ε-greedy).

**SARSA Update**:
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

**Name Origin**: The update uses (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}) tuple.

**Algorithm Flow**:
1. Initialize Q(s,a) arbitrarily
2. For each episode:
   - Initialize S
   - Choose A from S using policy derived from Q (ε-greedy)
   - For each step:
     - Take action A, observe R, S'
     - Choose A' from S' using policy derived from Q
     - Update: Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
     - S ← S', A ← A'

**Characteristics**:
- On-policy: learns about the policy it's following
- Conservative: accounts for exploration in value estimates
- Converges to optimal policy under standard conditions
- Safe for real-time learning (accounts for exploration costs)

### 3. Q-Learning - Off-Policy TD Control

Q-Learning learns the optimal action values Q*(s,a) regardless of the policy being followed.

**Q-Learning Update**:
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

**Key Difference from SARSA**: Uses max_a Q(S',a) instead of Q(S',A') where A' is actually chosen.

**Algorithm Flow**:
1. Initialize Q(s,a) arbitrarily
2. For each step:
   - Choose A from S using policy derived from Q (ε-greedy)
   - Take action A, observe R, S'
   - Update: Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
   - S ← S'

**Characteristics**:
- Off-policy: learns optimal policy while following exploratory policy
- Aggressive: assumes optimal actions will be taken
- Can learn from demonstrations or old policies
- Simple and widely used in practice
- Suffers from maximization bias (see Double Q-Learning)

### 4. Expected SARSA - Variance Reduction

Expected SARSA takes the expected value over next actions instead of sampling.

**Expected SARSA Update**:
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ Σ_a π(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]
```

**Advantages**:
- Lower variance than SARSA (uses expected value, not sample)
- Can be on-policy or off-policy
- Often performs better than SARSA empirically
- Eliminates variance due to random action selection

**When π is greedy**: Expected SARSA = Q-Learning
```
Σ_a π(a|s) Q(s,a) = max_a Q(s,a)  (if π is greedy)
```

**Computational Cost**: Must compute expectation over all actions (more expensive per step).

### 5. Double Q-Learning - Fixing Maximization Bias

**The Problem**: Q-Learning overestimates action values due to using max for both selecting and evaluating actions.

**Maximization Bias**:
```
max_a Q(s,a) ≥ E[Q(s,a)]
```

Taking the maximum of noisy estimates systematically overestimates the true maximum.

**Double Q-Learning Solution**:
Maintain two independent Q-functions: Q1 and Q2.

**Update (alternating)**:
```
With probability 0.5:
    Q1(S,A) ← Q1(S,A) + α[R + γ Q2(S', argmax_a Q1(S',a)) - Q1(S,A)]
Otherwise:
    Q2(S,A) ← Q2(S,A) + α[R + γ Q1(S', argmax_a Q2(S',a)) - Q2(S,A)]
```

**Key Idea**: Use one Q-function to select the action, the other to evaluate it. This decorrelates the selection and evaluation, reducing bias.

**Final Policy**: π(s) = argmax_a [Q1(s,a) + Q2(s,a)]

## Key Equations

### TD(0) Prediction
```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]

TD error: δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
```

### SARSA (On-Policy TD Control)
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
```

### Q-Learning (Off-Policy TD Control)
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
```

### Expected SARSA
```
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ Σ_a π(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)]
```

### Double Q-Learning
```
Q1(S,A) ← Q1(S,A) + α[R + γ Q2(S', argmax_a Q1(S',a)) - Q1(S,A)]
Q2(S,A) ← Q2(S,A) + α[R + γ Q1(S', argmax_a Q2(S',a)) - Q2(S,A)]
```

## Textbook References

- **Sutton & Barto**: Chapter 6 - Temporal-Difference Learning
  - Section 6.1: TD Prediction
  - Section 6.2: Advantages of TD Prediction Methods
  - Section 6.3: Optimality of TD(0)
  - Section 6.4: SARSA: On-Policy TD Control
  - Section 6.5: Q-Learning: Off-Policy TD Control
  - Section 6.6: Expected SARSA
  - Section 6.7: Maximization Bias and Double Learning

- **David Silver's RL Course**:
  - Lecture 4: Model-Free Prediction (TD prediction)
  - Lecture 5: Model-Free Control (SARSA, Q-Learning)

- **CS234 (Stanford)**: Week 5 - TD Learning and Q-Learning

## Implementation Tasks

### Task 1: Cliff Walking - SARSA vs Q-Learning

Implement both SARSA and Q-Learning on the Cliff Walking environment:

**Environment**:
- Grid world with cliff at bottom
- Start: bottom-left, Goal: bottom-right
- Actions: up, down, left, right
- Reward: -1 per step, -100 for falling off cliff

**Expected Observations**:
- **SARSA** (on-policy): Learns safe path away from cliff
- **Q-Learning** (off-policy): Learns optimal risky path along cliff edge
- SARSA's policy accounts for exploration; Q-Learning's assumes optimal execution

**Implementation Steps**:
1. Implement both algorithms with ε-greedy exploration (ε=0.1)
2. Track cumulative rewards per episode
3. Visualize learned policies and value functions
4. Compare convergence speed and final performance

### Task 2: Taxi-v3 Environment

Apply TD control methods to Gymnasium's Taxi-v3:

**Task**: Pickup and drop off passengers in a grid world.

**State**: (taxi_row, taxi_col, passenger_location, destination)

**Actions**: move north/south/east/west, pickup, dropoff

**Implementation**:
1. Start with Q-Learning
2. Implement Expected SARSA
3. Implement Double Q-Learning
4. Compare learning curves and sample efficiency

**Expected Results**:
- All methods should learn near-optimal policy
- Double Q-Learning may show more stable learning
- Expected SARSA may converge faster than SARSA

### Task 3: TD(0) Prediction on Random Walk

Implement TD(0) prediction for a simple random walk:

**Environment**:
- States: 1-2-3-4-5 (start at 3)
- Terminal states: 0 (left), 6 (right)
- Reward: 0 for left terminal, 1 for right terminal
- Random walk: equal probability left/right

**True Values**: [1/6, 2/6, 3/6, 4/6, 5/6]

**Tasks**:
1. Implement TD(0) with different learning rates α
2. Compare with Monte Carlo prediction
3. Measure RMSE vs episodes
4. Visualize convergence

### Task 4: Windy Gridworld

Implement SARSA on the Windy Gridworld (S&B Example 6.5):

**Challenge**: Wind pushes agent upward in some columns.

**Implementation**:
1. Build custom environment with wind effects
2. Implement SARSA
3. Extend to King's moves (8 directions)
4. Implement stochastic wind variant

## Comparison: TD vs MC vs DP

| Property | TD | Monte Carlo | Dynamic Programming |
|----------|----|--------------|--------------------|
| Model Required | No | No | Yes |
| Bootstrapping | Yes | No | Yes |
| Episodic Only | No | Yes | No |
| Online Learning | Yes | No | N/A |
| Bias | Initial | None | None |
| Variance | Medium | High | None |
| Convergence Speed | Fast | Slow | Fast (if model available) |
| Works with Continuing | Yes | No | Yes |

## Detailed Comparison: SARSA vs Q-Learning

| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| Policy Type | On-policy | Off-policy |
| Update | Uses actual next action A' | Uses max over next actions |
| Learning | About policy being followed | About optimal policy |
| Safety | More conservative | More aggressive |
| Exploration Cost | Accounted in values | Not accounted |
| Convergence | To optimal (with GLIE) | To optimal (always) |
| Use Case | Online learning, robots | Offline data, simulation |

**GLIE (Greedy in the Limit with Infinite Exploration)**:
- All state-action pairs visited infinitely often
- Policy converges to greedy policy
- Example: ε-greedy with ε_n → 0 such that Σε_n = ∞

## Advantages of TD Learning

1. **Online and Incremental**: Learn from each step, no waiting for episodes
2. **Lower Variance**: Bootstrap reduces dependence on future randomness
3. **Continuing Tasks**: Works without episodes
4. **Sample Efficient**: Often learns faster than MC in practice
5. **Flexible**: Basis for many advanced algorithms (Actor-Critic, DQN, etc.)

## Practical Tips

### Learning Rate (α)
- Start with α = 0.1 or 0.5
- Decay over time: α_t = α_0 / (1 + decay * t)
- Smaller α for tabular (exact convergence), larger for approximation

### Exploration (ε)
- Start with ε = 0.1 to 0.3
- Decay to ε_min = 0.01 or 0
- Decay schedule: ε_t = ε_min + (ε_0 - ε_min) * exp(-decay * t)

### Initialization
- **Optimistic**: Initialize Q(s,a) high (e.g., 0 or max possible return) to encourage exploration
- **Pessimistic**: Initialize Q(s,a) low if you want cautious early behavior
- **Zero**: Simple default, but no initialization bonus for exploration

### Discount Factor (γ)
- γ = 0.99 (common default for episodic tasks)
- γ = 0.95 (faster learning, less emphasis on distant future)
- γ = 1.0 (only for episodic tasks with guaranteed termination)

### Debugging Tips
1. Check if Q-values are updating (print TD errors)
2. Verify environment rewards are correct
3. Ensure all state-action pairs are visited
4. Monitor learning curves (cumulative reward per episode)
5. Visualize learned policy periodically

## Common Pitfalls

1. **Forgetting to update both S and A**: In SARSA, must update both state and action after each step
2. **Using wrong action in update**: SARSA uses A', Q-Learning uses argmax
3. **Not decreasing α or ε**: May prevent convergence
4. **Initializing Q incorrectly**: Can slow learning significantly
5. **Not handling terminal states**: Q(terminal, a) should be 0

## Questions to Consider

1. Why does TD(0) have lower variance than MC despite bootstrapping from potentially incorrect estimates?
2. When would you prefer SARSA over Q-Learning in practice?
3. How does the cliff walking example illustrate the on-policy vs off-policy distinction?
4. Why is maximization bias a problem and when does it matter most?
5. How do TD methods relate to the Bellman equation?

## Next Steps

After mastering TD learning, you'll be ready for:
- **Week 6**: n-step methods and eligibility traces (TD(λ))
- Understanding the spectrum between TD and MC
- Multi-step bootstrapping
- Credit assignment with eligibility traces

## Connection to Deep RL

TD learning forms the foundation for modern deep reinforcement learning:
- **DQN** (Deep Q-Network): Q-Learning with neural networks
- **A3C** (Asynchronous Advantage Actor-Critic): TD error for advantage estimation
- **TD3** (Twin Delayed DDPG): Double Q-Learning for continuous control
- **SAC** (Soft Actor-Critic): TD learning with entropy regularization

Understanding tabular TD methods is essential before moving to function approximation and deep RL.
