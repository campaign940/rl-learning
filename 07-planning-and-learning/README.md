# Week 7: Planning and Learning

## Learning Objectives

By the end of this week, you should be able to:

1. Distinguish between **model-based** and **model-free** reinforcement learning
2. Understand **distribution models** vs **sample models**
3. Implement the **Dyna architecture** that integrates planning and learning
4. Apply **Dyna-Q** algorithm to accelerate learning
5. Understand **prioritized sweeping** for efficient planning
6. Implement **Monte Carlo Tree Search (MCTS)** for planning
7. Recognize when and how to combine model-based and model-free approaches

## Key Concepts

### 1. Models in Reinforcement Learning

A **model** of the environment predicts what will happen next.

**Distribution Model** (perfect model):
```
p(s', r | s, a) = probability of next state s' and reward r given state s and action a
```

Provides complete distribution over all possible outcomes.

**Sample Model** (approximate model):
```
sample(s, a) → (s', r)
```

Generates individual samples from the transition distribution.

**Model Learning**:
- **Supervised learning problem**: Given experience (s, a, r, s'), learn to predict (r, s')
- **Table lookup**: Store actual transitions
- **Function approximation**: Neural network predicts next state and reward

---

### 2. Model-Based vs Model-Free

**Model-Free RL**:
- Learn policy or value function directly from experience
- Examples: Q-Learning, SARSA, Policy Gradient
- No explicit model of environment
- Sample efficient once learned, but learning can be slow

**Model-Based RL**:
- Learn a model of environment dynamics
- Use model for planning (simulate future)
- Examples: Dyna, MCTS, AlphaZero
- Can be more sample efficient (reuse experience)
- Planning requires computational resources

**Comparison**:

| Aspect | Model-Free | Model-Based |
|--------|------------|-------------|
| **Learns** | Value/Policy directly | Environment model |
| **Sample Efficiency** | Lower | Higher |
| **Computation** | Low | Higher (planning) |
| **Generalization** | Limited | Better (reuse model) |
| **Robustness** | More robust | Sensitive to model errors |
| **Examples** | Q-Learning, SARSA | Dyna, MCTS |

---

### 3. Dyna Architecture - Integrating Planning and Learning

Dyna combines:
1. **Direct RL**: Learn from real experience (model-free component)
2. **Planning**: Learn from simulated experience using learned model
3. **Model Learning**: Improve model from real experience

**Dyna Cycle**:
```
1. Take action in real environment
2. Update model from real experience
3. Update value function from real experience (direct RL)
4. Repeat n times (planning):
   a. Sample previously experienced state-action pair
   b. Use model to predict next state and reward
   c. Update value function from simulated experience
```

**Key Insight**: Planning (step 4) allows value function to improve between real environment interactions, dramatically accelerating learning.

---

### 4. Dyna-Q Algorithm

**Tabular Dyna-Q** (combines Q-Learning with planning):

```
Initialize Q(s,a) arbitrarily, Model(s,a) = null for all s,a

For each episode:
    Initialize S

    Loop for each step:
        # (a) Acting
        Choose A from S using ε-greedy policy derived from Q

        # (b) Direct RL (Q-Learning update)
        Take action A, observe R, S'
        Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]

        # (c) Model Learning
        Model(S,A) ← (R, S')  # Store transition

        # (d) Planning (n times)
        Repeat n times:
            S_sim ← random previously observed state
            A_sim ← random action previously taken in S_sim
            (R_sim, S'_sim) ← Model(S_sim, A_sim)
            Q(S_sim, A_sim) ← Q(S_sim, A_sim) + α[R_sim + γ max_a Q(S'_sim, a) - Q(S_sim, A_sim)]

        S ← S'
```

**Parameters**:
- **n**: Number of planning steps per real step (typically 5-50)
- Higher n → more planning → faster learning but more computation

**Advantages**:
- Dramatically faster learning than Q-Learning alone
- Reuses experience efficiently
- Simple to implement

---

### 5. Prioritized Sweeping

**Problem with Random Planning**: Dyna-Q samples random state-action pairs for planning. Many have no useful updates.

**Solution**: Prioritize updates that are likely to change value function significantly.

**Key Idea**: Track which states had large TD errors, plan backward from those.

**Algorithm**:

```
Initialize Q(s,a), Model(s,a), PQueue (priority queue of state-action pairs)

Loop:
    # Real experience
    Take action A, observe R, S'
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
    Model(S,A) ← (R, S')

    # Priority for this transition
    P = |R + γ max_a Q(S',a) - Q(S,A)|
    if P > θ:  # θ is small threshold
        Insert (S,A) into PQueue with priority P

    # Planning
    Repeat n times:
        if PQueue is empty: break
        (S_sim, A_sim) ← PQueue.pop() # Highest priority
        (R_sim, S'_sim) ← Model(S_sim, A_sim)
        Q(S_sim, A_sim) ← Q(S_sim, A_sim) + α[R_sim + γ max_a Q(S'_sim, a) - Q(S_sim, A_sim)]

        # Backward planning: update predecessors
        For each (S_pred, A_pred) predicted to lead to S_sim:
            R_pred ← predicted reward
            P_pred = |R_pred + γ max_a Q(S_sim, a) - Q(S_pred, A_pred)|
            if P_pred > θ:
                Insert (S_pred, A_pred) into PQueue with priority P_pred
```

**Advantages**:
- Much more efficient than random planning
- Focuses computation where it matters
- Particularly effective in large state spaces

---

### 6. Monte Carlo Tree Search (MCTS)

**Application**: Planning in large or continuous state spaces, especially games.

**Core Idea**: Build a search tree incrementally, focusing on promising regions.

**Four Phases per Iteration**:

1. **Selection**: Starting from root, traverse tree using selection policy (e.g., UCT)
2. **Expansion**: When leaf reached, add one or more child nodes
3. **Simulation**: From new node, run random policy until termination
4. **Backpropagation**: Propagate result back up the tree, updating node statistics

**UCT Selection Policy** (Upper Confidence bounds for Trees):
```
a* = argmax_a [Q(s,a) + c √(ln(N(s)) / N(s,a))]
              └─ exploitation ─┘  └─ exploration ─┘
```

Where:
- Q(s,a): Average value of taking action a in state s
- N(s): Visit count for state s
- N(s,a): Visit count for state-action pair
- c: Exploration constant (typically √2)

**Algorithm**:

```
function MCTS(root_state, iterations):
    for i = 1 to iterations:
        node = root

        # 1. Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child()  # UCT

        # 2. Expansion
        if not node.is_terminal():
            node = node.expand()  # Add child

        # 3. Simulation (rollout)
        reward = simulate_random_policy(node.state)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # Return best action from root
    return argmax_a root.children[a].visits
```

**Advantages**:
- Works with huge/continuous state spaces
- Anytime algorithm (better with more iterations)
- No need for explicit value function
- Balances exploration and exploitation naturally

**Applications**:
- AlphaGo/AlphaZero (combined with deep learning)
- Game playing (Go, Chess, Shogi)
- Robotics planning
- General sequential decision making

---

### 7. Model Errors and Robustness

**The Challenge**: Learned models are imperfect. Planning with wrong model can lead to poor policies.

**Types of Model Errors**:

1. **Optimistic Model**: Predicts better outcomes than reality
   - Agent exploits model errors
   - Converges to suboptimal policy
   - Example: Model thinks cliff is safe, agent falls repeatedly

2. **Pessimistic Model**: Predicts worse outcomes than reality
   - Agent misses good opportunities
   - Less dangerous but still suboptimal

**Handling Model Errors**:

1. **Model Uncertainty**: Track confidence in model predictions
2. **Optimism in Face of Uncertainty**: Explore areas with high model uncertainty
3. **Ensemble Models**: Use multiple models, aggregate predictions
4. **Model-Based + Model-Free**: Dyna combines both, model-free provides safety net
5. **Adversarial Planning**: Assume worst-case within model uncertainty

---

### 8. When to Use Model-Based RL

**Prefer Model-Based when**:
- Environment is expensive to interact with (real robots, clinical trials)
- Dynamics are relatively simple and learnable
- Need to plan for multiple goals (model reusable)
- Can simulate cheaply (games, simulators)
- Sample efficiency is critical

**Prefer Model-Free when**:
- Environment is complex, hard to model (pixel observations, high-dimensional)
- Cheap to interact with (simulators)
- Model errors are costly
- Simple policy is sufficient
- Computational resources limited

**Hybrid Approaches** (best of both):
- Dyna: Model-based planning + model-free learning
- Model-based value expansion: Use model for short-horizon lookahead
- AlphaZero: MCTS planning + neural network policy/value

---

## Key Equations

### Dyna-Q Update (Planning Step)
```
# Sample from model
(R, S') ← Model(S, A)

# Q-Learning update using simulated experience
Q(S,A) ← Q(S,A) + α[R + γ max_{a'} Q(S', a') - Q(S,A)]
```

### Prioritized Sweeping Priority
```
P(s,a) = |R + γ max_{a'} Q(S', a') - Q(S,A)|

Higher priority → update sooner
```

### UCT (MCTS Selection)
```
a* = argmax_a [Q(s,a) + c√(ln(N(s)) / N(s,a))]

c ≈ √2 (theoretical optimum)
```

### Model Learning (Table Lookup)
```
Model(s,a) ← (r, s')  # After observing transition (s, a, r, s')
```

## Textbook References

- **Sutton & Barto**: Chapter 8 - Planning and Learning with Tabular Methods
  - Section 8.1: Models and Planning
  - Section 8.2: Dyna: Integrated Planning, Acting, and Learning
  - Section 8.3: When the Model is Wrong
  - Section 8.4: Prioritized Sweeping
  - Section 8.5: Expected vs Sample Updates
  - Section 8.11: Monte Carlo Tree Search

- **David Silver's RL Course**: Lecture 8 - Integrating Learning and Planning
  - [Lecture Slides](https://www.davidsilver.uk/wp-content/uploads/2020/03/dyna.pdf)
  - [Video Lecture](https://www.youtube.com/watch?v=ItMutbeOHtc)

## Implementation Tasks

### Task 1: Dyna-Q on Dyna Maze

Implement Dyna-Q on the classic Dyna Maze (S&B Example 8.1).

**Environment**:
- 9×6 grid world
- Start: top-left
- Goal: top-right (reward +1)
- Obstacles: barriers blocking paths
- All other transitions: reward 0

**Implementation**:
1. Implement standard Q-Learning (n=0)
2. Implement Dyna-Q with n=5, 10, 50
3. Compare episodes to reach goal
4. Measure cumulative reward vs computation time

**Expected Observations**:
- Dyna-Q learns dramatically faster (10-100x speedup)
- Higher n → fewer episodes but more computation per episode
- Optimal n balances learning speed and computational cost

**Extensions**:
- Changing maze (add/remove obstacles mid-training)
- Test Dyna-Q's adaptation

### Task 2: Prioritized Sweeping vs Dyna-Q

Compare random planning (Dyna-Q) with prioritized sweeping.

**Environment**: Dyna Maze or larger grid world

**Implementation**:
1. Implement Dyna-Q as baseline
2. Implement prioritized sweeping with priority queue
3. Track:
   - Episodes to convergence
   - Planning updates used
   - Computation time

**Expected Observations**:
- Prioritized sweeping converges with ~10x fewer updates
- More beneficial in larger state spaces
- Overhead of priority queue management

### Task 3: MCTS for Tic-Tac-Toe

Implement Monte Carlo Tree Search for a simple game.

**Game**: Tic-Tac-Toe (3×3 grid)

**Implementation**:
1. Implement game rules and random player
2. Implement basic MCTS (selection, expansion, simulation, backpropagation)
3. Implement UCT selection policy
4. Vary number of iterations per move
5. Evaluate win rate vs random player

**Expected Observations**:
- With 100 iterations: beats random ~70-80%
- With 1000 iterations: beats random ~90-95%
- With 10000 iterations: near-perfect play
- UCT balances exploration/exploitation effectively

**Extensions**:
- Implement for Connect Four (more complex)
- Add neural network evaluation (mini-AlphaZero)

### Task 4: Model Learning and Planning

Build a simple model learner and planner.

**Environment**: FrozenLake (deterministic version)

**Implementation**:
1. Learn transition model: count(s, a, s') → estimate p(s'|s,a)
2. Learn reward model: average observed rewards
3. Value iteration using learned model
4. Compare with model-free Q-Learning

**Analysis**:
- Sample efficiency (episodes to converge)
- Robustness to model errors
- Effect of model accuracy on policy quality

## Comparison Tables

### Model-Based vs Model-Free

| Property | Model-Free | Model-Based |
|----------|------------|-------------|
| **Learns** | Policy/Values | Environment Model |
| **Sample Efficiency** | Lower | Higher |
| **Computation per Step** | O(1) | O(n) planning steps |
| **Generalization** | Limited | Model reusable |
| **Robustness** | More robust | Sensitive to errors |
| **Memory** | Value table | Model + Value table |
| **Examples** | Q-Learning | Dyna-Q, MCTS |

### Planning Methods

| Method | State Space | Updates | Efficiency | Best For |
|--------|-------------|---------|------------|----------|
| Dyna-Q | Small-Medium | Random | Moderate | Tabular, simple |
| Prioritized Sweeping | Medium-Large | Prioritized | High | Large tabular |
| MCTS | Huge/Continuous | Tree-based | Very High | Games, large spaces |
| Value Iteration | Small | Systematic | Low (needs model) | Known model |

## Advantages and Limitations

### Model-Based Advantages

1. **Sample Efficiency**: Learn more from each real experience
2. **Planning**: Can think ahead without acting
3. **Transfer**: Model reusable for different goals
4. **Exploration**: Model enables intelligent exploration
5. **Safety**: Test policies in simulation before deployment

### Model-Based Limitations

1. **Model Errors**: Wrong model → wrong policy
2. **Computational Cost**: Planning requires computation
3. **Model Complexity**: Hard to learn accurate models for complex environments
4. **Curse of Dimensionality**: Model size grows with state space

### Dyna-Specific Advantages

1. **Hybrid Approach**: Combines model-based and model-free
2. **Simple**: Easy to understand and implement
3. **Flexible**: Can tune planning amount (n parameter)
4. **Robust**: Model-free component provides safety net

## Practical Tips

### Tuning Dyna-Q

**Planning Steps (n)**:
- Start with n=10
- Increase if real experience is expensive
- Decrease if computation is bottleneck
- Typical range: 5-50

**Learning Rate (α)**:
- Same as Q-Learning: 0.1 to 0.5
- Can be slightly lower due to extra updates from planning

**Model Representation**:
- Tabular: Store all observed transitions
- Deterministic: Store single (s,a) → (r,s') mapping
- Stochastic: Store distribution or sample multiple

### Implementing Prioritized Sweeping

**Priority Threshold (θ)**:
- Typical: θ = 0.0001 to 0.01
- Too low: Queue explodes (all states)
- Too high: Miss important updates
- Start with 0.001

**Queue Size**:
- Limit queue size to prevent memory issues
- Keep only top-k priorities
- Typical k = 1000 to 10000

### MCTS Hyperparameters

**Exploration Constant (c)**:
- Theoretical optimum: c = √2 ≈ 1.41
- Games: c = 1.0 to 2.0
- More exploration: increase c
- More exploitation: decrease c

**Iterations**:
- More iterations → better play but slower
- Anytime algorithm: allocate based on time budget
- Typical: 100-10000 per decision

**Rollout Policy**:
- Random: Simple, unbiased
- Heuristic: Faster convergence
- Learned: Best performance (AlphaZero)

## Common Pitfalls

1. **Overfitting to Model**: Planning too much with inaccurate model
2. **Ignoring Model Uncertainty**: Not tracking where model is unreliable
3. **Computational Waste**: Planning on irrelevant states
4. **Model Staleness**: Not updating model from new experience
5. **Memory Explosion**: Storing too many transitions or tree nodes

## Connection to Modern Deep RL

**Model-Based Deep RL**:
- **World Models**: Learn latent dynamics model with neural networks
- **MuZero**: MCTS + learned model + learned value/policy
- **Dreamer**: Plan in learned latent space

**Hybrid Approaches**:
- **MBPO**: Model-Based Policy Optimization (short rollouts from learned model)
- **AlphaZero**: MCTS planning + neural network learning
- **TD-MPC**: Model predictive control with learned model

**Value Expansion**:
- Use model for n-step backup
- Combines model-free value learning with model-based lookahead
- Improves sample efficiency without full planning

## Questions to Consider

1. When is model-based RL worth the added complexity?
2. How do you detect and handle model errors?
3. Why does Dyna-Q use Q-Learning updates instead of SARSA for planning?
4. How does MCTS balance exploration and exploitation?
5. What are the tradeoffs between random planning and prioritized sweeping?

## Next Steps

After Week 7, you have a strong foundation in core RL algorithms! You're ready for:

- **Function Approximation**: Scale to large state spaces (neural networks)
- **Policy Gradient Methods**: Learn policies directly
- **Deep Reinforcement Learning**: DQN, A3C, PPO, SAC
- **Advanced Topics**: Multi-agent RL, Meta-RL, Inverse RL
- **Applications**: Robotics, games, optimization problems

Week 7 completes the fundamentals of tabular RL. You now understand:
- Value-based methods (DP, MC, TD)
- Policy-based methods (covered in later weeks)
- Model-free vs model-based approaches
- How to integrate planning and learning

The Dyna architecture shows that model-based and model-free are not opposing approaches but complementary techniques that can be combined for better performance!
