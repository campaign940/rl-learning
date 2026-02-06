# Week 2: Markov Decision Processes

## Learning Objectives

- [ ] Understand the formal MDP definition and components
- [ ] Master state-value and action-value functions
- [ ] Derive and apply Bellman expectation equations
- [ ] Understand Bellman optimality equations
- [ ] Comprehend the role of return and discounting

## Key Concepts

### 1. MDP Formalism

A Markov Decision Process is a tuple (S, A, P, R, γ) where:

- **S**: Set of states (state space)
- **A**: Set of actions (action space)
- **P**: State transition probability function
  - P(s'|s,a) = Pr{S_{t+1} = s' | S_t = s, A_t = a}
  - Or more generally: p(s',r|s,a) = Pr{S_{t+1}=s', R_{t+1}=r | S_t=s, A_t=a}
- **R**: Reward function
  - r(s,a) = E[R_{t+1} | S_t = s, A_t = a]
  - Or: r(s,a,s') = E[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']
- **γ**: Discount factor, γ ∈ [0,1]

**The Markov Property**:
```
Pr{S_{t+1}=s', R_{t+1}=r | S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0}
  = Pr{S_{t+1}=s', R_{t+1}=r | S_t, A_t}
```

The future depends only on the present state and action, not on the history. The state is a sufficient statistic of the history.

**Policy**: A policy π is a mapping from states to probability distributions over actions
- π(a|s) = Pr{A_t = a | S_t = s}
- Deterministic policy: π(s) = a
- Stochastic policy: π(a|s) gives probability of taking action a in state s

### 2. Return and Discounting

**Return** (G_t): The cumulative discounted reward from time t onward

**Episodic tasks** (with terminal state):
```
G_t = R_{t+1} + R_{t+2} + ... + R_T
```

**Continuing tasks** (no terminal state):
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ_{k=0}^∞ γ^k R_{t+k+1}
```

**Why discount (γ < 1)?**
1. **Mathematical convenience**: Ensures infinite sums converge
2. **Uncertainty about future**: Future rewards are less certain
3. **Preference for immediate reward**: Matches human/animal behavior
4. **Avoiding infinite returns**: Prevents unbounded values
5. **Encourages bounded solutions**: Even with infinite horizons

**Recursive property of return**:
```
G_t = R_{t+1} + γG_{t+1}
```

This recursive relationship is the foundation of Bellman equations.

### 3. Value Functions

**State-Value Function** v_π(s):
The expected return starting from state s and following policy π thereafter
```
v_π(s) = E_π[G_t | S_t = s]
        = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s]
```

**Action-Value Function** q_π(s,a):
The expected return starting from state s, taking action a, then following policy π
```
q_π(s,a) = E_π[G_t | S_t = s, A_t = a]
          = E_π[Σ_{k=0}^∞ γ^k R_{t+k+1} | S_t = s, A_t = a]
```

**Relationship between v and q**:
```
v_π(s) = Σ_a π(a|s) q_π(s,a)
```

The value of a state is the expected value of actions from that state under the policy.

### 4. Bellman Equations

**Bellman Expectation Equation for v_π**:
```
v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
```

Interpretation: The value of a state is the expected immediate reward plus the discounted value of the next state.

**Bellman Expectation Equation for q_π**:
```
q_π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γ Σ_{a'} π(a'|s')q_π(s',a')]
```

**Bellman Optimality Equation for v_***:
```
v_*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv_*(s')]
        = max_a q_*(s,a)
```

where v_*(s) = max_π v_π(s) is the optimal state-value function.

**Bellman Optimality Equation for q_***:
```
q_*(s,a) = Σ_{s',r} p(s',r|s,a)[r + γ max_{a'} q_*(s',a')]
```

where q_*(s,a) = max_π q_π(s,a) is the optimal action-value function.

**Optimal Policy**:
Any policy π_* that satisfies:
```
π_*(s) = argmax_a q_*(s,a)
```
is an optimal policy. There may be multiple optimal policies, but they all share the same v_* and q_*.

### 5. Backup Diagrams

**Visual representation of Bellman equations**:

For v_π:
```
    s
   /|\
  / | \  π(a|s)
 a  a  a
 |  |  |
s' s' s'  (successor states)
```

For q_π:
```
    s
    |
    a
   /|\
  / | \  p(s'|s,a)
s' s' s'
```

Each node represents a state or state-action pair. The backup is the update of a value based on successor values.

## Textbook References

- Sutton & Barto Chapter 3: Finite Markov Decision Processes
  - 3.1: The Agent-Environment Interface
  - 3.2: Goals and Rewards
  - 3.3: Returns and Episodes
  - 3.4: Unified Notation for Episodic and Continuing Tasks
  - 3.5: Policies and Value Functions
  - 3.6: Optimal Policies and Optimal Value Functions
- David Silver Lecture 2: Markov Decision Processes
- CS234 Week 2: How to Act Given Know How the World Works (MDPs)

## Implementation Tasks

Implement a GridWorld environment from scratch:

### GridWorld Specification

**Environment**:
- 4x4 grid
- States: 16 positions (0,0) to (3,3)
- Terminal states: (0,0) and (3,3)
- Actions: {UP, DOWN, LEFT, RIGHT}
- Rewards: -1 for all transitions (encouraging shortest path)
- Transitions: Deterministic (if you try to move off grid, stay in place)

**Tasks**:

1. **Environment Implementation**:
   ```python
   class GridWorld:
       def __init__(self):
           # Initialize grid, terminal states

       def step(self, state, action):
           # Return next_state, reward, done

       def get_all_states(self):
           # Return list of all states

       def get_possible_actions(self, state):
           # Return valid actions from state
   ```

2. **Policy Representation**:
   - Implement uniform random policy: π(a|s) = 1/4 for all non-terminal states
   - Implement deterministic policy storage

3. **Value Function Computation**:
   - Manual calculation of v_π for uniform random policy
   - Verify using the Bellman equation
   - Visualize value function as a 4x4 grid

4. **Optimal Policy Finding** (preview for next week):
   - What is v_* for this GridWorld?
   - What is π_*?

## Key Equations

**Return (Recursive Form)**:
```
G_t = R_{t+1} + γG_{t+1}
```

**Bellman Expectation Equation for v_π**:
```
v_π(s) = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
```

**Simplified form** (when rewards are deterministic):
```
v_π(s) = Σ_a π(a|s) Σ_{s'} p(s'|s,a)[r(s,a,s') + γv_π(s')]
```

**Bellman Expectation Equation for q_π**:
```
q_π(s,a) = Σ_{s',r} p(s',r|s,a)[r + γ Σ_{a'} π(a'|s')q_π(s',a')]
```

**Bellman Optimality Equation for v_***:
```
v_*(s) = max_a Σ_{s',r} p(s',r|s,a)[r + γv_*(s')]
```

**Bellman Optimality Equation for q_***:
```
q_*(s,a) = Σ_{s',r} p(s',r|s,a)[r + γ max_{a'} q_*(s',a')]
```

**Matrix Form** (for linear systems):
For a fixed policy π and finite state space, the Bellman expectation equation can be written as:
```
v_π = R_π + γP_π v_π
```
where v_π is a vector of state values, R_π is the expected immediate reward vector, and P_π is the state transition matrix under policy π.

Solving for v_π:
```
v_π = (I - γP_π)^{-1} R_π
```

This direct solution is only practical for small state spaces.

## Review Questions

1. **Why is the Markov property important for MDPs?**
   - It allows us to make decisions based only on the current state
   - Without it, we'd need to consider the entire history
   - It enables the recursive structure of Bellman equations
   - It's what makes the problem tractable

2. **What is the difference between v_π(s) and v_*(s)?**
   - v_π(s): Expected return following a specific policy π from state s
   - v_*(s): Maximum expected return achievable from state s over all policies
   - v_*(s) ≥ v_π(s) for all s and all π
   - v_*(s) = v_π*(s) for optimal policy π*

3. **How does the discount factor γ affect the optimal policy?**
   - γ → 0: Only immediate rewards matter (myopic behavior)
   - γ → 1: Future rewards valued almost equally (far-sighted behavior)
   - Different γ values can lead to different optimal policies
   - Example: γ=0 might prefer small immediate reward over larger delayed reward
   - γ < 1 necessary for convergence in infinite-horizon problems

4. **What is the relationship between Bellman expectation and optimality equations?**
   - Expectation: For a given policy π, recursive definition of v_π
   - Optimality: For the optimal policy, recursive definition of v_*
   - Expectation uses expectation over actions (Σ_a π(a|s))
   - Optimality uses max over actions (max_a)
   - Optimality equations are the fixed point of the Bellman optimality operator

5. **Can you solve an MDP if you know the Bellman optimality equation?**
   - Theoretically yes: v_* is the unique solution to the Bellman optimality equation
   - Once you have v_*, the optimal policy is greedy: π*(s) = argmax_a q_*(s,a)
   - Practically challenging: Non-linear system of equations
   - Direct solution only feasible for small state spaces
   - Next week: Dynamic Programming methods to solve iteratively

## Next Steps

After completing this week:
- Move to Week 3: Dynamic Programming
- Learn algorithms to compute v_π and v_* efficiently
- Implement Policy Iteration and Value Iteration
- Understand the principle of Generalized Policy Iteration (GPI)
