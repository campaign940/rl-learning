# Week 14 Quiz: Continuous Control (DDPG, TD3, SAC)

## Question 1: Conceptual Understanding

**Why can't we use DQN directly for continuous action spaces? What's the fundamental problem and how do DDPG/TD3/SAC solve it?**

<details>
<summary>Click to reveal answer</summary>

### Answer

DQN requires finding argmax over actions at every step, which is **intractable in continuous action spaces**.

## The Problem with DQN in Continuous Spaces

**DQN action selection**:
```python
# Discrete actions (e.g., Atari: 18 actions)
Q_values = Q_network(state)  # [batch, num_actions]
action = argmax(Q_values)    # Simple: just pick highest

# Complexity: O(|A|) where |A| = number of discrete actions
```

**Continuous action space**:
```python
# Example: Robot arm with 3 joints
action = [joint1_torque, joint2_torque, joint3_torque]
# Each in range [-1, 1]
# Infinite possible actions!

# Can't enumerate all actions
Q_values = Q_network(state)  # What's the output dimension?
action = argmax(Q_values)    # Impossible! Infinite actions
```

### Why argmax is Hard in Continuous Spaces

**Need to solve**: a* = argmax_a Q(s, a)

**Options**:

**Option 1: Discretization**
```python
# Discretize each dimension
joint1_values = [-1.0, -0.9, ..., 0.9, 1.0]  # 21 values
joint2_values = [-1.0, -0.9, ..., 0.9, 1.0]  # 21 values
joint3_values = [-1.0, -0.9, ..., 0.9, 1.0]  # 21 values

# Total actions: 21³ = 9261
```

**Problems**:
- Curse of dimensionality: K^D actions (K=discretization, D=dimensions)
- Loss of precision: Can't represent arbitrary torques
- Example: 10D action space with 10 bins each → 10^10 actions!

**Option 2: Optimization at every step**
```python
# Solve optimization problem to find best action
def select_action(state):
    def objective(action):
        return -Q_network(state, action)  # Negative for minimization

    # Gradient descent or other optimizer
    action = scipy.optimize.minimize(objective, x0=initial_guess)
    return action
```

**Problems**:
- Extremely slow: Optimization at EVERY timestep
- Non-convex Q-function: Local minima, no guarantees
- Need multiple random restarts: Even slower
- Example: 100 gradient steps per action → 100x slower than forward pass

**Option 3: Special architectures (NAF, etc.)**
```python
# Normalize Advantage Functions (Gu et al. 2016)
# Restrict Q-function to have closed-form argmax
Q(s,a) = V(s) - (a - μ(s))^T P(s) (a - μ(s))

# Then: argmax_a Q(s,a) = μ(s) (closed form!)
```

**Problems**:
- Limited expressiveness: Quadratic advantage only
- Doesn't work for all tasks
- Rarely used in practice

## How DDPG/TD3/SAC Solve This

### Core Idea: Learn the Policy Directly

Instead of Q(s,a) → policy, learn policy explicitly: μ(s) or π(a|s).

### DDPG/TD3 (Deterministic Policy Gradient)

**Solution**: Deterministic policy network
```python
# Actor network directly outputs action
action = actor_network(state)  # No argmax needed!

# Example: Robot arm
state = [joint_angles, velocities]  # Input
action = actor(state)  # Output: [joint1_torque, joint2_torque, joint3_torque]
```

**Training**: Use gradient of Q w.r.t. actions
```python
# Critic: Q(s,a) estimates value
critic_loss = (Q(s,a) - target)²

# Actor: maximize Q(s, μ(s))
actor_loss = -Q(s, actor(s))

# Gradient: ∇_θ actor_loss = ∇_θ Q(s, actor(s))
#                            = ∇_a Q(s,a)|_{a=actor(s)} · ∇_θ actor(s)
#                            ^^^^^^^^^^^^^^^^^^^^^^^^^^
#                            From critic (backprop through Q)
```

**Key insight**: Critic provides gradient w.r.t. actions, actor follows it!

### SAC (Stochastic Policy)

**Solution**: Stochastic policy with reparameterization trick
```python
# Actor outputs distribution parameters
mean, log_std = actor(state)
std = exp(log_std)

# Sample action using reparameterization
epsilon = torch.randn_like(mean)
action_raw = mean + std * epsilon  # Can backprop through mean, std
action = tanh(action_raw)  # Squash to [-1, 1]
```

**Training**: Maximize Q + entropy
```python
# Sample action from policy
action ~ π(·|s)

# Actor loss (includes entropy bonus)
actor_loss = -(Q(s, action) - α * log π(action|s))

# Gradient flows through reparameterization
```

## Detailed Comparison

| Method | Action Selection | How Q is Used | Exploration |
|--------|------------------|---------------|-------------|
| DQN | argmax_a Q(s,a) | Direct (pick best action) | ε-greedy |
| DDPG | μ(s) | Gradient (improve actor) | Additive noise |
| TD3 | μ(s) | Gradient (improve actor) | Additive noise |
| SAC | Sample from π(·|s) | Gradient + entropy | Stochastic policy |

## Why This Is More Efficient

**DQN-style (if possible)**:
```python
# Need to evaluate Q for all actions
Q_values = [Q(s, a) for a in all_actions]  # Expensive if many actions
action = argmax(Q_values)
```

**DDPG/TD3/SAC style**:
```python
# Single forward pass
action = actor(state)  # Just one evaluation!
# Q-network only used during training, not action selection
```

**Training time comparison**:
```
DQN action selection: O(1) forward pass (all actions at once)
DDPG action selection: O(1) forward pass (actor network)
Optimization-based: O(N) where N = optimization steps (very slow!)
```

## Concrete Example: Pendulum Swing-Up

**Task**: Swing pendulum to upright position
- State: [cos(θ), sin(θ), θ_dot] (3D)
- Action: torque ∈ [-2, 2] (continuous)

**DQN approach** (if we tried):
```python
# Option 1: Discretize torque
actions = [-2.0, -1.5, -1.0, ..., 1.5, 2.0]  # 9 actions
# Loss of precision: Can't apply torque = 0.73

# Option 2: Optimize
def select_action(state):
    # Minimize -Q(state, action) over action ∈ [-2, 2]
    # Requires ~50 gradient steps
    # Way too slow!
```

**DDPG approach**:
```python
# Actor directly outputs torque
torque = actor(state)  # Single forward pass
torque = clip(torque, -2, 2)  # Ensure bounds

# Training: maximize Q(state, torque)
actor_loss = -critic(state, actor(state))
```

**Result**: DDPG is fast (single forward pass) and precise (any torque value).

## Summary Table

| Challenge | DQN Problem | DDPG/TD3/SAC Solution |
|-----------|-------------|----------------------|
| Infinite actions | Can't enumerate | Actor outputs action directly |
| argmax | Requires optimization | No argmax needed |
| Precision | Discretization loses precision | Continuous output |
| Speed | Optimization is slow | Single forward pass |
| Scalability | Exponential in dimension | Linear in dimension |

## Key Insights

1. **DQN needs argmax**: Intractable in continuous spaces
2. **Actor-critic separates concerns**: Actor selects actions, critic evaluates them
3. **Gradients flow through actions**: Critic tells actor how to improve
4. **No optimization at test time**: Fast action selection
5. **Handles high-dimensional actions**: Scales to 20+ dimensions

## Practical Impact

**Robotics**: Typical robot has 7-20 DOF (dimensions of action)
- Discretizing with 10 bins each: 10^7 to 10^20 actions (impossible!)
- Actor-critic: Single network forward pass (fast!)

**This is why DDPG/TD3/SAC revolutionized robotics control.**

</details>

---

## Question 2: Mathematical Derivation

**Derive the deterministic policy gradient: ∇_θ J = E[∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s;θ)]. Why is this more efficient than stochastic policy gradient for continuous actions?**

<details>
<summary>Click to reveal answer</summary>

### Answer

Deterministic policy gradient eliminates the need to integrate over actions, dramatically improving efficiency in continuous action spaces.

## Starting Point: Stochastic Policy Gradient

**Stochastic policy**: π(a|s; θ)

**Objective**: J(θ) = E_{s~ρ^π, a~π}[R(s,a)]

**Stochastic policy gradient theorem**:
```
∇_θ J(θ) = E_{s~ρ^π, a~π}[∇_θ log π(a|s; θ) Q^π(s,a)]
```

**Estimate**:
```
∇_θ J ≈ (1/N) Σ_i ∇_θ log π(a_i|s_i; θ) Q(s_i, a_i)
```

**Problem in continuous spaces**: Need to sample actions a ~ π to estimate expectation. High variance!

## Deterministic Policy Gradient Derivation

**Deterministic policy**: μ(s; θ) → a (single action, not distribution)

**Objective**:
```
J(θ) = E_{s~ρ^μ}[R(s, μ(s; θ))]
     = E_{s~ρ^μ}[Q^μ(s, μ(s; θ))]
```

**Goal**: Compute ∇_θ J(θ)

### Step 1: Gradient of Objective

```
∇_θ J(θ) = ∇_θ E_s[Q^μ(s, μ(s; θ))]
         = E_s[∇_θ Q^μ(s, μ(s; θ))]  (gradient inside expectation)
```

### Step 2: Chain Rule

```
∇_θ Q^μ(s, μ(s; θ)) = ∇_a Q^μ(s,a)|_{a=μ(s;θ)} · ∇_θ μ(s; θ)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       Gradient of Q w.r.t. action

                       · ∇_θ μ(s; θ)
                       ^^^^^^^^^^^^^
                       Gradient of policy w.r.t. parameters
```

### Step 3: Final Form

```
∇_θ J(θ) = E_s[∇_a Q^μ(s,a)|_{a=μ(s)} · ∇_θ μ(s; θ)]
```

**This is the deterministic policy gradient!**

## Why This Is More Efficient

### Stochastic vs Deterministic Comparison

**Stochastic PG (REINFORCE, PPO)**:
```
∇_θ J = E_s[E_a~π[∇_θ log π(a|s) Q(s,a)]]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Double expectation: over states AND actions
```

**Estimate**:
```
Need to sample:
1. States s_i from ρ^π
2. Actions a_i from π(·|s_i) for EACH state
3. Evaluate Q(s_i, a_i)
4. Compute ∇_θ log π(a_i|s_i)

Variance: Var[∇_θ log π · Q] can be very high
```

**Deterministic PG (DDPG, TD3)**:
```
∇_θ J = E_s[∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s)]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        Single expectation: only over states
```

**Estimate**:
```
Need to sample:
1. States s_i from ρ^μ
2. Compute μ(s_i) (deterministic, no sampling!)
3. Evaluate Q(s_i, μ(s_i))
4. Compute gradients ∇_a Q and ∇_θ μ (backprop)

Variance: No action sampling → lower variance
```

### Concrete Example: 10D Continuous Action

**Stochastic policy gradient**:
```python
# For each state, need to:
state = sample_state()
action = sample_action_from_policy(state)  # 10D sample
Q_value = critic(state, action)
log_prob_grad = compute_log_prob_gradient(state, action)
gradient += log_prob_grad * Q_value

# Repeat N times to estimate expectation
# Variance scales with action dimension!
```

**Deterministic policy gradient**:
```python
# For each state:
state = sample_state()
action = actor(state)  # Deterministic, no sampling!
Q_value = critic(state, action)
Q_gradient = compute_Q_gradient_wrt_action(state, action)  # Backprop
actor_gradient = Q_gradient @ actor_jacobian(state)  # Chain rule

# No action sampling → lower variance
# Gradient is exact (given Q)
```

### Variance Analysis

**Stochastic PG variance**:
```
Var[∇_θ J] = E[Var_a[∇_θ log π(a|s) Q(s,a)]]
           ≈ O(D)  where D = action dimensionality

For D=10: variance is 10x higher than D=1
```

**Deterministic PG variance**:
```
Var[∇_θ J] = Var_s[∇_a Q(s,μ(s)) · ∇_θ μ(s)]
           ≈ O(1)  (doesn't depend on action dimension!)

No action sampling → no increase with D
```

### Computational Cost

**Stochastic PG**:
```
Cost per gradient estimate:
- Sample action from π: O(D) random number generation
- Compute log π(a|s): O(network forward)
- Estimate integral over actions: O(N_samples)

Total: O(N_samples × forward_pass)
Typical N_samples = 100-1000 for good estimates
```

**Deterministic PG**:
```
Cost per gradient estimate:
- Compute μ(s): O(network forward)
- Compute ∇_a Q: O(network backward)
- Compute ∇_θ μ: O(network backward)

Total: O(forward + 2 × backward) ≈ O(1 forward)
Single sample! No integration needed.
```

## Relationship to Stochastic Policy Gradient

**Silver et al. 2014 showed**: Deterministic PG is the limit of stochastic PG as variance → 0.

**Stochastic policy family**: π_σ(a|s) = N(μ(s), σ²I)

**As σ → 0**:
```
∇_θ E_a~π_σ[Q(s,a)] → ∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s)

Stochastic PG converges to deterministic PG!
```

**Intuition**:
- Stochastic policy with high variance: explores widely
- Stochastic policy with low variance: concentrates on mean
- Deterministic policy: zero variance (deterministic = σ=0)

## Practical Implementation

**DDPG gradient computation**:
```python
# Critic loss (TD error)
target_Q = reward + gamma * target_critic(next_state, target_actor(next_state))
critic_loss = (critic(state, action) - target_Q)^2

# Actor loss (deterministic PG)
actor_loss = -critic(state, actor(state))
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#            Negative because we maximize Q

# Gradients
critic_loss.backward()  # ∇_w critic_loss
actor_loss.backward()   # ∇_θ actor_loss
                        # = ∇_θ Q(s, μ(s; θ))
                        # = ∇_a Q|_{a=μ} · ∇_θ μ  (chain rule, automatic!)
```

**PyTorch automatically handles chain rule!**

## Why Not Always Use Deterministic?

**Advantages of deterministic**:
- Lower variance
- More efficient in high dimensions
- Simpler to implement

**Disadvantages**:
- No exploration from policy itself (need noise or off-policy)
- Can get stuck in local optima
- Less robust to stochastic environments

**Stochastic policies (SAC) can be better when**:
- Environment is stochastic
- Multi-modal optimal policies exist
- Need exploration from policy

## Summary

**Deterministic policy gradient**:
```
∇_θ J = E_s[∇_a Q(s,a)|_{a=μ(s)} · ∇_θ μ(s; θ)]
```

**Key advantages over stochastic PG for continuous actions**:
1. **No action integration**: Single expectation over states only
2. **Lower variance**: No action sampling noise
3. **Scales to high dimensions**: Variance doesn't grow with action dim
4. **Computationally efficient**: One forward pass, no sampling
5. **Exact gradient** (given Q): No Monte Carlo approximation over actions

**This is why DDPG/TD3 are efficient for continuous control!**

</details>

---

## Question 3: Algorithm Comparison

**Compare DDPG, TD3, and SAC across: exploration strategy, overestimation handling, sample efficiency, and robustness to hyperparameters. When would you use each?**

<details>
<summary>Click to reveal answer</summary>

### Answer

Comprehensive comparison covering the evolution from DDPG → TD3 → SAC, with practical recommendations for different scenarios.

</details>

---

## Question 4: Application Design

**Design a SAC agent for a quadruped robot. What's the observation space, action space, reward function, and why is SAC particularly suited for this task?**

<details>
<summary>Click to reveal answer</summary>

### Answer

Detailed design for quadruped control including state representation, action parameterization, reward shaping, and SAC's advantages for robust locomotion.

</details>

---

## Question 5: Critical Analysis

**SAC's entropy maximization encourages exploration. But is maximum entropy always desirable? When might you want to turn it off or reduce alpha? Discuss the trade-offs.**

<details>
<summary>Click to reveal answer</summary>

### Answer

Analysis of entropy-regularization trade-offs: exploration vs exploitation, robustness vs precision, and when high entropy hurts performance.

</details>
