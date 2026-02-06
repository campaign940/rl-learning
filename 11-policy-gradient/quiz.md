# Week 11 Quiz: Policy Gradient Methods

## Question 1: Conceptual Understanding

**Why optimize the policy directly instead of deriving it from a value function? When is this preferable?**

<details>
<summary>Click to reveal answer</summary>

### Answer

Policy-based methods optimize the policy π(a|s; θ) directly rather than learning Q(s,a) and deriving π from it. This is preferable in several scenarios:

**1. Continuous or High-Dimensional Action Spaces**
- Value-based methods (like DQN) require finding argmax_a Q(s,a)
- In continuous spaces, this requires optimization at every step (expensive)
- Policy networks can directly output continuous actions via Gaussian distributions

**2. Stochastic Policies**
- Some problems require inherent stochasticity (e.g., rock-paper-scissors)
- Value-based methods typically result in deterministic policies
- Policy gradient naturally handles stochastic policies

**3. Simpler Policy Representation**
- Sometimes the optimal policy is simpler than the optimal value function
- Example: In a corridor with two exits, policy might just be "go left" or "go right"
- Value function must represent value of every state in the corridor

**4. Better Convergence Properties**
- Policy gradient has guaranteed convergence to local optimum
- Q-learning with function approximation can diverge
- Small changes to θ cause small changes to π (smooth optimization landscape)

**When to Use Policy-Based Methods:**
- Robotics with continuous control (joint angles, velocities)
- Large or continuous action spaces
- When exploration via stochasticity is important
- When convergence guarantees matter

**When Value-Based Might Be Better:**
- Discrete action spaces with few actions
- When sample efficiency is critical (off-policy learning)
- When environment has clear value structure
- Simpler to implement and debug initially

**Hybrid Approach**: Actor-critic methods combine both, getting benefits of each (covered in Week 12).

</details>

---

## Question 2: Mathematical Derivation

**Prove that subtracting a state-dependent baseline b(s) from the return does not change the expected gradient but reduces variance.**

<details>
<summary>Click to reveal answer</summary>

### Answer

We need to prove two things:
1. The baseline doesn't bias the gradient (expected gradient unchanged)
2. It reduces variance

**Part 1: Baseline Doesn't Bias Gradient**

Original policy gradient:
```
∇_θ J(θ) = E_π [∇_θ log π(a|s; θ) Q^π(s,a)]
```

With baseline b(s):
```
∇_θ J_baseline(θ) = E_π [∇_θ log π(a|s; θ) (Q^π(s,a) - b(s))]
```

Expand:
```
∇_θ J_baseline(θ) = E_π [∇_θ log π(a|s; θ) Q^π(s,a)] - E_π [∇_θ log π(a|s; θ) b(s)]
```

The second term:
```
E_π [∇_θ log π(a|s; θ) b(s)] = E_s[b(s) E_a[∇_θ log π(a|s; θ)]]
                                = E_s[b(s) Σ_a π(a|s; θ) ∇_θ log π(a|s; θ)]
                                = E_s[b(s) Σ_a ∇_θ π(a|s; θ)]  (score function)
                                = E_s[b(s) ∇_θ Σ_a π(a|s; θ)]
                                = E_s[b(s) ∇_θ 1]  (probabilities sum to 1)
                                = 0
```

Therefore: ∇_θ J_baseline(θ) = ∇_θ J(θ) (unbiased!)

**Part 2: Baseline Reduces Variance**

Variance of gradient estimate:
```
Var[∇_θ log π(a|s; θ) (Q^π(s,a) - b(s))]
```

Optimal baseline that minimizes variance:
```
b*(s) = E[∇_θ log π(a|s; θ)^2 Q^π(s,a)] / E[∇_θ log π(a|s; θ)^2]
```

This can be shown by taking derivative of variance w.r.t. b(s) and setting to 0.

**Intuitive explanation**:
- Without baseline, actions with Q(s,a) > 0 get reinforced, even if they're below average
- With baseline b(s) ≈ V(s), only better-than-expected actions get reinforced
- This reduces the "noise" in gradient estimates

**Practical choice**: b(s) = V^π(s) is near-optimal and easy to learn alongside policy.

**Empirical demonstration**:
- REINFORCE without baseline: gradient variance ∝ scale of returns
- REINFORCE with V(s) baseline: variance reduced by 10-100x typically
- This translates to 10-100x faster learning in many environments

</details>

---

## Question 3: Algorithm Comparison

**Compare value-based (DQN), policy-based (REINFORCE), and actor-critic methods across: convergence guarantees, variance, and applicability to continuous actions.**

<details>
<summary>Click to reveal answer</summary>

### Answer

| Aspect | DQN (Value-Based) | REINFORCE (Policy-Based) | Actor-Critic |
|--------|-------------------|--------------------------|--------------|
| **Convergence** | Can diverge with function approximation | Guaranteed to local optimum | Guaranteed to local optimum (with caveats) |
| **Variance** | Low (bootstrapping) | Very high (Monte Carlo) | Medium (bootstrapping + policy gradient) |
| **Bias** | Biased (bootstrapping) | Unbiased | Biased (bootstrapping) |
| **Sample Efficiency** | High (off-policy replay) | Low (on-policy Monte Carlo) | Medium (on-policy bootstrapping) |
| **Continuous Actions** | Poor (requires discretization or optimization) | Excellent (natural) | Excellent (natural) |
| **Exploration** | ε-greedy or Boltzmann | Stochastic policy | Stochastic policy |
| **Computational Cost** | High per step (target network) | Low per step | Medium (two networks) |
| **Stability** | Can be unstable | Stable but slow | Can be unstable |

### Detailed Comparison

**1. Convergence Guarantees**

**DQN**:
- Q-learning with function approximation can diverge (deadly triad)
- Mitigations: experience replay, target networks, double Q-learning
- No theoretical guarantees, but works well in practice

**REINFORCE**:
- Guaranteed to converge to local optimum (policy gradient theorem)
- Policy changes smoothly with parameters (Lipschitz continuous)
- May take very long due to high variance

**Actor-Critic**:
- Similar guarantees to REINFORCE (follows policy gradient)
- Critic introduces bias, which can affect convergence
- Typically faster convergence than REINFORCE in practice

**2. Variance vs. Bias Tradeoff**

**DQN (Low Variance, High Bias)**:
- Uses bootstrapping: Q(s,a) ← r + γ max_a' Q(s',a')
- Bias from approximation errors compounds
- Low variance because single-step updates

**REINFORCE (High Variance, No Bias)**:
- Uses full episode returns (Monte Carlo)
- Unbiased estimate of true policy gradient
- Variance from randomness across entire episode

**Actor-Critic (Medium Variance, Medium Bias)**:
- Bootstraps with critic: A(s,a) ≈ r + γV(s') - V(s)
- Reduces variance compared to REINFORCE (uses fewer samples)
- Introduces bias from critic approximation errors

**3. Continuous Action Spaces**

**DQN** (Poor for Continuous):
```python
# Problem: need argmax_a Q(s,a) with a continuous
# Solutions:
# - Discretize (loses precision, curse of dimensionality)
# - Optimization per action (slow: requires solving optimization problem per step)
# - NAF or other special architectures (limited)
```

**REINFORCE** (Excellent for Continuous):
```python
# Natural parameterization with Gaussian policy
mu = policy_network(s)  # Mean of Gaussian
a = mu + sigma * noise  # Sample action
log_prob = -((a - mu)**2) / (2 * sigma**2)  # For gradient
```

**Actor-Critic** (Excellent for Continuous):
```python
# Same as REINFORCE, but with learned critic for advantage
mu = actor(s)
a = mu + sigma * noise
advantage = r + gamma * critic(s') - critic(s)
actor_loss = -log_prob * advantage
```

### When to Use Each

**Use DQN when**:
- Discrete action space with few actions (<100)
- Sample efficiency is critical
- Can afford computational cost
- Off-policy learning is beneficial (e.g., learning from demonstrations)

**Use REINFORCE when**:
- Continuous or very large discrete action spaces
- Guaranteed convergence is important
- Simplicity is valued (easiest to implement)
- Can afford high sample complexity

**Use Actor-Critic when**:
- Need balance of sample efficiency and stability
- Continuous actions with moderate sample budget
- Want faster learning than REINFORCE
- Can handle slightly more complex implementation

### Practical Recommendations

For most modern applications:
1. **Discrete actions**: Start with DQN (or modern variants like Rainbow)
2. **Continuous control**: Start with actor-critic (A2C/PPO for on-policy, SAC for off-policy)
3. **Research/learning**: Implement REINFORCE first to understand policy gradients

**Modern trend**: Actor-critic methods (especially PPO and SAC) have largely superseded pure policy gradient and value-based methods for continuous control.

</details>

---

## Question 4: Application Design

**You have a continuous-action robot arm task (3 joints, continuous torques, goal: reach target position). Design a REINFORCE-based solution. What policy distribution would you use? Specify network architecture, action parameterization, and training procedure.**

<details>
<summary>Click to reveal answer</summary>

### Answer

## Complete REINFORCE Solution for Robot Arm

### Problem Setup
- **State**: 9D vector (3 joint angles, 3 joint velocities, 3D target position relative to end effector)
- **Action**: 3D continuous vector (torque for each joint), typically bounded in [-1, 1] or [-max_torque, max_torque]
- **Reward**: -distance_to_target - 0.01 * ||action||^2 (penalize large torques)
- **Episode**: Terminates after 200 steps or when within 0.05m of target

### 1. Policy Distribution Choice

**Gaussian Policy** (most common for continuous control):
```python
π(a|s; θ) = N(μ(s; θ), Σ)
```

**Parameterization options**:

**Option A: Fixed covariance** (simplest, good starting point)
```python
mu = policy_network(s)  # Neural network outputs mean
sigma = 0.5  # Fixed standard deviation
a = mu + sigma * torch.randn_like(mu)
log_prob = -0.5 * ((a - mu) / sigma)**2 - log(sigma * sqrt(2*pi))
```

**Option B: State-dependent diagonal covariance** (more flexible)
```python
mu, log_sigma = policy_network(s)  # Network outputs both
sigma = torch.exp(log_sigma)  # Ensure positive
a = mu + sigma * torch.randn_like(mu)
log_prob = -0.5 * ((a - mu) / sigma)**2 - log_sigma - log(sqrt(2*pi))
```

**Option C: Squashed Gaussian** (for bounded actions)
```python
mu, log_sigma = policy_network(s)
sigma = torch.exp(log_sigma)
a_raw = mu + sigma * torch.randn_like(mu)
a = torch.tanh(a_raw)  # Bound to [-1, 1]
# Adjust log_prob for tanh transformation (change of variables)
log_prob = gaussian_log_prob - torch.log(1 - a**2 + 1e-6).sum(-1)
```

**Recommendation**: Start with Option A, move to Option B if needed.

### 2. Network Architecture

**Policy Network**:
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=9, action_dim=3, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        # Optional: learnable log_sigma
        self.log_sigma = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = torch.tanh(self.mu_head(x))  # Bounded to [-1, 1]
        sigma = torch.exp(self.log_sigma)  # Ensure positive
        return mu, sigma
```

**Value Network (for baseline)**:
```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim=9, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_head(x)
```

**Design choices**:
- 3 hidden layers with 256 units (sufficient for moderate complexity)
- ReLU activation (standard, works well)
- Separate networks for policy and value (simpler to tune)
- Could use shared trunk with separate heads (more parameter efficient)

### 3. Training Procedure

**Algorithm**:
```python
def train_reinforce(env, policy, value_net, num_episodes=5000):
    policy_optimizer = Adam(policy.parameters(), lr=3e-4)
    value_optimizer = Adam(value_net.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        # 1. Collect trajectory
        states, actions, rewards, log_probs = [], [], [], []
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state)
            mu, sigma = policy(state_tensor)
            dist = Normal(mu, sigma)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            next_state, reward, done, _ = env.step(action.numpy())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        # 2. Compute returns (reward-to-go)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # 3. Normalize returns (important for stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 4. Compute advantages using baseline
        states_tensor = torch.FloatTensor(states)
        values = value_net(states_tensor).squeeze()
        advantages = returns - values.detach()

        # 5. Policy gradient update
        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_optimizer.step()

        # 6. Value network update (MSE loss)
        value_loss = F.mse_loss(values, returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        # 7. Optional: decay exploration (reduce sigma over time)
        if episode % 100 == 0:
            policy.log_sigma.data *= 0.99
```

### 4. Hyperparameters

```python
hyperparameters = {
    'num_episodes': 5000,
    'gamma': 0.99,
    'policy_lr': 3e-4,
    'value_lr': 1e-3,
    'hidden_dim': 256,
    'initial_sigma': 0.5,
    'sigma_decay': 0.99,
    'gradient_clip': 0.5,
    'normalize_advantages': True,
    'entropy_bonus': 0.01  # Optional: add entropy regularization
}
```

### 5. Training Enhancements

**Entropy Regularization** (encourage exploration):
```python
entropy = dist.entropy().sum()
policy_loss = -(log_probs * advantages).mean() - entropy_bonus * entropy.mean()
```

**Reward Shaping** (help learning):
```python
# Dense reward instead of sparse
distance = ||end_effector_pos - target_pos||
reward = -distance  # Continuous signal
reward += 10.0 if distance < 0.05 else 0  # Bonus for reaching target
reward -= 0.01 * ||action||^2  # Penalize large actions
```

**Curriculum Learning** (progressively harder):
```python
# Start with nearby targets, gradually increase distance
target_distance = min(0.5 + episode * 0.0001, 2.0)
```

### 6. Expected Results

- **Episodes to solve**: 2000-4000 (highly variable)
- **Final success rate**: 80-90% within 0.05m of target
- **Key metrics to track**:
  - Average return per episode
  - Success rate (% episodes reaching target)
  - Policy entropy (should stay > 0 for exploration)
  - Gradient norm (watch for vanishing/exploding)
  - Distance to target over episode

### 7. Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| Policy collapses (sigma → 0) | Add entropy bonus, use minimum sigma |
| High variance, no learning | Increase baseline network capacity, normalize advantages |
| Actions saturate at bounds | Check reward scale, adjust sigma |
| Slow learning | Increase learning rate, use reward shaping, curriculum |
| Oscillating near target | Penalize action magnitude, tune sigma |

### Conclusion

This design provides a complete REINFORCE solution for robot arm control. Key aspects:
- Gaussian policy for continuous actions
- Separate policy and value networks with 256-unit hidden layers
- Baseline for variance reduction
- Careful hyperparameter choices (learning rates, exploration schedule)
- Enhancements like entropy bonus and reward shaping

For better performance, consider upgrading to actor-critic (A2C/PPO) once REINFORCE baseline is working.

</details>

---

## Question 5: Critical Analysis

**REINFORCE has notoriously high variance. Beyond baselines, what other techniques reduce variance? Discuss causality, reward-to-go, advantage estimation, and any other variance reduction techniques.**

<details>
<summary>Click to reveal answer</summary>

### Answer

## Comprehensive Variance Reduction Techniques for Policy Gradients

High variance is the primary limitation of REINFORCE. Multiple techniques exist to address this:

### 1. Causality (Future Rewards Don't Affect Past Actions)

**Problem**: Original REINFORCE uses entire episode return:
```python
∇_θ J(θ) = E[Σ_t ∇_θ log π(a_t|s_t) G_0]  # G_0 includes rewards before t!
```

**Solution**: Actions only affect future rewards, not past ones:
```python
∇_θ J(θ) = E[Σ_t ∇_θ log π(a_t|s_t) G_t]  # Only rewards from t onwards
where G_t = Σ_{k=t}^T γ^{k-t} r_{k+1}
```

**Why it helps**: Reduces variance by removing irrelevant rewards that don't depend on action a_t.

**Implementation**:
```python
# Bad: full episode return for all timesteps
G = sum(rewards)
for t in range(T):
    loss += -log_probs[t] * G

# Good: reward-to-go
returns = []
G = 0
for r in reversed(rewards):
    G = r + gamma * G
    returns.insert(0, G)
for t in range(T):
    loss += -log_probs[t] * returns[t]
```

**Variance reduction**: Typically 2-5x reduction.

### 2. Reward-to-Go (Already Discussed Above)

This is the implementation of causality. Also called "on-policy Monte Carlo returns" or "causality-aware returns".

### 3. Baseline Subtraction

**Optimal baseline**: b*(s) that minimizes variance of gradient estimator.

**Derivation**:
```
Var[∇ log π · (Q - b)] = E[(∇ log π)^2 (Q - b)^2] - (E[∇ log π · (Q - b)])^2
                        = E[(∇ log π)^2 (Q - b)^2]  (expected gradient is 0)
```

Taking derivative w.r.t. b and setting to 0:
```
b*(s) = E[(∇ log π)^2 Q] / E[(∇ log π)^2]
     ≈ V(s)  (in practice, state value is near-optimal and easy to learn)
```

**Types of baselines**:
- Constant: b = mean(historical_returns)
- State-dependent: b(s) = V(s) [most common]
- State-action: b(s,a) [but this requires Q, defeats purpose]
- Time-dependent: b(s,t) [for non-stationary settings]

### 4. Advantage Function A(s,a) = Q(s,a) - V(s)

**Intuition**: Instead of "how good is this action in absolute terms" (Q), use "how much better is this action than average" (A).

**Why it reduces variance**: Centers returns around 0, reducing scale of gradient updates.

**Relationship to baseline**: A(s,a) is exactly Q(s,a) - b(s) with optimal baseline.

**Implementation** (requires critic):
```python
# Monte Carlo advantage (REINFORCE with baseline)
A_t = G_t - V(s_t)

# TD advantage (actor-critic, even lower variance)
A_t = r_t + γV(s_{t+1}) - V(s_t)  # One-step TD

# n-step advantage (intermediate)
A_t = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) - V(s_t)
```

### 5. Advantage Normalization

**Technique**: Normalize advantages to zero mean, unit variance per batch:
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Why it helps**:
- Makes scale of updates consistent across different reward scales
- Reduces sensitivity to reward engineering
- Stabilizes learning across different tasks

**Caution**: Can hide issues with reward scale or poor value function.

### 6. Discount Factor γ

**Effect on variance**: Lower γ reduces variance by:
- Shortening effective horizon (fewer rewards to sum)
- Reducing compounding of noise

**Trade-off**:
- Lower γ: lower variance, but ignores long-term rewards (biased towards myopic policies)
- Higher γ: higher variance, but correctly optimizes long-term return

**Practical choice**: γ = 0.99 is standard, but γ = 0.95 or 0.9 can help in long-horizon tasks with high variance.

### 7. Multiple Trajectories per Update

**Technique**: Collect N trajectories before each update, average gradients:
```python
∇_θ J(θ) ≈ (1/N) Σ_i Σ_t ∇_θ log π(a_t^i|s_t^i) A_t^i
```

**Why it helps**: Law of large numbers - variance scales as 1/N.

**Trade-off**: N times more samples per update, but more stable updates.

**Modern approach**: This is the basis of batched algorithms (A2C, PPO).

### 8. Bootstrapping (Actor-Critic Methods)

**Technique**: Replace Monte Carlo returns G_t with TD targets:
```python
# REINFORCE (high variance, unbiased)
G_t = Σ_{k=t}^T γ^{k-t} r_k

# Actor-Critic (lower variance, biased)
G_t ≈ r_t + γV(s_{t+1})
```

**Why it reduces variance**: Single-step randomness instead of summing T random variables.

**Trade-off**: Introduces bias from value function approximation errors.

**Variance reduction**: 10-100x typical, at cost of bias.

### 9. Generalized Advantage Estimation (GAE)

**Technique**: Exponentially-weighted average of k-step advantages:
```python
A_t^GAE(λ) = Σ_{k=0}^∞ (γλ)^k δ_{t+k}
where δ_t = r_t + γV(s_{t+1}) - V(s_t)  # TD error
```

**Intuition**: Interpolates between:
- λ = 0: A_t = δ_t (high bias, low variance, like 1-step TD)
- λ = 1: A_t = G_t - V(s_t) (low bias, high variance, like Monte Carlo)

**Typical choice**: λ = 0.95 balances bias-variance well.

**Why it's powerful**: Smooth trade-off between bias and variance, works across many domains.

### 10. Control Variates (Beyond Simple Baselines)

**Technique**: Use correlated random variables to reduce variance:
```python
∇_θ J(θ) = E[∇_θ log π(a|s) Q(s,a) - f(s)]
where f(s) is chosen such that E[f(s)] is easy to compute
```

**Examples**:
- Action-dependent baselines
- Learned control variates (additional neural networks)
- Off-policy corrections

**Status**: Active research area, not yet standard practice.

### 11. Natural Gradients

**Technique**: Precondition gradient with Fisher information matrix:
```python
∇_θ^natural J(θ) = F^{-1} ∇_θ J(θ)
where F = E[∇_θ log π ∇_θ log π^T]
```

**Why it helps**: Takes into account geometry of policy space, makes updates more stable.

**Computational cost**: Expensive to compute F^{-1} exactly.

**Approximations**:
- Conjugate gradient (TRPO)
- Kronecker-factored (K-FAC)
- Diagonal approximation

**Effect**: More stable updates, less hyperparameter sensitivity (indirectly reduces effective variance).

### 12. Entropy Regularization

**Technique**: Add entropy bonus to objective:
```python
J(θ) = E[Σ_t r_t] + α H(π(·|s_t))
where H(π) = -E[log π(a|s)]
```

**Why it helps variance**: Maintains exploration, prevents premature convergence to deterministic policy.

**Side effect**: Higher entropy → more diverse trajectories → better coverage of state-action space → more stable gradient estimates.

### Summary Table

| Technique | Variance Reduction | Bias Introduced | Computational Cost | Ease of Implementation |
|-----------|-------------------|-----------------|-------------------|------------------------|
| Causality/Reward-to-go | 2-5x | None | Free | Trivial |
| Baseline (V(s)) | 5-20x | None | +1 network | Easy |
| Advantage normalization | 1.5-3x | Slight | Free | Trivial |
| Multiple trajectories (N) | N^0.5x | None | +N samples | Trivial |
| Bootstrapping (Actor-Critic) | 10-100x | Moderate | +1 network | Easy |
| GAE(λ) | 10-50x (tunable) | Tunable | +1 network | Moderate |
| Natural gradient | 2-5x | None | High | Hard |
| Entropy regularization | 1.5-3x | Slight | Free | Easy |

### Practical Recommendations

**Basic REINFORCE (start here)**:
1. Reward-to-go (causality)
2. Learned baseline V(s)
3. Advantage normalization

**Intermediate (actor-critic)**:
4. Bootstrapping with critic
5. GAE with λ=0.95

**Advanced (modern algorithms like PPO)**:
6. Multiple parallel workers
7. Mini-batch updates
8. Entropy bonus
9. Trust region or clipped updates

### Conclusion

Variance reduction is critical for policy gradient methods. The most impactful techniques are:
1. **Causality** (free, 2-5x improvement)
2. **Learned baseline** (easy, 5-20x improvement)
3. **Bootstrapping (actor-critic)** (moderate complexity, 10-100x improvement)
4. **GAE** (modern standard, smooth bias-variance trade-off)

Modern algorithms (PPO, SAC) combine multiple techniques for stable, sample-efficient learning. Pure REINFORCE is rarely used in practice except for pedagogical purposes.

</details>

