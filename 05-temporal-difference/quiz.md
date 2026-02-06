# Week 5 Quiz: Temporal-Difference Learning

Test your understanding of TD learning methods with these questions.

## Question 1: Understanding Bootstrapping

**What is bootstrapping in the context of reinforcement learning, and how does TD learning use it? Why is bootstrapping both an advantage and a disadvantage?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Definition of Bootstrapping**:

Bootstrapping means updating an estimate based on other estimates rather than only on actual outcomes. In RL, it means using the current value function estimate V(s') to update V(s), rather than waiting for the actual return.

**How TD Learning Uses Bootstrapping**:

TD(0) update:
```
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
                           ↑
                    bootstrapped estimate
```

Instead of the actual return:
```
G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T
```

TD uses the one-step lookahead:
```
TD target = R_{t+1} + γV(S_{t+1})
```

Where V(S_{t+1}) is an **estimate** (not the true value), serving as a proxy for all future rewards.

**Visual Comparison**:

```
Monte Carlo (no bootstrapping):
S_t → R_{t+1} → R_{t+2} → ... → R_T
└──────── wait for all actual rewards ─────┘

TD(0) (bootstrapping):
S_t → R_{t+1} → V(S_{t+1})
└── one actual reward + estimate ──┘

Dynamic Programming (full bootstrapping):
S_t → Σ p(s',r|s,a)[r + γV(s')]
└──── all estimates, no samples ────┘
```

---

**Advantages of Bootstrapping**:

1. **Online Learning**:
   - Update after every step
   - Don't need to wait for episode termination
   - Can learn during an episode and adjust behavior

2. **Works for Continuing Tasks**:
   - No episodes required
   - Suitable for infinite-horizon problems
   - Can't use MC (needs complete episodes)

3. **Lower Variance**:
   - Only one random reward R_{t+1}
   - Future uncertainty captured in estimate V(S_{t+1})
   - MC has variance from all future rewards

   Mathematically:
   ```
   Var[G_t] = Var[R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...]
            = Var[R_{t+1}] + γ²Var[R_{t+2}] + ... (if independent)
            = much higher

   Var[R_{t+1} + γV(S_{t+1})] = Var[R_{t+1}]  (V is deterministic given S_{t+1})
                                = lower
   ```

4. **Faster Learning** (often):
   - Lower variance means more consistent updates
   - Information propagates backward faster
   - Can learn from incomplete episodes

5. **Computational Efficiency**:
   - O(1) computation per step
   - No need to store or process long trajectories

---

**Disadvantages of Bootstrapping**:

1. **Initial Bias**:
   - V(S_{t+1}) starts incorrect
   - Early updates based on wrong estimates
   - Bias propagates through value function

   Example:
   ```
   True: V*(s) = 10
   Initial: V(s) = 0

   First update uses V(s') = 0 (wrong!)
   Gradually corrects over many updates
   ```

2. **Slower Propagation** (in some cases):
   - Values propagate one step at a time
   - Terminal rewards take many episodes to reach initial states
   - MC would get terminal reward to start state in one episode

3. **Sensitive to Initialization**:
   - Poor initialization can slow learning
   - Bias persists longer with small learning rates
   - MC eventually overcomes any initialization

4. **Convergence Depends on Learning Rate**:
   - Need appropriate α schedule
   - Too large: oscillation; too small: slow learning
   - MC averages returns (simple, robust)

5. **Not Guaranteed Unbiased**:
   - MC is unbiased (averages actual returns)
   - TD is biased toward initial estimates
   - Bias reduces but may not fully vanish

---

**When Bootstrapping Helps Most**:

1. **Long Episodes**: TD updates many times before MC updates once
2. **High Variance Environments**: TD's lower variance is crucial
3. **Continuing Tasks**: Only option (MC doesn't apply)
4. **Quick Policy Improvement**: Need to adjust behavior during episode
5. **Smooth Value Functions**: Bootstrap estimates are reasonable proxies

**When Bootstrapping Hurts**:

1. **Function Approximation**: Can cause divergence (deadly triad)
2. **Poor Initialization**: Wrong bootstrapped values mislead learning
3. **Highly Stochastic**: Bootstrap estimates unreliable
4. **Short Episodes**: MC's variance isn't as problematic
5. **Off-Policy with Divergence Risk**: Bootstrap + off-policy + FA = danger

---

**The Bias-Variance Tradeoff**:

Bootstrapping is fundamentally a bias-variance tradeoff:

| Method | Bias | Variance | Tradeoff |
|--------|------|----------|----------|
| MC | Zero | High | All variance, no bias |
| TD | Initially high → low | Low | Small bias, much lower variance |
| DP | Zero (if model correct) | Zero | No sampling involved |

In practice, **lower variance usually wins** because:
- Faster learning from consistent updates
- Bias decreases as estimates improve
- Sample efficiency is often the bottleneck

---

**Key Insight**:

Bootstrapping is essentially "pulling yourself up by your bootstraps" - using current (imperfect) knowledge to improve that knowledge. It's remarkably effective because:

1. Errors average out over many updates
2. Local value estimates are often reasonable
3. The Bellman equation provides a consistency target
4. Lower variance accelerates convergence despite bias

The TD algorithm is converging toward a fixed point where V(s) = E[R + γV(S')] for all s, which is exactly the Bellman equation. Bootstrapping exploits this self-consistency property.

</details>

---

## Question 2: Deriving TD(0) from Bellman Equation

**Derive the TD(0) update rule from the Bellman equation for v_π. Show how the TD error δ_t relates to the Bellman error.**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Starting Point: Bellman Equation for v_π**

The Bellman equation for the state value function under policy π is:

```
v_π(s) = E_π[R_{t+1} + γv_π(S_{t+1}) | S_t = s]
       = Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γv_π(s')]
```

This is the **consistency condition** that the true value function must satisfy.

---

**Step 1: Sample-Based Estimation**

Since we don't have the model p(s',r|s,a), we can't compute the expectation directly. Instead, we take a **sample**:

Following policy π, we observe:
- Current state: S_t
- Action: A_t ~ π(·|S_t)
- Reward: R_{t+1}
- Next state: S_{t+1}

The sample estimate of the expectation is:
```
v_π(S_t) ≈ R_{t+1} + γv_π(S_{t+1})
```

This is a **noisy unbiased sample** of the true expectation.

---

**Step 2: Using Current Estimate**

We don't know the true v_π(S_{t+1}), so we use our current estimate V(S_{t+1}):

```
v_π(S_t) ≈ R_{t+1} + γV(S_{t+1})
```

The right side is called the **TD target**.

---

**Step 3: Defining the TD Error**

The **TD error** (δ_t) measures how much our current estimate V(S_t) differs from the TD target:

```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
     └────── TD target ──────┘   └─ current estimate ─┘
```

If δ_t > 0: We underestimated V(S_t)
If δ_t < 0: We overestimated V(S_t)
If δ_t = 0: Our estimate is consistent (locally satisfies Bellman equation)

---

**Step 4: Incremental Update**

To move V(S_t) toward the TD target, we use a **gradient descent** style update:

```
V(S_t) ← V(S_t) + α · δ_t
       = V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

Where α ∈ (0,1] is the learning rate (step size).

This is the **TD(0) update rule**.

---

**Alternative Derivation: Moving Average**

We can also think of it as a moving average toward the TD target:

```
New estimate ← Old estimate + α(Target - Old estimate)

V(S_t) ← V(S_t) + α[(R_{t+1} + γV(S_{t+1})) - V(S_t)]
       = V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
       = (1-α)V(S_t) + α[R_{t+1} + γV(S_{t+1})]
```

This is a weighted average:
- Weight (1-α) on the old estimate
- Weight α on the new target

---

**Relationship to Bellman Error**

The **Bellman error** for a value function V is:

```
BE(s) = V(s) - [E_π[R_{t+1} + γV(S_{t+1}) | S_t = s]]
      = V(s) - Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
```

This measures how much V violates the Bellman equation.

The TD error is a **sample-based estimate** of the negative Bellman error:

```
δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)
    = -(V(S_t) - [R_{t+1} + γV(S_{t+1})])
    ≈ -BE(S_t)  (in expectation)
```

Taking expectation:
```
E[δ_t | S_t = s] = E[R_{t+1} + γV(S_{t+1}) | S_t = s] - V(s)
                 = [Bellman right side] - [Bellman left side]
                 = -BE(s)
```

---

**Why This Works: Convergence Intuition**

The TD(0) algorithm performs **stochastic gradient descent** on the mean squared Bellman error:

Objective:
```
J = E_π[(V(S_t) - [R_{t+1} + γV(S_{t+1})])²]
```

Gradient with respect to V(S_t):
```
∂J/∂V(S_t) = 2(V(S_t) - [R_{t+1} + γV(S_{t+1})])
           = -2δ_t
```

Gradient descent update:
```
V(S_t) ← V(S_t) - (α/2) · (∂J/∂V(S_t))
       = V(S_t) - (α/2) · (-2δ_t)
       = V(S_t) + α · δ_t
```

This is exactly the TD(0) update!

---

**Fixed Point Analysis**

At convergence, the expected update should be zero:

```
E[δ_t | S_t = s] = 0 for all s

E[R_{t+1} + γV(S_{t+1}) | S_t = s] = V(s)
```

This is exactly the Bellman equation! So TD(0) converges to a fixed point that satisfies the Bellman equation, which is v_π.

---

**Comparison: TD vs MC Update Derivation**

**Monte Carlo**:
```
Target: G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
Update: V(S_t) ← V(S_t) + α[G_t - V(S_t)]
```

MC directly estimates E[G_t], which equals v_π(s) by definition.

**TD(0)**:
```
Target: R_{t+1} + γV(S_{t+1})  (substitute V for future returns)
Update: V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
```

TD uses the Bellman equation structure to substitute current estimates for actual future returns.

---

**Key Insights**:

1. **TD exploits Bellman equation structure**: Uses self-consistency rather than waiting for actual returns

2. **TD error is a local consistency check**: Measures violation of Bellman equation at current state

3. **Convergence to fixed point**: Repeated updates drive expected TD error to zero, which means Bellman equation is satisfied

4. **Bootstrap enables online learning**: Don't need complete trajectories, just one-step transitions

5. **Stochastic approximation**: TD(0) is stochastic gradient descent on a well-defined objective

---

**Mathematical Summary**:

```
Bellman Equation:     v_π(s) = E_π[R + γv_π(S') | S=s]
Sample Estimate:      v_π(s) ≈ R + γv_π(S')
Use Current Estimate: V(s) ≈ R + γV(S')
TD Error:            δ = [R + γV(S')] - V(s)
TD Update:           V(s) ← V(s) + α·δ
```

This derivation shows that TD learning is a principled algorithm rooted in the fundamental Bellman equation, using sampling and bootstrapping to enable online, model-free learning.

</details>

---

## Question 3: SARSA vs Q-Learning on Cliff Walking

**Compare SARSA and Q-Learning on the Cliff Walking problem. Why do they find different paths? Draw both paths and explain the fundamental difference in what each algorithm learns.**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Cliff Walking Environment**:

```
┌─────────────────────┐
│ S . . . . . . . . . │  Row 3 (safe)
│ . . . . . . . . . . │  Row 2 (safe)
│ . . . . . . . . . . │  Row 1 (safe)
│ S C C C C C C C C G │  Row 0: Start(S), Cliff(C), Goal(G)
└─────────────────────┘
```

- Start: Bottom-left
- Goal: Bottom-right
- Cliff: Bottom row (except start and goal)
- Actions: up, down, left, right
- Rewards: -1 per step, -100 for falling off cliff
- Falling resets to start

---

**SARSA Path (Safe Path)**:

```
┌─────────────────────┐
│ → → → → → → → → ↓ . │  Goes up first
│ . . . . . . . . ↓ . │
│ . . . . . . . . ↓ . │
│ ↑ C C C C C C C C G │  Avoids cliff
└─────────────────────┘
```

SARSA learns to go:
1. Up (away from cliff)
2. Right (across top of grid)
3. Down (to goal)

Total steps: ~20, Never falls off cliff during optimal execution

---

**Q-Learning Path (Risky Path)**:

```
┌─────────────────────┐
│ . . . . . . . . . . │
│ . . . . . . . . . . │
│ . . . . . . . . . . │
│ S → → → → → → → → G │  Right along cliff edge
└─────────────────────┘
```

Q-Learning learns to go:
- Right (directly along cliff edge)

Total steps: ~12, Optimal but risky

---

**Why The Difference?**

The key is **on-policy vs off-policy** learning:

**SARSA (On-Policy)**:
```
Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
                              ↑
                         actual action taken
```

- Learns about the policy it's **actually following** (ε-greedy)
- ε-greedy means occasionally taking random actions
- If agent is near cliff and takes random action, it might fall
- Q-values reflect: "Expected return if I follow ε-greedy policy"

**Near-cliff Q-values under ε-greedy**:
```
State = (row 0, col 5)
Q(s, right) = (1-ε)×(-1) + ε×(-100) = -1 - 10 = -11 (with ε=0.1)
                  ↑           ↑
              greedy      random (might go down into cliff)
```

The possibility of random exploration into the cliff makes near-cliff states have lower Q-values. So SARSA learns: "Stay away from cliff because I might accidentally fall while exploring."

---

**Q-Learning (Off-Policy)**:
```
Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
                           ↑
                      best possible action
```

- Learns about the **optimal policy** regardless of what it's following
- Assumes optimal actions will be taken (no exploration errors)
- Q-values reflect: "Expected return if I act optimally from here"

**Near-cliff Q-values under optimal policy**:
```
State = (row 0, col 5)
Q(s, right) = -1 (if always taking optimal action: right)
```

Q-Learning ignores exploration risk because it assumes optimal execution. So it learns: "Cliff edge is fine if you always act optimally."

---

**Visual Comparison of Learning**:

**SARSA's Perspective**:
```
"I'm following ε-greedy policy. Near cliff, there's 10% chance I randomly
 step into cliff (R=-100). Even though straight path is shorter, the
 risk-adjusted value is lower. Safer to go around."
```

**Q-Learning's Perspective**:
```
"I'm learning optimal Q*. Optimally, I'd never step into cliff. Straight
 path is 12 steps, safe path is 20 steps. Q*(s,right) > Q*(s,up) near
 cliff. I learn the risky path is optimal."
```

---

**Detailed Q-Value Comparison**:

**State: One step right of start (row 0, col 1), just before cliff**

SARSA Q-values:
```
Q(s, up)    = -20 (safe, long path)
Q(s, right) = -30 (risky, might fall while exploring)
Q(s, left)  = -40 (goes backward)
Q(s, down)  = -100 (cliff!)

Policy: Choose up (safer expected return under exploration)
```

Q-Learning Q-values:
```
Q(s, up)    = -20 (safe, long path)
Q(s, right) = -12 (optimal path, no exploration penalty)
Q(s, left)  = -40 (goes backward)
Q(s, down)  = -100 (cliff!)

Policy: Choose right (optimal path assuming perfect execution)
```

---

**During Training**:

**SARSA**:
- Initially explores both paths
- Falls off cliff many times during exploration
- Q-values for near-cliff states become very negative
- Gradually learns safe path has higher value under ε-greedy
- Converges to safe path

**Q-Learning**:
- Initially explores both paths
- Falls off cliff many times during exploration
- Q-values ignore these falls (off-policy correction)
- Learns optimal path based on max future value
- Converges to risky path
- **But agent still falls during training!** (following ε-greedy for exploration)

---

**After Training (with ε=0, pure greedy)**:

**SARSA**: Follows safe path (because that's what it learned under ε-greedy)

**Q-Learning**: Follows risky path (optimal path it learned)
- **Key**: With ε=0, Q-Learning path is actually safe!
- The risk only exists during training with exploration

---

**Cumulative Reward During Training**:

**SARSA**:
- Higher cumulative reward during training
- Fewer falls (learns to avoid cliff faster)
- More conservative

**Q-Learning**:
- Lower cumulative reward during training
- More falls (explores risky regions more)
- Finds shorter path eventually

---

**Which is Better?**

**It depends on your objective:**

**Prefer SARSA when**:
- Training performance matters (real robot, can't afford falls)
- Safety is critical
- Online learning on actual system
- Cost of exploration errors is high

**Prefer Q-Learning when**:
- Only final performance matters
- Can train in simulation (falls don't matter)
- Want to find optimal policy
- Can afford exploratory mistakes

---

**The Fundamental Distinction**:

| Aspect | SARSA | Q-Learning |
|--------|-------|------------|
| Learns | Value of being in state and following current policy | Value of being in state and acting optimally |
| Update | Q(s,a) based on action actually taken | Q(s,a) based on best possible action |
| Target | R + γQ(S', A') where A' is sampled | R + γ max_a Q(S', a) |
| Policy | Safe under exploration | Optimal without exploration |
| Exploration risk | Accounted in values | Ignored in values |

---

**Mathematical Explanation**:

**SARSA fixed point**:
```
Q_π(s,a) = E_π[R + γQ_π(S',A') | S=s, A=a]
```
This is the action-value under policy π (ε-greedy in practice).

**Q-Learning fixed point**:
```
Q*(s,a) = E[R + γ max_{a'} Q*(S',a') | S=s, A=a]
```
This is the optimal action-value Q*.

SARSA converges to Q_π (value under behavior policy).
Q-Learning converges to Q* (value under optimal policy).

---

**Practical Example Values** (simplified):

After convergence near cliff:

**SARSA** (under ε-greedy):
```
Q(near_cliff, right) = -15  (accounts for 10% chance of falling)
Q(near_cliff, up)    = -20  (safe path)
→ Chooses up (safe)
```

**Q-Learning** (optimal):
```
Q(near_cliff, right) = -12  (optimal, assuming no falls)
Q(near_cliff, up)    = -20  (safe path)
→ Chooses right (risky)
```

---

**Key Insight**:

Cliff Walking elegantly demonstrates that:
- **On-policy** algorithms (SARSA) are **conservative**: they account for exploration noise
- **Off-policy** algorithms (Q-Learning) are **aggressive**: they assume optimal execution

The "best" choice depends on whether you can afford exploratory mistakes during learning.

</details>

---

## Question 4: Implementing Expected SARSA

**Write pseudocode for Expected SARSA for a given MDP with discrete actions. Compare it to SARSA and Q-Learning, explaining when Expected SARSA has advantages.**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Expected SARSA Pseudocode**:

```python
# Expected SARSA Algorithm
# For estimating Q(s,a) ≈ q_*(s,a)

Initialize Q(s,a) arbitrarily for all s ∈ S, a ∈ A
Initialize Q(terminal, ·) = 0

Parameters:
  α: learning rate (e.g., 0.1)
  γ: discount factor (e.g., 0.99)
  ε: exploration rate for ε-greedy (e.g., 0.1)

For each episode:
    Initialize S (start state)

    For each step of episode:
        # Choose action using behavior policy (ε-greedy)
        A = choose_action(S, Q, ε)

        # Take action, observe outcome
        Take action A, observe R, S'

        # Compute expected value under target policy
        if S' is terminal:
            expected_q = 0
        else:
            expected_q = compute_expected_q(S', Q, π)

        # TD update using expected value
        Q(S,A) ← Q(S,A) + α[R + γ·expected_q - Q(S,A)]

        S ← S'

    Until S is terminal


# Helper function: compute expected Q-value
def compute_expected_q(state, Q, policy):
    """
    Compute E_π[Q(state, a)]
    = Σ_a π(a|state) Q(state, a)
    """
    expected_value = 0
    for action in actions(state):
        prob = policy(action | state)  # Probability of taking action
        expected_value += prob * Q(state, action)
    return expected_value


# Helper function: ε-greedy action selection
def choose_action(state, Q, ε):
    """
    Choose action using ε-greedy policy
    """
    if random() < ε:
        return random_action(state)  # Explore
    else:
        return argmax_a Q(state, a)  # Exploit


# Helper function: ε-greedy policy probabilities
def epsilon_greedy_policy(action, state, Q, ε):
    """
    Return π(action|state) for ε-greedy policy
    """
    n_actions = len(actions(state))
    greedy_action = argmax_a Q(state, a)

    if action == greedy_action:
        return 1 - ε + ε/n_actions
    else:
        return ε/n_actions
```

---

**Complete Implementation with Details**:

```python
import numpy as np
from collections import defaultdict

class ExpectedSARSA:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, state):
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def get_expected_value(self, state):
        """
        Compute expected value: E_π[Q(state, a)]
        """
        n_actions = self.env.action_space.n
        greedy_action = np.argmax(self.Q[state])

        expected_value = 0.0
        for action in range(n_actions):
            # Compute probability of this action under ε-greedy
            if action == greedy_action:
                prob = 1 - self.epsilon + self.epsilon / n_actions
            else:
                prob = self.epsilon / n_actions

            # Add weighted Q-value
            expected_value += prob * self.Q[state][action]

        return expected_value

    def update(self, state, action, reward, next_state, done):
        """
        Expected SARSA update
        """
        # Compute expected Q-value of next state
        if done:
            expected_next_q = 0.0
        else:
            expected_next_q = self.get_expected_value(next_state)

        # TD target
        td_target = reward + self.gamma * expected_next_q

        # TD error
        td_error = td_target - self.Q[state][action]

        # Update Q-value
        self.Q[state][action] += self.alpha * td_error

    def train(self, num_episodes):
        """
        Train the agent
        """
        returns = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # Choose and take action
                action = self.get_action(state)
                next_state, reward, done, info = self.env.step(action)

                # Update Q-value
                self.update(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            returns.append(total_reward)

        return returns
```

---

**Comparison with SARSA and Q-Learning**:

**SARSA Update**:
```python
# Sample next action from policy
A_next = choose_action(S_next, Q, ε)

# Update using sampled action
Q(S,A) ← Q(S,A) + α[R + γ·Q(S_next, A_next) - Q(S,A)]
```

**Q-Learning Update**:
```python
# Use maximum Q-value (no sampling)
max_q_next = max_a Q(S_next, a)

# Update using maximum
Q(S,A) ← Q(S,A) + α[R + γ·max_q_next - Q(S,A)]
```

**Expected SARSA Update**:
```python
# Compute expected value under policy
expected_q_next = Σ_a π(a|S_next) Q(S_next, a)

# Update using expectation
Q(S,A) ← Q(S,A) + α[R + γ·expected_q_next - Q(S,A)]
```

---

**Side-by-Side Comparison**:

| Aspect | SARSA | Q-Learning | Expected SARSA |
|--------|-------|------------|----------------|
| **Next Value** | Q(S', A') sampled | max_a Q(S', a) | E_π[Q(S', a)] |
| **Sampling** | Sample A' | No sampling | No sampling |
| **Policy** | On-policy | Off-policy | Can be either |
| **Variance** | High | Medium | Low |
| **Bias** | Low | Low | Low |
| **Computation** | O(1) | O(\|A\|) | O(\|A\|) |
| **Convergence** | To Q_π | To Q* | To Q_π or Q* |

---

**When Expected SARSA Has Advantages**:

**1. Lower Variance**:

SARSA samples next action:
```
Target = R + γ Q(S', A')  where A' ~ π(·|S')
```
Variance from sampling A'.

Expected SARSA takes expectation:
```
Target = R + γ E_π[Q(S', a)]
```
No variance from action selection!

**Example**:
```
State S': Q-values = [10, 10, 10, -100]
ε-greedy (ε=0.2):

SARSA:
  95% chance: sample good action → target ≈ 10
  5% chance: sample bad action → target = -100
  High variance across episodes!

Expected SARSA:
  Expected value: 0.95×10 + 0.05×(-100) = 9.5 - 5 = 4.5
  Same every time → no sampling variance
```

**2. Better Performance in Stochastic Environments**:

When action selection is noisy (high ε), Expected SARSA is more stable:
- SARSA updates can be noisy due to random action sampling
- Expected SARSA smooths over the noise

**3. Faster Convergence**:

Lower variance → more consistent updates → faster learning

Empirically, Expected SARSA often converges in fewer episodes than SARSA.

**4. Flexibility**:

Can be on-policy or off-policy:
- Use behavior policy b to choose actions
- Compute expectation under target policy π
- If b = π: on-policy
- If π is greedy: equivalent to Q-Learning

**5. Generalizes Both SARSA and Q-Learning**:

```
When π is greedy (ε=0):
  E_π[Q(S',a)] = max_a Q(S',a)
  Expected SARSA = Q-Learning

When π is the behavior policy:
  Expected SARSA is like SARSA but with reduced variance
```

---

**When to Use Each**:

**Use SARSA when**:
- Simplicity is important (slightly easier to implement)
- Computational cost must be minimal (O(1) update)
- On-policy learning is required
- Low-dimensional action spaces

**Use Q-Learning when**:
- Want to learn optimal policy explicitly
- Off-policy learning is beneficial
- Can tolerate initial aggressiveness
- Action space is large (expectation expensive)

**Use Expected SARSA when**:
- Variance reduction is important
- Can afford O(|A|) computation per step
- Stochastic environment or high exploration
- Want more stable learning than SARSA
- Want flexibility (on/off-policy)

---

**Empirical Performance**:

On many benchmark tasks:
```
Expected SARSA > SARSA > Q-Learning (during training)
```

But all converge to similar performance eventually.

**Example (Cliff Walking)**:
- Expected SARSA: Finds safe path, converges fast
- SARSA: Finds safe path, more noise
- Q-Learning: Finds risky path, most noise during training

---

**Practical Implementation Tips**:

1. **Efficient Expectation Computation**:
```python
# For ε-greedy, can simplify:
n = len(actions)
greedy_action = argmax(Q[state])
greedy_value = Q[state][greedy_action]
avg_value = mean(Q[state])

expected = (1-ε)*greedy_value + ε*avg_value
```

2. **Softmax Policy**:
```python
# Expected SARSA works great with softmax (Boltzmann) exploration
probs = softmax(Q[state] / temperature)
expected = sum(probs * Q[state])
```

3. **Action Space Size**:
- Small action space (< 10): Expected SARSA is great
- Large action space (> 100): Expectation becomes expensive, prefer sampling

4. **Learning Rate**:
- Expected SARSA can use slightly larger α due to lower variance
- Try α = 0.2 to 0.5 instead of 0.1

---

**Key Insight**:

Expected SARSA eliminates the variance from sampling the next action, making it a more stable version of SARSA with minimal added cost. It's often the best choice for tabular methods with moderate action spaces.

</details>

---

## Question 5: Maximization Bias and Double Q-Learning

**Explain maximization bias in Q-Learning with a concrete example. How does Double Q-Learning address this problem, and what does it sacrifice (if anything)?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Maximization Bias: The Problem**

Q-Learning uses the maximum Q-value for bootstrapping:
```
Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
                            ↑
                    This causes overestimation!
```

**Why It Happens**:

When Q-values have estimation noise (they always do), taking the maximum systematically overestimates:

```
True values: Q_true = [1, 2, 3]
Noisy estimates: Q_est = [1.5, 1.5, 3.5]  (noise: [+0.5, -0.5, +0.5])

max(Q_true) = 3
max(Q_est) = 3.5  > 3   (overestimation due to noise)
```

The maximum picks out positive noise more than negative noise!

**Mathematical Explanation**:

For random variables X_1, ..., X_n:
```
E[max(X_1, ..., X_n)] ≥ max(E[X_1], ..., E[X_n])
```

When Q-values are noisy estimates:
```
Q(s,a) = Q_true(s,a) + ε_a  where E[ε_a] = 0

max_a Q(s,a) = max_a [Q_true(s,a) + ε_a]
             ≥ max_a Q_true(s,a)  (in expectation)
```

The maximum operator is **biased upward** when applied to noisy estimates.

---

**Concrete Example: Simple MDP**

Consider a state with 3 actions:

```
True Q-values:
Q*(s, a1) = 0
Q*(s, a2) = 0
Q*(s, a3) = 0
All actions equally valuable!

After some experience, our estimates (with noise):
Q(s, a1) = 0.2   (noise: +0.2)
Q(s, a2) = -0.1  (noise: -0.1)
Q(s, a3) = 0.3   (noise: +0.3)

Q-Learning chooses: max_a Q(s,a) = 0.3 (from a3)

Expected value if all were sampled equally:
E[Q(s,a)] = (0.2 - 0.1 + 0.3) / 3 ≈ 0.13

But we use: max = 0.3

Overestimation: 0.3 > 0.13 (and both > 0, the true value)
```

This overestimation **propagates** through bootstrapping, getting worse over time!

---

**Detailed Example: Two-State MDP**

```
States: A, B (B is terminal)
From A: 10 actions, all go to B with reward 0
True values: Q*(A, a_i) = 0 for all i

Initial Q-values: Q(A, a_i) ~ N(0, 1) (random initialization)

Q-Learning update from A:
Q(A, a_i) ← Q(A, a_i) + α[0 + γ·max_a Q(B,a) - Q(A, a_i)]
                                    ↑
                              this is 0 (terminal)

Wait, no propagation here. Let me use a better example...
```

**Better Example: Chain MDP**

```
States: S1 → S2 → S3 (terminal, reward +1)
All rewards during chain: 0
True values: V(S1) = 1, V(S2) = 1

From S2: 5 actions, all go to S3
Q*(S2, a_i) = 0 + γ·1 = 1 for all i

Noisy estimates:
Q(S2, a1) = 1.2
Q(S2, a2) = 0.8
Q(S2, a3) = 1.3   ← max
Q(S2, a4) = 0.9
Q(S2, a5) = 1.1

True max: 1.0
Estimated max: 1.3 (overestimation: +0.3)

Now update S1:
From S1: 1 action goes to S2
Q(S1, a) ← α[0 + γ·max_a Q(S2, a)] = α[γ·1.3]

Should be: γ·1.0
Actually using: γ·1.3

Overestimation propagates from S2 to S1!
```

---

**Why Maximization Bias Matters**:

1. **Compounds Over Time**: Overestimates at later states propagate to earlier states through bootstrapping

2. **Poor Action Selection**: May prefer actions with higher estimation noise rather than truly better actions

3. **Slower Convergence**: Takes longer to correct overoptimistic values

4. **Suboptimal Policies**: In some cases, can lead to persistently suboptimal behavior

**When It's Especially Problematic**:
- Many actions with similar values (more chances for positive noise to appear large)
- Early in learning (estimates are very noisy)
- Stochastic environments (more noise in Q-values)
- Long chains of bootstrapping (error compounds)

---

**Double Q-Learning Solution**

**Key Idea**: Use two independent Q-functions to **decorrelate** action selection and evaluation.

- Q1 and Q2 both estimate Q*(s,a)
- Use one to **select** the best action
- Use the other to **evaluate** that action

**Double Q-Learning Update** (randomly alternate):

```python
With probability 0.5:
    # Update Q1
    # Select action using Q1
    a_max = argmax_a Q1(S', a)

    # Evaluate action using Q2
    Q1(S,A) ← Q1(S,A) + α[R + γ·Q2(S', a_max) - Q1(S,A)]

Otherwise:
    # Update Q2
    # Select action using Q2
    a_max = argmax_a Q2(S', a)

    # Evaluate action using Q1
    Q2(S,A) ← Q2(S,A) + α[R + γ·Q1(S', a_max) - Q2(S,A)]
```

**Final Policy**:
```
π(s) = argmax_a [Q1(s,a) + Q2(s,a)]  or just use either Q-function
```

---

**Why Double Q-Learning Works**:

**The Problem with Q-Learning**:
```
max_a Q(s,a) = Q(s, argmax_a Q(s,a))
     ↑               ↑           ↑
  evaluate       action      select

Same Q-function for both selection and evaluation!
If Q(s,a_i) is overestimated, we both:
  1. Select a_i (because it looks best)
  2. Use that overestimated value in the update

This creates a positive feedback loop.
```

**Double Q-Learning Solution**:
```
argmax_a Q1(s,a) gives action a*
Then evaluate using Q2(s,a*)

If Q1 overestimates a_i:
  1. We select a_i (because Q1 thinks it's best)
  2. But we evaluate using Q2(a_i)

If Q2 doesn't also overestimate a_i (likely, if Q1 and Q2 are independent):
  - We get a less biased estimate
  - The overestimation doesn't reinforce itself
```

**Mathematical Intuition**:

Q-Learning:
```
E[Q(s, argmax_a Q(s,a))] ≥ max_a E[Q(s,a)]  (biased)
```

Double Q-Learning:
```
E[Q2(s, argmax_a Q1(s,a))] ≈ max_a E[Q(s,a)]  (less biased)
```

Because Q1 and Q2 have independent noise, the selection (by Q1) and evaluation (by Q2) are decorrelated.

---

**Concrete Example with Double Q-Learning**:

```
State S2: 5 actions to terminal (reward 1)
True: Q*(S2, a_i) = 1 for all i

Q1 estimates:
Q1(S2, a1) = 1.2
Q1(S2, a2) = 0.8
Q1(S2, a3) = 1.3  ← max in Q1
Q1(S2, a4) = 0.9
Q1(S2, a5) = 1.1

Q2 estimates (independent noise):
Q2(S2, a1) = 0.9
Q2(S2, a2) = 1.1
Q2(S2, a3) = 0.8  ← evaluation for a3
Q2(S2, a4) = 1.2
Q2(S2, a5) = 1.0

Standard Q-Learning:
  max_a Q(S2,a) = 1.3 (overestimate)

Double Q-Learning:
  a* = argmax_a Q1(S2,a) = a3
  value = Q2(S2, a3) = 0.8 (underestimate this time!)

On average across many updates:
  Q-Learning: consistently overestimates
  Double Q-Learning: sometimes over, sometimes under → averages to correct value
```

---

**What Double Q-Learning Sacrifices**:

**1. Computational Cost**:
- Must maintain two Q-tables (2× memory)
- Each update requires lookup in both tables
- Negligible for tabular, but matters for deep RL (Double DQN)

**2. Sample Efficiency** (slightly):
- Each Q-function gets half the updates
- Q1 and Q2 converge slightly slower than single Q
- But reduced bias can compensate, often learning faster overall

**3. Implementation Complexity**:
- Slightly more complex (maintain two tables, randomize updates)
- More code to debug

**4. Not Zero Bias**:
- Still has some bias (just much less than Q-Learning)
- When Q1 and Q2 have correlated errors, bias remains
- Asymptotically unbiased, but not unbiased at every step

**5. Can Underestimate**:
- Q-Learning always overestimates
- Double Q-Learning can underestimate if the selected action by Q1 happens to have negative noise in Q2
- However, this averages out over time

---

**Empirical Comparison**:

**On Simple MDP** (like above chain):

Q-Learning Q-values:
```
Iteration 1000: Q(S1) = 1.8 (true: 1.0)
Iteration 2000: Q(S1) = 1.6 (true: 1.0)
Iteration 5000: Q(S1) = 1.3 (true: 1.0)
```
Overestimates, slowly converges.

Double Q-Learning Q-values:
```
Iteration 1000: Q(S1) ≈ 1.1 (true: 1.0)
Iteration 2000: Q(S1) ≈ 1.05 (true: 1.0)
Iteration 5000: Q(S1) ≈ 1.01 (true: 1.0)
```
Less bias, faster convergence.

---

**When to Use Double Q-Learning**:

**Use Double Q-Learning when**:
- Many actions with similar values
- Stochastic rewards or transitions
- Overestimation causes problems (safety-critical)
- Have memory for two Q-tables
- Want more stable learning

**Stick with Q-Learning when**:
- Few actions (< 5), maximization bias is small
- Memory is very constrained
- Simplicity is paramount
- Overestimation isn't harmful (e.g., optimistic initialization is desired)

---

**Modern Extensions**:

1. **Double DQN** (Deep Q-Network):
   - Uses target network as second Q-function
   - Huge improvement over DQN in practice
   - Standard in modern deep RL

2. **Clipped Double Q-Learning**:
   - Use: min(Q1, Q2) instead of Q2(argmax Q1)
   - Even more conservative
   - Used in TD3 algorithm

3. **Averaged Q-Learning**:
   - Maintain multiple Q-functions, use average
   - Further variance reduction

---

**Key Insights**:

1. **Maximization bias is inherent in Q-Learning**: Using max on noisy estimates always overestimates

2. **Decorrelation is the solution**: Separate selection and evaluation using independent estimates

3. **Tradeoff**: Slightly more computation and complexity for significantly less bias

4. **Practical impact**: Double Q-Learning often learns faster and more stably, especially in complex environments

5. **Foundation for modern methods**: Double DQN is standard in deep RL, based on this idea

---

**Summary Table**:

| Aspect | Q-Learning | Double Q-Learning |
|--------|------------|-------------------|
| **Bias** | Overestimates | Much less bias |
| **Variance** | Low | Low |
| **Memory** | 1× | 2× |
| **Update Cost** | O(1) | O(1) but 2 lookups |
| **Implementation** | Simple | Moderate |
| **Convergence** | Slower (biased) | Faster (less biased) |
| **Action Selection** | argmax Q(s,a) | argmax [Q1(s,a)+Q2(s,a)] |
| **Use Case** | Simple problems | Complex, stochastic environments |

The extra complexity of Double Q-Learning is almost always worth it in practice!

</details>

---

## Additional Practice Problems

1. Prove that TD(0) converges to v_π under tabular representation with appropriate learning rate schedule.

2. Implement all four methods (SARSA, Q-Learning, Expected SARSA, Double Q-Learning) on the same environment and compare learning curves.

3. Derive the update rule for SARSA(λ), the eligibility trace version of SARSA.

4. Analyze the computational complexity of each TD method per step and per episode.

5. Design an environment where maximization bias causes Q-Learning to fail but Double Q-Learning succeeds.
