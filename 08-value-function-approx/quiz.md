# Week 8 Quiz: Value Function Approximation

## Question 1: Conceptual Understanding

**What is the deadly triad in reinforcement learning? Explain why these three elements together can cause instability and give an example of an algorithm that faces this challenge.**

<details>
<summary>Answer</summary>

The deadly triad consists of three elements that, when combined, can lead to divergence and instability:

1. **Function Approximation**: Using parameterized functions (linear, neural networks) instead of tables to represent value functions
2. **Bootstrapping**: Using estimates to update estimates (e.g., TD learning where we use v(S_{t+1}) to update v(S_t))
3. **Off-Policy Learning**: Learning about a target policy while following a different behavior policy

**Why they cause instability together**:

- **Function approximation** means that updating one state affects the values of other states through shared parameters. This creates dependencies and can amplify errors.

- **Bootstrapping** means our targets themselves depend on our current estimates. We're using v(S_{t+1}, w) to update v(S_t, w), creating a feedback loop.

- **Off-Policy learning** introduces distribution mismatch. We're training on data from one distribution (behavior policy) but trying to learn values for another (target policy). This breaks the assumptions needed for convergence.

**Why each pair alone is safer**:

- Function approximation + bootstrapping (on-policy): Like on-policy SARSA with function approximation - generally stable
- Function approximation + off-policy (no bootstrapping): Like Monte Carlo off-policy - can work but slower
- Bootstrapping + off-policy (tabular): Like tabular Q-learning - proven convergent

**Example algorithm facing this challenge**:

**Q-learning with neural networks** (before DQN innovations):
- Uses neural network function approximation
- TD learning (bootstraps): Q(s,a) ← r + γ max_a' Q(s',a')
- Off-policy: learns optimal Q* while acting ε-greedily

This combination led to instability and divergence in practice, which is why DQN introduced:
- Experience replay (decorrelates samples)
- Target network (stabilizes bootstrapping target)
- Other tricks to make deep Q-learning work

**Historical significance**: Baird's counterexample (1995) demonstrated that even with linear function approximation, the deadly triad could cause divergence. This motivated decades of research into stable off-policy learning algorithms.

</details>

---

## Question 2: Mathematical Derivation

**Derive the semi-gradient TD(0) update rule for linear value function approximation. Start from the gradient descent principle and explain why it's called "semi-gradient." Show the final update rule for the weight vector w.**

<details>
<summary>Answer</summary>

**Starting point: Gradient Descent**

In supervised learning, we minimize the mean squared error:
```
J(w) = E[(v_π(S) - v(S, w))²]
```

The gradient descent update would be:
```
w ← w - α/2 · ∇_w J(w)
  = w - α/2 · ∇_w E[(v_π(S) - v(S, w))²]
  = w + α · E[(v_π(S) - v(S, w)) · ∇_w v(S, w)]
```

For stochastic gradient descent (single sample):
```
w ← w + α · [v_π(S) - v(S, w)] · ∇_w v(S, w)
```

**Problem**: We don't know the true value v_π(S)!

**TD(0) Solution**: Replace v_π(S) with the TD target:
```
v_π(S_t) ≈ R_{t+1} + γ · v(S_{t+1}, w)
```

This gives the **semi-gradient TD(0) update**:
```
w ← w + α · [R_{t+1} + γ · v(S_{t+1}, w) - v(S_t, w)] · ∇_w v(S_t, w)
  = w + α · δ_t · ∇_w v(S_t, w)
```

Where δ_t = R_{t+1} + γ · v(S_{t+1}, w) - v(S_t, w) is the TD error.

**Why "semi-gradient"?**

If we took the true gradient of the TD error squared, we would differentiate the entire expression:
```
True gradient of [R_{t+1} + γ · v(S_{t+1}, w) - v(S_t, w)]²
would include: ∂/∂w [γ · v(S_{t+1}, w)] = γ · ∇_w v(S_{t+1}, w)
```

But in semi-gradient methods, we **treat the target as constant**:
```
Semi-gradient treats [R_{t+1} + γ · v(S_{t+1}, w)] as if it doesn't depend on w
Only differentiates: -v(S_t, w)
```

**For linear function approximation**: v(s, w) = w^T · x(s)

The gradient is simply:
```
∇_w v(s, w) = x(s)
```

So the update becomes:
```
w ← w + α · [R_{t+1} + γ · w^T · x(S_{t+1}) - w^T · x(S_t)] · x(S_t)
  = w + α · δ_t · x(S_t)
```

**Component-wise** (for each weight w_i):
```
w_i ← w_i + α · δ_t · x_i(S_t)
```

**Key insights**:

1. Only features active in S_t get updated (where x_i(S_t) ≠ 0)
2. Update magnitude proportional to feature value and TD error
3. Features in S_{t+1} affect the update through δ_t but aren't directly differentiated
4. This is computationally efficient but not a true gradient descent

**What's wrong with full gradient?**

The semi-gradient ignores the dependency of the target on w. While this seems wrong, it:
- Is computationally simpler
- Often works well in practice
- Converges for on-policy linear TD (proven)
- Matches the Bellman operator contraction

The full gradient (gradient TD methods like TDC, GTD2) can be more stable for off-policy learning but is more complex.

</details>

---

## Question 3: Comparison

**Compare three approaches to representing value functions: (1) Tabular methods, (2) Linear function approximation with tile coding, (3) Neural network function approximation. Discuss their representational power, sample efficiency, computational cost, and convergence guarantees.**

<details>
<summary>Answer</summary>

| Aspect | Tabular | Linear + Tile Coding | Neural Networks |
|--------|---------|---------------------|-----------------|
| **Representational Power** | Perfect for discrete spaces | Piecewise linear approximation | Universal function approximators |
| **Sample Efficiency** | Each state learned independently | Medium (generalization helps) | Can be very sample efficient with good features |
| **Computational Cost** | Low (table lookup) | Low (sparse features) | High (forward/backward pass) |
| **Convergence Guarantees** | Strong (proven for most algorithms) | Strong for on-policy (proven) | Weak (few guarantees) |
| **State Space** | Small, discrete only | Medium, continuous OK | Large, continuous, high-dimensional |
| **Generalization** | None | Good for similar states | Excellent |

## Detailed Comparison

### 1. Tabular Methods

**Representational Power**: ⭐⭐⭐⭐⭐
- Can represent ANY value function exactly (given enough states)
- Each state has independent value
- No approximation error

**Sample Efficiency**: ⭐⭐
- Must visit each state many times
- No generalization between states
- Requires complete exploration

**Computational Cost**: ⭐⭐⭐⭐⭐
- O(1) lookup and update
- Minimal memory for small spaces
- No complex computations

**Convergence**: ⭐⭐⭐⭐⭐
- Q-learning: proven convergent
- SARSA: proven convergent
- Monte Carlo: convergent by averaging
- Well-understood theory

**Limitations**:
- Only works for discrete, small state spaces
- Doesn't scale to continuous or large spaces
- No knowledge transfer between similar states

**Best for**: GridWorld, simple board games, small discrete MDPs

---

### 2. Linear Function Approximation + Tile Coding

**Representational Power**: ⭐⭐⭐
- Piecewise linear approximation
- Can approximate continuous functions well with enough tiles
- Limited to relatively smooth functions
- Cannot represent arbitrary complex functions

**Sample Efficiency**: ⭐⭐⭐⭐
- Generalization across similar states
- Updates affect nearby states
- Faster learning than tabular
- Still needs reasonable coverage

**Computational Cost**: ⭐⭐⭐⭐
- Sparse feature activation: O(num_tilings)
- Fast feature computation
- Low memory with hashing
- Efficient updates

**Convergence**: ⭐⭐⭐⭐
- On-policy TD: proven convergent (Sutton & Barto Theorem 9.1)
- Semi-gradient SARSA: convergent under conditions
- Off-policy can be unstable (deadly triad)
- Well-understood for linear methods

**Tile Coding Specifics**:
- Adaptive resolution: coarse features with fine discrimination
- Number of active features constant (one per tiling)
- Easy to tune (num_tilings, tiles_per_dim)
- Works well for 2-4 dimensional continuous spaces

**Limitations**:
- Representational power limited to piecewise linear
- Feature engineering required
- Curse of dimensionality (tiles grow exponentially with dimensions)
- May need domain knowledge for good features

**Best for**: Mountain Car, CartPole, Acrobot, continuous control with low dimensions

---

### 3. Neural Network Function Approximation

**Representational Power**: ⭐⭐⭐⭐⭐
- Universal function approximators
- Can represent arbitrarily complex functions
- Automatically learns features
- No dimensionality limit in theory

**Sample Efficiency**: ⭐⭐⭐ (varies widely)
- Can be very efficient with good architecture
- Transfer learning and pre-training help
- But can require millions of samples (Atari DQN)
- Depends heavily on network design and hyperparameters

**Computational Cost**: ⭐⭐
- Forward pass: O(weights)
- Backward pass: O(weights)
- GPU acceleration helps
- Memory intensive for large networks
- Training can be slow

**Convergence**: ⭐⭐
- Few theoretical guarantees
- Non-convex optimization
- Can diverge (especially with deadly triad)
- Requires careful stabilization (target networks, etc.)
- Hyperparameter sensitive

**Advantages**:
- Handles high-dimensional inputs (images, etc.)
- Automatic feature learning
- Can leverage modern deep learning tools
- State-of-the-art performance on complex tasks

**Challenges**:
- Instability in RL setting
- Requires large datasets
- Hard to interpret
- Many hyperparameters to tune
- Overfitting possible

**Best for**: Atari games, image-based tasks, robotics with visual input, high-dimensional spaces

---

## When to Choose Each

**Choose Tabular if**:
- State space is small (<10,000 states)
- States are discrete
- You need guaranteed convergence
- Interpretability is important

**Choose Linear + Tile Coding if**:
- State space is continuous but low-dimensional (2-4D)
- You want convergence guarantees
- Computational resources are limited
- You can design good features
- Classic control problems (Mountain Car, CartPole)

**Choose Neural Networks if**:
- State space is high-dimensional (images, etc.)
- You have lots of data
- You need maximum representational power
- You can afford computational cost
- You're willing to tune hyperparameters carefully
- Modern deep RL (DQN, PPO, SAC)

## Hybrid Approaches

In practice, combinations can work well:
- **Tile coding features → linear layer**: Gets benefits of both
- **Neural network for feature extraction → linear value head**: Common in modern RL
- **Ensemble methods**: Multiple neural networks for robustness

**Key Insight**: There's a fundamental tradeoff between representational power and guarantees. Tabular is safe but limited; neural networks are powerful but tricky; linear methods with good features are often the sweet spot for medium-complexity problems.

</details>

---

## Question 4: Application

**You need to implement tile coding for the Mountain Car state space, where position ∈ [-1.2, 0.6] and velocity ∈ [-0.07, 0.07]. Design a tile coding scheme with specific parameters. Then show how you would compute the feature vector for state s = (-0.5, 0.02) with action a = 1 (accelerate right). Explain your parameter choices.**

<details>
<summary>Answer</summary>

## Tile Coding Design for Mountain Car

### Parameter Choices

```python
# State space bounds
POSITION_BOUNDS = (-1.2, 0.6)  # Range: 1.8
VELOCITY_BOUNDS = (-0.07, 0.07)  # Range: 0.14
NUM_ACTIONS = 3  # {0: left, 1: none, 2: right}

# Tile coding parameters
NUM_TILINGS = 8  # Number of overlapping grids
TILES_PER_DIM_POSITION = 8  # Tiles per dimension for position
TILES_PER_DIM_VELOCITY = 8  # Tiles per dimension for velocity

# Derived values
POSITION_TILE_WIDTH = 1.8 / 8 = 0.225  # Width of each tile in position
VELOCITY_TILE_WIDTH = 0.14 / 8 = 0.0175  # Width of each tile in velocity
```

### Justification for Parameters

**NUM_TILINGS = 8**:
- Standard choice in literature
- Provides good balance between resolution and learning speed
- Each state activates 8 features (one per tiling)
- Learning rate should be α/8 to normalize updates

**TILES_PER_DIM = 8**:
- Position: 8 tiles cover range [-1.2, 0.6], each 0.225 wide
  - Provides reasonable discretization of position
  - Not too coarse (would miss important features)
  - Not too fine (would slow learning)
- Velocity: 8 tiles cover range [-0.07, 0.07], each 0.0175 wide
  - Captures velocity changes adequately
  - Matches position resolution

**Total features per action**: 8 tilings × 8 × 8 tiles = 512 features
**Total features (all actions)**: 512 × 3 = 1,536 features
- Manageable size for linear methods
- Sparse representation (only 8 active at a time)

### Computing Feature Vector for s = (-0.5, 0.02), a = 1

#### Step 1: Initialize Tiling Offsets

For 8 tilings, create uniform offsets in [0, tile_width]:

```python
# Offset fraction for tiling i: i / NUM_TILINGS
# Tiling 0: offset = (0, 0)
# Tiling 1: offset = (0.028125, 0.0021875)
# Tiling 2: offset = (0.05625, 0.004375)
# ...
# Tiling 7: offset = (0.196875, 0.0153125)

offsets = []
for i in range(8):
    pos_offset = i * 0.225 / 8  # i * POSITION_TILE_WIDTH / NUM_TILINGS
    vel_offset = i * 0.0175 / 8  # i * VELOCITY_TILE_WIDTH / NUM_TILINGS
    offsets.append((pos_offset, vel_offset))
```

#### Step 2: Compute Tile Indices for Each Tiling

For state s = (-0.5, 0.02):

```python
state = (-0.5, 0.02)
active_tiles = []

for tiling_idx in range(8):
    # Apply offset
    pos_offset, vel_offset = offsets[tiling_idx]

    # Position coordinate
    position_adjusted = -0.5 - pos_offset
    position_from_min = position_adjusted - (-1.2)  # Shift to start from 0
    position_tile = int(position_from_min / 0.225)
    position_tile = max(0, min(7, position_tile))  # Clamp to [0, 7]

    # Velocity coordinate
    velocity_adjusted = 0.02 - vel_offset
    velocity_from_min = velocity_adjusted - (-0.07)  # Shift to start from 0
    velocity_tile = int(velocity_from_min / 0.0175)
    velocity_tile = max(0, min(7, velocity_tile))  # Clamp to [0, 7]

    # Convert 2D coordinates to 1D tile index
    # Index within this tiling: position_tile * 8 + velocity_tile
    # Offset by tiling: tiling_idx * 64
    tile_index = tiling_idx * 64 + position_tile * 8 + velocity_tile
    active_tiles.append(tile_index)
```

**Concrete calculation for Tiling 0** (offset = 0):
```
Position:
  position_adjusted = -0.5 - 0 = -0.5
  position_from_min = -0.5 - (-1.2) = 0.7
  position_tile = int(0.7 / 0.225) = int(3.11) = 3

Velocity:
  velocity_adjusted = 0.02 - 0 = 0.02
  velocity_from_min = 0.02 - (-0.07) = 0.09
  velocity_tile = int(0.09 / 0.0175) = int(5.14) = 5

Tile index = 0 * 64 + 3 * 8 + 5 = 29
```

**Concrete calculation for Tiling 1** (offset = (0.028125, 0.0021875)):
```
Position:
  position_adjusted = -0.5 - 0.028125 = -0.528125
  position_from_min = -0.528125 - (-1.2) = 0.671875
  position_tile = int(0.671875 / 0.225) = int(2.99) = 2

Velocity:
  velocity_adjusted = 0.02 - 0.0021875 = 0.0178125
  velocity_from_min = 0.0178125 - (-0.07) = 0.0878125
  velocity_tile = int(0.0878125 / 0.0175) = int(5.02) = 5

Tile index = 1 * 64 + 2 * 8 + 5 = 85
```

Continuing for all 8 tilings:
```
active_tiles = [29, 85, 141, 197, 253, 309, 365, 421]
```

#### Step 3: Offset by Action

For action a = 1 (none/coast), offset each tile by action × 512:

```python
action = 1
action_offset = 512  # 512 tiles per action

feature_indices = [tile + action_offset for tile in active_tiles]
feature_indices = [541, 597, 653, 709, 765, 821, 877, 933]
```

#### Step 4: Create Sparse Feature Vector

The feature vector x(s, a) ∈ ℝ^1536 where:
- x[i] = 1 if i ∈ feature_indices
- x[i] = 0 otherwise

```python
# Sparse representation (efficient)
x_sparse = feature_indices  # [541, 597, 653, 709, 765, 821, 877, 933]

# Dense representation (for illustration)
x_dense = np.zeros(1536)
x_dense[feature_indices] = 1

# Q-value computation: q(s, a, w) = w^T · x = sum(w[i] for i in feature_indices)
q_value = sum(w[i] for i in feature_indices)
```

### Why This Design Works

1. **Generalization**: States with similar position/velocity activate similar tiles
   - State (-0.51, 0.02) would activate mostly the same tiles
   - Smooth value function approximation

2. **Discrimination**: Multiple tilings provide fine resolution
   - Single tiling: coarse (8×8 = 64 possible combinations)
   - 8 tilings: effective resolution much finer
   - Can distinguish states that differ slightly

3. **Efficiency**: Sparse activation
   - Only 8 out of 1,536 features are non-zero
   - Fast computation: O(8) instead of O(1536)
   - Memory efficient with sparse representation

4. **Action separation**: Separate features for each action
   - No interference between actions
   - Can learn different Q-values for each action independently

5. **Learning dynamics**:
   - Update affects all 8 active features equally
   - Learning rate α/8 keeps update magnitude consistent
   - Nearby states share features → generalization

### Practical Implementation Tips

```python
class TileCoding:
    def __init__(self):
        self.num_tilings = 8
        self.tiles_per_dim = 8
        self.num_actions = 3
        self.total_tiles = 512 * 3  # 1536

    def get_features(self, position, velocity, action):
        """Returns list of 8 active feature indices"""
        # Use iht (Index Hash Table) for efficient implementation
        # or compute explicitly as shown above
        pass

    def q_value(self, position, velocity, action, weights):
        """Compute Q(s,a) = sum of weights for active features"""
        features = self.get_features(position, velocity, action)
        return sum(weights[f] for f in features)

    def update(self, position, velocity, action, weights, delta, alpha):
        """Update weights: w[i] += alpha * delta for active features"""
        features = self.get_features(position, velocity, action)
        for f in features:
            weights[f] += alpha * delta
```

This design has been proven effective for Mountain Car, typically solving it in 500-2000 episodes with proper hyperparameters.

</details>

---

## Question 5: Critical Thinking

**Explain why the semi-gradient TD methods are called "semi-gradient" rather than true gradient descent. What would a "true gradient" update look like? Why do we use semi-gradient methods despite them not being true gradient descent? What problems could arise, and what are the trade-offs?**

<details>
<summary>Answer</summary>

## Understanding Semi-Gradient vs. True Gradient

### Why "Semi-Gradient"?

Semi-gradient methods treat the **bootstrapping target as constant** when computing gradients, differentiating only with respect to the current state value, not the next state value.

#### Semi-Gradient TD(0) Update

```
Target: R_{t+1} + γ · v(S_{t+1}, w_t)
Error: δ_t = R_{t+1} + γ · v(S_{t+1}, w_t) - v(S_t, w_t)

Semi-gradient update:
w ← w + α · δ_t · ∇_w v(S_t, w)

Only differentiates v(S_t, w), treats v(S_{t+1}, w) as constant!
```

#### True Gradient Update (What We're NOT Doing)

If we took the true gradient of the squared TD error:

```
Loss: L(w) = [R_{t+1} + γ · v(S_{t+1}, w) - v(S_t, w)]²

True gradient:
∇_w L = 2 · [R_{t+1} + γ · v(S_{t+1}, w) - v(S_t, w)] · ∇_w [R_{t+1} + γ · v(S_{t+1}, w) - v(S_t, w)]
      = 2 · δ_t · [γ · ∇_w v(S_{t+1}, w) - ∇_w v(S_t, w)]
```

This includes the term **γ · ∇_w v(S_{t+1}, w)**, which semi-gradient ignores!

### Concrete Example

**Linear value function**: v(s, w) = w^T · x(s)

**Semi-gradient TD(0)**:
```
w ← w + α · [r + γ · w^T · x(s') - w^T · x(s)] · x(s)
  = w + α · δ · x(s)
```

**True gradient**:
```
∇_w L = 2 · δ · [γ · x(s') - x(s)]

w ← w - α/2 · ∇_w L
  = w + α · δ · [x(s) - γ · x(s')]
```

**Key difference**: The true gradient includes the **-γ · x(s')** term!

---

## Why Use Semi-Gradient Despite Not Being True Gradient?

### Reason 1: Computational Simplicity

**Semi-gradient**: Only need features of current state x(S_t)
- Simple, efficient implementation
- Natural online learning
- Matches intuition of TD learning

**True gradient**: Need to track and update based on both x(S_t) and x(S_{t+1})
- More complex bookkeeping
- Less intuitive
- Requires more memory

### Reason 2: Matches Bellman Operator

The semi-gradient update corresponds to the **Bellman operator**:
```
T^π v(s) = E[R + γ · v(S') | S = s]
```

We're trying to solve: v = T^π v (Bellman equation)

Semi-gradient TD implements this operator, treating the target as the "true" value we're trying to match.

**Intuition**: We're not minimizing a loss function globally; we're doing **successive approximation** toward the Bellman fixed point.

### Reason 3: Works Well in Practice

For **on-policy linear TD**:
- Proven convergent (Sutton & Barto Theorem 9.1)
- Converges to within bounded error of optimal
- Stable and reliable
- Used successfully for decades

### Reason 4: True Gradient Can Be Worse

True gradient descent (naive implementation) can:
- Converge more slowly
- Be less stable in some cases
- Not match the Bellman operator semantics

**The semi-gradient isn't a bug; it's a feature!** It's designed for the structure of RL problems.

---

## Problems That Can Arise

### Problem 1: Off-Policy Divergence (Deadly Triad)

Semi-gradient methods can **diverge** when combined with:
- Function approximation
- Bootstrapping
- Off-policy learning

**Famous example**: Baird's counterexample (1995)
- Simple linear function approximation
- Off-policy updates
- Weights diverge to infinity!

**Why**: The semi-gradient doesn't minimize a real objective function. Off-policy distribution mismatch amplifies errors that the semi-gradient can't correct.

### Problem 2: No Global Optimality Guarantee

Semi-gradient methods don't minimize a global loss function:
- Don't necessarily reach the best possible approximation
- May get stuck in poor local configurations
- Sensitive to initialization and step size

**True gradient methods** (e.g., minimizing mean squared projection error) have clearer optimality properties.

### Problem 3: Chattering and Instability

Even on-policy, semi-gradient can:
- Oscillate around fixed point
- Be sensitive to step size α
- Show high variance in parameter updates

### Problem 4: Not a True Gradient in Optimization Sense

Because it's not true gradient descent:
- Can't use standard optimization theory
- Convergence analysis is more complex
- Harder to apply advanced optimization techniques (momentum, Adam, etc.)

---

## Trade-offs Summary

| Aspect | Semi-Gradient TD | True Gradient TD (e.g., GTD, TDC) |
|--------|------------------|-----------------------------------|
| **Simplicity** | ✅ Simple, intuitive | ❌ More complex |
| **Computation** | ✅ Efficient | ❌ More expensive |
| **On-policy convergence** | ✅ Proven for linear | ✅ Also proven |
| **Off-policy convergence** | ❌ Can diverge | ✅ Guaranteed convergence |
| **Sample efficiency** | ✅ Often faster | ❌ Often slower |
| **Theoretical foundation** | ⚠️ Heuristic (but works) | ✅ True gradient descent |
| **Practical success** | ✅ Widely used | ⚠️ Less common |

---

## True Gradient Alternatives

**Gradient TD methods** (Sutton et al. 2008-2009) perform true gradient descent:

**GTD2** (Gradient TD 2):
```
Minimizes projected Bellman error
Uses two sets of parameters
True gradient descent
Guaranteed convergence even off-policy
```

**TDC** (TD with correction):
```
Similar to GTD2
Slightly different formulation
Also convergent off-policy
```

**Update equations** (simplified):
```
w ← w + α · δ · x - α · γ · (w^T · x) · (x - γ · x')
```

This includes the correction term that makes it a true gradient!

---

## When to Use Each

**Use semi-gradient TD when**:
- On-policy learning
- Simplicity is important
- Computational efficiency matters
- You want fast learning
- Standard problems (Mountain Car, CartPole)

**Use true gradient TD (GTD/TDC) when**:
- Off-policy learning required
- Stability is critical
- You need convergence guarantees
- You can afford extra computation
- Research or theoretical work

**Use deep learning / neural networks**:
- Modern approach: DQN, PPO, etc.
- Uses semi-gradient but with stabilization tricks
- Target networks, experience replay
- Accepts lack of guarantees for practical performance

---

## Philosophical Perspective

**Semi-gradient methods reflect a fundamental insight**:

RL is not just optimization! It's about:
1. **Prediction**: What will happen (value estimation)
2. **Control**: What should I do (policy improvement)
3. **Successive approximation**: Iteratively solving Bellman equation

The semi-gradient naturally fits this framework. It's not trying to minimize a global loss; it's doing **local consistency** updates toward the Bellman fixed point.

**True gradient descent** tries to minimize a global objective, which is sometimes:
- Not what we really want
- Slower to optimize
- Less aligned with RL semantics

**The lesson**: In RL, the "right" algorithm isn't always true gradient descent. Sometimes, well-designed heuristics (like semi-gradient) that match the problem structure work better!

</details>

