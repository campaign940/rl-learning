# Week 8: Value Function Approximation

## Learning Objectives

- [ ] Understand the limitations of tabular methods and the need for function approximation
- [ ] Learn linear function approximation and feature construction methods
- [ ] Implement tile coding for continuous state spaces
- [ ] Understand gradient descent methods for value function learning
- [ ] Recognize the deadly triad and its implications
- [ ] Apply semi-gradient methods to Mountain Car problem

## Key Concepts

### Function Approximation

**Definition**: Using parameterized functions to represent value functions instead of tables, enabling generalization across states.

**Intuition**: When the state space is too large or continuous, we can't store a value for every state. Instead, we learn a function v(s, w) with parameters w that approximates the true value function.

**Key insight**: Updates to one state affect the values of similar states, enabling generalization.

### Linear Function Approximation

**Definition**: Value function represented as a linear combination of features.

**Equation**:
```
v(s, w) = w^T · x(s) = Σ_i w_i · x_i(s)
```

Where:
- w is the weight vector (parameters)
- x(s) is the feature vector for state s
- Each x_i(s) is a feature of state s

**Intuition**: Each feature captures some aspect of the state, and we learn how much each feature contributes to the value.

**Properties**:
- Convergence guarantees under certain conditions
- Efficient computation
- Interpretable weights
- Limited representational power

### Feature Construction

**Definition**: Designing or learning representations x(s) that capture relevant aspects of states.

**Common methods**:

1. **Polynomial features**: x(s) = [1, s, s^2, s^3, ...]
2. **Fourier basis**: trigonometric functions
3. **Tile coding**: partition state space into overlapping grids
4. **Radial basis functions**: Gaussian-like features centered at different points

**Key principle**: Good features should capture structure in the value function while enabling generalization.

### Tile Coding

**Definition**: A method of feature construction that uses multiple overlapping grid-like tilings of the state space.

**Intuition**:
- Single grid: coarse approximation
- Multiple offset grids: fine-grained approximation through overlap
- Each tile is a binary feature (1 if state is in tile, 0 otherwise)

**Properties**:
- Computationally efficient
- Good generalization
- Adjustable resolution via number and size of tilings
- Popular for continuous state spaces

**Example**: For 2D state space with 4 tilings:
```
Tiling 1: grid with offset (0, 0)
Tiling 2: grid with offset (0.25, 0)
Tiling 3: grid with offset (0, 0.25)
Tiling 4: grid with offset (0.25, 0.25)
```

Each state activates exactly 4 tiles (one per tiling).

### Stochastic Gradient Descent (SGD)

**Definition**: Updating parameters in the direction that reduces error for the current sample.

**General update rule**:
```
w_{t+1} = w_t - α/2 · ∇_w [v_π(s) - v(s, w)]^2
       = w_t + α · [v_π(s) - v(s, w)] · ∇_w v(s, w)
```

Where:
- α is the step size (learning rate)
- v_π(s) is the true value (target)
- v(s, w) is our approximation
- ∇_w v(s, w) is the gradient of our approximation

**For linear approximation**: ∇_w v(s, w) = x(s)

**Challenge**: We don't know the true value v_π(s), so we substitute it with estimates.

### Semi-Gradient Methods

**Definition**: Gradient descent methods that use bootstrapped targets, treating the target as constant (not differentiating through it).

**Semi-gradient TD(0) update**:
```
w_{t+1} = w_t + α · [R_{t+1} + γ · v(S_{t+1}, w_t) - v(S_t, w_t)] · ∇_w v(S_t, w_t)
        = w_t + α · δ_t · ∇_w v(S_t, w_t)
```

Where δ_t = R_{t+1} + γ · v(S_{t+1}, w_t) - v(S_t, w_t) is the TD error.

**Why "semi-gradient"?**: We don't differentiate through the target R_{t+1} + γ · v(S_{t+1}, w_t). A true gradient would include ∂v(S_{t+1}, w)/∂w, but we treat the target as constant.

**For linear approximation**:
```
w_{t+1} = w_t + α · [R_{t+1} + γ · w_t^T · x(S_{t+1}) - w_t^T · x(S_t)] · x(S_t)
```

### The Deadly Triad

**Definition**: Three elements that, when combined, can cause instability and divergence in reinforcement learning.

**The three elements**:

1. **Function Approximation**: Using parameterized functions instead of tables
   - Necessary for large/continuous state spaces
   - Updates affect multiple states

2. **Bootstrapping**: Using estimates to update estimates
   - TD learning: using v(S_{t+1}) to update v(S_t)
   - Faster learning than Monte Carlo
   - Target depends on current parameters

3. **Off-Policy Learning**: Learning about one policy while following another
   - Important for learning optimal policy
   - Experience replay in DQN
   - Breaks distribution assumptions

**Why dangerous together?**:
- Function approximation amplifies errors
- Bootstrapping propagates these errors
- Off-policy learning compounds instability with distribution mismatch

**Historical example**: Baird's counterexample showed divergence with all three present.

**Mitigation strategies**:
- Use on-policy methods (remove off-policy)
- Use Monte Carlo instead of TD (remove bootstrapping)
- Careful algorithm design (DQN uses target networks, experience replay)
- Gradient TD methods (true gradient descent)

### Semi-Gradient SARSA

**Update rule**:
```
w_{t+1} = w_t + α · [R_{t+1} + γ · q(S_{t+1}, A_{t+1}, w_t) - q(S_t, A_t, w_t)] · ∇_w q(S_t, A_t, w_t)
```

**On-policy**: Learns about the policy being followed, avoiding one element of the deadly triad.

**Linear form**:
```
q(s, a, w) = w^T · x(s, a)
w_{t+1} = w_t + α · δ_t · x(S_t, A_t)
```

Where δ_t = R_{t+1} + γ · q(S_{t+1}, A_{t+1}, w_t) - q(S_t, A_t, w_t)

## Textbook References

- **Sutton & Barto, 2nd Edition**:
  - Chapter 9: On-policy Prediction with Approximation
    - 9.1-9.3: Value-function approximation, prediction objective
    - 9.4: Stochastic gradient and semi-gradient methods
    - 9.5: Linear methods
    - 9.5.4: Tile coding
  - Chapter 10: On-policy Control with Approximation
    - 10.1: Episodic semi-gradient SARSA
    - 10.2: n-step semi-gradient SARSA
  - Chapter 11: Off-policy Methods with Approximation
    - 11.2: The deadly triad
    - 11.3: Linear value-function geometry

- **David Silver's RL Course**:
  - Lecture 6: Value Function Approximation
    - Feature vectors
    - Linear value function approximation
    - Incremental prediction algorithms
    - Convergence of prediction algorithms

## Implementation Details

### Mountain Car with Linear Function Approximation + Tile Coding

**Environment**: Mountain Car
- State: (position, velocity) ∈ [-1.2, 0.6] × [-0.07, 0.07]
- Actions: {left, none, right}
- Goal: Reach position ≥ 0.5
- Challenge: Not enough power to drive straight up, must build momentum

**Feature Construction**:
```python
class TileCoder:
    def __init__(self, num_tilings=8, tiles_per_dim=8, state_bounds=None):
        """
        num_tilings: Number of overlapping tilings
        tiles_per_dim: Number of tiles per dimension in each tiling
        state_bounds: [(low, high), ...] for each dimension
        """
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.state_bounds = state_bounds

        # Calculate tile width for each dimension
        self.tile_widths = [
            (high - low) / tiles_per_dim
            for low, high in state_bounds
        ]

        # Create offsets for each tiling
        self.offsets = [
            [i * width / num_tilings for width in self.tile_widths]
            for i in range(num_tilings)
        ]

    def get_tiles(self, state):
        """
        Returns indices of active tiles for the given state.
        Returns list of num_tilings integers.
        """
        tiles = []
        for tiling_idx in range(self.num_tilings):
            # Compute tile coordinates for this tiling
            coords = []
            for dim_idx, (value, (low, high)) in enumerate(zip(state, self.state_bounds)):
                # Apply offset for this tiling
                offset_value = value - self.offsets[tiling_idx][dim_idx]
                # Compute tile coordinate
                tile_coord = int((offset_value - low) / self.tile_widths[dim_idx])
                tile_coord = max(0, min(self.tiles_per_dim - 1, tile_coord))
                coords.append(tile_coord)

            # Convert coordinates to single index
            tile_idx = tiling_idx * (self.tiles_per_dim ** len(state))
            for coord in coords:
                tile_idx = tile_idx * self.tiles_per_dim + coord
            tiles.append(tile_idx)

        return tiles

    def get_feature_vector(self, state, action, num_actions):
        """
        Returns sparse feature vector for (state, action) pair.
        Only indices corresponding to active tiles are non-zero.
        """
        tiles = self.get_tiles(state)
        # Offset tiles by action * total_num_tiles
        total_tiles = self.num_tilings * (self.tiles_per_dim ** len(self.state_bounds))
        feature_indices = [tile + action * total_tiles for tile in tiles]
        return feature_indices
```

**Semi-Gradient SARSA Implementation**:
```python
class LinearSARSA:
    def __init__(self, tile_coder, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.tile_coder = tile_coder
        self.num_actions = num_actions
        self.alpha = alpha / tile_coder.num_tilings  # Divide by num active features
        self.gamma = gamma
        self.epsilon = epsilon

        # Weight vector (initialize to zeros)
        total_tiles = tile_coder.num_tilings * (tile_coder.tiles_per_dim ** 2)
        self.w = np.zeros(total_tiles * num_actions)

    def q_value(self, state, action):
        """Compute q(s, a, w) = w^T · x(s, a)"""
        feature_indices = self.tile_coder.get_feature_vector(state, action, self.num_actions)
        return np.sum(self.w[feature_indices])

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            q_values = [self.q_value(state, a) for a in range(self.num_actions)]
            return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action, done):
        """Semi-gradient SARSA update"""
        # Compute TD error
        q_current = self.q_value(state, action)
        if done:
            td_target = reward
        else:
            q_next = self.q_value(next_state, next_action)
            td_target = reward + self.gamma * q_next

        td_error = td_target - q_current

        # Update weights for active features
        feature_indices = self.tile_coder.get_feature_vector(state, action, self.num_actions)
        self.w[feature_indices] += self.alpha * td_error

    def train(self, env, num_episodes):
        """Training loop"""
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            action = self.select_action(state)
            total_reward = 0
            done = False

            while not done:
                next_state, reward, done, _ = env.step(action)
                next_action = self.select_action(next_state)

                self.update(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action
                total_reward += reward

            episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")

        return episode_rewards
```

**Key Implementation Notes**:

1. **Learning rate scaling**: Divide α by the number of active features (num_tilings) to keep update magnitude consistent.

2. **Feature representation**: Use sparse representation (list of active indices) for efficiency with large feature spaces.

3. **Exploration**: Epsilon-greedy is simple but effective. Consider decreasing epsilon over time.

4. **Hyperparameters**:
   - num_tilings: 8-16 typical (more = finer resolution)
   - tiles_per_dim: 8-16 typical (more = more features, slower)
   - alpha: 0.1-0.5 (after scaling by num_tilings)
   - epsilon: 0.1 for exploration

5. **Visualization**: Plot value function v(s) or q(s, a) over state space to see learned approximation.

## Review Questions

1. **Why can't we use tabular methods for Mountain Car?**
   - The state space is continuous: (position, velocity) ∈ ℝ²
   - Infinite number of possible states
   - Can't maintain a table with entries for every state

2. **What is the difference between supervised learning and reinforcement learning with function approximation?**
   - Supervised: Have true labels y for inputs x, minimize E[(y - f(x))²]
   - RL: Don't have true values, use bootstrapped estimates as targets
   - RL targets are non-stationary (change as we learn)
   - RL faces exploration-exploitation tradeoff

3. **Why use multiple tilings instead of a single fine-grained grid?**
   - Single fine grid: Many features, slow learning, poor generalization
   - Multiple coarse tilings: Efficient generalization across similar states
   - Overlapping tilings provide fine discrimination where needed
   - Computationally efficient (only few features active per state)

4. **Prove that linear semi-gradient TD(0) converges for on-policy prediction.**
   - This follows from stochastic approximation theory
   - Key conditions: α decreases appropriately, on-policy guarantees correct distribution
   - Weight updates are contractions in expectation
   - See Sutton & Barto Theorem 9.1 for formal proof

5. **What happens if we use Q-learning (off-policy) with linear function approximation on Mountain Car?**
   - Potential for instability due to deadly triad
   - Function approximation + bootstrapping + off-policy
   - May diverge in some cases (though often works in practice)
   - On-policy SARSA is more stable

6. **How would you extend tile coding to handle 4-dimensional state spaces?**
   - Same principle: Create num_tilings offset grids in 4D
   - Each state activates one tile per tiling
   - Computational cost: tiles_per_dim^4 * num_tilings features
   - May need fewer tiles_per_dim to keep feature count manageable

7. **Compare the representational power of linear vs. polynomial features.**
   - Linear: Can only represent linear functions (hyperplanes)
   - Polynomial (degree d): Can represent more complex functions with curves and interactions
   - Higher degree = more expressiveness but more parameters
   - Tile coding (piecewise linear) can approximate any continuous function

8. **Why do we divide the learning rate by the number of active tilings?**
   - Each update affects multiple weights (one per active tiling)
   - Without scaling: update magnitude would grow with num_tilings
   - Division keeps expected update magnitude constant
   - Ensures consistent learning dynamics regardless of num_tilings

9. **What are the trade-offs between table lookup and function approximation?**
   - Tabular: Exact representation, no generalization, only discrete/small spaces
   - Function approximation: Generalization, compact, works with large/continuous spaces
   - Function approximation: May not represent optimal policy perfectly, can be unstable

10. **How would you diagnose if your tile coding has too few or too many tiles?**
    - Too few: Coarse approximation, can't represent value function well, poor performance
    - Too many: Slow learning, requires more samples, potential overfitting
    - Visualize learned value function to see if it captures structure
    - Monitor learning curves: faster learning with fewer tiles, better asymptotic performance with more
