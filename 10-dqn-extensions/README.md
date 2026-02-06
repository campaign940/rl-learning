# Week 10: DQN Extensions

## Learning Objectives

- [ ] Understand the overestimation bias in Q-learning and how Double DQN addresses it
- [ ] Learn the Dueling architecture and why separating V(s) and A(s,a) helps
- [ ] Implement Prioritized Experience Replay (PER) with importance sampling
- [ ] Understand Rainbow DQN as a combination of improvements
- [ ] Apply these extensions to improve DQN performance on challenging environments

## Key Concepts

### Double DQN: Fixing Overestimation Bias

**The Problem**: Standard DQN overestimates Q-values due to max operator

**Standard DQN target**:
```
y = r + γ · max_a' Q(s', a'; θ^-)
      = r + γ · Q(s', argmax_a' Q(s', a'; θ^-); θ^-)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     Same network selects and evaluates action
```

**Why overestimation occurs**:
- max(noisy estimates) > true max
- Selection bias: Pick action with positive noise
- Propagates through bootstrapping

**Double DQN solution**: Decouple selection and evaluation

```
y = r + γ · Q(s', argmax_a' Q(s', a'; θ); θ^-)
                  ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^
                  Select with online network   Evaluate with target network
```

**Key insight**: Selection and evaluation errors are independent, reducing bias

**Paper**: van Hasselt et al. (2016)

**Improvement**: 5-20% better performance on Atari

### Dueling Architecture: Separating Value and Advantage

**Motivation**: Q(s,a) = V(s) + A(s,a)
- V(s): How good is state s?
- A(s,a): How much better is action a than average?

**Standard DQN**:
```
State → CNN → Flatten → FC → Q(s,a1), Q(s,a2), ..., Q(s,a|A|)
```

**Dueling DQN**:
```
State → CNN → Flatten → Split:
                         ├─ Value stream → V(s)
                         └─ Advantage stream → A(s,a1), ..., A(s,a|A|)

Combine: Q(s,a) = V(s) + (A(s,a) - mean_a' A(s,a'))
```

**Why the mean subtraction?**
- Identifiability: Otherwise V and A not unique
- Forces A to have zero mean
- Makes V represent true state value

**Benefits**:
- Faster learning: V learned once, shared across actions
- Better generalization: Separates state quality from action choice
- Sample efficiency: 20-50% improvement

**Paper**: Wang et al. (2016)

### Prioritized Experience Replay (PER)

**Motivation**: Not all transitions are equally useful
- High TD-error: Network hasn't learned this transition well
- Low TD-error: Already learned, less informative
- Rare important events: Should be sampled more often

**Standard replay**: Uniform sampling P(i) = 1/N

**Prioritized replay**: Sample based on TD error

**Priority definition**:
```
p_i = |δ_i| + ε

where:
- δ_i = r + γ max_a' Q(s', a'; θ^-) - Q(s, a; θ) is TD error
- ε = small constant (e.g., 0.01) ensures p_i > 0
```

**Sampling probability**:
```
P(i) = p_i^α / Σ_j p_j^α

where α controls prioritization strength:
- α = 0: uniform (standard replay)
- α = 1: proportional to priority
- typical: α = 0.6
```

**Importance sampling correction**:
Prioritized sampling introduces bias, correct with importance sampling weights:

```
w_i = (N · P(i))^{-β} / max_j w_j

where β ∈ [0, 1]:
- β = 0: no correction
- β = 1: full correction
- typical: anneal from 0.4 to 1.0 over training
```

**Modified loss**:
```
L = (1/B) Σ_i w_i · [y_i - Q(s_i, a_i; θ)]²
```

**Implementation**: Use sum-tree data structure for efficient O(log N) sampling

**Benefits**:
- 30-50% faster learning
- Huge gains on sparse reward tasks
- Focuses learning where it matters

**Paper**: Schaul et al. (2016)

### Rainbow: Combining Everything

**Motivation**: Individual improvements help, but combining them helps more

**Rainbow combines 6 extensions**:

1. **Double DQN**: Reduces overestimation
2. **Prioritized replay**: Samples important transitions
3. **Dueling networks**: Separates V and A
4. **Multi-step learning**: n-step returns (better credit assignment)
5. **Distributional RL**: Learns value distribution, not just mean
6. **Noisy Nets**: Learned exploration noise

**Results**: State-of-the-art on Atari as of 2017

**Ablation study findings**:
- Prioritized replay: Most important single component
- Distributional RL: Second most important
- Multi-step: Also very important
- All components contribute positively

**Paper**: Hessel et al. (2018)

## Textbook References

- **Sutton & Barto, 2nd Edition**:
  - Chapter 16: Applications (mentions modern deep RL)

- **Berkeley CS285**:
  - Lecture 8: Deep RL with Q-Functions (Advanced)

## Key Papers

### Double DQN
- **van Hasselt, H., Guez, A., & Silver, D. (2016).** "Deep Reinforcement Learning with Double Q-Learning"
  - AAAI 2016
  - Addresses overestimation bias

### Dueling DQN
- **Wang, Z., et al. (2016).** "Dueling Network Architectures for Deep Reinforcement Learning"
  - ICML 2016
  - Separates value and advantage streams

### Prioritized Experience Replay
- **Schaul, T., et al. (2016).** "Prioritized Experience Replay"
  - ICLR 2016
  - Non-uniform sampling based on TD error

### Rainbow DQN
- **Hessel, M., et al. (2018).** "Rainbow: Combining Improvements in Deep Reinforcement Learning"
  - AAAI 2018
  - Comprehensive ablation study

## Implementation Details

### Double DQN Implementation

**Minimal change from DQN**:

```python
# Standard DQN target
with torch.no_grad():
    max_next_q = target_network(next_states).max(1)[0]
    target_q = rewards + gamma * max_next_q * (1 - dones)

# Double DQN target
with torch.no_grad():
    # Select action with online network
    next_actions = q_network(next_states).argmax(1)
    # Evaluate with target network
    next_q_values = target_network(next_states).gather(1, next_actions.unsqueeze(1))
    target_q = rewards + gamma * next_q_values.squeeze() * (1 - dones)
```

**Key difference**: Two-line change!

### Dueling DQN Implementation

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()

        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine with mean subtraction
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
```

### Prioritized Experience Replay Implementation

**Sum-tree data structure**:

```python
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def update(self, idx, p):
        """Update priority at leaf idx"""
        tree_idx = idx + self.capacity - 1
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # Propagate change up tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, p, data):
        """Add new transition"""
        self.data[self.write] = data
        self.update(self.write, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get(self, s):
        """Sample transition with priority sum s"""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.epsilon = 0.01
        self.max_priority = 1.0

    def push(self, transition):
        """Add transition with max priority"""
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        """Sample batch with priorities"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total_p / batch_size

        # Anneal beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        # Compute importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_p
        weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        weights /= weights.max()  # Normalize

        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(idxs, td_errors):
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
```

**Training loop with PER**:

```python
# Sample batch
batch, idxs, weights = replay_buffer.sample(batch_size)
states, actions, rewards, next_states, dones = zip(*batch)

# Convert to tensors
weights = torch.FloatTensor(weights)

# Compute loss (same as before)
current_q = q_network(states).gather(1, actions)
target_q = compute_target(next_states, rewards, dones)
td_errors = target_q - current_q

# Weighted loss
loss = (weights * td_errors.pow(2)).mean()

# Update priorities
replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy())
```

## Review Questions

1. **Why does Q-learning overestimate values? Provide a concrete numerical example.**

2. **How does Double DQN reduce overestimation bias? Explain the decoupling of selection and evaluation.**

3. **What is the identifiability problem in Dueling DQN? Why do we subtract the mean advantage?**

4. **Derive the importance sampling weight formula for PER. Why is β annealed from 0.4 to 1.0?**

5. **Compare uniform replay vs proportional PER vs rank-based PER. What are trade-offs?**

6. **If you could only add one extension to DQN, which would you choose and why?**

7. **What does Rainbow's ablation study tell us about combining improvements?**

8. **How would you implement rank-based prioritization instead of proportional?**

9. **Why does Dueling architecture help more in some environments than others?**

10. **What are the computational costs of each extension? Which is most expensive?**
