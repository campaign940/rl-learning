# Week 9: Deep Q-Networks (DQN)

## Learning Objectives

- [ ] Understand why naive Q-learning with neural networks is unstable
- [ ] Learn the key innovations of DQN: experience replay and target networks
- [ ] Implement DQN for CartPole environment
- [ ] Understand the DQN loss function and Huber loss
- [ ] Scale DQN to Atari games with visual inputs
- [ ] Analyze the convergence properties and limitations of DQN

## Key Concepts

### The Problem: Instability of Naive Neural Q-Learning

**Why doesn't standard Q-learning work with neural networks?**

Traditional Q-learning update:
```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]
```

With neural networks Q(s, a; θ), this becomes unstable due to:

1. **Correlated samples**: Sequential states are highly correlated
   - Consecutive experiences: (s_t, a_t, r_t, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2}), ...
   - Neural networks assume i.i.d. data
   - Correlation leads to overfitting to recent experiences

2. **Non-stationary targets**: Target changes with every update
   - Target: r + γ · max_a' Q(s', a'; θ)
   - As θ updates, target shifts
   - Chasing a moving target → oscillations

3. **Deadly triad**: Function approximation + bootstrapping + off-policy
   - Neural networks are powerful function approximators
   - Q-learning bootstraps (uses Q(s') to update Q(s))
   - Q-learning is off-policy (learns optimal Q while acting ε-greedily)

**Historical context**: Before DQN (2013), deep RL was considered impractical. Researchers believed neural networks couldn't work with RL.

### Deep Q-Network (DQN) Architecture

**Definition**: A neural network that approximates the action-value function Q(s, a; θ).

**Standard architecture** (for visual inputs like Atari):

```
Input: 84×84×4 grayscale frames (last 4 frames stacked)
    ↓
Conv Layer 1: 32 filters, 8×8, stride 4, ReLU
    ↓
Conv Layer 2: 64 filters, 4×4, stride 2, ReLU
    ↓
Conv Layer 3: 64 filters, 3×3, stride 1, ReLU
    ↓
Flatten
    ↓
Fully Connected: 512 units, ReLU
    ↓
Output: |A| units (one per action, linear activation)
```

**For CartPole** (low-dimensional state):
```
Input: 4-dimensional state vector
    ↓
Fully Connected: 128 units, ReLU
    ↓
Fully Connected: 128 units, ReLU
    ↓
Output: 2 units (left/right actions)
```

**Key design choices**:
- **Output representation**: One output per action (not state-action pairs as input)
- **Activation**: ReLU for hidden layers, linear for output
- **No softmax**: Output is Q-values, not probabilities

### Experience Replay

**Definition**: Store experiences in a replay buffer and sample random minibatches for training.

**How it works**:

1. **Replay buffer**: D = {e_1, e_2, ..., e_N}
   - Each experience: e_t = (s_t, a_t, r_t, s_{t+1}, done_t)
   - Fixed size buffer (e.g., 1M transitions)
   - Oldest experiences removed when full

2. **Training**: Sample random minibatch from D
   - Batch: {(s_i, a_i, r_i, s'_i, done_i)} for i = 1...B
   - Typically B = 32 or 64

3. **Update**: Use minibatch for gradient descent

**Why it works**:

**Breaks correlations**:
- Random sampling from buffer → decorrelated experiences
- Neural network sees diverse state distribution
- Reduces overfitting to recent trajectory

**Data efficiency**:
- Each experience can be used multiple times
- Learn from rare experiences repeatedly
- 10-100x sample efficiency improvement

**Stabilizes learning**:
- Smooth out variance in updates
- More stable gradient estimates
- Prevents catastrophic forgetting

**Mathematical insight**: Experience replay approximates the true expectation over the state distribution:
```
∇_θ J = E_{(s,a,r,s')~D} [∇_θ L(θ)]

Instead of following trajectory distribution, sample from D ≈ stationary distribution
```

### Target Network

**Definition**: A separate network with frozen parameters θ^- used to compute TD targets.

**How it works**:

1. **Two networks**:
   - Online network: Q(s, a; θ) — updated every step
   - Target network: Q(s, a; θ^-) — updated periodically

2. **TD target computation**: Use target network
   ```
   y_i = r_i + γ · max_a' Q(s'_i, a'; θ^-)
   ```

3. **Loss**: Minimize difference between online and target
   ```
   L(θ) = E[(y_i - Q(s_i, a_i; θ))²]
   ```

4. **Target network update**: Every C steps (e.g., C = 10,000)
   ```
   θ^- ← θ
   ```

**Why it works**:

**Stabilizes targets**:
- Targets stay fixed for C steps
- No moving target problem
- Reduces oscillations

**Breaks feedback loop**:
- Without target network: Q affects target, target affects Q → instability
- With target network: Q affects target slowly → stability

**Analogy**: Like fitting a function to data points that don't move while you're fitting.

**Trade-off**: Slower to incorporate improvements (θ^- lags behind θ), but much more stable.

### DQN Loss Function

**Standard squared error loss**:
```
L(θ) = E[(y - Q(s, a; θ))²]

where y = r + γ · max_a' Q(s', a'; θ^-)
```

**Huber loss** (used in practice):
```
L_δ(θ) = E[ℓ_δ(y - Q(s, a; θ))]

where ℓ_δ(e) = {
    (1/2) · e²           if |e| ≤ δ
    δ · (|e| - (1/2)δ)   if |e| > δ
}
```

**Why Huber loss?**

- **Robust to outliers**: Squared error sensitive to large TD errors
- **Smooth gradient**: Differentiable everywhere (unlike absolute error)
- **Combines best of both**: Quadratic for small errors, linear for large errors

**Gradient**:
```
∇_θ ℓ_δ(e) = {
    e · ∇_θ Q(s, a; θ)           if |e| ≤ δ
    δ · sign(e) · ∇_θ Q(s, a; θ)  if |e| > δ
}
```

**Typical choice**: δ = 1

### DQN Training Algorithm

**Complete algorithm**:

```
Initialize replay buffer D with capacity N
Initialize Q-network with random weights θ
Initialize target network with weights θ^- = θ

for episode = 1 to M:
    Initialize state s_1

    for t = 1 to T:
        # Select action
        With probability ε: select random action a_t
        Otherwise: a_t = argmax_a Q(s_t, a; θ)

        # Execute action
        Execute a_t, observe r_t, s_{t+1}, done

        # Store transition
        Store (s_t, a_t, r_t, s_{t+1}, done) in D

        # Sample minibatch
        Sample random minibatch of B transitions from D

        # Compute targets
        for each transition (s_i, a_i, r_i, s'_i, done_i):
            if done_i:
                y_i = r_i
            else:
                y_i = r_i + γ · max_a' Q(s'_i, a'; θ^-)

        # Gradient descent
        L(θ) = (1/B) · Σ_i [y_i - Q(s_i, a_i; θ)]²
        θ ← θ - α · ∇_θ L(θ)

        # Update target network
        Every C steps: θ^- ← θ

        if done: break
```

**Key hyperparameters**:
- Buffer size N: 100,000 - 1,000,000
- Minibatch size B: 32 - 64
- Target update frequency C: 1,000 - 10,000
- Learning rate α: 0.00001 - 0.0001 (often 0.00025)
- Discount factor γ: 0.99
- Exploration ε: 1.0 → 0.01 over 1M steps

## Textbook References

- **Sutton & Barto, 2nd Edition**:
  - Chapter 11: Off-policy Methods with Approximation
    - 11.3: The deadly triad
    - 11.7: Deep Q-learning (brief overview)
  - Chapter 16: Applications and Case Studies
    - 16.6: TD-Gammon (precursor to DQN)

- **Berkeley CS285 (Deep RL)**:
  - Lecture 7: Value Function Methods
    - Q-learning with function approximation
    - DQN innovations
    - Atari results

## Key Papers

### The DQN Paper

**Mnih, V., et al. (2015).** "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

**Key contributions**:
1. Experience replay for decorrelating samples
2. Target network for stabilizing learning
3. Frame preprocessing for Atari games
4. Demonstrated human-level performance on 49 Atari games

**Impact**: Revived interest in deep RL, proved neural networks can work with RL.

### Precursor Paper

**Mnih, V., et al. (2013).** "Playing Atari with Deep Reinforcement Learning." *NIPS Deep Learning Workshop*.

**First version** of DQN, introduced main ideas.

## Implementation Details

### CartPole DQN Implementation

**Environment**:
- State: (position, velocity, angle, angular_velocity) ∈ ℝ⁴
- Actions: {0: push left, 1: push right}
- Reward: +1 for each timestep alive
- Goal: Balance pole for 195+ steps (solved)

**Network architecture**:
```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Linear output (Q-values)
```

**Replay buffer**:
```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```

**DQN agent**:
```python
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.target_update_freq = 100
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample minibatch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, num_episodes):
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                # Select and execute action
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # Store transition
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Update agent
                self.update()

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return episode_rewards
```

**Usage**:
```python
import gym

env = gym.make('CartPole-v1')
agent = DQNAgent(state_dim=4, action_dim=2)
rewards = agent.train(env, num_episodes=500)
```

### Atari Pong DQN Implementation

**Key differences from CartPole**:

1. **Frame preprocessing**:
```python
def preprocess_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84))
    # Normalize
    normalized = resized / 255.0
    return normalized

def get_state(frame_buffer):
    # Stack last 4 frames
    return np.stack(frame_buffer, axis=0)
```

2. **CNN architecture**:
```python
class AtariDQN(nn.Module):
    def __init__(self, action_dim):
        super(AtariDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

3. **Larger replay buffer**: 1,000,000 transitions

4. **Longer training**: 10-50 million frames

5. **Frame skipping**: Repeat action for 4 frames (reduces computation)

6. **Reward clipping**: Clip rewards to [-1, +1] (stabilizes learning across games)

**Training time**: ~1-2 days on GPU for decent Pong performance.

## Review Questions

1. **Why does naive Q-learning fail with neural networks?**
   - Correlated sequential samples violate i.i.d. assumption
   - Non-stationary targets (target changes with every update)
   - Deadly triad: function approximation + bootstrapping + off-policy
   - Result: Catastrophic forgetting, divergence, oscillations

2. **How does experience replay address correlation?**
   - Stores past experiences in buffer
   - Samples random minibatches for training
   - Breaks temporal correlations
   - Neural network sees diverse state distribution
   - More sample efficient (reuse experiences)

3. **Why use a target network instead of a single network?**
   - Single network: target y = r + γ max_a' Q(s', a'; θ) depends on θ
   - Every update changes both Q(s,a) and target
   - Chasing moving target → instability
   - Target network: freeze parameters θ^- for C steps → stable targets

4. **What are the trade-offs of the target network update frequency C?**
   - Small C: More up-to-date targets, but less stable
   - Large C: More stable, but slower to incorporate improvements
   - Typical: C = 1,000 - 10,000 steps
   - Too large: target network too stale, slow learning

5. **Why use Huber loss instead of MSE?**
   - MSE sensitive to outliers (large TD errors)
   - Squared error amplifies large errors
   - Huber loss: quadratic for small errors, linear for large
   - More robust, smoother training
   - Prevents gradient explosion from outliers

6. **How would DQN perform without experience replay?**
   - Severe overfitting to recent experiences
   - Catastrophic forgetting of past knowledge
   - Very sample inefficient
   - Unstable learning curves
   - Empirically: often fails to learn

7. **How would DQN perform without target network?**
   - Oscillating Q-values
   - Divergence possible
   - Very unstable learning
   - Empirically: sometimes works but much worse

8. **Why stack 4 frames as input for Atari?**
   - Single frame: no velocity information
   - Need temporal information to infer motion
   - 4 frames: captures recent trajectory
   - Alternatives: recurrent networks (RNNs), but more complex

9. **What is the computational bottleneck in DQN?**
   - For Atari: Forward passes through CNN (mitigated by frame skipping)
   - Experience replay buffer memory (1M transitions × frame size)
   - Gradient computation through deep network
   - Not environment interaction (Atari is fast simulator)

10. **Can DQN be used for continuous action spaces?**
    - No! DQN requires max_a' Q(s', a') over discrete actions
    - Cannot maximize over continuous action space efficiently
    - Need different algorithms: DDPG, TD3, SAC (Week 14)
    - Workaround: Discretize action space (but loses precision)
