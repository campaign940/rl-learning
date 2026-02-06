# Week 7 Quiz: Planning and Learning

Test your understanding of model-based RL, Dyna architecture, and Monte Carlo Tree Search.

## Question 1: Model-Based vs Model-Free Advantage

**What is the key advantage of model-based over model-free RL? Under what circumstances can this advantage be lost or reversed?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**The Key Advantage: Sample Efficiency**

Model-based RL can achieve significantly better **sample efficiency** - learning more from each interaction with the real environment.

**Why This Matters**:
- Real-world interactions are often expensive (robots can break, clinical trials have costs, A/B tests have revenue impact)
- Sample efficiency determines how quickly an agent can learn a good policy
- In simulation, this matters less, but in reality, it's often the bottleneck

---

### How Model-Based Achieves Better Sample Efficiency

**1. Experience Replay on Steroids**

Model-free (e.g., Q-Learning):
```
Real experience: (s, a, r, s')
Update: Q(s,a) once using this transition
Discard transition (or store in replay buffer)
```

Model-based (e.g., Dyna-Q):
```
Real experience: (s, a, r, s')
Update model: Model(s,a) ← (r, s')
Direct update: Q(s,a) once from real experience

Then, repeatedly sample from model:
  For i = 1 to n:
    (s_sim, a_sim) ~ previously seen (s,a)
    (r_sim, s'_sim) ← Model(s_sim, a_sim)
    Update: Q(s_sim, a_sim) using simulated experience

Total: 1 + n updates from a single real experience!
```

**Multiplier Effect**: If n=10, model-based gets 11 updates per real experience vs 1 for model-free.

---

**2. Planning Before Acting**

Model-free:
- Must take action to learn
- Trial-and-error in real world
- Expensive mistakes

Model-based:
- Can simulate outcomes before acting
- Try many actions mentally (in model)
- Choose best action based on planning
- Avoid costly mistakes

**Example: Robot Grasping**
```
Model-free: Try 1000 random grasps in real world (hours, wear and tear)
Model-based: Learn model from 100 grasps, simulate 10,000 grasps (minutes), achieve same performance
```

---

**3. Generalization Through Model**

Model-free:
- Value function: maps (s,a) → value
- Must learn separately for each state-action pair
- Limited generalization

Model-based:
- Model: maps (s,a) → (s',r)
- Model can generalize (especially with function approximation)
- Plan for new states using the same model

**Example: Navigation**
```
Model-free: Learn Q(s,a) for going north in every location separately
Model-based: Learn "going north moves you 1 grid cell north"
  This single rule applies to all locations!
```

---

**4. Reusability for Multiple Goals**

Model-free:
- Learn policy/value for specific reward
- Change reward → must relearn from scratch

Model-based:
- Model is task-independent (just environment dynamics)
- Change reward → replan using same model
- No additional environment interaction needed

**Example: Robot in Kitchen**
```
Model-free:
  Learn "grab cup" → train 1000 episodes
  Learn "grab plate" → train another 1000 episodes

Model-based:
  Learn physics model → 500 episodes
  Plan "grab cup" → use model (0 episodes)
  Plan "grab plate" → use same model (0 episodes)
```

---

### When the Advantage is Lost or Reversed

**1. Model Errors (Most Critical)**

**The Problem**: If the model is inaccurate, planning with it leads to suboptimal or catastrophic policies.

**Example: Cliff Walking**
```
True dynamics: Going right near cliff → fall (R=-100)
Learned model (wrong): Going right near cliff → reach goal (R=+1)

Planning with wrong model:
  Agent plans risky path along cliff (looks optimal in model)
  Executes in reality → falls repeatedly
  Model-based policy: Terrible
  Model-free policy: Learns to avoid cliff, much better
```

**When This Happens**:
- Environment is complex (high-dimensional observations, non-linear dynamics)
- Limited data for model learning (model overfits or underfits)
- Environment is stochastic (model can't capture full distribution)
- Model class is insufficiently expressive

**Result**: Model-based can be **worse** than model-free if model errors dominate.

---

**2. Computational Cost of Planning**

**The Problem**: Planning requires computation. If computation is expensive relative to environment interaction, model-based is less attractive.

**Example: Simulated Atari Games**
```
Model-free (DQN):
  Environment interaction: Nearly free (simulator runs fast)
  Computation: Forward pass through network (milliseconds)
  Bottleneck: Neither, can collect millions of transitions

Model-based:
  Environment interaction: Still free
  Computation: Plan with learned model (seconds per decision)
  Bottleneck: Planning computation!

Result: Model-free is faster and simpler
```

**When This Happens**:
- Environment simulator is fast and available
- Planning is expensive (large state space, deep search)
- Computational resources are limited
- Real-time performance required

---

**3. Simplicity of Direct Learning**

**The Problem**: Model learning adds complexity. For simple tasks, direct learning may be easier.

**Example: CartPole**
```
Model-free:
  State: 4D (position, velocity, angle, angular velocity)
  Q-function: Maps 4D state → 2 actions (simple)
  Training: 100 episodes, converges quickly

Model-based:
  Must learn: p(s' | s, a) for 4D continuous state
  Model: More complex (4D → 4D mapping)
  Training: Learn model + plan (extra complexity)

If task is simple enough, model-free is simpler and sufficient
```

**When This Happens**:
- Task is simple (low-dimensional, short horizon)
- Model-free already sample-efficient enough
- Development time is limited
- Don't need reusability for multiple goals

---

**4. Model Exploitation (Adversarial Environment)**

**The Problem**: Agent finds and exploits errors in model that don't exist in reality.

**Example: Simulated Robot**
```
True dynamics: Robot falls if tipping angle > 30°
Learned model (imperfect): Doesn't capture tipping dynamics correctly

Planning:
  Agent discovers: "If I tip to 45°, I move faster" (in model)
  Executes: Tips to 45° in reality → falls!

Agent optimizes for model flaws, not reality
```

**When This Happens**:
- Model has systematic errors
- Agent is powerful enough to find and exploit them
- Environment has safety constraints (falling, breaking)

**Solution**: Uncertainty-aware planning, adversarial robustness, ensemble models

---

**5. High-Dimensional Observations**

**The Problem**: Learning accurate models from high-dimensional observations (images, audio) is much harder than learning policies.

**Example: Atari Games from Pixels**
```
Model-free (DQN):
  Learn: pixels → Q-values
  Easier: Direct mapping from observation to value

Model-based:
  Learn: pixels → predict next pixels + reward
  Much harder: Must model entire visual scene evolution
  Model errors in pixels → bad planning

State-of-the-art:
  Model-free (DQN): Superhuman performance
  Naive model-based: Struggles due to pixel prediction errors
```

**When This Happens**:
- High-dimensional observations (images, video, audio)
- Observation space much larger than action space
- Prediction is harder than control

**Modern Solution**: Learn latent models (MuZero) or plan in learned abstract spaces

---

### Quantitative Comparison

**Sample Efficiency** (Typical):
```
Simple task (CartPole):
  Model-free: 100 episodes
  Model-based: 50 episodes (2x better)

Medium task (Dyna Maze):
  Model-free: 1000 episodes
  Model-based: 50 episodes (20x better)

Complex task (Atari, pixels):
  Model-free: 10M frames
  Naive model-based: Doesn't converge (model errors)
  Advanced model-based (MuZero): 5M frames (2x better)
```

**Computation** (Typical):
```
Model-free: 1x computation (baseline)
Model-based: 5-50x computation (planning overhead)

Only worthwhile if real environment interaction is much more expensive than computation
```

---

### When to Use Model-Based

**Strong Indicators**:
1. Real environment interactions are expensive (robotics, healthcare, manufacturing)
2. Environment dynamics are relatively simple and learnable
3. Need to solve multiple tasks in same environment
4. Have computational resources for planning
5. Can collect diverse data for model learning
6. Safety is critical (simulate dangerous scenarios)

**Example Domains**:
- Robotics (real robot time is expensive)
- Healthcare (clinical trials cost lives and money)
- Manufacturing (downtime is costly)
- Finance (real trades have market impact)
- Strategy games (planning is natural)

---

### When to Use Model-Free

**Strong Indicators**:
1. Have access to cheap simulation (Atari, simulated robotics)
2. Environment dynamics are complex (vision, physics with contact)
3. High-dimensional observations (images, audio)
4. Simple task where direct learning suffices
5. Need fast real-time inference (no planning time)
6. Computational resources are limited

**Example Domains**:
- Video games (simulation available)
- Simulated robotics (Mujoco, PyBullet)
- Recommendation systems (can A/B test cheaply online)
- Simple control tasks (low-dimensional state)

---

### Hybrid Approaches (Best of Both)

**Dyna Architecture**:
- Use model for planning (sample efficiency)
- Use model-free for safety net (robustness to model errors)
- Gradually trust model more as it improves

**Short-Horizon Planning**:
- Use model only for short-term (1-5 steps)
- Model errors don't compound over long horizons
- Get some sample efficiency without catastrophic model errors

**Model-Based Value Expansion**:
- Use model to generate n-step returns
- Update value function (model-free component)
- Combines benefits of both

**Examples**:
- AlphaZero: MCTS (model-based planning) + neural network (learned policy/value)
- MuZero: Learned latent model + planning + policy/value learning
- MBPO: Short model rollouts + SAC (model-free)

---

### Key Insights

1. **Main advantage of model-based: Sample efficiency** through experience reuse and planning

2. **Advantage is lost when**:
   - Model errors dominate (complex environments)
   - Computation cost outweighs sample savings (cheap simulation)
   - Task is simple enough for direct learning
   - High-dimensional observations make model learning hard

3. **Not binary**: Hybrid approaches combine strengths of both

4. **Task-dependent**: No universal winner; depends on cost of samples vs computation, model learnability, and task complexity

5. **Modern trend**: Sophisticated model-based methods (MuZero, Dreamer) can match or exceed model-free even in complex domains

The choice between model-based and model-free is fundamentally about **tradeoffs**: sample efficiency vs robustness, planning ability vs simplicity, generalization vs specialization. Understanding when each advantage matters is key to choosing the right approach!

</details>

---

## Question 2: Dyna-Q Algorithm

**Describe the Dyna-Q algorithm step by step. Why does it combine Q-Learning updates for both real and simulated experience? What role does the parameter n play?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Dyna-Q** integrates three components: (1) direct RL (learning from real experience), (2) model learning (building environment model), and (3) planning (learning from simulated experience).

---

### Step-by-Step Dyna-Q Algorithm

**Initialization**:
```python
Q(s,a) ← 0 for all s ∈ S, a ∈ A
Model(s,a) ← null for all s, a
```

**Main Loop** (for each episode):

```python
Initialize state S

while S is not terminal:
    # ===== STEP 1: Acting =====
    Choose action A from S using ε-greedy policy based on Q
    # A = argmax_a Q(S,a) with probability 1-ε, random with probability ε

    # ===== STEP 2: Observe Result =====
    Take action A in real environment
    Observe reward R and next state S'

    # ===== STEP 3: Direct RL (Model-Free Learning) =====
    # Q-Learning update from real experience
    Q(S,A) ← Q(S,A) + α[R + γ·max_{a'} Q(S',a') - Q(S,A)]

    # ===== STEP 4: Model Learning =====
    # Update model with observed transition
    Model(S,A) ← (R, S')  # Store or update model entry

    # ===== STEP 5: Planning (n times) =====
    for i in range(n):
        # 5a. Sample a previously observed state-action pair
        S_sim ← random state from states visited so far
        A_sim ← random action taken in S_sim (that was stored in Model)

        # 5b. Query model for predicted outcome
        (R_sim, S'_sim) ← Model(S_sim, A_sim)

        # 5c. Q-Learning update from simulated experience
        Q(S_sim, A_sim) ← Q(S_sim, A_sim) + α[R_sim + γ·max_a Q(S'_sim,a) - Q(S_sim, A_sim)]

    # ===== STEP 6: Move to Next State =====
    S ← S'
```

---

### Why Combine Q-Learning for Both Real and Simulated Experience?

**1. Direct RL (Step 3) - Ground Truth Updates**

```python
Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) - Q(S,A)]
```

**Purpose**: Learn directly from real experience.

**Why Essential**:
- Real experience is **ground truth** (no model errors)
- Ensures Q-function is grounded in reality
- Provides safety net if model is wrong
- Allows model-free learning to happen concurrently

**Without this step**: Dyna-Q would be purely model-based, vulnerable to model errors.

---

**2. Planning (Step 5c) - Accelerated Learning**

```python
Q(S_sim, A_sim) ← Q(S_sim, A_sim) + α[R_sim + γ·max_a Q(S'_sim,a) - Q(S_sim, A_sim)]
```

**Purpose**: Learn from simulated experience generated by model.

**Why Essential**:
- **Multiplies** the impact of each real experience
- Real experience updates Q once; model allows n additional updates
- Propagates value information faster through state space
- "Thinks ahead" between real environment interactions

**Without this step**: Dyna-Q would be just Q-Learning (model-free only).

---

**Why Both Are Needed Together**:

**Model-Free Component (Direct RL)**:
- ✅ Robust to model errors
- ✅ Ground truth from real experience
- ❌ Sample inefficient (learns only from real experience)
- ❌ Slow value propagation

**Model-Based Component (Planning)**:
- ✅ Sample efficient (reuses experience)
- ✅ Fast value propagation
- ❌ Vulnerable to model errors
- ❌ Depends on model quality

**Combined (Dyna-Q)**:
- ✅ Sample efficient AND robust
- ✅ Fast learning with safety net
- ✅ Gradually trusts model as it improves
- ✅ Best of both worlds

---

### The Role of Parameter n

**n** = Number of planning steps per real environment step.

**What n Controls**:
- How much planning happens between real actions
- Computational budget allocated to thinking vs acting
- Balance between sample efficiency and computational efficiency

---

**Small n (e.g., n=0)**:
```
n=0: Pure Q-Learning (no planning)
  - Most robust (no model dependence)
  - Least sample efficient
  - Fastest per step (no planning overhead)
  - Use when: Environment is cheap to interact with
```

**Medium n (e.g., n=5 to 50)**:
```
n=10:
  - Each real step: 1 real update + 10 simulated updates
  - 11x more updates per environment interaction
  - Moderate computational cost
  - Good balance for most tasks
  - Use when: Want faster learning without excessive computation
```

**Large n (e.g., n=100 to 1000)**:
```
n=100:
  - Each real step: 1 real + 100 simulated updates
  - 101x more updates per real interaction
  - Very sample efficient
  - Slow per step (expensive planning)
  - Use when: Environment interaction is very expensive
```

**Extreme n (n→∞)**:
```
n→∞:
  - Planning until convergence between actions
  - Maximum sample efficiency
  - Essentially model-based planning (like Dyna-PI)
  - Extremely slow
  - Use when: Environment interaction is prohibitively expensive
```

---

### Empirical Effects of n

**Dyna Maze Example**:

```
Episodes to reach goal:

n=0 (Q-Learning):    ~800 episodes
n=5:                 ~100 episodes  (8x improvement)
n=10:                ~50 episodes   (16x improvement)
n=50:                ~25 episodes   (32x improvement)

Computation time per episode:

n=0:     1x  (baseline)
n=5:     6x  (5 planning steps + 1 real step)
n=10:    11x
n=50:    51x

Wallclock time (episodes × time per episode):

n=0:     800 × 1x = 800 units
n=5:     100 × 6x = 600 units  (faster!)
n=10:    50 × 11x = 550 units  (even faster)
n=50:    25 × 51x = 1275 units (slower due to planning overhead)

Optimal n ≈ 10 for this task
```

---

### Choosing n: Practical Guidelines

**Consider**:
1. **Cost of real environment interaction** (C_real)
2. **Cost of planning step** (C_plan)
3. **Speedup from planning** (learning rate improvement)

**Optimal n** roughly when:
```
Marginal benefit of planning = Marginal cost

speedup(n) / C_plan ≈ constant

Typically: n ∈ [5, 50] for most tasks
```

**Decision Framework**:

| Situation | Recommended n | Reasoning |
|-----------|---------------|-----------|
| Fast simulator | 0-5 | Real experience is cheap |
| Real robot | 50-1000 | Real experience very expensive |
| Complex environment | 10-50 | Need more planning for propagation |
| Simple environment | 5-10 | Less planning needed |
| Online real-time | 0-10 | Need fast response |
| Offline batch | 100+ | Can afford planning time |
| Early training | 10-50 | Fast learning needed |
| Late training | 0-5 | Fine-tuning, model may have errors |

---

### Why Use Q-Learning Specifically?

**Q-Learning properties that matter**:

1. **Off-policy**: Learns Q* regardless of behavior policy
   - Planning updates are "off-policy" (simulated states don't follow current policy)
   - Q-Learning handles this naturally
   - SARSA wouldn't work as well (on-policy)

2. **Sample backup**: Uses single sample instead of expectation
   - Efficient with sample model
   - Model provides single (r, s') sample

3. **Max operator**: Learns optimal values
   - Planning should learn about best actions
   - Max operator focuses on optimistic values

**Could use other algorithms**:
- Expected SARSA: Works, may be better
- SARSA: Doesn't work well (on-policy mismatch with planning)
- TD: Requires policy, Q-Learning learns Q* directly

---

### Variants of Dyna-Q

**Dyna-Q+**: Exploration bonuses for unvisited states
```python
# Add bonus to simulated rewards for states not seen recently
R_sim = R_sim + κ·sqrt(time_since_visit(S_sim))
```

**Prioritized Dyna**: Priority queue instead of random sampling
```python
# Sample states with largest TD errors
(S_sim, A_sim) = priority_queue.pop()  # highest priority
```

**Dyna-2**: Two value functions (long-term, short-term)
- Long-term: learns from real experience
- Short-term: learns from simulated + real
- Combines their estimates

---

### Common Pitfalls

1. **Not storing model updates**:
```python
# Wrong: model never updates
Model(S,A) = (R, S')  # Overwrites each time? Need data structure

# Correct: deterministic model
Model[(S,A)] = (R, S')  # Dictionary

# Correct: stochastic model (average outcomes)
Model[(S,A)].add_sample(R, S')
```

2. **Sampling unvisited states**:
```python
# Wrong: may sample states never visited
S_sim = random_state()

# Correct: only sample visited states
S_sim = random.choice(visited_states)
```

3. **Same learning rate for planning as direct RL**:
```python
# Works but may want to tune separately
α_real = 0.5   # for real experience
α_sim = 0.3    # for simulated (if less confident in model)
```

---

### Key Insights

1. **Dyna-Q unifies model-free and model-based RL** in a simple, elegant way

2. **Direct RL provides robustness**, planning provides sample efficiency

3. **n is the key hyperparameter**: Controls planning amount
   - Higher n → fewer episodes but more computation per episode
   - Optimal n balances learning speed and computational cost

4. **Q-Learning is used because**:
   - Off-policy (works with planning)
   - Sample-based (works with sample model)
   - Learns optimal Q* directly

5. **Practical impact**: Dyna-Q typically achieves **10-50x** sample efficiency improvement over pure Q-Learning with modest computational cost

Dyna-Q demonstrates that model-based and model-free are not opposing paradigms but complementary approaches that can be seamlessly integrated!

</details>

---

## Question 3: Dyna-Q Performance Comparison

**Compare Dyna-Q with and without planning steps on the Cliff Walking environment. Explain why the performance difference might be larger or smaller than on the Dyna Maze. What determines when Dyna-Q provides the most benefit?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

(Due to length constraints, this answer is abbreviated. A full answer would include detailed comparison of Dyna-Q performance on Cliff Walking vs Dyna Maze, analysis of when planning helps most, and empirical results.)

**Key Points**:

**Cliff Walking Characteristics**:
- Simple layout, one optimal path
- Sparse rewards (-100 for cliff, 0 elsewhere, -1 per step)
- Short episodes once policy learned
- Deterministic transitions

**Dyna-Q Performance on Cliff Walking**:

```
Without planning (n=0, pure Q-Learning):
  - Episodes to convergence: ~200-500
  - Learns slowly due to sparse rewards

With planning (n=10):
  - Episodes to convergence: ~50-100
  - 5-10x improvement
  - Planning propagates cliff penalty backward quickly
```

**Comparison to Dyna Maze**:

| Factor | Dyna Maze | Cliff Walking | Impact on Dyna Benefit |
|--------|-----------|---------------|------------------------|
| State space size | 54 states | 48 states | Similar |
| Episode length | 100-300 steps | 20-40 steps | Shorter → less benefit |
| Reward sparsity | Sparse (only goal) | Very sparse (only cliff/goal) | More sparse → more benefit |
| Path complexity | Multiple paths | One clear optimal | Less complex → less benefit |
| Backups needed | Many | Fewer | Fewer → less benefit |

**Why Dyna-Q Helps More on Dyna Maze**:

1. **Longer episodes**: More steps mean more states to update, planning spreads value faster

2. **Complex exploration**: Multiple paths require exploring many states; planning leverages all explored states

3. **Deep value propagation**: Terminal reward is far from start; planning bridges the gap

**When Dyna-Q Provides Most Benefit**:

1. **Sparse rewards**: Planning propagates rare rewards to many states
2. **Large state spaces**: More states to visit, planning uses model to update unvisited
3. **Long episodes**: More time between rewards, planning connects distant states
4. **Expensive real experience**: Cost of real interaction >> cost of planning
5. **Reusable knowledge**: States visited in one episode help plan for others

**When Dyna-Q Provides Least Benefit**:

1. **Dense rewards**: Value information already local
2. **Small state spaces**: Visit all states quickly anyway
3. **Short episodes**: Less value propagation needed
4. **Cheap simulation**: Might as well collect real experience
5. **Highly stochastic**: Model is inaccurate, planning is misleading

</details>

---

## Question 4: Monte Carlo Tree Search (MCTS)

**Explain MCTS and how AlphaGo used it. Walk through one iteration of the four phases (selection, expansion, simulation, backpropagation) with a concrete example.**

<details>
<summary>Click to reveal answer</summary>

### Answer:

**Monte Carlo Tree Search (MCTS)** is a planning algorithm that builds a search tree incrementally, focusing on promising regions using Monte Carlo sampling.

---

### The Four Phases of MCTS

Let's use **Tic-Tac-Toe** as a concrete example.

**Setup**:
```
Current board:
X | O | .
---------
. | X | .
---------
. | . | .

It's X's turn. Root node represents this state.
```

---

**Phase 1: Selection**

Starting from root, **traverse tree using selection policy** (typically UCT) until reaching a leaf node.

**UCT Formula**:
```
Select action: a* = argmax_a [Q(s,a) + c·sqrt(ln(N(s)) / N(s,a))]
                               └ exploitation ┘  └─ exploration ─┘
```

Where:
- Q(s,a) = average value of action a from state s
- N(s) = visit count for state s
- N(s,a) = visit count for state-action pair
- c = exploration constant (typically √2)

**Example Iteration 5** (after 4 previous iterations):
```
Root state (X to move):
  Action "top-right": Q=0.5, N=2  → UCT = 0.5 + √2·sqrt(ln(4)/2) ≈ 1.18
  Action "center-left": Q=0.67, N=1 → UCT = 0.67 + √2·sqrt(ln(4)/1) ≈ 2.34 ← highest!
  Action "bottom-left": Q=0, N=1 → UCT = 0 + √2·sqrt(ln(4)/1) ≈ 1.67

Select "center-left" (highest UCT)
```

**After selection**:
```
Move to child node:
X | O | .
---------
X | X | .   ← This state (after X plays center-left)
---------
. | . | .
```

**Continue selection** if this child has been visited:
```
If this node has children (been expanded), apply UCT again
Otherwise, stop (reached leaf)
```

In this case, assume this node has **not** been expanded yet (leaf node).

---

**Phase 2: Expansion**

**Add one or more child nodes** to the tree.

From the leaf state, generate possible next states (available actions):

```
Leaf state (O to move after X played center-left):
X | O | .
---------
X | X | .
---------
. | . | .

Available actions for O:
  1. top-right
  2. center-right
  3. bottom-left
  4. bottom-center
  5. bottom-right

Expansion: Add all 5 children to the tree
(or just one child, depending on implementation)
```

Let's add one child: O plays "center-right" to block X.

```
New node added:
X | O | .
---------
X | X | O  ← O blocks
---------
. | . | .

Initialize: Q=0, N=0 for this new node
```

---

**Phase 3: Simulation (Rollout)**

From the new node, **simulate a random game** to completion (terminal state).

**Random Rollout**:
```
Start state:
X | O | .
---------
X | X | O
---------
. | . | .

Move 1: X plays randomly → bottom-left
X | O | .
---------
X | X | O
---------
X | . | .

Move 2: O plays randomly → bottom-center
X | O | .
---------
X | X | O
---------
X | O | .

Move 3: X plays randomly → top-right
X | O | X  ← X wins! (diagonal)
---------
X | X | O
---------
X | O | .

Result: X wins (+1 for X, -1 for O)
```

**Simulation Result**: +1 (from root player X's perspective)

---

**Phase 4: Backpropagation**

**Propagate the result back up the tree**, updating statistics for all nodes visited.

**Update Path**:
```
Root → "center-left" action → O's turn → "center-right" action → (simulation)

Backpropagate +1 (X win)
```

**Updates** (from leaf to root):

1. **New node** (O blocked):
```
Before: Q=0, N=0
After: Q=1, N=1   (value=+1, visits=1)
Wait, this is O's node, so from O's perspective: Q=-1, N=1
```

2. **Parent node** (X played center-left):
```
Before: Q=0.67·1 = 0.67, N=1
After: Total value = 0.67 + 1 = 1.67, N=2
       Q = 1.67/2 = 0.835
```

3. **Root node**:
```
Before: N=4
After: N=5  (increment visit count)
```

All nodes on the path from root to simulation node are updated.

---

### Complete MCTS Algorithm

```python
def mcts(root_state, iterations, c=sqrt(2)):
    root = Node(root_state)

    for i in range(iterations):
        node = root

        # 1. Selection
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child_UCT(c)

        # 2. Expansion
        if not node.is_terminal():
            node = node.expand()  # Add one child

        # 3. Simulation
        reward = simulate_random_game(node.state)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_value += reward
            reward = -reward  # Flip reward for alternating players
            node = node.parent

    # Return action with most visits (most reliable)
    return argmax_a root.children[a].visits
```

---

### How AlphaGo Used MCTS

**AlphaGo combines MCTS with deep neural networks**:

**Classic MCTS**:
- Selection: UCT based on visit counts and average values
- Simulation: Random rollouts until game end

**AlphaGo MCTS Enhancements**:

1. **Neural Network Policy** (Selection):
```
Instead of random: a ~ π_θ(a|s)  (learned policy)
UCT: Q(s,a) + c·P(s,a)·sqrt(N(s)) / (1 + N(s,a))
                  ↑
          network prior P(s,a) = π_θ(a|s)
```

Neural network guides which moves to explore.

2. **Neural Network Value** (Simulation):
```
Instead of random rollout to end:
  v = V_θ(s)  (learned value function)

Combine with rollout:
  Final value = λ·v + (1-λ)·rollout_value
```

Neural network evaluates positions without full simulation.

3. **Much Stronger**:
- Policy network learned from expert games (supervised)
- Value network learned from self-play (reinforcement learning)
- MCTS uses networks to search smarter
- Networks improved by MCTS search results

**AlphaGo Zero** (even better):
- No human expert data
- Single neural network for both policy and value
- Pure self-play
- MCTS + neural network training loop

**Result**: Superhuman Go playing.

---

### Why MCTS Works

**Strengths**:
1. **Anytime algorithm**: More iterations → better performance
2. **Focuses on promising regions**: UCT balances exploration/exploitation
3. **No need for value function**: Simulations provide estimates
4. **Handles huge state spaces**: Builds tree adaptively
5. **Asymmetric tree growth**: Explores good moves more deeply

**Example**:
```
After 1000 iterations:
  Good moves: 800 visits (deep exploration)
  Bad moves: 50 visits each (shallow exploration)

Computational budget spent where it matters!
```

**Weaknesses**:
1. **Computationally expensive**: Many simulations needed
2. **Random rollouts are weak**: Unless using learned policy
3. **Slow convergence**: In very large spaces
4. **Need terminal rewards**: Harder for continuing tasks

---

### Key Insights

1. **MCTS builds tree incrementally**, focusing on promising regions via UCT

2. **Four phases work together**:
   - Selection: Navigate to promising leaf
   - Expansion: Add new node
   - Simulation: Estimate value
   - Backpropagation: Update statistics

3. **UCT balances exploration and exploitation** naturally

4. **AlphaGo's innovation**: Combine MCTS with deep learning
   - Neural networks guide search
   - MCTS refines neural network play
   - Synergy achieves superhuman performance

5. **Modern trend**: MCTS + learned models/values (MuZero, AlphaZero)

MCTS is one of the most successful planning algorithms, enabling breakthrough performance in games (Go, Chess, Shogi) and beyond!

</details>

---

## Question 5: Model Errors and Exploitation

**When can a learned model hurt performance? Describe the "model exploitation problem" where an agent finds and exploits flaws in its model. How can this be mitigated?**

<details>
<summary>Click to reveal answer</summary>

### Answer:

(Due to length, this is abbreviated. Full answer would include detailed examples of model exploitation, mathematical analysis of compounding errors, and mitigation strategies.)

**The Model Exploitation Problem**:

An agent can discover and exploit systematic errors in its learned model, leading to policies that work well in the model but fail catastrophically in reality.

**Why It Happens**:

1. **Optimization Pressure**: Agent actively searches for high-reward policies
2. **Model Errors**: Learned models are imperfect approximations
3. **Distribution Shift**: Agent visits states where model is inaccurate
4. **Compounding**: Errors accumulate over multiple time steps

**Concrete Example**:

```
True environment: Robot walking, falls if leans > 30°
Learned model: Incorrectly predicts lean angle dynamics

Agent discovers:
  "In my model, leaning 45° makes me walk faster"
  Plans: Lean 45° for speed

Reality:
  Leans 45° → Falls!

Model error exploited catastrophically
```

**Mitigation Strategies**:

1. **Uncertainty Quantification**: Track model confidence, avoid uncertain regions
2. **Ensemble Models**: Use multiple models, if they disagree, don't trust
3. **Adversarial Robustness**: Plan for worst-case within uncertainty
4. **Short Horizons**: Use model only for short-term (errors don't compound)
5. **Model-Free Safety Net**: Combine with model-free learning (Dyna)
6. **Careful Exploration**: Don't visit states where model is untrained
7. **Reality Check**: Periodically validate model predictions vs reality

**Key Insight**: Model-based RL must handle model errors carefully to avoid catastrophic exploitation of model flaws.

</details>

---

## Additional Practice Problems

1. Implement Dyna-Q and compare with Q-Learning on a custom maze environment.
2. Implement prioritized sweeping and measure planning efficiency vs Dyna-Q.
3. Build MCTS for Connect Four; analyze how performance scales with iteration count.
4. Design an environment where a learned model's errors lead to catastrophic exploitation.
5. Implement model-based value expansion (n-step with learned model) and compare with n-step TD.

