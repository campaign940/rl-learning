# Week 7 Resources: Planning and Learning

## Primary Textbooks

### Sutton & Barto: Reinforcement Learning - An Introduction (2nd Edition)

**Chapter 8: Planning and Learning with Tabular Methods**
- [Full Book PDF](http://incompleteideas.net/book/RLbook2020.pdf)
- [Chapter 8 Direct Link](http://incompleteideas.net/book/RLbook2020.pdf#page=193)
- Essential sections:
  - 8.1: Models and Planning
  - 8.2: Dyna: Integrated Planning, Acting, and Learning
  - 8.3: When the Model is Wrong
  - 8.4: Prioritized Sweeping
  - 8.5: Expected vs Sample Updates
  - 8.6: Trajectory Sampling
  - 8.11: Monte Carlo Tree Search
  - 8.12: Summary of the Chapter

### David Silver's RL Course

- **Lecture 8: Integrating Learning and Planning**
  - [Lecture Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/dyna.pdf)
  - [Video Lecture](https://www.youtube.com/watch?v=ItMutbeOHtc)
  - Topics: Model-based RL, Dyna, MCTS, AlphaGo
  - Comprehensive coverage of planning methods

## University Courses

### CS234: Reinforcement Learning (Stanford)
- **Lecture 6: Model-Based RL**
  - [Course Website](http://web.stanford.edu/class/cs234/)
  - Model learning and planning

### CS285: Deep Reinforcement Learning (UC Berkeley)
- **Lecture 11: Model-Based RL**
  - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
  - Modern deep model-based methods
  - [Slides](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-11.pdf)

## Research Papers

### Classic Papers on Dyna

- **Sutton (1990): Integrated Architectures for Learning, Planning, and Reacting**
  - [Paper PDF](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c2e8ac9e05c3e5b39c3f7b8c9a5c6e2e6e5f2f8c)
  - Original Dyna architecture
  - Foundational paper

- **Sutton (1991): Dyna, an Integrated Architecture for Learning, Planning, and Reacting**
  - [Paper PDF](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Dyna-AIJ.pdf)
  - Extended Dyna paper
  - Must-read for Dyna understanding

- **Moore & Atkeson (1993): Prioritized Sweeping**
  - [Paper PDF](https://www.cs.cmu.edu/~cga/ai-course/prioritized.pdf)
  - Efficient planning with priorities
  - Significant improvement over random planning

### MCTS Papers

- **Coulom (2006): Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search**
  - [Paper PDF](https://www.remi-coulom.fr/CG2006/CG2006.pdf)
  - Introduced MCTS to computer Go

- **Kocsis & Szepesvári (2006): Bandit-based Monte-Carlo Planning (UCT)**
  - [Paper PDF](https://link.springer.com/content/pdf/10.1007/11871842_29.pdf)
  - UCT algorithm (foundation of modern MCTS)
  - Theoretical analysis

- **Browne et al. (2012): A Survey of Monte Carlo Tree Search Methods**
  - [Paper PDF](https://ieeexplore.ieee.org/document/6145622)
  - Comprehensive MCTS survey
  - Variants and applications

### AlphaGo / AlphaZero

- **Silver et al. (2016): Mastering the Game of Go with Deep Neural Networks and Tree Search**
  - [Nature Paper](https://www.nature.com/articles/nature16961)
  - AlphaGo: MCTS + deep learning
  - Breakthrough achievement

- **Silver et al. (2017): Mastering the Game of Go without Human Knowledge (AlphaGo Zero)**
  - [Nature Paper](https://www.nature.com/articles/nature24270)
  - Pure self-play learning
  - Stronger than original AlphaGo

- **Silver et al. (2018): A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go (AlphaZero)**
  - [Science Paper](https://www.science.org/doi/10.1126/science.aar6404)
  - Generalized to multiple games
  - Single algorithm for Chess, Shogi, Go

- **Schrittwieser et al. (2020): Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)**
  - [Nature Paper](https://www.nature.com/articles/s41586-020-03051-4)
  - Learns model, doesn't need true rules
  - State-of-the-art model-based RL

### Modern Model-Based Deep RL

- **Chua et al. (2018): Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)**
  - [Paper PDF](https://arxiv.org/pdf/1805.12114.pdf)
  - Uncertainty-aware model-based RL
  - Very sample efficient

- **Janner et al. (2019): When to Trust Your Model: Model-Based Policy Optimization (MBPO)**
  - [Paper PDF](https://arxiv.org/pdf/1906.08253.pdf)
  - Short model rollouts to avoid compounding errors
  - State-of-the-art continuous control

- **Hafner et al. (2020): Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)**
  - [Paper PDF](https://arxiv.org/pdf/1912.01603.pdf)
  - Learn and plan in latent space
  - Efficient image-based RL

- **Hansen et al. (2022): Temporal Difference Learning for Model Predictive Control (TD-MPC)**
  - [Paper PDF](https://arxiv.org/pdf/2203.04955.pdf)
  - Combines TD learning with model predictive control
  - Simple and effective

## Blog Posts and Tutorials

### Model-Based RL

- **Lil'Log: Model-Based RL**
  - [Blog Post](https://lilianweng.github.io/posts/2019-06-23-meta-rl/#model-based-rl)
  - Clear explanations of model-based approaches
  - Modern algorithms covered

- **Towards Data Science: Dyna-Q Explained**
  - [Article](https://towardsdatascience.com/dyna-q-integrating-planning-and-learning-8c3f7d1e7e2a)
  - Practical implementation guide

- **Medium: Model-Based RL Overview**
  - [Article](https://medium.com/@jonathan_hui/rl-model-based-reinforcement-learning-3c2b6f0aa323)
  - Comparison of different approaches

### MCTS

- **Jeff Bradberry: Monte Carlo Tree Search**
  - [Tutorial](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/)
  - Excellent step-by-step explanation
  - Python implementation

- **The Nature of Code: MCTS**
  - [Tutorial](https://natureofcode.com/book/chapter-9-the-evolution-of-code/#921-monte-carlo-tree-search)
  - Visual explanations

- **AI Game Dev: MCTS Tutorial**
  - [Series](http://www.aifactory.co.uk/newsletter/2013_01_reduce_burden.htm)
  - Detailed implementation guide

### AlphaGo / AlphaZero

- **DeepMind Blog: AlphaGo**
  - [Blog Post](https://www.deepmind.com/research/highlighted-research/alphago)
  - Official explanation from DeepMind

- **Towards Data Science: AlphaZero Explained**
  - [Article](https://towardsdatascience.com/alphazero-explained-8b8b5af0c5f1)
  - Accessible breakdown

- **Medium: MuZero Explained**
  - [Article](https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a)
  - Implementation guide

## Implementation Resources

### Code Repositories

**Dyna-Q Implementations**:
- **Dennybritz RL Repository**
  - [GitHub](https://github.com/dennybritz/reinforcement-learning)
  - [Dyna-Q Notebook](https://github.com/dennybritz/reinforcement-learning/tree/master/Planning)

- **Sutton & Barto Official Code**
  - [GitHub](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
  - [Chapter 8 Code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter08)
  - Dyna Maze, Prioritized Sweeping implementations

**MCTS Implementations**:
- **Simple MCTS (Python)**
  - [GitHub](https://github.com/pbsinclair42/MCTS)
  - Clean, educational implementation

- **MCTS for Games**
  - [GitHub](https://github.com/int8/monte-carlo-tree-search)
  - Tic-Tac-Toe, Connect Four examples

- **AlphaZero-style MCTS**
  - [GitHub](https://github.com/suragnair/alpha-zero-general)
  - Neural network + MCTS
  - Multiple games

**Model-Based Deep RL**:
- **MBPO Implementation**
  - [GitHub](https://github.com/janner/mbpo)
  - Official implementation

- **Dreamer**
  - [GitHub](https://github.com/danijar/dreamer)
  - TensorFlow implementation

- **TD-MPC**
  - [GitHub](https://github.com/nicklashansen/tdmpc)
  - PyTorch implementation

### Environments for Testing

**Dyna-Q**:
```python
# Dyna Maze (from S&B)
# Implement custom grid world
# Or use Gymnasium's FrozenLake

import gymnasium as gym
env = gym.make('FrozenLake-v1', is_slippery=False)  # Deterministic
```

**MCTS**:
```python
# Games: Tic-Tac-Toe, Connect Four, Go
# Libraries: python-chess, gym-go

# Simple Tic-Tac-Toe
class TicTacToe:
    # ... implement game logic ...
```

**Model-Based RL**:
```python
# MuJoCo: Continuous control
import gymnasium as gym
env = gym.make('HalfCheetah-v4')

# Or simpler: CartPole, Pendulum
env = gym.make('CartPole-v1')
```

## Video Lectures and Talks

### David Silver's Lecture

- **Lecture 8: Integrating Learning and Planning**
  - [YouTube](https://www.youtube.com/watch?v=ItMutbeOHtc)
  - Timestamp 0:00 - 30:00: Dyna
  - Timestamp 30:00 - 60:00: MCTS
  - Best single video on these topics

### DeepMind Talks

- **AlphaGo Documentary**
  - [YouTube](https://www.youtube.com/watch?v=WXuK6gekU1Y)
  - Full documentary about AlphaGo vs Lee Sedol
  - Compelling story

- **DeepMind x UCL: Model-Based RL**
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
  - Lecture 8: Planning and Models

### Conference Talks

- **David Silver: AlphaZero**
  - [YouTube](https://www.youtube.com/watch?v=Wujy7OzvdJk)
  - NIPS 2017 talk

- **Richard Sutton: Dyna Architecture**
  - Classic talks on Dyna (search on YouTube)

## Interactive Demos

### MCTS Visualizations

- **MCTS Visualization Tool**
  - [Demo](https://vgarciasc.github.io/mcts-viz/)
  - Interactive tree visualization
  - See selection, expansion, simulation in action

- **AlphaGo Visualization**
  - [Demo](https://alphagomovie.com/)
  - Visualize AlphaGo's thinking

### Dyna Experiments

- **Implement in Colab**
  - Build Dyna-Q from scratch
  - Visualize value function evolution
  - Compare with Q-Learning

## Tools and Libraries

### Python Libraries

**Core RL**:
```bash
pip install gymnasium       # Environments
pip install numpy           # Numerical computing
pip install matplotlib      # Visualization
```

**MCTS Libraries**:
```bash
pip install mcts            # MCTS implementation
pip install python-chess    # Chess for MCTS testing
```

**Model-Based RL**:
```bash
pip install torch           # PyTorch for neural networks
pip install mujoco          # Physics simulation
pip install dm-control      # DeepMind control suite
```

**Visualization**:
```bash
pip install graphviz        # Tree visualization
pip install networkx        # Graph visualization
```

### Game Libraries

**For MCTS Testing**:
- **python-chess**: Chess implementation
- **gym-go**: Go environment
- **pettingzoo**: Multi-agent environments

## Theoretical Foundations

### Books

- **Sutton & Barto (2018): Reinforcement Learning: An Introduction**
  - Chapter 8: Essential reading

- **Dimitri Bertsekas (2019): Reinforcement Learning and Optimal Control**
  - Chapter on model-based methods
  - More mathematical treatment

- **Csaba Szepesvári (2010): Algorithms for Reinforcement Learning**
  - Theoretical analysis of planning

### Papers on Theory

- **Sample Complexity of Model-Based RL**
  - Analysis of when model-based is more efficient

- **UCT Convergence**
  - Proof that UCT converges to optimal policy

- **Model Error Compounding**
  - Analysis of how model errors accumulate

## Practical Guides

### Implementing Dyna-Q

**Step-by-Step Guide**:

1. **Start with Q-Learning**:
```python
Q = defaultdict(lambda: np.zeros(n_actions))

# Standard Q-Learning loop
```

2. **Add Model Storage**:
```python
Model = {}  # (s,a) -> (r, s')

# After each step:
Model[(s,a)] = (r, s_next)
```

3. **Add Planning Loop**:
```python
for _ in range(n_planning):
    s_sim = random.choice(visited_states)
    a_sim = random.choice(actions_taken_in[s_sim])
    r_sim, s_next_sim = Model[(s_sim, a_sim)]
    # Q-Learning update with simulated experience
```

4. **Tune n**: Start with n=5, increase if real experience is expensive

### Implementing MCTS

**Step-by-Step Guide**:

1. **Node Structure**:
```python
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
```

2. **Selection** (UCT):
```python
def select_child(self, c=sqrt(2)):
    return max(self.children.values(),
               key=lambda n: n.value/n.visits +
                             c*sqrt(log(self.visits)/n.visits))
```

3. **Expansion**:
```python
def expand(self):
    actions = get_legal_actions(self.state)
    for action in actions:
        next_state = apply_action(self.state, action)
        self.children[action] = Node(next_state, parent=self)
```

4. **Simulation**:
```python
def simulate(state):
    while not is_terminal(state):
        action = random.choice(get_legal_actions(state))
        state = apply_action(state, action)
    return get_reward(state)
```

5. **Backpropagation**:
```python
def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        reward = -reward  # Flip for adversary
        node = node.parent
```

## Common Mistakes and Debugging

### Dyna-Q Mistakes

1. **Not storing model**:
```python
# Wrong: model is never updated
Model = (r, s_next)

# Correct: use dictionary
Model[(s,a)] = (r, s_next)
```

2. **Sampling unvisited states**:
```python
# Wrong: may sample never-visited states
s_sim = random.randint(0, n_states)

# Correct: sample from visited states
s_sim = random.choice(list(Model.keys()))[0]
```

3. **Not enough planning**:
```python
# n=1 is too low, won't see benefit
# Try n=10 or higher
```

### MCTS Mistakes

1. **Wrong UCT constant**:
```python
# Theoretical: c = sqrt(2)
# Practical: tune c between 0.5 and 2.0
```

2. **Not enough iterations**:
```python
# 10 iterations: too few
# 1000 iterations: good starting point
# 10000 iterations: strong play
```

3. **Biased rollouts**:
```python
# Random rollouts can be weak
# Consider using heuristic or learned policy
```

## Advanced Topics

### Research Frontiers

- **Latent Space Models**: Plan in learned abstract space (Dreamer, MuZero)
- **Uncertainty Quantification**: Ensemble models, Bayesian neural networks
- **Meta-Learning Models**: Learn to learn models across tasks
- **World Models**: Generative models for planning
- **Adversarial Robustness**: Robust to model errors

### Modern Applications

**Games**:
- AlphaZero (Chess, Go, Shogi)
- MuZero (Atari without rules)
- OpenAI Five (Dota 2)

**Robotics**:
- Model-based manipulation
- Locomotion with learned models
- Sim-to-real transfer

**Other Domains**:
- Recommendation systems (sequential decision making)
- Resource allocation (planning with models)
- Autonomous driving (planning with learned dynamics)

## Quick Reference

### Dyna-Q

```python
# Main loop
for each step:
    1. Act in environment (ε-greedy)
    2. Observe (s, a, r, s')
    3. Update Q from real experience
    4. Update Model[(s,a)] = (r, s')
    5. Planning (n times):
       - Sample (s_sim, a_sim) from visited
       - Query model: (r_sim, s'_sim) = Model[(s_sim, a_sim)]
       - Update Q from simulated experience

Hyperparameters:
  - n: planning steps (5-50)
  - α: learning rate (0.1-0.5)
  - ε: exploration (0.1)
```

### MCTS

```python
# Main loop
for each iteration:
    1. Selection: UCT from root to leaf
    2. Expansion: Add child(ren) to leaf
    3. Simulation: Random rollout to terminal
    4. Backpropagation: Update all nodes on path

UCT: argmax[Q(s,a) + c·sqrt(ln(N(s))/N(s,a))]

Hyperparameters:
  - iterations: 100-10000
  - c: exploration constant (√2)
  - rollout policy: random or heuristic
```

## Next Steps

After completing Week 7:

1. **Solid Tabular Foundation**: You've mastered all core tabular RL algorithms
2. **Ready for Function Approximation**: Scale to large/continuous state spaces
3. **Prepared for Deep RL**: DQN, A3C, PPO build on these foundations
4. **Understand Model-Based vs Model-Free**: When to use each approach

### Recommended Learning Path

1. Read S&B Chapter 8 thoroughly
2. Watch David Silver Lecture 8
3. Implement Dyna-Q on Dyna Maze
4. Implement MCTS for Tic-Tac-Toe
5. Compare Dyna-Q with Q-Learning empirically
6. Read AlphaGo paper (at least introduction and methods)
7. Complete quiz problems
8. Move to advanced topics (function approximation, policy gradients)

### Bridge to Advanced RL

Planning and learning integration is crucial for:
- **Modern deep RL**: MuZero, Dreamer, MBPO
- **Robotics**: Model-based control essential
- **Sample efficiency**: Critical for real-world applications
- **Transfer learning**: Models generalize across tasks

Week 7 completes your foundation in reinforcement learning. You now understand all major paradigms: value-based, policy-based, model-free, and model-based. The integration of planning and learning (Dyna) shows these are complementary approaches, not opposing philosophies!

## Summary of Weeks 4-7

You've covered:
- **Week 4**: Monte Carlo (model-free, no bootstrapping)
- **Week 5**: Temporal-Difference (model-free, bootstrapping)
- **Week 6**: n-step & Eligibility Traces (unifying TD and MC)
- **Week 7**: Planning and Learning (model-based + model-free)

These four weeks complete the core of tabular RL. You're now ready for function approximation, deep RL, and advanced topics!
