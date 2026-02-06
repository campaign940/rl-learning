# RL Learning Roadmap

Designed around **Sutton & Barto (2nd Ed.)** chapter progression, supplemented by **David Silver (UCL)**, **Stanford CS234**, **UC Berkeley CS285**, and **MIT 6.7950**.

---

## Phase 1: Foundations (Weeks 1-3)

> S&B Part I (Ch.1-4) | Silver Lectures 1-3 | CS234 Weeks 1-3

### Week 1: Introduction to Reinforcement Learning
- **S&B**: Ch.1 Introduction, Ch.2 Multi-armed Bandits
- **Silver**: Lecture 1 - Introduction to RL
- **Core concepts**: Agent, Environment, Reward Hypothesis, Exploration vs Exploitation
- **Implementation**: k-armed Bandit (epsilon-greedy, UCB, gradient bandit)
- **Key equations**: Action-value methods, incremental update rule

### Week 2: Markov Decision Processes
- **S&B**: Ch.3 Finite Markov Decision Processes
- **Silver**: Lecture 2 - Markov Decision Processes
- **Core concepts**: MDP tuple (S, A, P, R, gamma), State/Action Value Functions, Bellman Equations
- **Implementation**: GridWorld environment, Bellman equation solver
- **Key equations**: Bellman Expectation Equation, Bellman Optimality Equation

### Week 3: Dynamic Programming
- **S&B**: Ch.4 Dynamic Programming
- **Silver**: Lecture 3 - Planning by Dynamic Programming
- **Core concepts**: Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration
- **Implementation**: Gambler's Problem, Jack's Car Rental (simplified)
- **Key equations**: Iterative policy evaluation, GPI (Generalized Policy Iteration)

---

## Phase 2: Tabular Methods (Weeks 4-7)

> S&B Part I (Ch.5-8) | Silver Lectures 4-5, 8 | CS234 Weeks 4-6

### Week 4: Monte Carlo Methods
- **S&B**: Ch.5 Monte Carlo Methods
- **Silver**: Lecture 4 - Model-Free Prediction
- **Core concepts**: First-visit MC, Every-visit MC, MC Control, Exploring Starts, Off-policy MC
- **Implementation**: Blackjack agent (OpenAI Gym)
- **Key equations**: Importance sampling ratio, weighted importance sampling

### Week 5: Temporal-Difference Learning
- **S&B**: Ch.6 Temporal-Difference Learning
- **Silver**: Lecture 4-5 (Model-Free Prediction & Control)
- **Core concepts**: TD(0), SARSA, Q-Learning, Expected SARSA, Double Q-Learning
- **Implementation**: Cliff Walking, Windy GridWorld
- **Key equations**: TD error (delta), TD(0) update rule, Q-learning update

### Week 6: n-step Methods & Eligibility Traces
- **S&B**: Ch.7 n-step Bootstrapping, Ch.12 Eligibility Traces
- **Silver**: (supplementary material)
- **Core concepts**: n-step TD, n-step SARSA, TD(lambda), Forward/Backward view
- **Implementation**: Random Walk (19-state), Mountain Car with eligibility traces
- **Key equations**: n-step return, lambda-return, accumulating/replacing traces

### Week 7: Planning and Learning
- **S&B**: Ch.8 Planning and Tabular Methods
- **Silver**: Lecture 8 - Integrating Learning and Planning
- **Core concepts**: Dyna-Q, Model learning, MCTS, Simulation-based search
- **Implementation**: Dyna Maze, simple MCTS for Tic-Tac-Toe
- **Key equations**: Dyna-Q architecture, planning/learning unification

---

## Phase 3: Function Approximation (Weeks 8-10)

> S&B Part II (Ch.9-11) | Silver Lecture 6 | CS285 Lectures 7-8

### Week 8: Value Function Approximation
- **S&B**: Ch.9 On-policy Prediction with Approximation, Ch.10 On-policy Control with Approximation
- **Silver**: Lecture 6 - Value Function Approximation
- **Core concepts**: Linear FA, Tile Coding, SGD, Semi-gradient methods, Deadly triad
- **Implementation**: Mountain Car with linear FA + tile coding
- **Key equations**: Semi-gradient TD(0), linear value function, feature vectors

### Week 9: Deep Q-Networks (DQN)
- **S&B**: Ch.11 (Off-policy Methods with Approximation)
- **CS285**: Lecture 7 - Value Function Methods
- **Core concepts**: Experience Replay, Target Network, epsilon decay, Huber loss
- **Implementation**: CartPole-v1 DQN, Atari Pong DQN
- **Key paper**: Mnih et al. (2015) "Human-level control through deep RL"
- **Key equations**: DQN loss function, target network update

### Week 10: DQN Extensions
- **CS285**: Lecture 8 - Deep RL with Q-Functions
- **Core concepts**: Double DQN, Dueling DQN, Prioritized Experience Replay, Rainbow
- **Implementation**: Atari Breakout with Double DQN + PER
- **Key papers**: Hasselt (2015), Wang (2015), Schaul (2015), Hessel (2017)
- **Key equations**: Double Q-learning target, dueling architecture, priority calculation

---

## Phase 4: Policy Optimization (Weeks 11-14)

> S&B Ch.13 | Silver Lecture 7 | CS285 Lectures 5, 9 | CS234 Weeks 7-8

### Week 11: Policy Gradient Methods
- **S&B**: Ch.13 Policy Gradient Methods
- **Silver**: Lecture 7 - Policy Gradient Methods
- **Core concepts**: REINFORCE, Policy gradient theorem, Baseline, Score function estimator
- **Implementation**: CartPole with REINFORCE + baseline
- **Key equations**: Policy gradient theorem, REINFORCE update, variance reduction with baseline

### Week 12: Actor-Critic Methods
- **CS285**: Lecture 5 - Policy Gradients
- **Core concepts**: A2C, A3C, Advantage function, GAE (Generalized Advantage Estimation)
- **Implementation**: LunarLander-v2 with A2C
- **Key papers**: Mnih et al. (2016) A3C, Schulman et al. (2015) GAE
- **Key equations**: Advantage function, GAE(lambda), A2C loss

### Week 13: TRPO & PPO
- **CS285**: Lecture 9 - Advanced Policy Gradients
- **Core concepts**: Trust regions, KL constraint, Clipped surrogate objective, Value clipping
- **Implementation**: MuJoCo HalfCheetah with PPO
- **Key papers**: Schulman et al. (2015) TRPO, Schulman et al. (2017) PPO
- **Key equations**: TRPO constrained optimization, PPO clipped objective, KL penalty

### Week 14: Continuous Control
- **CS285**: (Advanced actor-critic)
- **Core concepts**: DDPG, TD3, SAC, Reparameterization trick, Entropy regularization
- **Implementation**: Pendulum/BipedalWalker with SAC
- **Key papers**: Lillicrap (2015) DDPG, Fujimoto (2018) TD3, Haarnoja (2018) SAC
- **Key equations**: Deterministic policy gradient, twin Q-networks, maximum entropy objective

---

## Phase 5: Advanced Topics (Weeks 15-16)

> S&B Ch.8,14 | Silver Lectures 8-9 | CS285 Lectures 11-12

### Week 15: Model-Based Reinforcement Learning
- **Silver**: Lecture 8 - Integrating Learning and Planning
- **CS285**: Lecture 11 - Model-Based RL
- **Core concepts**: Learned dynamics models, MBPO, World Models, MuZero
- **Implementation**: CartPole with learned model + planning
- **Key papers**: Ha & Schmidhuber (2018) World Models, Schrittwieser (2020) MuZero

### Week 16: Exploration & Exploitation
- **Silver**: Lecture 9 - Exploration and Exploitation
- **CS285**: Lecture 12 - Exploration
- **Core concepts**: Optimism (UCB), Count-based, Curiosity (ICM, RND), Go-Explore
- **Implementation**: Sparse-reward GridWorld with RND
- **Key papers**: Bellemare (2016) Count-based, Pathak (2017) ICM, Burda (2018) RND

---

## Phase 6: RLHF & Alignment (Weeks 17-18)

> CS234 RLHF module | InstructGPT | Constitutional AI | DPO

### Week 17: Reward Modeling & RLHF
- **CS234**: RLHF module
- **Core concepts**: Preference learning, Bradley-Terry model, Reward model training, PPO with KL penalty
- **Implementation**: Reward model on synthetic preferences, small RLHF loop
- **Key papers**: Christiano et al. (2017) Deep RL from Human Preferences, Ouyang et al. (2022) InstructGPT

### Week 18: Beyond RLHF
- **Core concepts**: DPO, IPO, KTO, Constitutional AI, Online RLHF, RLVR
- **Implementation**: DPO fine-tuning on a small language model
- **Key papers**: Rafailov et al. (2023) DPO, Bai et al. (2022) Constitutional AI
- **Key equations**: DPO loss derivation from RLHF objective, KTO asymmetric loss

---

## Reading Order Suggestion

For each week:
1. Read the **Sutton & Barto chapter** first (mathematical foundation)
2. Watch the **David Silver lecture** (intuition and visual explanation)
3. Review **CS234/CS285 slides** (additional perspectives)
4. **Implement** the algorithm from scratch
5. **Take the quiz** (5 questions)
6. Write a **summary** for NotebookLM archiving
