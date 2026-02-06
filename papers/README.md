# Reinforcement Learning Papers

A comprehensive, curated list of reinforcement learning papers organized by topic with a tier system to guide your learning journey.

## Tier System

- **Tier 1**: Foundational, must-read papers. Start here for each topic.
- **Tier 2**: Important extensions and practical improvements. Read after understanding Tier 1.
- **Tier 3**: Advanced, cutting-edge research. For deep dives and staying current.

## Reading Order Recommendation

1. Start with **Foundations & Theory (Tier 1)** - Build your theoretical foundation
2. Move to **Value-Based Methods (Tier 1)** - Understand deep Q-learning
3. Progress to **Policy Gradient Methods (Tier 1)** - Learn policy optimization
4. Explore **Continuous Control (Tier 1)** - Master actor-critic methods
5. Branch into specialized topics (Model-Based, Exploration, Multi-Agent, Hierarchical)
6. Deep dive into **RLHF & Alignment** if interested in LLM applications
7. Always reference **Reproducibility & Analysis** when implementing

## Table of Contents

1. [Foundations & Theory](#1-foundations--theory)
2. [Value-Based Methods (DQN family)](#2-value-based-methods)
3. [Policy Gradient Methods](#3-policy-gradient-methods)
4. [Continuous Control & Deterministic PG](#4-continuous-control--deterministic-pg)
5. [Model-Based RL](#5-model-based-rl)
6. [Exploration](#6-exploration)
7. [Multi-Agent RL](#7-multi-agent-rl)
8. [Hierarchical RL](#8-hierarchical-rl)
9. [RLHF & Alignment](#9-rlhf--alignment)
10. [Reproducibility & Analysis](#10-reproducibility--analysis)

---

## Papers by Topic

### 1. Foundations & Theory

**Tier 1:**

- Sutton & Barto (2018) [Reinforcement Learning: An Introduction, 2nd Edition](http://incompleteideas.net/book/the-book-2nd.html). The definitive textbook covering all fundamental concepts from MDPs to deep RL. **Related: Week 1-7**

- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf) - Sutton et al. (2000). Established the foundation for modern policy gradient algorithms with convergence guarantees. **Related: Week 11**

- [A Natural Policy Gradient](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf) - Kakade (2002). Introduced natural gradients to policy optimization, leading to faster convergence. **Related: Week 13**

- [An Analysis of Temporal-Difference Learning with Function Approximation](https://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf) - Tsitsiklis & Van Roy (1997). Rigorous theoretical analysis of TD learning convergence properties. **Related: Week 8**

- [Algorithms for Reinforcement Learning](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) - Szepesvari (2009). Concise survey covering TD, Monte Carlo, and policy gradient methods. **Related: Week 1-11**

**Tier 2:**

- [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) - Kakade & Langford (2002). Conservative policy iteration with performance bounds. **Related: Week 13**

- [Reinforcement Learning of Motor Skills with Policy Gradients](https://www.kyb.tuebingen.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf) - Peters & Schaal (2008). Applied policy gradients to robotics and continuous control. **Related: Week 11**

---

### 2. Value-Based Methods

**Tier 1:**

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Mnih et al. (2015). Introduced DQN, the breakthrough combining Q-learning with deep neural networks. **Related: Week 9**

- [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) - van Hasselt et al. (2015). Addressed overestimation bias in Q-learning with a simple but effective modification. **Related: Week 10**

- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) - Wang et al. (2015). Separated value and advantage streams for better value estimation. **Related: Week 10**

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) - Schaul et al. (2015). Sample important transitions more frequently for faster learning. **Related: Week 10**

**Tier 2:**

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) - Hessel et al. (2017). Combined six DQN extensions into a single powerful agent. **Related: Week 10**

- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) - Bellemare et al. (2017). Modeled the full distribution of returns instead of just the expectation. **Related: Week 10**

- [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) - Dabney et al. (2017). Used quantile regression for distributional RL with improved stability. **Related: Week 10**

**Tier 3:**

- [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) - Hausknecht & Stone (2015). Extended DQN to handle partial observability with LSTMs.

- [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923) - Dabney et al. (2018). Flexible distributional RL without pre-specified quantiles.

---

### 3. Policy Gradient Methods

**Tier 1:**

- [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf) - Williams (1992). Introduced REINFORCE, the foundational Monte Carlo policy gradient algorithm. **Related: Week 11**

- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) - Schulman et al. (2015). Monotonic policy improvement with trust region constraints. **Related: Week 13**

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al. (2017). Simplified TRPO with clipped surrogate objective, the standard for RLHF. **Related: Week 13**

- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - Schulman et al. (2015). Introduced GAE for variance reduction in policy gradients. **Related: Week 12**

**Tier 2:**

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) - Mnih et al. (2016). Introduced A3C and other asynchronous variants for parallel training. **Related: Week 12**

- [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224) - Wang et al. (2016). Combined off-policy learning with actor-critic for better sample efficiency. **Related: Week 12**

---

### 4. Continuous Control & Deterministic PG

**Tier 1:**

- [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf) - Silver et al. (2014). Introduced deterministic policy gradients for continuous action spaces. **Related: Week 14**

- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) - Lillicrap et al. (2015). Combined DPG with deep learning to create DDPG for continuous control. **Related: Week 14**

- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) - Fujimoto et al. (2018). TD3 improved DDPG with clipped double Q-learning and delayed policy updates. **Related: Week 14**

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) - Haarnoja et al. (2018). Maximum entropy RL for robust and sample-efficient continuous control. **Related: Week 14**

**Tier 2:**

- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) - Haarnoja et al. (2018). Improved SAC with automatic temperature tuning (SAC v2). **Related: Week 14**

---

### 5. Model-Based RL

**Tier 1:**

- [World Models](https://arxiv.org/abs/1803.10122) - Ha & Schmidhuber (2018). Learn a compressed spatial-temporal representation for model-based control. **Related: Week 15**

- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) - Schrittwieser et al. (2020). MuZero achieved superhuman performance without knowing game rules. **Related: Week 15**

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) - Silver et al. (2017). AlphaZero generalized AlphaGo's approach to multiple games. **Related: Week 7, 15**

**Tier 2:**

- [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253) - Janner et al. (2019). MBPO balanced real and model-generated data for stable learning. **Related: Week 15**

- [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) - Hafner et al. (2019). Dreamer learned behaviors entirely from latent imagination. **Related: Week 15**

- [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596) - Nagabandi et al. (2017). Combined model-based pre-training with model-free fine-tuning. **Related: Week 15**

**Tier 3:**

- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) - Hafner et al. (2023). DreamerV3 scaled to diverse domains with a single set of hyperparameters.

- [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/abs/2203.04955) - Hansen et al. (2022). TD-MPC combined implicit models with model predictive control.

---

### 6. Exploration

**Tier 1:**

- [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868) - Bellemare et al. (2016). Pseudo-count density models for exploration bonuses. **Related: Week 16**

- [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) - Pathak et al. (2017). ICM used prediction error as intrinsic reward for exploration. **Related: Week 16**

- [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) - Burda et al. (2018). RND predicted random network outputs for scalable exploration bonuses. **Related: Week 16**

**Tier 2:**

- [First return, then explore](https://arxiv.org/abs/2004.12919) - Ecoffet et al. (2019). Go-Explore combined archive-based exploration with robustification. **Related: Week 16**

- [VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674) - Houthooft et al. (2016). Used information gain about environment dynamics for exploration. **Related: Week 16**

- [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355) - Burda et al. (2018). Empirical analysis of curiosity in diverse environments. **Related: Week 16**

**Tier 3:**

- [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070) - Eysenbach et al. (2018). DIAYN learned diverse skills through mutual information maximization.

---

### 7. Multi-Agent RL

**Tier 1:**

- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) - Lowe et al. (2017). MADDPG extended DDPG to multi-agent settings with centralized training.

- [QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) - Rashid et al. (2018). Factorized Q-learning for cooperative multi-agent tasks.

**Tier 2:**

- [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z) - Vinyals et al. (2019). AlphaStar achieved grandmaster level in StarCraft II.

- [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528) - Baker et al. (2019). OpenAI hide-and-seek demonstrated emergent complexity in multi-agent RL.

---

### 8. Hierarchical RL

**Tier 1:**

- [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161) - Vezhnevets et al. (2017). Manager-worker hierarchy with goal-conditioned policies.

- [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296) - Nachum et al. (2018). HIRO learned hierarchical policies with off-policy correction.

**Tier 2:**

- [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140) - Bacon et al. (2017). End-to-end learning of options with policy gradient methods.

---

### 9. RLHF & Alignment

**Tier 1:**

- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) - Christiano et al. (2017). The original RLHF paper learning reward functions from human comparisons. **Related: Week 17**

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) - Ouyang et al. (2022). InstructGPT applied RLHF to align large language models with human intent. **Related: Week 17**

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) - Rafailov et al. (2023). DPO eliminated the reward model by optimizing preferences directly. **Related: Week 18**

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Bai et al. (2022). Used AI-generated feedback for alignment with reduced human annotation. **Related: Week 18**

**Tier 2:**

- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) - Stiennon et al. (2020). Applied RLHF to text summarization at scale. **Related: Week 17**

- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) - Ziegler et al. (2019). Early work on using human preferences to fine-tune language models. **Related: Week 17**

- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) - Ethayarajh et al. (2024). Aligned models using only binary signals without paired preferences. **Related: Week 18**

- [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) - Hong et al. (2024). Combined SFT and preference learning in a single monolithic objective. **Related: Week 18**

- [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) - Azar et al. (2024). IPO provided theoretical foundations for preference optimization. **Related: Week 18**

**Tier 3:**

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - DeepSeek AI (2025). Used RL to develop reasoning capabilities matching OpenAI o1. **Related: Week 18**

- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) - Gao et al. (2023). Characterized the overoptimization problem in RLHF. **Related: Week 18**

- [Nash Learning from Human Feedback](https://arxiv.org/abs/2312.00886) - Munos et al. (2023). Framed RLHF as a two-player game between policy and reward model.

- [Human Alignment of Large Language Models through Online Preference Optimization](https://arxiv.org/abs/2403.08635) - Calandriello et al. (2024). Online learning approach for real-time preference optimization.

---

### 10. Reproducibility & Analysis

**Tier 1:**

- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560) - Henderson et al. (2017). Exposed reproducibility crisis in deep RL with analysis of variance sources.

- [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729) - Engstrom et al. (2020). Showed that implementation details matter as much as algorithms.

**Tier 2:**

- [Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control](https://arxiv.org/abs/1708.04133) - Islam et al. (2017). Analyzed reproducibility challenges in continuous control benchmarks.

- [Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/abs/1803.07055) - Mania et al. (2018). Showed that simple random search can compete with sophisticated RL algorithms.

---

## Additional Resources

### Classic Papers Not Listed Above

- Watkins & Dayan (1992) Q-Learning - The original Q-learning paper
- Mnih et al. (2013) Playing Atari with Deep Reinforcement Learning - The original DQN paper (pre-Nature)
- Silver et al. (2016) Mastering the game of Go with deep neural networks and tree search - AlphaGo

### Survey Papers

- [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274) - Li (2017)
- [A Survey on Policy Search for Robotics](https://www.nowpublishers.com/article/Details/ROB-021) - Deisenroth et al. (2013)
- [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/1604.06778) - Duan et al. (2016)

### Recommended Reading Paths

**For beginners:** Start with Sutton & Barto (2018), then DQN → Policy Gradients → PPO

**For LLM practitioners:** Focus on section 9 (RLHF & Alignment), especially InstructGPT → DPO → Recent advances

**For robotics:** Policy Gradients → Continuous Control → Model-Based RL → Hierarchical RL

**For researchers:** All Tier 1 papers, then Tier 2 in your area of interest, supplement with Reproducibility papers

---

## Contributing

Found a missing paper or broken link? Please open an issue or submit a pull request.

## License

This curated list is provided for educational purposes. Please refer to individual papers for their respective licenses and citation requirements.
