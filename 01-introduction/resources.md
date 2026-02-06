# Week 1 Resources: Introduction to Reinforcement Learning

## Primary Textbook

### Sutton & Barto: Reinforcement Learning: An Introduction (2nd Edition)

**Chapter 1: Introduction**
- [Online version](http://incompleteideas.net/book/RLbook2020.pdf)
- Covers: RL problem formulation, agent-environment interface, reward hypothesis
- Key sections:
  - 1.1: Reinforcement Learning
  - 1.2: Examples
  - 1.3: Elements of RL
  - 1.5: An Extended Example: Tic-Tac-Toe

**Chapter 2: Multi-armed Bandits**
- [Online version](http://incompleteideas.net/book/RLbook2020.pdf)
- Covers: k-armed bandit, action-value methods, exploration strategies
- Key sections:
  - 2.1: A k-armed Bandit Problem
  - 2.2: Action-value Methods
  - 2.3: The 10-armed Testbed
  - 2.4: Incremental Implementation
  - 2.5: Tracking a Nonstationary Problem
  - 2.6: Optimistic Initial Values
  - 2.7: Upper-Confidence-Bound Action Selection
  - 2.8: Gradient Bandit Algorithms

## Video Lectures

### David Silver's RL Course

**Lecture 1: Introduction to Reinforcement Learning**
- [YouTube Link](https://www.youtube.com/watch?v=2pWv7GOvuf0)
- Duration: ~90 minutes
- Topics covered:
  - What is RL?
  - The RL problem
  - Agent and environment
  - Rewards
  - History and state
  - Major components of an RL agent
- [Lecture slides](https://www.davidsilver.uk/wp-content/uploads/2020/03/intro_RL.pdf)

### Stanford CS234: Reinforcement Learning

**Week 1: Introduction and Course Overview**
- [Course website](http://web.stanford.edu/class/cs234/index.html)
- [Lecture videos on YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
- Topics:
  - Sequential decision making under uncertainty
  - Exploration vs exploitation
  - Model-based vs model-free learning
  - Course logistics and overview

## Blog Posts and Tutorials

### Lilian Weng's Blog

**A (Long) Peek into Reinforcement Learning**
- [Blog post](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
- Comprehensive introduction to RL
- Covers:
  - Key concepts and terminology
  - Various RL algorithms
  - Historical context
- Excellent visual diagrams

### OpenAI Spinning Up

**Introduction to RL**
- [Spinning Up documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- Practical introduction with code examples
- Part 1: Key concepts
- Part 2: Kinds of RL algorithms
- Part 3: Introduction to policy optimization

## Interactive Resources

### Reinforcement Learning Visualizations

**Multi-Armed Bandit Playground**
- [Interactive demo](https://rlvizkit.com/)
- Visualize different exploration strategies
- Compare epsilon-greedy, UCB, and Thompson sampling
- Adjust parameters in real-time

## Research Papers

### Classic Papers on Bandits

**Finite-time Analysis of the Multiarmed Bandit Problem**
- Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002)
- [Paper link](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
- Introduces UCB algorithm with theoretical guarantees

**A Tutorial on Thompson Sampling**
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018)
- [Paper link](https://arxiv.org/abs/1707.02038)
- Comprehensive tutorial on Bayesian approach to bandits

## Code Repositories

### Sutton & Barto Code Examples

**Official Repository**
- [GitHub link](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
- Python implementations of all examples from the textbook
- Chapter 2: Multi-armed bandit implementations

### OpenAI Gym

**Bandit Environments**
- [Gym documentation](https://www.gymlibrary.dev/)
- Standard interface for RL environments
- Useful for implementing your own bandit experiments

## Additional Reading

### Deep Dive into Exploration

**Exploration in Reinforcement Learning (Survey)**
- [Article link](https://arxiv.org/abs/2109.00157)
- Comprehensive survey of exploration methods
- Covers count-based, uncertainty-based, and information-theoretic approaches

### Practical RL Resources

**Practical RL Course by Yandex**
- [GitHub repository](https://github.com/yandexdataschool/Practical_RL)
- Practical assignments and tutorials
- Week 1 includes bandit problems

## Tools and Libraries

### Python Libraries for Bandits

**MABWiser**
- [Documentation](https://github.com/fidelity/mabwiser)
- Context-free and contextual multi-armed bandit library
- Implementations of many algorithms

**Vowpal Wabbit**
- [Website](https://vowpalwabbit.org/)
- Fast online learning library
- Includes contextual bandit algorithms

## Discussion and Community

### Reddit Communities

**r/reinforcementlearning**
- [Subreddit link](https://www.reddit.com/r/reinforcementlearning/)
- Active community for discussions
- Good for asking questions

### Discord Servers

**Deep RL Discord**
- Community of RL researchers and practitioners
- Real-time discussions and help

## Recommended Study Path

1. **Start**: Read S&B Chapters 1-2 (3-4 hours)
2. **Watch**: David Silver Lecture 1 (1.5 hours)
3. **Read**: Lilian Weng's blog post (1 hour)
4. **Implement**: k-armed bandit testbed (4-6 hours)
5. **Practice**: Quiz questions and exercises (2 hours)
6. **Explore**: Interactive visualizations (1 hour)

**Total estimated time**: 12-15 hours

## Next Week Preview

Week 2 will cover Markov Decision Processes (MDPs):
- Formal MDP definition
- State and action value functions
- Bellman equations
- Dynamic programming preview

**Prepare by**:
- Reviewing basic probability (expectations, conditional probability)
- Refreshing linear algebra (matrix operations)
- Understanding recursive equations
