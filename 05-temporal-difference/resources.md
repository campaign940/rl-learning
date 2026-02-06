# Week 5 Resources: Temporal-Difference Learning

## Primary Textbooks

### Sutton & Barto: Reinforcement Learning - An Introduction (2nd Edition)
- **Chapter 6: Temporal-Difference Learning**
  - [Full Book PDF](http://incompleteideas.net/book/RLbook2020.pdf)
  - [Chapter 6 Direct Link](http://incompleteideas.net/book/RLbook2020.pdf#page=133)
  - Essential sections:
    - 6.1: TD Prediction
    - 6.2: Advantages of TD Prediction Methods
    - 6.3: Optimality of TD(0)
    - 6.4: SARSA: On-Policy TD Control
    - 6.5: Q-learning: Off-Policy TD Control
    - 6.6: Expected SARSA
    - 6.7: Maximization Bias and Double Learning

### David Silver's RL Course
- **Lecture 4: Model-Free Prediction**
  - [Lecture Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)
  - [Video Lecture](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
  - Topics: MC vs TD, TD(λ), convergence properties

- **Lecture 5: Model-Free Control**
  - [Lecture Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)
  - [Video Lecture](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
  - Topics: On-policy (SARSA) vs Off-policy (Q-Learning), convergence

## University Courses

### CS234: Reinforcement Learning (Stanford)
- **Lecture 3: Model-Free Policy Evaluation**
  - [Course Website](http://web.stanford.edu/class/cs234/index.html)
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)

- **Lecture 4: Model-Free Control**
  - Q-Learning, SARSA comparison
  - Maximization bias discussion

### CS285: Deep Reinforcement Learning (UC Berkeley)
- **Lecture 7: Value Function Methods**
  - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
  - [Lecture Slides](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf)
  - Covers TD learning as foundation for DQN

## Blog Posts and Tutorials

### Detailed Explanations

- **Lil'Log: RL Overview (TD Section)**
  - [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#temporal-difference-learning)
  - Clear visual explanations of TD methods

- **Towards Data Science: Understanding TD Learning**
  - [TD Learning Explained](https://towardsdatascience.com/introduction-to-reinforcement-learning-td-learning-sarsa-vs-q-learning-4be68b6b16c4)
  - Practical Python implementations

- **Medium: SARSA vs Q-Learning**
  - [On-Policy vs Off-Policy](https://medium.com/@jonathan_hui/rl-on-policy-vs-off-policy-q-learning-sarsa-4ffe3ebdbb09)
  - Cliff Walking example explained

- **Towards Data Science: Double Q-Learning**
  - [Fixing Maximization Bias](https://towardsdatascience.com/double-q-learning-the-easy-way-a924c4085ec3)
  - Intuitive explanation with code

### Interactive Resources

- **OpenAI Spinning Up: TD Methods**
  - [Intro to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#temporal-difference-learning)
  - Clean mathematical exposition

- **RL Cheat Sheet**
  - [GitHub: Deep RL Bootcamp](https://github.com/williamFalcon/DeepRLHacks)
  - Quick reference for TD algorithms

## Implementation Resources

### Gymnasium Environments

- **Gymnasium Documentation**
  - [Official Docs](https://gymnasium.farama.org/)
  - [CliffWalking-v0](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)
  - [Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/)

### Code Repositories

- **Dennybritz RL Repository**
  - [GitHub](https://github.com/dennybritz/reinforcement-learning)
  - Excellent clean implementations:
    - [TD Prediction](https://github.com/dennybritz/reinforcement-learning/tree/master/TD)
    - [SARSA](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb)
    - [Q-Learning](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb)

- **Sutton & Barto Official Code**
  - [GitHub](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
  - Python implementations matching textbook examples
  - Chapter 6 code for all TD methods

- **CleanRL: Simple RL Implementations**
  - [GitHub](https://github.com/vwxyzjn/cleanrl)
  - [Q-Learning Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/q_learning.py)
  - Single-file, well-documented implementations

## Research Papers

### Classic Papers

- **Richard Sutton (1988): Learning to Predict by the Methods of Temporal Differences**
  - [Paper PDF](https://link.springer.com/article/10.1007/BF00115009)
  - Original TD learning paper
  - Foundation of modern RL

- **Watkins & Dayan (1992): Q-Learning**
  - [Paper PDF](https://link.springer.com/article/10.1007/BF00992698)
  - Original Q-Learning paper with convergence proof
  - Must-read classic

- **van Hasselt (2010): Double Q-learning**
  - [Paper PDF](https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf)
  - Introduces Double Q-Learning
  - Elegant solution to maximization bias

### Modern Applications

- **Mnih et al. (2015): Human-level control through deep reinforcement learning (DQN)**
  - [Nature Paper](https://www.nature.com/articles/nature14236)
  - Deep Q-Network: Q-Learning with neural networks
  - Breakthrough in deep RL

- **van Hasselt et al. (2016): Deep Reinforcement Learning with Double Q-learning**
  - [Paper PDF](https://arxiv.org/pdf/1509.06461.pdf)
  - Double DQN: extends Double Q-Learning to deep RL
  - Significant improvement over DQN

- **Hessel et al. (2018): Rainbow: Combining Improvements in Deep RL**
  - [Paper PDF](https://arxiv.org/pdf/1710.02298.pdf)
  - Combines Double Q-Learning with other improvements
  - State-of-the-art DQN variant

## Video Lectures

### YouTube Tutorials

- **Stanford CS234 (Emma Brunskill)**
  - [Full Course Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
  - Lectures 3-4 cover TD methods thoroughly

- **DeepMind x UCL RL Lecture Series**
  - [Full Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
  - Lecture 4: Model-Free Prediction and Control

- **Mutual Information: TD Learning**
  - [TD vs MC](https://www.youtube.com/watch?v=AJiG3ykOxmY)
  - Visual intuition building

- **Arxiv Insights: Q-Learning**
  - [Q-Learning Explained](https://www.youtube.com/watch?v=qhRNvCVVJaA)
  - Accessible introduction

## Tools and Libraries

### Python Libraries for RL

- **Gymnasium** (OpenAI Gym successor)
  ```bash
  pip install gymnasium
  ```
  - [Documentation](https://gymnasium.farama.org/)
  - Standard RL environment interface

- **Stable-Baselines3**
  ```bash
  pip install stable-baselines3
  ```
  - [Documentation](https://stable-baselines3.readthedocs.io/)
  - Includes DQN (deep Q-Learning)

- **TF-Agents** (TensorFlow)
  ```bash
  pip install tf-agents
  ```
  - [Documentation](https://www.tensorflow.org/agents)
  - Production-ready RL library

- **RLlib** (Ray)
  ```bash
  pip install ray[rllib]
  ```
  - [Documentation](https://docs.ray.io/en/latest/rllib/)
  - Scalable RL with DQN support

### Visualization and Analysis

- **TensorBoard**
  ```bash
  pip install tensorboard
  ```
  - Track Q-values, losses, rewards over time

- **Weights & Biases**
  ```bash
  pip install wandb
  ```
  - [Documentation](https://docs.wandb.ai/)
  - Advanced experiment tracking

- **Matplotlib for RL**
  - Visualize policies, value functions, learning curves
  - [Seaborn](https://seaborn.pydata.org/) for statistical plots

## Convergence Theory

### Theoretical Foundations

- **Bertsekas & Tsitsiklis: Neuro-Dynamic Programming**
  - [Book](http://web.mit.edu/dimitrib/www/NDP_Book.html)
  - Rigorous convergence proofs for TD methods

- **Tsitsiklis (1994): Asynchronous Stochastic Approximation**
  - Convergence theory for TD learning

- **Jaakkola et al. (1994): Convergence of Stochastic Iterative Dynamic Programming**
  - Q-Learning convergence proof

### Online Lectures on Theory

- **MIT OCW: Dynamic Programming and Stochastic Control**
  - [Course Materials](https://ocw.mit.edu/courses/6-231-dynamic-programming-and-stochastic-control-fall-2015/)
  - Mathematical foundations

## Interactive Demos and Visualizations

- **GridWorld Visualizer**
  - [Demo](https://cs.stanford.edu/people/karpathy/reinforcejs/gridworld_td.html)
  - Andrej Karpathy's interactive TD demo
  - See TD vs MC in real-time

- **Q-Learning Visualizer**
  - [Demo](https://www.mladdict.com/q-learning-interactive-demo)
  - Step-by-step Q-Learning visualization

- **Cliff Walking Visualization**
  - Implement yourself using matplotlib and Gymnasium
  - Essential for understanding on-policy vs off-policy

## Practice Environments

### Simple Environments (Start Here)

1. **Random Walk**
   - Classic TD prediction example
   - Implement from Sutton & Barto

2. **CliffWalking-v0**
   - Perfect for SARSA vs Q-Learning comparison
   - Built into Gymnasium

3. **Taxi-v3**
   - Discrete state/action space
   - Tests exploration strategies

### Intermediate Environments

4. **FrozenLake-v1**
   - Stochastic transitions
   - Tests robustness to noise

5. **CartPole-v1**
   - Can discretize for tabular methods
   - Or use as intro to function approximation

6. **Mountain Car**
   - Sparse rewards (hard!)
   - Good for testing convergence

### Advanced Environments

7. **Atari Games** (with DQN)
   - ALE (Arcade Learning Environment)
   - Requires function approximation

8. **MuJoCo** (with continuous action extensions)
   - Robotics simulations
   - Advanced continuous control

## Books

### Comprehensive Textbooks

- **Sutton & Barto (2018): Reinforcement Learning: An Introduction**
  - [Free PDF](http://incompleteideas.net/book/the-book-2nd.html)
  - The bible of RL

- **Dimitri Bertsekas (2019): Reinforcement Learning and Optimal Control**
  - [Book Website](http://web.mit.edu/dimitrib/www/RLbook.html)
  - More mathematical perspective

- **Csaba Szepesvári (2010): Algorithms for Reinforcement Learning**
  - [Free PDF](https://sites.ualberta.ca/~szepesva/RLBook.html)
  - Concise, theoretical treatment

### Practical Books

- **Phil Winder (2020): Reinforcement Learning**
  - Industrial RL applications
  - Focus on practical deployment

- **Maxim Lapan (2020): Deep Reinforcement Learning Hands-On**
  - PyTorch implementations
  - DQN and variants in detail

## Discussion Forums

- **Reddit: r/reinforcementlearning**
  - [Subreddit](https://www.reddit.com/r/reinforcementlearning/)
  - Active community discussions

- **Stack Overflow: [reinforcement-learning] tag**
  - [Questions](https://stackoverflow.com/questions/tagged/reinforcement-learning)
  - Implementation help

- **r/learnmachinelearning**
  - [Subreddit](https://www.reddit.com/r/learnmachinelearning/)
  - Beginner-friendly

- **Discord Servers**
  - ML Collective
  - Yannic Kilcher's server
  - OpenAI Scholars

## Blogs to Follow

- **Lil'Log by Lilian Weng**
  - [Blog](https://lilianweng.github.io/)
  - Excellent RL posts

- **Arthur Juliani's Medium**
  - [Simple RL with TensorFlow series](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)

- **Andrej Karpathy's Blog**
  - [Deep RL Policy Gradients](http://karpathy.github.io/2016/05/31/rl/)

- **DeepMind Blog**
  - [Blog](https://www.deepmind.com/blog)
  - Latest research from DQN creators

## Cheat Sheets and Quick References

- **Stanford CS234 Cheat Sheet**
  - Summary of key algorithms
  - [PDF if available from course]

- **TD Methods Quick Reference**
  ```
  TD(0): V(s) ← V(s) + α[r + γV(s') - V(s)]
  SARSA: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
  Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
  Expected SARSA: Q(s,a) ← Q(s,a) + α[r + γΣ π(a'|s')Q(s',a') - Q(s,a)]
  Double Q: Q1(s,a) ← Q1(s,a) + α[r + γQ2(s',argmax_a' Q1(s',a')) - Q1(s,a)]
  ```

## Debugging and Common Issues

### Common Pitfalls

- **Q-values diverging**: Check learning rate, ensure proper terminal state handling
- **Not exploring enough**: Increase ε, ensure GLIE conditions
- **Slow convergence**: Try different learning rates, check if all states visited
- **Maximization bias problems**: Switch to Double Q-Learning

### Debugging Resources

- **RL Debugging Guide**
  - [Debugging RL Paper](https://arxiv.org/pdf/1709.06560.pdf)
  - Systematic debugging strategies

- **Common RL Mistakes**
  - [Blog Post](https://andyljones.com/posts/rl-debugging.html)
  - Practical debugging tips

## Next Steps

After mastering Week 5 TD Learning:

1. **Implement all four algorithms** (SARSA, Q-Learning, Expected SARSA, Double Q-Learning)
2. **Compare them empirically** on Cliff Walking and Taxi
3. **Understand convergence conditions** (GLIE, learning rate schedules)
4. **Move to Week 6**: n-step TD and eligibility traces
5. **Prepare for function approximation**: Understand tabular methods as foundation

### Recommended Learning Path

1. Read S&B Chapter 6 (sections 6.1-6.5)
2. Watch David Silver Lectures 4-5
3. Implement TD(0) prediction on Random Walk
4. Implement SARSA and Q-Learning on CliffWalking
5. Read section 6.7 on Double Q-Learning
6. Implement Expected SARSA and Double Q-Learning
7. Complete quiz problems
8. Ready for Week 6: n-step methods!

### Bridge to Deep RL

TD learning is the foundation for:
- **DQN**: Q-Learning + neural networks
- **DDPG**: Actor-critic with TD learning
- **TD3**: Twin Delayed DDPG (uses Double Q-Learning idea)
- **SAC**: Soft Actor-Critic (entropy-regularized TD)

Understanding tabular TD deeply is essential before moving to deep RL!
