# Week 10 Resources: DQN Extensions

## Primary Papers

### Double DQN
- **van Hasselt, H., Guez, A., & Silver, D. (2016).** "Deep Reinforcement Learning with Double Q-Learning"
  - [arXiv](https://arxiv.org/abs/1509.06461)
  - AAAI 2016
  - Addresses Q-value overestimation

### Dueling DQN
- **Wang, Z., Schaul, T., et al. (2016).** "Dueling Network Architectures for Deep Reinforcement Learning"
  - [arXiv](https://arxiv.org/abs/1511.06581)
  - ICML 2016
  - Separates value and advantage

### Prioritized Experience Replay
- **Schaul, T., Quan, J., et al. (2016).** "Prioritized Experience Replay"
  - [arXiv](https://arxiv.org/abs/1511.05952)
  - ICLR 2016
  - Non-uniform sampling

### Rainbow DQN (Must Read)
- **Hessel, M., et al. (2018).** "Rainbow: Combining Improvements in Deep Reinforcement Learning"
  - [arXiv](https://arxiv.org/abs/1710.02298)
  - AAAI 2018
  - Comprehensive ablation study

## Additional Papers

### Distributional RL (C51)
- **Bellemare, M. G., Dabney, W., & Munos, R. (2017).** "A Distributional Perspective on Reinforcement Learning"
  - [arXiv](https://arxiv.org/abs/1707.06887)
  - ICML 2017
  - Rainbow component

### Noisy Networks
- **Fortunato, M., et al. (2017).** "Noisy Networks for Exploration"
  - [arXiv](https://arxiv.org/abs/1706.10295)
  - ICLR 2018
  - Learned exploration

### Multi-Step Learning
- **Sutton & Barto Chapter 7**: n-step Bootstrapping
  - Theoretical foundation

## Video Lectures

### Berkeley CS285
- **Lecture 8: Deep RL with Q-Functions (Advanced Topics)**
  - [YouTube](https://www.youtube.com/watch?v=Psrhxy88zww)
  - Covers Double DQN, Dueling, PER

### DeepMind Lectures
- **Advanced Value-Based Methods**
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLqYmG7hTraZDVH599EItlEWsUOsJbAodm)

## Blog Posts and Tutorials

### Comprehensive Guides
- **Lilian Weng: "Deep Reinforcement Learning"**
  - [Blog Post](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
  - Section on value-based improvements

- **OpenAI Spinning Up: DQN Extensions**
  - [Documentation](https://spinningup.openai.com/en/latest/)

### Implementation Guides
- **Prioritized Experience Replay Tutorial**
  - [Blog](https://danieltakeshi.github.io/2019/07/14/per/)
  - Detailed PER implementation

- **Dueling DQN PyTorch Tutorial**
  - [Medium](https://medium.com/@parsa_h_m/deep-reinforcement-learning-dqn-double-dqn-dueling-dqn-noisy-dqn-and-dqn-with-prioritized-551f621a9823)

## Implementation Resources

### Official Implementations

**Stable-Baselines3**
- [DQN Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- Includes Double DQN, Dueling

**Dopamine (Google)**
- [GitHub](https://github.com/google/dopamine)
- Clean Rainbow implementation
- Excellent starting point

**CleanRL**
- [DQN Variants](https://github.com/vwxyzjn/cleanrl)
- Single-file implementations
- Educational code

### From-Scratch Implementations

**PyTorch DQN Extensions**
- [GitHub Tutorial](https://github.com/higgsfield/RL-Adventure)
- Double DQN, Dueling, PER, Noisy

**Rainbow PyTorch**
- [GitHub](https://github.com/Kaixhin/Rainbow)
- Full Rainbow implementation
- Well-documented

## Tools and Libraries

### Prioritized Replay Implementations

**Sum-Tree Library**
- [GitHub](https://github.com/jaromiru/AI-blog/blob/master/SumTree.py)
- Efficient sum-tree for PER

**Segment-Tree (Alternative)**
- [GitHub](https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py)
  - OpenAI Baselines implementation

## Textbook References

- **Sutton & Barto, 2nd Edition**
  - Chapter 7: n-step Bootstrapping
  - Chapter 12: Eligibility Traces

- **Deep Reinforcement Learning** by Aske Plaat
  - Chapter on value-based methods
  - DQN extensions coverage

## Benchmarks and Datasets

### Atari Benchmarks
- **Atari Human Normalized Scores**
  - Rainbow paper results
  - Standard evaluation protocol

### MinAtar (Simplified Atari)
- [GitHub](https://github.com/kenjyoung/MinAtar)
  - Fast prototyping environment
  - Test extensions quickly

## Research Papers (Advanced)

### Quantile Regression DQN
- **Dabney, W., et al. (2018).** "Distributional Reinforcement Learning with Quantile Regression"
  - [arXiv](https://arxiv.org/abs/1710.10044)
  - Improvement over C51

### Implicit Quantile Networks
- **Dabney, W., et al. (2018).** "Implicit Quantile Networks for Distributional Reinforcement Learning"
  - [arXiv](https://arxiv.org/abs/1806.06923)
  - State-of-the-art distributional RL

### Never Give Up (Exploration)
- **Badia, A. P., et al. (2020).** "Never Give Up: Learning Directed Exploration Strategies"
  - [arXiv](https://arxiv.org/abs/2002.06038)
  - Advanced exploration for Montezuma

## Community Resources

### Reddit Discussions
- **r/reinforcementlearning**
  - [Discussion on PER](https://www.reddit.com/r/reinforcementlearning/comments/comments)
  - Implementation tips

### Stack Overflow
- **[RL Tag](https://stackoverflow.com/questions/tagged/reinforcement-learning)**
  - Debugging PER, Dueling implementations

## Debugging Resources

### Common Issues

**Prioritized Replay Debugging**
- [Blog Post](https://danieltakeshi.github.io/2019/07/14/per/)
- Sum-tree verification
- Importance sampling checks

**Dueling Architecture Pitfalls**
- Mean vs max subtraction
- Initialization issues

## Advanced Topics

### Recurrent Experience Replay
- **Lin, Z., et al. (2018).** "R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning"
  - [OpenReview](https://openreview.net/forum?id=r1lyTjAqYX)

### Ape-X (Distributed PER)
- **Horgan, D., et al. (2018).** "Distributed Prioritized Experience Replay"
  - [arXiv](https://arxiv.org/abs/1803.00933)
  - Scalable PER

## Visualization Tools

### TensorBoard
- Log Q-values over time
- Priority distributions
- Value/advantage separation

### Weights & Biases
- Compare extensions
- Ablation study tracking

## Recommended Learning Path

1. **Week 1**: Implement Double DQN (easiest)
2. **Week 2**: Add Dueling architecture
3. **Week 3**: Implement PER (most complex)
4. **Week 4**: Combine all three
5. **Week 5**: Read Rainbow paper, understand other components

## Books

- **Deep Reinforcement Learning Hands-On** by Maxim Lapan
  - Chapter on DQN improvements
  - Code examples

- **Reinforcement Learning: An Introduction** by Sutton & Barto
  - Theoretical foundations
