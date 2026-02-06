# Week 9 Resources: Deep Q-Networks (DQN)

## Primary Papers

### The DQN Paper (Must Read)

- **Mnih, V., et al. (2015).** "Human-level control through deep reinforcement learning"
  - [Nature Paper](https://www.nature.com/articles/nature14236)
  - [PDF](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
  - The landmark paper that started deep RL revolution
  - Introduces experience replay and target networks
  - Results on 49 Atari games

### Precursor Paper

- **Mnih, V., et al. (2013).** "Playing Atari with Deep Reinforcement Learning"
  - [arXiv](https://arxiv.org/abs/1312.5602)
  - NIPS Deep Learning Workshop version
  - First version of DQN
  - Good for understanding development of ideas

## Textbook References

- **Sutton & Barto, 2nd Edition**
  - [Free online](http://incompleteideas.net/book/the-book-2nd.html)
  - Chapter 11.7: Brief overview of Deep Q-learning
  - Chapter 16: Applications and case studies

- **Reinforcement Learning and Optimal Control** by Dimitri Bertsekas
  - Chapter 6: Approximate Dynamic Programming
  - Theoretical foundations

## Video Lectures

### DeepMind Lectures

- **David Silver's RL Course - Lecture 6**
  - [YouTube](https://www.youtube.com/watch?v=UoPei5o4fps)
  - Value function approximation foundations

- **DeepMind x UCL RL Lecture Series 2021 - Lecture 7**
  - [YouTube](https://www.youtube.com/watch?v=TCCjZe0y4Qc)
  - Modern perspective on DQN

### Berkeley CS285

- **Lecture 7: Value Function Methods**
  - [YouTube](https://www.youtube.com/watch?v=Psrhxy88zww)
  - [Slides](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-7.pdf)
  - Comprehensive coverage of DQN
  - Sergey Levine's excellent explanations

### Stanford CS234

- **Lecture 7: Deep RL**
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
  - Emma Brunskill's course

## Blog Posts and Tutorials

### Comprehensive Tutorials

- **Lilian Weng: "A (Long) Peek into Reinforcement Learning"**
  - [Blog Post](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
  - Excellent mathematical exposition
  - Section on DQN and extensions

- **Arthur Juliani: "Simple Reinforcement Learning with TensorFlow"**
  - [Medium Series](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
  - Part 4: DQN implementation
  - Very beginner-friendly

### Technical Deep Dives

- **Nervana Systems: "Demystifying Deep Reinforcement Learning"**
  - [Blog Post](https://www.intelnervana.com/demystifying-deep-reinforcement-learning/)
  - Clear explanations with diagrams

- **OpenAI Spinning Up: "Introduction to Deep RL"**
  - [Documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
  - DQN section with math and code

## Implementation Resources

### Official Implementations

- **PyTorch DQN Tutorial**
  - [Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
  - Official PyTorch tutorial
  - CartPole implementation
  - Well-commented code

- **TensorFlow Agents DQN**
  - [TF-Agents DQN](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)
  - Production-quality implementation
  - Atari examples

### Clean Implementations

- **CleanRL DQN**
  - [GitHub](https://github.com/vwxyzjn/cleanrl)
  - Single-file implementations
  - Benchmarked results

- **Stable-Baselines3 DQN**
  - [Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
  - [GitHub](https://github.com/DLR-RM/stable-baselines3)
  - Well-tested implementation
  - Easy to use API

### From-Scratch Implementations

- **Deep RL Course (Hugging Face)**
  - [Course Link](https://huggingface.co/learn/deep-rl-course/unit1/introduction)
  - Unit 3: DQN
  - Step-by-step tutorial

- **PyTorch DQN from Scratch**
  - [GitHub Repo](https://github.com/qfettes/DeepRL-Tutorials)
  - Educational implementations

## Environments

### OpenAI Gym

- **Gym Classic Control**
  - [CartPole Documentation](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)
  - [Mountain Car](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)
  - Good for testing DQN basics

### Atari Environments

- **Gym Atari**
  - [Documentation](https://www.gymlibrary.dev/environments/atari/)
  - Installation: `pip install gym[atari]`
  - 57 games available

- **Arcade Learning Environment (ALE)**
  - [GitHub](https://github.com/mgbellemare/Arcade-Learning-Environment)
  - Backend for Atari environments

## Tools and Libraries

### Core Libraries

- **PyTorch**
  - [Official Website](https://pytorch.org/)
  - [Tutorials](https://pytorch.org/tutorials/)

- **TensorFlow / Keras**
  - [Official Website](https://www.tensorflow.org/)
  - [RL Tutorials](https://www.tensorflow.org/agents)

### RL Frameworks

- **RLlib (Ray)**
  - [Documentation](https://docs.ray.io/en/latest/rllib/index.html)
  - Scalable RL library
  - Production-ready DQN

- **Dopamine**
  - [GitHub](https://github.com/google/dopamine)
  - Google's RL research framework
  - Clean DQN implementation

### Visualization

- **TensorBoard**
  - [Documentation](https://www.tensorflow.org/tensorboard)
  - Visualize training curves

- **Weights & Biases**
  - [Website](https://wandb.ai/)
  - ML experiment tracking
  - [RL integration](https://docs.wandb.ai/guides/integrations/other/reinforcement-learning)

## Research Papers (Extensions Preview)

### DQN Variants (Week 10 Preview)

- **Double DQN**
  - Van Hasselt et al. (2016)
  - [arXiv](https://arxiv.org/abs/1509.06461)

- **Dueling DQN**
  - Wang et al. (2016)
  - [arXiv](https://arxiv.org/abs/1511.06581)

- **Prioritized Experience Replay**
  - Schaul et al. (2016)
  - [arXiv](https://arxiv.org/abs/1511.05952)

- **Rainbow DQN**
  - Hessel et al. (2018)
  - [arXiv](https://arxiv.org/abs/1710.02298)

## Debugging and Tips

### Common Issues

- **PyTorch DQN Debugging Guide**
  - [Blog Post](https://andyljones.com/posts/rl-debugging.html)
  - Comprehensive debugging strategies

- **Reddit r/reinforcementlearning**
  - [Subreddit](https://www.reddit.com/r/reinforcementlearning/)
  - Active community for questions

### Performance Tips

- **Tips for Training RL Agents**
  - [OpenAI Blog](https://openai.com/research/openai-baselines-ppo)
  - General RL debugging advice

## Datasets and Benchmarks

### Atari Benchmarks

- **Atari Grand Challenge**
  - [Website](https://sites.google.com/view/atari-grand-challenge)
  - Benchmark results
  - Leaderboards

- **Atari Human Normalized Scores**
  - Standard evaluation metric
  - Reference: DQN Nature paper

## Interactive Demos

### Playable Demos

- **Karpathy's ConvNetJS Demos**
  - [Demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/rldemo.html)
  - Interactive DQN visualization

- **TensorFlow.js DQN**
  - [Demo](https://storage.googleapis.com/tfjs-examples/cart-pole/dist/index.html)
  - CartPole in browser

## Competitions and Challenges

- **OpenAI Gym Leaderboard** (Historical)
  - Past benchmark results

- **NeurIPS Competitions**
  - Periodic RL competitions
  - Check current year's challenges

## Community Resources

### Forums

- **RL Discord Servers**
  - Many active communities
  - Real-time help

- **AI Stack Exchange**
  - [RL Tag](https://ai.stackexchange.com/questions/tagged/reinforcement-learning)
  - Q&A format

### Courses

- **Coursera: Reinforcement Learning Specialization**
  - University of Alberta
  - Covers fundamentals including DQN

- **Udacity: Deep Reinforcement Learning Nanodegree**
  - Project-based learning
  - DQN implementation projects

## Advanced Topics

### Theoretical Analysis

- **Neural Fitted Q Iteration**
  - Riedmiller (2005)
  - Precursor to DQN

- **Finite-Sample Analysis of Q-Learning**
  - Recent theoretical work on sample complexity

### Extensions

- **Recurrent DQN (DRQN)**
  - Hausknecht & Stone (2015)
  - [arXiv](https://arxiv.org/abs/1507.06527)

- **Noisy DQN**
  - Fortunato et al. (2017)
  - [arXiv](https://arxiv.org/abs/1706.10295)

## Recommended Learning Path

1. **Week 1**: Read DQN Nature paper + watch CS285 Lecture 7
2. **Week 2**: Implement CartPole DQN from scratch (PyTorch tutorial)
3. **Week 3**: Scale to Atari (start with Pong)
4. **Week 4**: Experiment with hyperparameters, debug issues
5. **Week 5**: Read extension papers (prep for Week 10)

## Books

- **Deep Reinforcement Learning Hands-On** by Maxim Lapan
  - [Book](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
  - Practical DQN implementations

- **Grokking Deep Reinforcement Learning** by Miguel Morales
  - [Book](https://www.manning.com/books/grokking-deep-reinforcement-learning)
  - Intuitive explanations with code
