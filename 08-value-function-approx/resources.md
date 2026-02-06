# Week 8 Resources: Value Function Approximation

## Primary Textbook

- **Reinforcement Learning: An Introduction (2nd Edition)** by Sutton & Barto
  - [Free online version](http://incompleteideas.net/book/the-book-2nd.html)
  - Chapter 9: On-policy Prediction with Approximation
  - Chapter 10: On-policy Control with Approximation
  - Chapter 11: Off-policy Methods with Approximation (11.2 on Deadly Triad)

## Video Lectures

- **David Silver's RL Course - Lecture 6: Value Function Approximation**
  - [YouTube Link](https://www.youtube.com/watch?v=UoPei5o4fps)
  - [Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/FA.pdf)
  - Topics: Feature vectors, linear VFA, incremental methods, convergence

- **Stanford CS234 - Lecture on Function Approximation**
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
  - Emma Brunskill's course

## Papers

### Classic Papers

- **Baird, L. (1995).** "Residual Algorithms: Reinforcement Learning with Function Approximation"
  - [PDF Link](http://www.leemon.com/papers/1995b.pdf)
  - Introduces the famous counterexample showing divergence with deadly triad

- **Tsitsiklis, J. N., & Van Roy, B. (1997).** "An Analysis of Temporal-Difference Learning with Function Approximation"
  - [PDF Link](https://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf)
  - Theoretical analysis of TD with function approximation

### Gradient TD Methods

- **Sutton, R. S., et al. (2009).** "Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation"
  - [PDF Link](http://proceedings.mlr.press/v5/sutton09a/sutton09a.pdf)
  - Introduces GTD2 and TDC algorithms

## Implementation Resources

### Tile Coding

- **Sutton's Tile Coding Software**
  - [Python implementation](http://incompleteideas.net/tiles/tiles3.html)
  - Production-quality implementation with hashing

- **OpenAI Gym - Mountain Car**
  - [Environment documentation](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)
  - Standard benchmark for function approximation

### Code Examples

- **Reinforcement Learning: An Introduction - Code Repository**
  - [GitHub](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
  - Chapter 9 & 10 implementations in Python

- **Mountain Car with Tile Coding Tutorial**
  - [Medium Article](https://medium.com/@jaems33/reinforcement-learning-mountain-car-with-tile-coding-7b0b8c6c5e5d)
  - Step-by-step implementation guide

## Interactive Demos

- **Value Function Approximation Visualization**
  - [Demo Link](https://cs.stanford.edu/people/karpathy/reinforcejs/)
  - Visual demonstration of function approximation

## Additional Reading

### Blog Posts

- **Lilian Weng: "A (Long) Peek into Reinforcement Learning"**
  - [Blog Post](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
  - Section on function approximation

- **Arthur Juliani: "Simple Reinforcement Learning with Tensorflow"**
  - [Medium Series](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
  - Part 1.5 covers function approximation

### Advanced Topics

- **Geist, M., & Scherrer, B. (2014).** "Off-policy Learning with Eligibility Traces: A Survey"
  - [PDF Link](https://arxiv.org/pdf/1405.5098.pdf)
  - Deep dive into off-policy methods

- **Van Hasselt, H., & Sutton, R. S. (2015).** "Learning to Predict Independent of Span"
  - [PDF Link](https://arxiv.org/pdf/1508.04582.pdf)
  - Advanced theoretical perspective

## Software Libraries

- **OpenAI Gym**
  - [Documentation](https://www.gymlibrary.dev/)
  - Standard RL environments

- **Stable-Baselines3**
  - [GitHub](https://github.com/DLR-RM/stable-baselines3)
  - Reference implementations of RL algorithms

- **RLlib (Ray)**
  - [Documentation](https://docs.ray.io/en/latest/rllib/index.html)
  - Scalable RL library

## Practice Problems

- **Berkeley CS285 Homework 1**
  - [Homework Link](http://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/homeworks/hw1.pdf)
  - Includes value function approximation exercises

- **Stanford CS234 Assignments**
  - [Course Website](http://web.stanford.edu/class/cs234/index.html)
  - Assignments on function approximation

## Mathematical Background

- **Matrix Cookbook**
  - [PDF](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
  - Useful for gradients and matrix operations

- **Convex Optimization** by Boyd & Vandenberghe
  - [Free online book](https://web.stanford.edu/~boyd/cvxbook/)
  - Chapter 9 on gradient descent

## Discussion Forums

- **r/reinforcementlearning**
  - [Reddit Community](https://www.reddit.com/r/reinforcementlearning/)
  - Active discussions on RL topics

- **AI Stack Exchange**
  - [RL Tag](https://ai.stackexchange.com/questions/tagged/reinforcement-learning)
  - Q&A on function approximation

## Research Groups

- **Reinforcement Learning and Artificial Intelligence Lab (University of Alberta)**
  - [Website](http://rlai.ualberta.ca/)
  - Sutton's research group

- **MIT CSAIL - Distributed Robotics Lab**
  - [Website](https://groups.csail.mit.edu/drl/)
  - Research on RL and control

## Tools for Visualization

- **TensorBoard**
  - [Documentation](https://www.tensorflow.org/tensorboard)
  - Visualize learning curves and value functions

- **Matplotlib**
  - [Gallery](https://matplotlib.org/stable/gallery/index.html)
  - Create custom visualizations of value functions

- **Plotly**
  - [Documentation](https://plotly.com/python/)
  - Interactive 3D plots for value functions
