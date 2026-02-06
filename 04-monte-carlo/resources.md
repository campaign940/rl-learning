# Week 4 Resources: Monte Carlo Methods

## Primary Textbooks

### Sutton & Barto: Reinforcement Learning - An Introduction (2nd Edition)
- **Chapter 5: Monte Carlo Methods**
  - [Full Book PDF](http://incompleteideas.net/book/RLbook2020.pdf)
  - [Chapter 5 Direct Link](http://incompleteideas.net/book/RLbook2020.pdf#page=111)
  - Sections to focus on:
    - 5.1: Monte Carlo Prediction
    - 5.2: Monte Carlo Estimation of Action Values
    - 5.3: Monte Carlo Control
    - 5.4: Monte Carlo Control without Exploring Starts
    - 5.5: Off-policy Prediction via Importance Sampling
    - 5.6: Incremental Implementation
    - 5.7: Off-policy Monte Carlo Control

### David Silver's RL Course
- **Lecture 4: Model-Free Prediction**
  - [Lecture Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)
  - [Video Lecture](https://www.youtube.com/watch?v=PnHCvfgC_ZA)
  - Topics: Monte Carlo Learning, Temporal-Difference Learning, TD(λ)

- **Lecture 5: Model-Free Control**
  - [Lecture Slides PDF](https://www.davidsilver.uk/wp-content/uploads/2020/03/control.pdf)
  - [Video Lecture](https://www.youtube.com/watch?v=0g4j2k_Ggc4)
  - Topics: On-policy MC Control, Off-policy Learning

## University Courses

### CS234: Reinforcement Learning (Stanford)
- **Week 4: Model-Free Control**
  - [Course Website](http://web.stanford.edu/class/cs234/index.html)
  - [Lecture Slides](http://web.stanford.edu/class/cs234/slides/)
  - Topics: MC Control, Importance Sampling, Exploration

### CS285: Deep Reinforcement Learning (UC Berkeley)
- **Lecture 6: Actor-Critic Algorithms** (includes MC background)
  - [Course Website](http://rail.eecs.berkeley.edu/deeprlcourse/)
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PL_iWQOsE6TfXxKgI1GgyV1B_Xa0DxE5eH)

## Blog Posts and Tutorials

### Detailed Explanations

- **Lil'Log: Monte Carlo Methods**
  - [A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
  - Excellent visual explanations of MC prediction and control

- **Towards Data Science: MC Methods in RL**
  - [Introduction to Monte Carlo Methods](https://towardsdatascience.com/introduction-to-reinforcement-learning-monte-carlo-methods-5f4e9e8e7e15)
  - Practical Python examples

- **Medium: Off-Policy Learning**
  - [Understanding Importance Sampling](https://medium.com/@jonathan_hui/rl-importance-sampling-ebfb28b4a8c6)
  - Clear explanation of IS with visualizations

### Interactive Resources

- **OpenAI Spinning Up: Monte Carlo**
  - [MC Methods Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#model-free-rl)
  - Part of comprehensive RL education resource

- **RL Cheat Sheet**
  - [GitHub: RL Cheat Sheet](https://github.com/udacity/deep-reinforcement-learning)
  - Quick reference for MC algorithms

## Implementation Resources

### Gymnasium (OpenAI Gym) Environments

- **Gymnasium Documentation**
  - [Official Docs](https://gymnasium.farama.org/)
  - [Blackjack Environment](https://gymnasium.farama.org/environments/toy_text/blackjack/)

- **Sample Implementations**
  - [Dennybritz RL Repository](https://github.com/dennybritz/reinforcement-learning)
  - Clean Python implementations of MC algorithms
  - [MC Prediction Notebook](https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Prediction%20Solution.ipynb)
  - [MC Control Notebook](https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Control%20with%20Epsilon-Greedy%20Policies%20Solution.ipynb)

- **Sutton & Barto Code Repository**
  - [Official Code](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
  - Implementations matching textbook examples

## Video Lectures

### YouTube Tutorials

- **Stanford CS234 Full Course**
  - [YouTube Playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
  - Winter 2019, Emma Brunskill

- **DeepMind x UCL RL Lecture Series**
  - [Full Series](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
  - 2021 lectures by DeepMind researchers

- **Mutual Information: Monte Carlo Methods**
  - [MC in RL](https://www.youtube.com/watch?v=bpUszPiWM7o)
  - Visual intuition for MC concepts

## Research Papers

### Classic Papers

- **Monte Carlo Tree Search (MCTS)**
  - Browne et al. (2012): "A Survey of Monte Carlo Tree Search Methods"
  - [Paper Link](https://ieeexplore.ieee.org/document/6145622)

- **Importance Sampling in RL**
  - Precup et al. (2000): "Eligibility Traces for Off-Policy Policy Evaluation"
  - [Paper Link](https://scholarworks.umass.edu/cs_faculty_pubs/80/)

### Modern Applications

- **AlphaGo and Monte Carlo**
  - Silver et al. (2016): "Mastering the game of Go with deep neural networks and tree search"
  - [Nature Paper](https://www.nature.com/articles/nature16961)

## Tools and Libraries

### Python Libraries

- **Gymnasium** (successor to OpenAI Gym)
  - `pip install gymnasium`
  - [Documentation](https://gymnasium.farama.org/)

- **Stable-Baselines3**
  - `pip install stable-baselines3`
  - Includes MC-based algorithms
  - [Documentation](https://stable-baselines3.readthedocs.io/)

- **RLlib (Ray)**
  - `pip install ray[rllib]`
  - Scalable RL library
  - [Documentation](https://docs.ray.io/en/latest/rllib/)

### Visualization Tools

- **TensorBoard**
  - Track learning curves and value functions
  - `pip install tensorboard`

- **Matplotlib for RL**
  - Visualize policies and value functions
  - [Gallery Examples](https://matplotlib.org/stable/gallery/index.html)

## Practice Problems and Exercises

### Online Platforms

- **LeetCode RL Problems**
  - [Dynamic Programming Section](https://leetcode.com/tag/dynamic-programming/)
  - Foundational for understanding value iteration

- **Kaggle RL Competitions**
  - [Competitions](https://www.kaggle.com/competitions?search=reinforcement+learning)
  - Real-world RL challenges

### Textbook Exercises

- **S&B Chapter 5 Exercises**
  - Exercise 5.1: MC Prediction derivation
  - Exercise 5.4: Racetrack problem (challenging!)
  - Exercise 5.12: Importance sampling variance proof

## Discussion Forums and Communities

- **Reddit: r/reinforcementlearning**
  - [Subreddit](https://www.reddit.com/r/reinforcementlearning/)
  - Active community for questions

- **Stack Overflow: RL Tag**
  - [RL Questions](https://stackoverflow.com/questions/tagged/reinforcement-learning)

- **Discord: ML/RL Servers**
  - Yannic Kilcher Discord
  - OpenAI Scholars Program

## Quick Reference Sheets

- **RL Algorithms Cheat Sheet**
  - [GitHub Gist](https://gist.github.com/simoninithomas/7611db5d8a8f3e93f0a6b42e7fcc7e4b)

- **RL Terminology Glossary**
  - [Spinning Up Glossary](https://spinningup.openai.com/en/latest/spinningup/keypapers.html)

## Next Steps

After completing Week 4, you should:
1. Be comfortable implementing MC prediction and control
2. Understand importance sampling mechanics
3. Ready for Temporal-Difference learning (Week 5)
4. Able to compare model-free methods (MC vs TD)

**Recommended Path**:
1. Read S&B Chapter 5 (sections 5.1-5.4)
2. Watch David Silver Lecture 4
3. Implement Blackjack with MC control
4. Read sections 5.5-5.7 for off-policy methods
5. Complete quiz problems
6. Move to Week 5: Temporal-Difference Learning
