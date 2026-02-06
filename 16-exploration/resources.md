# Week 16 Resources: Exploration and Exploitation

## Primary Readings

### Lecture Videos

1. **David Silver's RL Course - Lecture 9: Exploration and Exploitation**
   - URL: https://www.youtube.com/watch?v=sGuiWX07sKw
   - Topics: Multi-armed bandits, UCB, contextual bandits, exploration in MDPs
   - Duration: ~90 minutes

2. **CS285 (Berkeley) - Lecture 12: Exploration (Part 1 & 2)**
   - Part 1: https://www.youtube.com/watch?v=SfCa1HQMkuw
   - Part 2: https://www.youtube.com/watch?v=Ol9OCTZUyj4
   - Topics: Count-based, pseudo-counts, curiosity, RND
   - Slides: https://rail.eecs.berkeley.edu/deeprlcourse/

### Key Papers

1. **Random Network Distillation (Burda et al., 2018)**
   - Paper: https://arxiv.org/abs/1810.12894
   - Blog: https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/
   - Code: https://github.com/openai/random-network-distillation
   - Key contribution: Simple exploration bonus outperforms complex methods
   - Result: First pure RL method to achieve >10K on Montezuma's Revenge

2. **Curiosity-Driven Exploration (ICM) (Pathak et al., 2017)**
   - Paper: https://arxiv.org/abs/1705.05363
   - Website: https://pathak22.github.io/noreward-rl/
   - Code: https://github.com/pathak22/noreward-rl
   - Key contribution: Learn features that capture controllable aspects
   - Result: Solves many sparse-reward games without external rewards

3. **Count-Based Exploration (Bellemare et al., 2016)**
   - Paper: https://arxiv.org/abs/1606.01868
   - Key contribution: Pseudo-counts from density models for exploration
   - Theory: Connects to PAC-MDP exploration

4. **Go-Explore (Ecoffet et al., 2019)**
   - Paper: https://arxiv.org/abs/1901.10995
   - Blog: https://www.uber.com/blog/go-explore/
   - Website: https://eng.uber.com/go-explore/
   - Key contribution: First to solve Montezuma's Revenge from pixels
   - Result: Superhuman performance on hard-exploration Atari games

5. **Never Give Up (NGU) (Badia et al., 2020)**
   - Paper: https://arxiv.org/abs/2002.06038
   - Key contribution: Episodic and lifelong novelty combined
   - Result: State-of-the-art on Atari exploration games

## Foundational Papers

1. **UCB Algorithm (Auer et al., 2002)**
   - Paper: https://link.springer.com/article/10.1023/A:1013689704352
   - "Finite-time Analysis of the Multiarmed Bandit Problem"
   - Classic bandits paper with optimal regret bounds

2. **Thompson Sampling (Thompson, 1933)**
   - Paper: https://academic.oup.com/biomet/article/25/3-4/285/203691
   - "On the Likelihood that One Unknown Probability Exceeds Another"
   - Bayesian approach to exploration

3. **R-MAX (Brafman & Tennenholtz, 2002)**
   - Paper: https://www.jmlr.org/papers/v3/brafman02a.html
   - PAC-MDP algorithm with optimism
   - Provably efficient exploration

4. **MBIE-EB (Strehl & Littman, 2008)**
   - Paper: https://www.jmlr.org/papers/v9/strehl08a.html
   - "An Analysis of Model-Based Interval Estimation"
   - Improved PAC-MDP bounds

## Supplementary Materials

### Blog Posts

1. **OpenAI: Reinforcement Learning with Prediction-Based Rewards**
   - URL: https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/
   - Excellent explanation of RND
   - Video demonstrations

2. **Lil'Log: Exploration Strategies in Deep RL**
   - URL: https://lilianweng.github.io/posts/2020-06-07-exploration-drl/
   - Comprehensive overview of exploration methods
   - Clear explanations with math

3. **Uber Engineering: Go-Explore**
   - URL: https://eng.uber.com/go-explore/
   - Accessible explanation of Go-Explore
   - Interactive visualizations

4. **The Promise of Curiosity-Driven Learning**
   - URL: https://pathak22.github.io/noreward-rl/
   - ICM project page with videos
   - Demos of curiosity-driven agents

### Textbook Sections

1. **Sutton & Barto - Chapter 2: Multi-Armed Bandits**
   - Epsilon-greedy, UCB, gradient bandits
   - PDF: http://incompleteideas.net/book/RLbook2020.pdf

2. **Tor Lattimore & Csaba Szepesvári - Bandit Algorithms**
   - URL: https://tor-lattimore.com/downloads/book/book.pdf
   - Comprehensive theory of bandits and exploration
   - Regret bounds and optimal algorithms

3. **Alekh Agarwal et al. - RL Theory Book (Chapter on Exploration)**
   - URL: https://rltheorybook.github.io/
   - PAC-MDP framework and sample complexity

## Code Repositories

1. **OpenAI Baselines - RND Implementation**
   - URL: https://github.com/openai/random-network-distillation
   - Official RND implementation
   - Includes Montezuma's Revenge experiments

2. **ICM Official Code**
   - URL: https://github.com/pathak22/noreward-rl
   - TensorFlow implementation
   - Multiple environments

3. **Go-Explore Implementation**
   - URL: https://github.com/uber-research/go-explore
   - Official implementation
   - Phase 1 and Phase 2 algorithms

4. **Stable-Baselines3 - Exploration Wrappers**
   - URL: https://github.com/DLR-RM/stable-baselines3
   - Clean implementations of various exploration strategies

5. **CleanRL - RND and ICM**
   - URL: https://github.com/vwxyzjn/cleanrl
   - Single-file implementations
   - Easy to understand and modify

## Tools and Environments

1. **Gymnasium (OpenAI Gym)**
   - URL: https://gymnasium.farama.org/
   - Standard RL environments
   - Atari games with sparse rewards

2. **MiniGrid**
   - URL: https://github.com/Farama-Foundation/Minigrid
   - Simple grid-world environments
   - Configurable sparse rewards
   - Good for testing exploration

3. **ProcGen**
   - URL: https://github.com/openai/procgen
   - Procedurally generated environments
   - Tests generalization and exploration

4. **NetHack Learning Environment**
   - URL: https://github.com/facebookresearch/nle
   - Extremely hard exploration
   - Research benchmark

## Advanced Topics

1. **Episodic Curiosity (Savinov et al., 2018)**
   - Paper: https://arxiv.org/abs/1810.02274
   - Reachability-based exploration
   - Combines with count-based methods

2. **Empowerment (Salge et al., 2014)**
   - Paper: https://arxiv.org/abs/1310.1863
   - Information-theoretic intrinsic motivation
   - Maximize agent's influence on environment

3. **Diversity is All You Need (Eysenbach et al., 2018)**
   - Paper: https://arxiv.org/abs/1802.06070
   - Learn diverse skills for exploration
   - Mutual information objective

4. **Disagreement-Based Exploration (Pathak et al., 2019)**
   - Paper: https://arxiv.org/abs/1906.04161
   - Use model ensemble disagreement for exploration

5. **Agent57 (Badia et al., 2020)**
   - Paper: https://arxiv.org/abs/2003.13350
   - First agent to achieve human-level on all 57 Atari games
   - Combines many exploration techniques

## Theoretical Papers

1. **PAC-MDP Framework (Strehl et al., 2006)**
   - Paper: https://www.jmlr.org/papers/v7/strehl06a.html
   - Theoretical foundation for exploration

2. **Sample Complexity of Reinforcement Learning (Kakade, 2003)**
   - Paper: https://homes.cs.washington.edu/~sham/papers/thesis/sham_thesis.pdf
   - PhD thesis on sample complexity

3. **Regret Bounds for RL (Jaksch et al., 2010)**
   - Paper: https://www.jmlr.org/papers/v11/jaksch10a.html
   - UCRL2 algorithm with optimal regret

4. **Posterior Sampling for RL (Osband et al., 2013)**
   - Paper: https://arxiv.org/abs/1306.0940
   - Thompson sampling in MDPs

5. **Information-Directed Sampling (Russo & Van Roy, 2014)**
   - Paper: https://arxiv.org/abs/1403.5556
   - Information theory for exploration

## Benchmarks and Datasets

1. **Atari-57 Suite**
   - Standard benchmark with hard exploration games
   - Montezuma's Revenge, Pitfall, Private Eye

2. **DeepMind Lab**
   - URL: https://github.com/deepmind/lab
   - 3D navigation environments
   - Sparse rewards

3. **MuJoCo Sparse Reward Variants**
   - Modified continuous control with sparse rewards

4. **Hard-Exploration Suite (Montezuma's Revenge, etc.)**
   - Specifically designed to test exploration
   - Standard benchmark for exploration papers

## Videos and Talks

1. **Deepak Pathak - Curiosity-Driven Learning (ICML 2018)**
   - URL: https://www.youtube.com/watch?v=l1FqtAHfJLI
   - Author presentation of ICM

2. **Sergey Levine - Exploration in Deep RL (CS285)**
   - URL: https://www.youtube.com/playlist?list=PL_iWQOsE6TfXxKgI1GgyV1B_Xa0DxE5eH
   - Comprehensive lecture series

3. **Emma Brunskill - Sample Complexity in RL**
   - URL: https://www.youtube.com/watch?v=H2GQDrqMsZA
   - Theoretical perspective on exploration

4. **Jeff Clune - Go-Explore (NeurIPS 2019)**
   - URL: https://www.youtube.com/watch?v=SWcuTgk2di4
   - Solving Montezuma's Revenge

## Discussion Forums

1. **Reddit r/reinforcementlearning**
   - URL: https://www.reddit.com/r/reinforcementlearning/
   - Active discussions on exploration methods

2. **RL Discord Servers**
   - Community discussions
   - Paper reading groups

3. **Stack Overflow RL Tag**
   - URL: https://stackoverflow.com/questions/tagged/reinforcement-learning

## Research Groups

1. **UC Berkeley BAIR**
   - URL: https://bair.berkeley.edu/
   - Sergey Levine's group
   - Exploration and model-based RL

2. **DeepMind**
   - Agent57, Never Give Up
   - State-of-the-art exploration research

3. **OpenAI**
   - RND, PPO with intrinsic rewards
   - Scaling exploration to hard games

4. **Uber AI Labs (archived)**
   - Go-Explore
   - Quality diversity algorithms

## Practical Tips

1. **Start Simple:**
   - Test on GridWorld before Atari
   - Verify exploration bonus is working

2. **Monitor Exploration:**
   - Track state visitation counts
   - Visualize intrinsic rewards
   - Check if agent finds sparse rewards

3. **Hyperparameters:**
   - Intrinsic reward coefficient: 0.01 to 1.0
   - Learning rate for predictor network
   - Observation normalization (critical for RND)

4. **Debugging:**
   - If no exploration: increase intrinsic coefficient
   - If too much exploration: decrease intrinsic coefficient
   - Check that intrinsic reward decreases over time (learning)

5. **Baselines:**
   - Always compare with no-exploration baseline
   - Track both extrinsic and intrinsic rewards separately

## Related Courses

1. **CS285: Deep Reinforcement Learning (UC Berkeley)**
   - URL: https://rail.eecs.berkeley.edu/deeprlcourse/
   - Lectures 12-13 on exploration

2. **CS234: Reinforcement Learning (Stanford)**
   - URL: https://web.stanford.edu/class/cs234/
   - Bandits and exploration theory

3. **Emma Brunskill's RL Course**
   - Sample complexity and PAC-MDP

## Historical Context

1. **Early Work (1990s):**
   - R-MAX, E3, optimistic initialization
   - Tabular methods with provable guarantees

2. **Deep Learning Era (2015+):**
   - Count-based exploration with neural density models
   - ICM: curiosity in high-dimensional spaces

3. **Modern Era (2018+):**
   - RND: simple and effective
   - Go-Explore: solving hardest Atari games
   - Agent57: human-level on all Atari 57

## Open Problems

1. **Scalable Exploration in Continuous Spaces:**
   - Most theory is for discrete spaces
   - How to explore efficiently in high dimensions?

2. **Safe Exploration:**
   - Avoid catastrophic failures
   - Exploration with constraints

3. **Transfer of Exploration Strategies:**
   - Can exploration skills transfer across tasks?
   - Meta-learning for exploration

4. **Exploration in POMDPs:**
   - Partial observability makes exploration harder
   - Need to explore observation and state spaces

5. **Sample Efficiency:**
   - Current methods still need millions of samples
   - How to explore with human-level efficiency?

## Next Week Preview

Week 17 introduces Reward Modeling and RLHF (Reinforcement Learning from Human Feedback), the technique powering ChatGPT and aligned AI systems. We'll learn how to train reward models from preference data and optimize policies for human values.
