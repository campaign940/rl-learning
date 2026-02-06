# Week 15 Resources: Model-Based Reinforcement Learning

## Primary Readings

### Lecture Videos

1. **David Silver's RL Course - Lecture 8: Model-Based RL**
   - URL: https://www.youtube.com/watch?v=ItMutbeOHtc
   - Topics: Dyna, integrated learning and planning, simulation-based search
   - Duration: ~90 minutes

2. **CS285 (Berkeley) - Lecture 11: Model-Based RL**
   - URL: https://www.youtube.com/watch?v=iC2a7M9voYU
   - Topics: Learned dynamics models, uncertainty, model-based policy optimization
   - Slides: https://rail.eecs.berkeley.edu/deeprlcourse/

### Key Papers

1. **World Models (Ha & Schmidhuber, 2018)**
   - Paper: https://arxiv.org/abs/1803.10122
   - Website: https://worldmodels.github.io/
   - Code: https://github.com/hardmaru/WorldModelsExperiments
   - Key contribution: Learn compressed spatial and temporal representation, train agent in dream
   - Result: Competitive performance on CarRacing with sample-efficient training

2. **MBPO - When to Trust Your Model (Janner et al., 2019)**
   - Paper: https://arxiv.org/abs/1906.08253
   - Code: https://github.com/jannerm/mbpo
   - Key contribution: Theoretical analysis of optimal model rollout length
   - Result: State-of-the-art sample efficiency on MuJoCo benchmarks

3. **MuZero (Schrittwieser et al., 2020)**
   - Paper: https://arxiv.org/abs/1911.08265
   - Blog: https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/
   - Key contribution: Learn latent model for planning without predicting observations
   - Result: Superhuman performance on Atari, Chess, Go without knowing rules

4. **Dreamer (Hafner et al., 2019)**
   - Paper: https://arxiv.org/abs/1912.01603
   - Website: https://danijar.com/project/dreamer/
   - Code: https://github.com/danijar/dreamer
   - Key contribution: Learn long-horizon behaviors purely by latent imagination
   - Result: Strong performance on visual control tasks

5. **PlaNet (Hafner et al., 2019)**
   - Paper: https://arxiv.org/abs/1811.04551
   - Key contribution: Deep planning network for model-based RL
   - Result: Competitive with model-free on image-based control

## Foundational Papers

1. **Dyna Architecture (Sutton, 1991)**
   - Paper: https://link.springer.com/article/10.1007/BF00115009
   - Classic work on integrating planning and learning

2. **PILCO (Deisenroth & Rasmussen, 2011)**
   - Paper: https://ieeexplore.ieee.org/document/6654139
   - Gaussian process dynamics models for data-efficient RL

3. **Neural Network Dynamics for Model-Based Deep RL (Nagabandi et al., 2018)**
   - Paper: https://arxiv.org/abs/1708.02596
   - Simple but effective approach to model-based deep RL

## Supplementary Materials

### Blog Posts

1. **World Models Blog Post**
   - URL: https://worldmodels.github.io/
   - Interactive visualization of learned world model
   - Highly recommended for intuition

2. **The Promise of Model-Based RL**
   - URL: https://bair.berkeley.edu/blog/2019/12/12/mbpo/
   - BAIR blog on MBPO and when to trust your model

3. **MuZero: Mastering Go, Chess, Shogi, and Atari Without Rules**
   - URL: https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/
   - DeepMind blog explaining MuZero

4. **Model-Based RL Tutorial (ICML 2020)**
   - URL: https://sites.google.com/view/mbrl-tutorial
   - Comprehensive tutorial slides and videos

### Textbook Sections

1. **Sutton & Barto - Chapter 8: Planning and Learning with Tabular Methods**
   - Covers Dyna architecture
   - PDF: http://incompleteideas.net/book/RLbook2020.pdf

2. **Sergey Levine's CS285 Notes**
   - URL: https://rail.eecs.berkeley.edu/deeprlcourse/
   - Excellent lecture notes on model-based RL

## Code Repositories

1. **MBPO Official Implementation**
   - URL: https://github.com/jannerm/mbpo
   - PyTorch implementation of MBPO

2. **World Models**
   - URL: https://github.com/hardmaru/WorldModelsExperiments
   - TensorFlow implementation

3. **Dreamer**
   - URL: https://github.com/danijar/dreamer
   - TensorFlow 2 implementation

4. **MuZero Pseudocode**
   - URL: https://github.com/deepmind/open_spiel/tree/master/open_spiel/python/algorithms/muzero
   - Reference implementation in Open Spiel

5. **Simple MBRL Implementations**
   - URL: https://github.com/openai/baselines
   - OpenAI Baselines includes model-based algorithms

6. **MBRL Library**
   - URL: https://github.com/facebookresearch/mbrl-lib
   - Facebook Research's model-based RL library
   - Includes PETS, MBPO, and other algorithms

## Tools and Libraries

1. **Gymnasium (OpenAI Gym)**
   - URL: https://gymnasium.farama.org/
   - Standard RL environments

2. **MuJoCo**
   - URL: https://mujoco.org/
   - Physics simulator for robotics (now free)

3. **dm_control**
   - URL: https://github.com/deepmind/dm_control
   - DeepMind Control Suite

4. **PlaNet Baseline**
   - URL: https://github.com/google-research/planet
   - Reference implementation

## Advanced Topics

1. **Model Ensembles and Uncertainty**
   - "Deep Exploration via Bootstrapped DQN" (Osband et al., 2016)
   - Paper: https://arxiv.org/abs/1602.04621

2. **Combining Model-Free and Model-Based RL**
   - "Imagination-Augmented Agents" (Weber et al., 2017)
   - Paper: https://arxiv.org/abs/1707.06203

3. **Visual Model-Based RL**
   - "Learning Latent Dynamics for Planning from Pixels" (Hafner et al., 2019)
   - Paper: https://arxiv.org/abs/1811.04551

4. **Model-Based Meta-RL**
   - "Model-Based Meta-Reinforcement Learning" (Fakoor et al., 2019)
   - Paper: https://arxiv.org/abs/1903.08254

## Datasets

1. **D4RL: Datasets for Deep Data-Driven RL**
   - URL: https://sites.google.com/view/d4rl/home
   - Can be used to evaluate learned models offline

2. **Real Robot Datasets**
   - URL: https://sites.google.com/view/reality-robonet
   - Real-world robot interaction data for model learning

## Videos and Talks

1. **David Ha - World Models (NeurIPS 2018)**
   - URL: https://www.youtube.com/watch?v=HzA8LRqhujk
   - Author presentation of World Models

2. **Sergey Levine - Model-Based Deep RL (ICML 2019)**
   - URL: https://sites.google.com/view/mbrl-tutorial
   - Comprehensive tutorial

3. **Julian Schrittwieser - MuZero (DeepMind)**
   - URL: https://www.youtube.com/watch?v=L0A86LmH7Yw
   - MuZero overview by lead author

## Discussion Forums

1. **Reddit r/reinforcementlearning**
   - URL: https://www.reddit.com/r/reinforcementlearning/

2. **Model-Based RL Discord**
   - Active community discussing MBRL research

3. **Stack Overflow RL Tag**
   - URL: https://stackoverflow.com/questions/tagged/reinforcement-learning

## Benchmarks

1. **MuJoCo Continuous Control**
   - Hopper, HalfCheetah, Ant, Humanoid
   - Standard benchmark for sample efficiency

2. **DMControl Suite**
   - URL: https://github.com/deepmind/dm_control
   - More challenging control tasks

3. **Atari Games**
   - Test model-based RL on pixel observations
   - Good for evaluating learned world models

## Related Courses

1. **CS285: Deep Reinforcement Learning (UC Berkeley)**
   - URL: https://rail.eecs.berkeley.edu/deeprlcourse/
   - Comprehensive modern RL course

2. **CS234: Reinforcement Learning (Stanford)**
   - URL: https://web.stanford.edu/class/cs234/

3. **Advanced Deep Learning & Reinforcement Learning (DeepMind/UCL)**
   - URL: https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs

## Historical Papers (Optional)

1. **Dyna-Q (Sutton, 1990)**
   - Foundation of model-based RL
   - Integrates planning and learning

2. **Prioritized Sweeping (Moore & Atkeson, 1993)**
   - Efficient planning with learned models

3. **TEXPLORE (Hester & Stone, 2012)**
   - Model-based exploration in continuous spaces

## Open Problems and Recent Research

1. **Overcoming Model Bias**
   - How to avoid exploiting model errors?

2. **Scaling to Complex Environments**
   - Real-world robotics with high-dimensional observations

3. **Combining Strengths**
   - Hybrid model-free and model-based methods

4. **Sample-Efficient Exploration**
   - Using model uncertainty for exploration

## Practical Tips

1. Start with simple environments (CartPole, Pendulum) before complex ones
2. Use model ensembles to estimate uncertainty
3. Keep rollouts short initially (k=1-3)
4. Monitor model prediction error over time
5. Compare model-based vs model-free on same task
6. Visualize model predictions to debug issues

## Next Week Preview

Week 16 focuses on exploration strategies: UCB, count-based exploration, curiosity-driven methods (ICM), and random network distillation (RND). We'll see how exploration bonuses help in sparse-reward environments like Montezuma's Revenge.
