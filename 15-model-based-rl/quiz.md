# Week 15 Quiz: Model-Based Reinforcement Learning

## Question 1: Advantages and Disadvantages of Learning an Environment Model

**What are the main advantages and disadvantages of learning an environment model compared to model-free RL? Discuss sample efficiency, asymptotic performance, and computational costs.**

<details>
<summary>Answer</summary>

**Advantages:**

1. **Sample Efficiency:**
   - Can generate unlimited synthetic experience from limited real data
   - 10-100x more sample efficient than model-free methods
   - Critical for real-world applications (robotics, expensive simulations)

2. **Planning and Lookahead:**
   - Can plan multiple steps ahead without environment interaction
   - Enables sophisticated search algorithms (MCTS, MPC)
   - Can evaluate many candidate actions quickly

3. **Transfer and Generalization:**
   - Learned model can transfer to new tasks in same environment
   - Can adapt to new reward functions without retraining policy
   - Model captures environment structure independent of task

4. **Interpretability:**
   - Can visualize and analyze model predictions
   - Easier to debug and understand agent behavior
   - Can identify where model is uncertain

**Disadvantages:**

1. **Model Errors:**
   - Prediction errors compound over long rollouts
   - Can lead to exploitation of model inaccuracies ("model hacking")
   - May hurt asymptotic performance even with perfect policy optimization

2. **Computational Cost:**
   - Must learn both model and policy
   - Planning can be expensive (especially tree search)
   - Model training requires significant computation

3. **Implementation Complexity:**
   - More components to tune (model architecture, rollout length, etc.)
   - Harder to implement correctly
   - Requires careful balance between real and synthetic data

4. **Domain Applicability:**
   - Some environments are too complex or stochastic to model accurately
   - High-dimensional observation spaces (images) are challenging
   - May not work well in environments with discontinuities

**When to Use:**
- Sample efficiency is critical
- Environment has learnable structure
- Planning is valuable
- Can accept lower asymptotic performance

**When to Avoid:**
- Unlimited samples available
- Environment is extremely complex or stochastic
- Asymptotic performance is most important
- Computational resources are limited

</details>

---

## Question 2: MBPO Short Rollouts

**Describe how Model-Based Policy Optimization (MBPO) uses short model rollouts to augment real data. What determines the optimal rollout length, and why is it important to limit rollout length?**

<details>
<summary>Answer</summary>

**MBPO Algorithm:**

1. **Data Collection:**
   - Collect real data from environment using current policy: D_env
   - Store all transitions in replay buffer

2. **Model Training:**
   - Train dynamics model on all real data
   - Minimize prediction error: L = E[||f_theta(s,a) - s'||^2]

3. **Synthetic Rollout Generation:**
   - For each real state s in D_env:
     - Perform k-step rollout using learned model
     - s_0 = s (real)
     - s_{t+1} = f_theta(s_t, a_t) for t = 0, ..., k-1
   - Store synthetic transitions in model buffer D_model

4. **Policy Training:**
   - Train policy (e.g., SAC) on mixture of D_env and D_model
   - Typically use equal amounts or more model data

**Optimal Rollout Length:**

The optimal rollout length k* depends on model error epsilon:

```
k* ~ O(sqrt(epsilon / (1 - gamma)))
```

**Intuition:**
- Model error epsilon per step compounds over rollout
- After k steps, cumulative error ~ k * epsilon (linear) or worse
- Want synthetic data quality similar to real data
- Too short: Don't benefit from model
- Too long: Synthetic data becomes inaccurate, hurts policy

**Empirical Finding:**
- k = 1-5 works well for most tasks
- Typical: k = 1 early in training, increase to k = 5 as model improves

**Why Limit Rollout Length:**

1. **Compounding Errors:**
   ```
   Error at step k ~ sum_{t=0}^{k-1} epsilon_t
   ```
   - Errors accumulate over trajectory
   - States drift to unrealistic regions

2. **Distribution Shift:**
   - Long rollouts explore states not seen in real data
   - Model extrapolates poorly to out-of-distribution states
   - Policy learns to exploit model errors

3. **Diminishing Returns:**
   - Most benefit comes from first few steps
   - Later steps in rollout have little advantage over real data

**Key Insight:**
Short rollouts provide sample efficiency gains while preventing model exploitation. MBPO achieves state-of-the-art sample efficiency by carefully balancing real and synthetic data.

</details>

---

## Question 3: Comparing MBRL Approaches

**Compare and contrast four approaches to model-based RL: Dyna-Q, MBPO, World Models, and MuZero. What are the key differences in how they use learned models?**

<details>
<summary>Answer</summary>

| Aspect | Dyna-Q | MBPO | World Models | MuZero |
|--------|--------|------|--------------|---------|
| **Model Type** | Tabular or deterministic | Probabilistic ensemble | VAE + RNN | Latent dynamics |
| **Model Output** | Next state s' | Next state distribution | Latent z_{t+1} | Latent s^{t+1} + reward |
| **Planning Method** | Value backup | None (data augmentation) | Train in dream | MCTS in latent space |
| **Policy Training** | Q-learning on real+synthetic | SAC on real+synthetic | CMA-ES in model | Self-play with search |
| **Rollout Length** | 1-step backups | Short (k=1-5) | Full episodes | Deep tree search |
| **Key Innovation** | Integrated learning+planning | Optimal rollout length | Latent space dynamics | Task-agnostic representation |

**Dyna-Q (Sutton 1991):**
- **Approach:** Integrate planning and learning
- **Model Use:** 1-step backups for value updates
- **Algorithm:**
  ```
  For each real experience (s,a,r,s'):
    Update Q(s,a) using real data
    Update model M(s,a) = s'
    Repeat n times:
      Sample random (s,a) from experience
      s', r = M(s,a)
      Update Q(s,a) toward r + gamma * max_a' Q(s',a')
  ```
- **Pros:** Simple, proven effective
- **Cons:** Doesn't scale to high-dimensional spaces

**MBPO (Janner et al. 2019):**
- **Approach:** Data augmentation with short rollouts
- **Model Use:** Generate synthetic transitions
- **Key insight:** Rollout length should match model accuracy
- **Algorithm:**
  ```
  Train ensemble of models on real data
  Generate k-step rollouts from real states
  Train SAC on mix of real + synthetic
  ```
- **Pros:** State-of-the-art sample efficiency, theoretically grounded
- **Cons:** Still requires significant hyperparameter tuning

**World Models (Ha & Schmidhuber 2018):**
- **Approach:** Learn compressed world model, train policy in dream
- **Model Use:** Full episode generation in latent space
- **Components:**
  - V (Vision): VAE encodes obs to latent z
  - M (Memory): RNN predicts z_{t+1} = M(z_t, a_t)
  - C (Controller): Policy in latent space
- **Training:**
  1. Collect random rollouts
  2. Train V and M on observations
  3. Train C using CMA-ES entirely in model
- **Pros:** Trains policy without environment access, fast
- **Cons:** Separate training stages, limited to simple environments

**MuZero (Schrittwieser et al. 2020):**
- **Approach:** Learn latent model for planning
- **Model Use:** MCTS in learned latent space
- **Key insight:** Don't need to predict observations, just value/policy
- **Components:**
  - h(o): Representation function (obs to latent)
  - g(s,a): Dynamics function (latent transition + reward)
  - f(s): Prediction function (policy and value)
- **Planning:** MCTS using learned dynamics g
- **Training:** End-to-end from self-play
- **Pros:** Handles images, long horizons, superhuman performance
- **Cons:** Requires massive compute, complex implementation

**Key Differences:**

1. **Model Fidelity:**
   - Dyna/MBPO: Predict actual next states
   - World Models: Predict in VAE latent space
   - MuZero: Task-specific latent representation

2. **Planning vs Data Augmentation:**
   - Dyna/MuZero: Use model for planning
   - MBPO/World Models: Use model to generate training data

3. **Online vs Offline:**
   - Dyna/MBPO: Interleaved with environment interaction
   - World Models/MuZero: Can train policy without real interaction

4. **Sample Efficiency:**
   - MuZero > MBPO > World Models > Dyna-Q

</details>

---

## Question 4: Designing a Model-Based Approach for Robot Arm

**Design a model-based RL approach for a robot arm reaching task. What would the model predict? What planning method would you use? How would you handle model uncertainty?**

<details>
<summary>Answer</summary>

**Task Specification:**
- **State:** Joint angles, velocities, end-effector position
- **Action:** Joint torques or velocity commands
- **Reward:** Negative distance to target + penalties for jerky motion
- **Challenges:** Contact dynamics, high-dimensional state, safety constraints

**Model Design:**

**Option 1: State-Space Model**
```
Model: (s, a) -> (s', r)
s = [joint_angles, joint_velocities, ee_position]
s' = f_theta(s, a) + epsilon, epsilon ~ N(0, Sigma)
```

**Option 2: Latent-Space Model**
```
Encoder: z = h(s)
Dynamics: z' = g(z, a)
Decoder: s' = d(z')
```
Better for high-dimensional observations (vision).

**Recommended: Ensemble of Probabilistic Models**
```
Train N=5 models {f_1, ..., f_5}
Prediction: s' ~ (1/N) sum_i N(mu_i(s,a), Sigma_i(s,a))
Uncertainty: Var = (1/N) sum_i [mu_i - mean_mu]^2
```

**Planning Method:**

**Recommended: Model Predictive Control (MPC)**

```python
def mpc_planning(current_state, target, horizon=10):
    best_action_sequence = None
    best_value = -inf

    # Sampling-based MPC
    for _ in range(num_samples):
        action_sequence = sample_actions(horizon)

        # Rollout in model
        state = current_state
        total_reward = 0
        uncertainty_penalty = 0

        for t in range(horizon):
            # Get ensemble predictions
            predictions = [model_i(state, action_sequence[t])
                          for model_i in ensemble]

            # Mean prediction
            next_state = mean(predictions)

            # Uncertainty penalty
            uncertainty = variance(predictions)
            uncertainty_penalty += uncertainty_weight * uncertainty

            # Reward
            reward = compute_reward(next_state, target)
            total_reward += (gamma ** t) * reward

            state = next_state

        value = total_reward - uncertainty_penalty

        if value > best_value:
            best_value = value
            best_action_sequence = action_sequence

    return best_action_sequence[0]  # Execute first action only
```

**Alternative: MBPO-Style Approach**
```
1. Collect real robot experience
2. Train ensemble of dynamics models
3. Generate short (k=5) rollouts from real states
4. Train SAC policy on real + synthetic data
5. Deploy policy (no planning at test time)
```

**Handling Model Uncertainty:**

1. **Model Ensemble:**
   - Train multiple models with different initializations
   - Variance indicates epistemic uncertainty
   - Penalize actions that lead to high disagreement

2. **Conservative Planning:**
   ```
   Value(s,a) = E[R] - beta * Var[R]
   ```
   Trade off expected return vs uncertainty

3. **Safety Constraints:**
   ```python
   def is_safe(state, action, ensemble):
       predictions = [model(state, action) for model in ensemble]

       # Check all predictions satisfy constraints
       for pred in predictions:
           if violates_joint_limits(pred) or risk_collision(pred):
               return False
       return True

   # Only consider safe actions in planning
   safe_actions = [a for a in action_candidates if is_safe(s, a, ensemble)]
   ```

4. **Active Learning:**
   - Explore regions with high uncertainty
   - Collect more data where model is uncertain
   - Retrain model with new data

**Full Algorithm:**

```
Initialize:
  - Train initial dynamics ensemble on random data
  - Initialize policy

Main Loop:
  For each timestep:
    1. Plan action using MPC with uncertainty penalty
    2. Execute action in real environment
    3. Store transition in buffer

    4. Periodically:
       - Retrain dynamics models on all data
       - Fine-tune policy (optional)

    5. Safety check:
       - If model uncertainty too high, request human demo
       - If unexpected outcome, update model immediately
```

**Practical Considerations:**

1. **Safety:** Always verify planned actions in simulator first
2. **Sim-to-Real:** Train model on real robot data, not just simulation
3. **Latency:** Planning must be fast (<100ms for real-time control)
4. **Fallback:** Have safe default policy if planning fails
5. **Monitoring:** Track model prediction error over time

**Expected Benefits:**
- 10-20x better sample efficiency than model-free
- Safer due to planning and uncertainty awareness
- Can adapt to new targets without retraining policy

</details>

---

## Question 5: When Does Model-Based RL Fail?

**When does model-based RL fail? Discuss compounding model errors, model exploitation, and scenarios where model-free methods are preferable.**

<details>
<summary>Answer</summary>

**Major Failure Modes:**

**1. Compounding Model Errors**

**Problem:**
Small per-step errors accumulate exponentially over long rollouts.

```
True state trajectory: s_0 -> s_1 -> s_2 -> ... -> s_T
Model prediction:      s_0 -> s_1' -> s_2'' -> ... -> s_T^(error)

Error at step k: ||s_k - s_k^(pred)|| ~ k * epsilon (optimistic)
```

**Example - Pendulum:**
- Model error: 0.1 radians per step
- After 10 steps: 1 radian error (significant)
- After 50 steps: completely wrong region of state space

**Why it happens:**
- Model learns average dynamics, misses details
- Small errors push trajectory to unvisited states
- Model extrapolates poorly to OOD states
- Errors in one prediction affect all future predictions

**Solutions:**
- Short rollouts (MBPO)
- Model ensembles with uncertainty
- Planning in latent space (MuZero)
- Frequent replanning (MPC)

**2. Model Exploitation (Goodhart's Law)**

**Problem:**
"When a measure becomes a target, it ceases to be a good measure."

Policy learns to exploit model errors to get artificially high predicted rewards.

**Example - Simulated Robot:**
```
Real environment: Robot falls if moves too fast
Model: Imperfectly captures physics at high speeds
Policy: Discovers that fast jerky motions get high reward in model
Result: Policy fails catastrophically in real environment
```

**Why it happens:**
- Model is imperfect approximation
- Policy optimization finds adversarial inputs to model
- Model errors are not random - they have structure
- Policy exploits systematic biases

**Detection:**
```python
# Check for exploitation
model_value = evaluate_policy_in_model(policy, model)
real_value = evaluate_policy_in_env(policy, env)

if model_value >> real_value:
    print("Warning: Model exploitation detected!")
```

**Solutions:**
- Conservative planning (penalize uncertainty)
- Adversarial training of model
- Regular real-world evaluation
- Limit policy optimization steps on synthetic data

**3. High-Dimensional Observations**

**Problem:**
Learning to predict pixel-level observations is extremely hard.

**Example - Atari from pixels:**
- State: 84x84x4 image = 28,224 dimensions
- Model must predict every pixel accurately
- Small errors in pixels compound
- Irrelevant details (background) must be modeled

**Why model-free wins:**
- DQN/A3C can learn directly from pixels
- Don't need to predict future observations
- Only need to predict value/policy

**Solutions:**
- Learn latent dynamics (World Models, Dreamer)
- Task-specific representations (MuZero)
- Use model-free for vision, model-based for dynamics

**4. Stochastic Environments**

**Problem:**
Environments with irreducible randomness are hard to model.

**Examples:**
- Weather prediction
- Financial markets
- Multi-agent games with unpredictable opponents

**Why it fails:**
```
s' = f(s, a) + noise
```
- Model must capture full distribution, not just mean
- High-variance outcomes are hard to learn
- Planning with stochastic model is computationally expensive

**When model-free is better:**
- Learn expected value directly: Q(s,a) = E[R|s,a]
- Averages over randomness automatically
- Don't need to model unpredictable details

**5. Discontinuities and Contacts**

**Problem:**
Discontinuous dynamics (collisions, friction) are hard to model with neural networks.

**Example - Ball bouncing:**
```
Before contact: smooth dynamics
At contact: velocity reverses instantly
After contact: smooth dynamics again
```

Neural network struggles with discontinuity.

**Solutions:**
- Use physics simulators (not learned models)
- Hybrid approaches (model-based for contacts, learning for other dynamics)
- Separate models for different contact modes

**6. Computational Cost**

**Problem:**
Model-based RL requires:
- Training dynamics model
- Planning (can be expensive)
- More hyperparameters to tune

**When model-free is better:**
- Cheap/fast environments (simulators, games)
- Unlimited samples available
- Simple environments where model-free works well

**Comparison: When to Use Each**

| Criterion | Model-Based | Model-Free |
|-----------|-------------|------------|
| **Sample efficiency** | 10-100x better | Baseline |
| **Asymptotic performance** | May be worse | Often better |
| **Computation** | Higher | Lower |
| **Implementation** | Complex | Simpler |
| **Environment** | Deterministic, learnable | Works everywhere |
| **Observations** | Low-dim preferred | Handles pixels well |

**Scenarios Favoring Model-Free:**

1. **Unlimited samples:** Games, fast simulators
2. **High-dimensional obs:** Vision-based tasks with model-free vision networks
3. **Stochastic environments:** Unpredictable dynamics
4. **Discontinuous dynamics:** Contacts, collisions
5. **Asymptotic performance critical:** Accept lower sample efficiency for better final policy
6. **Simple implementation needed:** Tight deadlines, limited engineering resources

**Best Practice:**

Consider hybrid approaches:
- Use model-free for image encoding
- Use model-based for low-level control
- Combine strengths of both paradigms

**Real-World Example:**

Google's robot grasping:
- Model-free: Visual perception (CNN)
- Model-based: Arm control (MPC with learned dynamics)
- Result: Sample-efficient learning with robust perception

</details>

---

## Scoring Rubric

**Question 1:** /20 points
- Advantages (8 pts): Sample efficiency, planning, transfer, interpretability
- Disadvantages (8 pts): Model errors, computational cost, complexity
- Tradeoffs (4 pts): When to use each approach

**Question 2:** /20 points
- MBPO algorithm (6 pts)
- Optimal rollout length formula and intuition (8 pts)
- Why limit rollout length (6 pts)

**Question 3:** /25 points
- Dyna-Q (5 pts)
- MBPO (5 pts)
- World Models (5 pts)
- MuZero (5 pts)
- Key differences (5 pts)

**Question 4:** /20 points
- Model design (5 pts)
- Planning method (5 pts)
- Uncertainty handling (6 pts)
- Practical considerations (4 pts)

**Question 5:** /25 points
- Compounding errors (5 pts)
- Model exploitation (5 pts)
- High-dimensional observations (5 pts)
- Stochastic environments (5 pts)
- When to use model-free (5 pts)

**Total:** /110 points

**Grading Scale:**
- 100+: Exceptional understanding
- 90-99: Strong understanding
- 80-89: Good understanding
- 70-79: Adequate understanding
- <70: Needs review
