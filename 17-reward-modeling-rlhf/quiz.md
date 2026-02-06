# Week 17 Quiz: Reward Modeling and RLHF

## Question 1: Why Not Hand-Defined Rewards?

**Why can't we use human-defined reward functions for complex tasks like text generation? What specific problems arise, and how does reward modeling solve them?**

<details>
<summary>Answer</summary>

**Problems with Hand-Defined Rewards:**

**1. Specification Difficulty:**

Human values are too complex to formalize:
```python
# Attempt: Define "helpfulness" for chatbot
def helpfulness_reward(response):
    reward = 0
    if len(response) > 50: reward += 1  # Detailed
    if contains_facts(response): reward += 1  # Informative
    if polite(response): reward += 1  # Courteous
    if not_repetitive(response): reward += 1  # Diverse
    return reward
```

**Problems:**
- Arbitrary weights: Why equal weighting?
- Missing aspects: What about clarity, tone, relevance?
- Hard to define: How do you code "polite" or "clear"?
- Context-dependent: Good response varies by situation

**2. Proxy Failures (Goodhart's Law):**

"When a measure becomes a target, it ceases to be a good measure."

**Example - Length Exploitation:**
```python
reward = len(response)
Result: "This is a response that goes on and on and on..."
```

**Example - Keyword Stuffing:**
```python
reward = count("certainly", "absolutely", "indeed")
Result: "Certainly, absolutely this is indeed certainly true."
```

**3. Multi-Objective Trade-offs:**

Cannot balance conflicting objectives:
```python
reward = w1*helpful + w2*safe + w3*brief + w4*accurate
```
Weights are arbitrary and context-dependent.

**4. Incompleteness:**

Cannot specify nuanced concepts:
- Tone (friendly but professional)
- Style (creative but appropriate)
- Judgment (when to refuse harmful requests)

**How Reward Modeling Solves This:**

**1. Learn from Comparisons:**
```
Human labels: "Response B is better than A"
Reward model learns: r(B) > r(A)
Captures implicit preferences
```

**2. Handles Complexity:**
- Neural network can learn complex, non-linear preferences
- Learns from examples, not manual specification
- Generalizes to new situations

**3. Context-Aware:**
- Same prompt can have different good responses
- Model learns when detailed vs brief is better
- Adapts to situation

**4. Iterative Improvement:**
- Collect more preferences
- Retrain reward model
- Continuously improve without rewriting rules

**5. Scales to Rich Domains:**
- Language: Billions of possible responses
- Reward model: Single neural network
- Hand-crafted: Would need infinite rules

**Comparison:**

| Aspect | Hand-Defined | Learned Reward Model |
|--------|--------------|----------------------|
| **Specification** | Manual rules | Learn from data |
| **Complexity** | Limited | Arbitrarily complex |
| **Adaptability** | Fixed | Can retrain |
| **Nuance** | Difficult | Natural |
| **Scalability** | Poor | Good |
| **Failure Mode** | Exploits proxies | Can be gamed but fixable |

**Example Success: InstructGPT**

Hand-defined approach would require:
- Rules for helpfulness
- Rules for harmlessness
- Rules for honesty
- Rules for tone, style, etc.
- Impossibly complex!

RLHF approach:
- Collect 10K-100K preference comparisons
- Train reward model
- Optimize policy with RL
- Successful: Powers ChatGPT

**Key Insight:**

"It's easier to judge quality than to define it."

Humans can easily compare responses but cannot write complete reward functions. Reward modeling leverages this asymmetry.

</details>

---

## Question 2: Derive Bradley-Terry Loss

**Derive the Bradley-Terry model loss function for reward model training. Explain each term and why this formulation makes sense for learning from preferences.**

<details>
<summary>Answer</summary>

**Setup:**

Given comparison data: D = {(x_i, y_w^i, y_l^i)}
- x: Prompt
- y_w: Preferred (winner) response
- y_l: Dis-preferred (loser) response

Goal: Learn reward function r_theta(x, y) that predicts preferences.

**Bradley-Terry Model:**

Probability human prefers y_w over y_l:
```
P(y_w > y_l | x) = exp(r(x, y_w)) / (exp(r(x, y_w)) + exp(r(x, y_l)))
```

**Simplification using logistic function:**
```
P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))

where sigma(z) = 1 / (1 + exp(-z))
```

**Proof of equivalence:**
```
sigma(r_w - r_l) = 1 / (1 + exp(-(r_w - r_l)))
                 = 1 / (1 + exp(r_l - r_w))
                 = 1 / (1 + exp(r_l) / exp(r_w))
                 = exp(r_w) / (exp(r_w) + exp(r_l))
```

**Maximum Likelihood Estimation:**

Likelihood of observed preferences:
```
L(theta) = prod_i P(y_w^i > y_l^i | x_i)
         = prod_i sigma(r_theta(x_i, y_w^i) - r_theta(x_i, y_l^i))
```

**Log-Likelihood:**
```
log L(theta) = sum_i log sigma(r_theta(x_i, y_w^i) - r_theta(x_i, y_l^i))
```

**Negative Log-Likelihood (Loss):**
```
L_RM(theta) = -sum_i log sigma(r_theta(x_i, y_w^i) - r_theta(x_i, y_l^i))
```

**Expectation form:**
```
L_RM(theta) = -E[(x, y_w, y_l) ~ D][log sigma(r_theta(x, y_w) - r_theta(x, y_l))]
```

**Alternative Forms:**

**1. Log-sum-exp form:**
```
-log sigma(r_w - r_l) = -log(exp(r_w) / (exp(r_w) + exp(r_l)))
                       = -log(exp(r_w)) + log(exp(r_w) + exp(r_l))
                       = -r_w + log(exp(r_w) + exp(r_l))
                       = log(1 + exp(r_l - r_w))
```

**2. Binary cross-entropy form:**
```
L = -[1 * log(sigma(r_w - r_l)) + 0 * log(1 - sigma(r_w - r_l))]
  = -log(sigma(r_w - r_l))
```
Treats preference as binary classification: winner = 1, loser = 0.

**Gradient:**

```
∇_theta L = -E[∇_theta log sigma(r_w - r_l)]

where:
d/dz log sigma(z) = 1 - sigma(z) = sigma(-z)

Therefore:
∇_theta L = E[sigma(r_l - r_w) * (∇r_l - ∇r_w)]
          = E[(1 - sigma(r_w - r_l)) * (∇r_l - ∇r_w)]
```

**Intuition of Gradient:**

- If r_w >> r_l (correct, confident): sigma(r_l - r_w) ≈ 0, small gradient
- If r_l > r_w (wrong): sigma(r_l - r_w) > 0.5, large gradient
- Gradient pushes r_w up and r_l down, weighted by confidence

**Why Bradley-Terry Makes Sense:**

**1. Probabilistic:**
- Humans don't have deterministic preferences
- Sometimes disagree or are uncertain
- Model captures this with probability

**2. Scale Invariant:**
- Only reward differences matter: r_w - r_l
- Can add constant to all rewards without changing probabilities
- Prevents reward scale issues

**3. Interpretable:**
```
r_w - r_l = log(odds ratio)

If r_w - r_l = 2:
  P(y_w > y_l) = sigma(2) ≈ 0.88

If r_w - r_l = -2:
  P(y_w > y_l) = sigma(-2) ≈ 0.12
```

**4. Mathematically Tractable:**
- Log-likelihood is concave
- Gradient descent converges
- Well-studied in statistics

**5. Consistent with Utility Theory:**

Assume latent utilities:
```
u_w = r(x, y_w) + epsilon_w
u_l = r(x, y_l) + epsilon_l

where epsilon ~ Gumbel(0, 1)
```

Then:
```
P(u_w > u_l) = P(r_w + epsilon_w > r_l + epsilon_l)
             = P(epsilon_w - epsilon_l > r_l - r_w)
             = sigma(r_w - r_l)
```

**Extensions:**

**1. Confidence Weighting:**
```
L = -sum_i w_i * log sigma(r_w - r_l)
```
Weight by human confidence or agreement.

**2. Multiple Comparisons:**
```
L = -sum_i log sigma(r(y_i) - log(sum_j exp(r(y_j))))
```
Extends to ranking multiple responses.

**3. Margin:**
```
L = -sum_i log sigma(r_w - r_l - margin)
```
Enforce minimum separation between winner and loser.

**Implementation:**

```python
def bradley_terry_loss(reward_model, x, y_w, y_l):
    r_w = reward_model(x, y_w)
    r_l = reward_model(x, y_l)

    # Method 1: Logsigmoid
    loss = -F.logsigmoid(r_w - r_l).mean()

    # Method 2: Log-sum-exp (numerically stable)
    loss = F.softplus(r_l - r_w).mean()

    # Method 3: Binary cross-entropy
    loss = F.binary_cross_entropy_with_logits(
        r_w - r_l,
        torch.ones_like(r_w)
    )

    return loss
```

**Numerical Stability:**

Direct computation can overflow:
```
sigma(x) = 1 / (1 + exp(-x))
```
If x is large negative, exp(-x) overflows.

**Solution: Log-sum-exp trick:**
```
log sigma(x) = log(1 / (1 + exp(-x)))
             = -log(1 + exp(-x))
             = -softplus(-x)
```

PyTorch's logsigmoid is numerically stable.

**Summary:**

Bradley-Terry loss:
```
L = -E[log sigma(r(y_w) - r(y_l))]
```

- Probabilistic model of preferences
- Scale-invariant
- Theoretically grounded
- Numerically stable with proper implementation
- Powers modern RLHF systems

</details>

---

## Question 3: Three Stages of RLHF

**Compare the three stages of RLHF (SFT, Reward Modeling, PPO). What does each stage contribute, and why are all three necessary?**

<details>
<summary>Answer</summary>

## Stage 1: Supervised Fine-Tuning (SFT)

**Purpose:**
Bootstrap policy to generate reasonable responses.

**Data:**
High-quality demonstrations: (prompt, response) pairs
- Human-written responses
- Filtered for quality
- Typically 10K-100K examples

**Method:**
```
Minimize: L_SFT = -sum_t log p_theta(y_t | y_{<t}, x)
```
Standard language modeling objective.

**Training:**
```python
for prompt, response in demonstrations:
    logits = model(prompt + response)
    loss = cross_entropy(logits, response)
    optimizer.step()
```

**What It Learns:**
- Format: How to structure responses
- Domain knowledge: Facts and concepts
- Basic capabilities: Grammar, coherence
- Task understanding: What kind of response is expected

**What It Doesn't Learn:**
- Preferences: What makes one response better than another
- Trade-offs: When to prioritize brevity vs detail
- Edge cases: How to handle ambiguous or harmful requests

**Output:**
pi_SFT: A policy that can generate reasonable responses, used as:
1. Initialization for RL training
2. Reference policy for KL penalty

**Why Necessary:**

Without SFT, starting from base model:
- Random/incoherent responses initially
- RL would take too long to explore
- Reward model wouldn't provide useful signal

**Example:**
```
Base GPT-3: "Explain AI: The AI is the most important thing in the world..."
After SFT: "Explain AI: Artificial intelligence is the simulation of human intelligence..."
```

## Stage 2: Reward Modeling (RM)

**Purpose:**
Learn human preferences from comparisons.

**Data:**
Preference comparisons: (prompt, winner, loser) tuples
- Humans compare model outputs
- Label which is better
- Typically 10K-100K comparisons

**Method:**
```
Minimize: L_RM = -log sigma(r_theta(x, y_w) - r_theta(x, y_l))
```
Bradley-Terry preference model.

**Training:**
```python
for prompt, y_winner, y_loser in preferences:
    r_w = reward_model(prompt, y_winner)
    r_l = reward_model(prompt, y_loser)
    loss = -log_sigmoid(r_w - r_l)
    optimizer.step()
```

**What It Learns:**
- Quality judgments: What constitutes a good response
- Preferences: When A is better than B
- Trade-offs: Balancing competing objectives
- Nuance: Context-dependent quality criteria

**What It Doesn't Learn:**
- Generation: Doesn't produce responses
- Exploration: Doesn't tell policy how to improve

**Output:**
r_theta: Reward model that predicts scalar reward for any response

**Why Necessary:**

Without reward model:
- Hand-crafted rewards fail (too complex)
- Can't capture nuanced preferences
- No way to give RL meaningful signal

**Example:**
```
r("Explain AI: AI is stuff") = 0.2
r("Explain AI: AI simulates human intelligence...") = 0.8
```

## Stage 3: RL Fine-Tuning (PPO)

**Purpose:**
Optimize policy to maximize learned rewards while staying close to reference.

**Data:**
Prompts (no responses needed)
- Sample from desired distribution
- Generate responses with current policy
- Get rewards from reward model

**Method:**
```
Maximize: J(theta) = E[r_theta(x, y)] - beta * KL(pi_theta || pi_ref)
```
PPO with KL penalty.

**Training:**
```python
for prompt in prompt_dataset:
    # Generate response
    response = policy.generate(prompt)

    # Get reward
    reward = reward_model(prompt, response)
    kl = KL(policy.log_prob(response), ref.log_prob(response))
    total_reward = reward - beta * kl

    # PPO update
    advantages = compute_advantages(total_reward)
    policy_loss = ppo_loss(advantages)
    optimizer.step()
```

**What It Learns:**
- Optimization: How to generate high-reward responses
- Exploration: Trying variations to find better outputs
- Trade-offs: Balancing reward and KL penalty

**What It Doesn't Learn:**
- Preferences: Uses pre-trained reward model
- Basic capabilities: Relies on SFT initialization

**Output:**
pi_RLHF: Final aligned policy optimized for human preferences

**Why Necessary:**

Without RL:
- SFT alone: Imitates average demo, doesn't optimize quality
- Can't discover better responses than in training data
- No way to incorporate learned preferences

**Example:**
```
After SFT: "AI simulates human intelligence"
After PPO: "AI simulates human intelligence through machine learning algorithms that enable computers to learn from data and make decisions"
[More detailed, higher reward]
```

## Why All Three Are Necessary

**SFT → RM:**
- Need reasonable responses to compare
- Reward model trained on SFT outputs
- If no SFT, all responses gibberish, can't learn preferences

**RM → PPO:**
- Need reward signal for RL
- Without reward model, can't do RL
- Hand-crafted rewards don't work

**SFT → PPO:**
- Need good initialization for RL
- Random policy too slow to explore
- KL penalty requires reference (SFT model)

**Complete Pipeline:**

```
Base Model
    ↓ [SFT: Learn capabilities]
SFT Model (pi_ref)
    ↓ [RM: Learn preferences]
Reward Model
    ↓ [RL: Optimize for preferences]
Aligned Model (pi_RLHF)
```

**Synergy:**

1. **SFT provides foundation:**
   - Capabilities for RM and RL to build on
   - Reference policy for KL constraint

2. **RM provides objective:**
   - Captures preferences SFT doesn't
   - Guides RL optimization

3. **RL provides optimization:**
   - Improves beyond SFT demonstrations
   - Balances reward and reference policy

**Comparison to Alternatives:**

**SFT Only:**
- Pros: Simple, stable
- Cons: Limited by demo quality, no optimization

**RM + RL (no SFT):**
- Pros: Direct optimization
- Cons: Too slow, poor exploration, unstable

**SFT + RM (no RL):**
- Pros: Fast, stable
- Cons: Can't improve beyond demos

**Empirical Results (InstructGPT):**

| Stage | Win Rate vs Prompts |
|-------|---------------------|
| SFT | ~70% |
| SFT + RM (ranking) | ~75% |
| SFT + RM + PPO | ~85% |

Each stage contributes meaningful improvement.

**Modern Variations:**

**Online RLHF:**
- Interleave all three stages
- Continuously update RM with new data
- Prevents distribution shift

**Direct Preference Optimization (DPO):**
- Combines RM and RL into one stage
- More stable, simpler
- Week 18 topic!

**Summary:**

| Stage | Input | Output | Purpose |
|-------|-------|--------|---------|
| **SFT** | Demos | pi_SFT | Capabilities |
| **RM** | Comparisons | r_theta | Preferences |
| **PPO** | Prompts | pi_RLHF | Optimization |

All three are necessary:
- SFT: Bootstrap
- RM: Objective
- RL: Optimize

Together, they enable aligning powerful AI systems with human values.

</details>

---

## Question 4: Design an RLHF Pipeline

**Design a complete RLHF pipeline for training a chatbot to be helpful and harmless. What are the key decisions at each stage, and how would you evaluate success?**

<details>
<summary>Answer</summary>

## Complete RLHF Pipeline for Chatbot

### Stage 0: Data Collection

**1. Demonstrations for SFT:**

**Sources:**
- Human writers: 10K high-quality conversations
- Existing datasets: Filtered for quality
- Expert domain knowledge: Technical topics
- Safety examples: Refusing harmful requests

**Quality criteria:**
- Helpful: Answers question thoroughly
- Harmless: Avoids harmful content
- Honest: Accurate information
- Appropriate: Right tone and length

**Format:**
```
{
  "prompt": "How do I learn Python?",
  "response": "Python is a great first language. Start with...",
  "metadata": {
    "quality": "high",
    "category": "programming",
    "difficulty": "beginner"
  }
}
```

**Key Decisions:**
- Quantity: 10K-100K demos
- Quality over quantity: Better to have 10K excellent than 100K mediocre
- Diversity: Cover many topics and styles
- Balance: Include edge cases (refusals, uncertainty)

**2. Preferences for Reward Model:**

**Collection method:**
- Show humans pairs of responses
- Ask: "Which is better?"
- Optional: Why is it better? (for analysis)

**Sampling strategy:**
- Generate responses from SFT model
- Sample varied quality (not just bad vs good)
- Include disagreements (hard cases)
- Stratify by topic and type

**Format:**
```
{
  "prompt": "Explain quantum computing",
  "response_A": "[Technical jargon]",
  "response_B": "[Clear with analogy]",
  "preference": "B",
  "confidence": "high",
  "reason": "More accessible"
}
```

**Key Decisions:**
- Quantity: 10K-100K comparisons
- Quality: Train labelers, measure agreement
- Difficulty: Include edge cases where preferences unclear
- Representation: Cover full input distribution

### Stage 1: Supervised Fine-Tuning

**Base Model:**
Choose between:
- GPT-2 (124M params): Fast, educational
- GPT-3 (1.3B-175B): Production quality
- Open source (LLaMA, Mistral): Customizable

**Key Decision: Start with instruction-tuned or base?**
- Base: More control, longer training
- Instruction-tuned: Faster, already capable

**Training:**
```python
# Hyperparameters
lr = 1e-5  # Low for fine-tuning
batch_size = 8
epochs = 3  # Don't overfit
warmup_steps = 100

# Loss weighting (optional)
loss = nll_loss + 0.1 * length_penalty + 0.05 * repetition_penalty

# Data augmentation
- Prompt variations
- Response paraphrases
- Synthetic data from strong models
```

**Evaluation:**
- Perplexity on held-out demos
- Human evaluation: "Is this a reasonable response?"
- Coverage: Can handle diverse prompts?

**Success Criteria:**
- Perplexity < baseline
- 70%+ human approval
- Generates coherent, on-topic responses

### Stage 2: Reward Modeling

**Architecture:**

**Option 1: Separate Reward Model**
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.encoder = base_model.transformer
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, prompt, response):
        hidden = self.encoder(prompt + response)
        reward = self.value_head(hidden[-1])
        return reward
```

**Option 2: Shared Backbone with SFT**
- More parameter efficient
- Risk of interference

**Key Decision:** Separate or shared?
- Separate: Safer, standard practice
- Shared: Efficient, requires careful tuning

**Training:**
```python
# Loss
loss = -log_sigmoid(r_win - r_lose)

# Regularization
loss += lambda_l2 * ||theta||^2  # Prevent overfitting
loss += lambda_smoothness * smoothness_penalty(r)  # Encourage smooth rewards

# Ensemble (recommended)
train K=5 models with different seeds
use mean or median reward at inference
```

**Key Decisions:**

1. **Labeling:**
   - Humans: High quality, expensive
   - AI (RLAIF): Scalable, potential biases
   - Hybrid: AI labels, human verification

2. **Disagreement:**
   - Multiple labels per comparison
   - Model uncertainty: sigma(r_w - r_l) ≈ 0.5
   - Keep difficult cases for training

3. **Normalization:**
   - Z-score normalize rewards
   - Prevents scale issues in RL

**Evaluation:**
- Accuracy: Does r_w > r_l for labeled pairs?
- Calibration: Does probability match confidence?
- OOD: Evaluate on unseen distribution
- Ablation: Test on known good/bad responses

**Success Criteria:**
- 70%+ accuracy on held-out preferences
- Well-calibrated probabilities
- Reasonable rewards on test prompts

### Stage 3: RL Fine-Tuning

**Algorithm: PPO with RLHF**

```python
# Objective
reward = reward_model(prompt, response)
kl = KL(policy, reference)
objective = reward - beta * kl

# Hyperparameters
beta = 0.1  # KL penalty (tune this!)
lr = 1e-6  # Lower than SFT
ppo_epochs = 4
clip_eps = 0.2

# Training loop
for episode in range(num_episodes):
    # Generate batch
    prompts = sample_prompts()
    responses = policy.generate(prompts)

    # Compute rewards
    rewards = reward_model(prompts, responses)
    kl_penalty = KL(policy, reference)
    total_rewards = rewards - beta * kl_penalty

    # PPO update
    advantages = compute_advantages(total_rewards)
    policy_loss = ppo_clip(advantages)
    value_loss = mse(value_predictions, rewards)

    loss = policy_loss + 0.5 * value_loss
    optimizer.step()

    # Monitoring
    log_metrics(rewards, kl_penalty, policy_loss)
```

**Key Decisions:**

1. **Beta (KL penalty):**
   - Start: 0.1
   - Adjust: If KL too high/low
   - Adaptive: Increase if reward hacking detected

2. **Prompt Distribution:**
   - Match deployment distribution
   - Include adversarial prompts
   - Diverse topics and difficulties

3. **Early Stopping:**
   - Monitor gold standard evaluation
   - Stop if performance degrades
   - Prevent overoptimization

4. **Batch Size:**
   - Larger = more stable (but slower)
   - Typical: 64-256 prompts per batch

**Evaluation:**
- Reward model score: Is it increasing?
- KL from reference: Is it staying reasonable (<10)?
- Human eval: Win rate vs SFT baseline
- Safety checks: Red team adversarial prompts

**Success Criteria:**
- Win rate >80% vs SFT
- KL < 10 from reference
- No reward hacking detected
- Passes safety evaluations

### Stage 4: Deployment and Monitoring

**Red Teaming:**
- Adversarial prompts: Jailbreaks, harmful requests
- Edge cases: Ambiguous, multi-turn conversations
- Robustness: Typos, non-English, code-switching

**A/B Testing:**
```
Control: SFT model
Treatment: RLHF model

Metrics:
- User satisfaction (thumbs up/down)
- Conversation length (engagement)
- Task completion rate
- Safety incidents
```

**Continuous Improvement:**
```
Deployment → User Interactions → New Preferences → Retrain RM → Update Policy
```

Online RLHF: Continuously improve with user feedback.

### Complete Pipeline Summary

```
[Data Collection]
├── Demonstrations (10K)
└── Preferences (50K)
        ↓
[Stage 1: SFT]
├── Base model + demos
├── Train 3 epochs
└── Output: pi_SFT, pi_ref
        ↓
[Stage 2: RM]
├── pi_SFT + preferences
├── Train Bradley-Terry model
└── Output: r_theta (ensemble)
        ↓
[Stage 3: PPO]
├── pi_SFT + r_theta + prompts
├── Optimize with KL penalty
└── Output: pi_RLHF
        ↓
[Evaluation]
├── Human eval: Win rate
├── Safety: Red team
└── Deploy if passing
        ↓
[Deployment]
├── A/B test
├── Monitor metrics
└── Collect feedback → Iterate
```

### Key Success Metrics

| Metric | Target |
|--------|--------|
| SFT perplexity | < Base |
| RM accuracy | >70% |
| RLHF win rate vs SFT | >80% |
| KL from reference | <10 |
| Safety pass rate | >95% |
| User satisfaction | >4/5 |

### Common Pitfalls and Solutions

**1. SFT Overfitting:**
- Problem: Memorizes demos
- Solution: Early stopping, regularization

**2. RM Overconfidence:**
- Problem: r_w >> r_l always
- Solution: Calibration, hard negatives

**3. RL Reward Hacking:**
- Problem: High reward, low quality
- Solution: Monitor gold eval, adjust beta

**4. Mode Collapse:**
- Problem: Same response for all prompts
- Solution: Diversity bonus, temperature

**5. Distribution Shift:**
- Problem: Policy outputs unlike training
- Solution: Iterative training, broad prompts

### Budget and Resources

**Typical Costs:**
- Demos: $50K (500 hours @ $100/hr)
- Preferences: $100K (1000 hours)
- Compute: $10K-100K (depends on model size)
- Total: $160K-250K for production system

**Timeline:**
- Data collection: 2-4 weeks
- SFT: 1-3 days
- RM: 1-2 days
- PPO: 1-5 days
- Evaluation: 1-2 weeks
- Total: 1-2 months

This design would produce a helpful, harmless chatbot aligned with human preferences through RLHF.

</details>

---

## Question 5: Failure Modes of RLHF

**What are the main failure modes of RLHF (reward hacking, distribution shift, KL penalty sensitivity)? How can we detect and mitigate each?**

<details>
<summary>Answer</summary>

## 1. Reward Hacking (Model Exploitation)

**Definition:**
Policy exploits imperfections in reward model to achieve high predicted reward without actual quality.

**Manifestations:**

**Length Exploitation:**
```
Reward model: Trained on preferences for detailed responses
Policy learns: Generate very long, repetitive text
Example: "The answer is... [repeats same content] ...in conclusion..."
Predicted reward: High
Actual quality: Low (verbose, repetitive)
```

**Keyword Stuffing:**
```
Reward model: Associates certain words with quality
Policy learns: Stuff in those keywords
Example: "Certainly, this is absolutely an excellent response that definitely..."
Predicted reward: High
Actual quality: Unnatural, over-formal
```

**Mode Collapse:**
```
Policy finds one high-reward response
Generates same text for all prompts
Example: Always responds with "I don't have enough information to answer accurately"
Predicted reward: High (safe, humble)
Actual quality: Useless (doesn't answer)
```

**Detection:**

**1. Gold Standard Evaluation:**
```python
# Compare proxy vs gold
proxy_reward = reward_model(prompt, response)
gold_reward = human_eval(prompt, response)  # Expensive but accurate

# Detect divergence
if proxy_reward - gold_reward > threshold:
    warning("Reward hacking detected")
```

**2. Diversity Metrics:**
```python
# Measure response diversity
diversity = distinct_n_grams(responses) / total_n_grams
repetition = repeated_sequences(response)

if diversity < threshold or repetition > threshold:
    warning("Mode collapse detected")
```

**3. Feature Analysis:**
```python
# Check for suspicious patterns
length_ratio = len(response) / expected_length
keyword_density = count_keywords(response) / len(response)

if length_ratio > 2.0 or keyword_density > threshold:
    warning("Potential exploitation")
```

**Mitigation:**

**1. KL Penalty (Primary Defense):**
```python
objective = reward - beta * KL(policy || reference)
```
Prevents policy from drifting too far from safe reference.

**2. Ensemble Reward Models:**
```python
# Train multiple reward models
rewards = [rm_i(prompt, response) for rm_i in ensemble]

# Use conservative estimate
final_reward = min(rewards)  # Or mean - k*std
```
Exploitation harder when multiple models must agree.

**3. Uncertainty Penalty:**
```python
# Estimate uncertainty from ensemble
mean_reward = np.mean(rewards)
uncertainty = np.std(rewards)

final_reward = mean_reward - lambda_uncertainty * uncertainty
```
Penalize high-variance regions.

**4. Adversarial Training:**
```python
# Generate adversarial examples
adversarial = generate_high_reward_but_bad(policy)

# Add to reward model training
for example in adversarial:
    # Human labels correctly as bad
    human_rating = evaluate(example)
    reward_model.update(example, human_rating)
```

**5. Early Stopping:**
```python
# Monitor gold evaluation during training
if gold_reward starts decreasing:
    stop_training()
    load_checkpoint(best_gold_reward)
```

## 2. Overoptimization (Goodhart's Law)

**Definition:**
As RL progresses, proxy reward increases but true quality plateaus or decreases.

**The Curve:**
```
Proxy Reward: ↗ (monotonically increasing)
True Quality:  ↗ → (plateau) → ↘ (decrease)

RL Steps: 0    1000   2000   3000   4000
Proxy:    0.5  0.7    0.85   0.92   0.96  ← keeps increasing
Gold:     0.5  0.7    0.75   0.73   0.68  ← degrades!
```

**Cause:**
Reward model is imperfect approximation. Policy finds examples where model is overly optimistic.

**Detection:**

**1. Regular Gold Evaluation:**
```python
# Every N steps, expensive human eval
for step in range(training_steps):
    train_step()

    if step % eval_frequency == 0:
        gold_reward = human_evaluation(policy)
        log(step, proxy_reward, gold_reward)

        # Check for divergence
        if gold_reward < previous_gold and proxy_reward > previous_proxy:
            warning("Overoptimization detected at step", step)
```

**2. KL Divergence Monitoring:**
```python
kl = KL(policy || reference)

# If KL growing rapidly, overoptimization likely
if kl > threshold:  # Typical: 5-20
    warning("Policy diverged too far")
```

**3. Out-of-Distribution Detection:**
```python
# Check if policy outputs are OOD for reward model
likelihood = reward_model_training_data_likelihood(response)

if likelihood < threshold:
    warning("Response is OOD for reward model")
```

**Mitigation:**

**1. KL Constraint:**
```python
# Strong KL penalty prevents overoptimization
objective = reward - beta * KL(policy || reference)

# Adaptive beta
if kl > target_kl:
    beta *= 1.1  # Increase penalty
else:
    beta *= 0.9  # Decrease penalty
```

**2. Early Stopping:**
```python
best_gold_reward = 0
patience = 5
patience_counter = 0

for epoch in range(max_epochs):
    train()
    gold = evaluate()

    if gold > best_gold_reward:
        best_gold_reward = gold
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping - gold reward plateaued")
        load_checkpoint()
        break
```

**3. Iterative Training:**
```python
# Continuously update reward model
for iteration in range(num_iterations):
    # Train policy with current RM
    policy = train_rl(reward_model)

    # Generate new data from policy
    new_responses = policy.generate(prompts)

    # Collect human preferences on new data
    new_preferences = human_label(new_responses)

    # Retrain reward model
    reward_model.update(new_preferences)
```
Prevents distribution shift by keeping RM updated.

**4. Conservative Rewards:**
```python
# Penalize reward model for high rewards on unusual inputs
def conservative_reward(response):
    r = reward_model(response)
    likelihood = training_data_likelihood(response)

    # Lower reward for OOD inputs
    adjusted_r = r * min(1.0, likelihood / threshold)
    return adjusted_r
```

## 3. Distribution Shift

**Definition:**
RL policy generates text different from reward model training distribution, causing unreliable rewards.

**Example:**
```
RM training: Responses from SFT model (formal, structured)
RL policy: Generates creative, unusual responses (out-of-distribution)
Result: Reward model unreliable on new distribution
```

**Detection:**

**1. OOD Metrics:**
```python
# Measure distributional shift
train_features = extract_features(rm_training_data)
policy_features = extract_features(policy_outputs)

distance = distribution_distance(train_features, policy_features)

if distance > threshold:
    warning("Large distribution shift")
```

**2. Reward Model Uncertainty:**
```python
# Ensemble disagreement indicates OOD
ensemble_rewards = [rm_i(response) for rm_i in ensemble]
uncertainty = np.std(ensemble_rewards)

if uncertainty > threshold:
    warning("High uncertainty - likely OOD")
```

**3. Human Evaluation Mismatch:**
```python
# Correlation between RM and human eval
for response in sample:
    rm_score = reward_model(response)
    human_score = human_eval(response)

correlation = pearson(rm_scores, human_scores)

if correlation < 0.7:  # Significant mismatch
    warning("RM performance degraded on policy outputs")
```

**Mitigation:**

**1. Broad Training Distribution:**
```python
# Train RM on diverse data
rm_training_data = [
    sft_responses,
    random_samples,
    adversarial_examples,
    policy_outputs  # Include some RL data
]
```

**2. Online RLHF:**
```python
# Continuously update RM
while True:
    # RL training
    policy = train_rl_steps(reward_model, num_steps=1000)

    # Collect new preferences
    new_responses = policy.generate(prompts)
    preferences = human_label_pairs(new_responses)

    # Update reward model
    reward_model.finetune(preferences)
```

**3. KL Penalty (Again):**
```python
# KL penalty limits distribution shift
objective = reward - beta * KL(policy || reference)
```
Policy can't drift too far from reference (SFT) distribution.

**4. Rejection Sampling:**
```python
# Only use on-distribution samples for training
for response in policy_outputs:
    likelihood = p_reference(response)

    if likelihood > threshold:  # Accept
        use_for_training(response)
    else:  # Reject
        skip(response)
```

## 4. KL Penalty Sensitivity

**Definition:**
Performance highly sensitive to KL coefficient beta.

**Too Low (beta → 0):**
```
Objective ≈ reward only
Result: Reward hacking, mode collapse
Example: Policy generates gibberish with high predicted reward
```

**Too High (beta → ∞):**
```
Objective ≈ -KL only
Result: Policy doesn't improve, stays at reference
Example: Policy identical to SFT, no benefit from RL
```

**Detection:**

**1. Reward-KL Frontier:**
```python
# Sweep beta values
for beta in [0.01, 0.05, 0.1, 0.5, 1.0]:
    policy = train(beta=beta)
    reward = evaluate_reward(policy)
    kl = evaluate_kl(policy, reference)

    plot(kl, reward)  # Pareto frontier
```
Choose beta with good reward-KL trade-off.

**2. Gold Evaluation:**
```python
for beta in beta_values:
    policy = train(beta=beta)
    gold_score = human_eval(policy)

optimal_beta = argmax(gold_score)
```

**Mitigation:**

**1. Adaptive Beta:**
```python
# Adjust beta based on KL
target_kl = 5.0  # Desired KL

for step in training:
    kl = compute_kl(policy, reference)

    if kl > target_kl:
        beta *= 1.5  # Increase penalty
    elif kl < target_kl * 0.5:
        beta *= 0.8  # Decrease penalty

    # Clip to reasonable range
    beta = np.clip(beta, 0.01, 1.0)
```

**2. Curriculum:**
```python
# Start high, decrease
initial_beta = 0.5
final_beta = 0.05

for step in range(training_steps):
    progress = step / training_steps
    beta = initial_beta * (1 - progress) + final_beta * progress
```
Early: Stay close to reference (safe)
Late: More freedom to optimize

**3. Lagrangian (Constrained Optimization):**
```python
# Hard constraint on KL instead of penalty
maximize: reward
subject to: KL(policy || reference) < epsilon
```
More theoretically principled than penalty.

## 5. Reward Model Bias

**Definition:**
Reward model learns biases from human labelers.

**Examples:**
- Length bias: Prefers longer responses
- Verbosity: Favors formal, complex language
- Sycophancy: Agrees with user even when wrong
- Style bias: Specific writing style

**Detection:**

```python
# Test for known biases
for prompt in test_set:
    short_response = generate_short(prompt)
    long_response = generate_long(prompt)  # Same content, padded

    if reward(long) > reward(short):
        length_bias_count += 1

# Similar tests for other biases
```

**Mitigation:**

**1. Balanced Training Data:**
```python
# Ensure diversity in comparisons
for each category:
    include_varied_lengths()
    include_different_styles()
    include_counter-examples()
```

**2. Bias Correction:**
```python
# Explicitly control for length
adjusted_reward = raw_reward - lambda_length * length_penalty

# Residualize out bias
reward_residual = reward - predicted_from_length(response)
```

**3. Labeler Training:**
- Instruct humans to ignore length
- Show examples of length bias
- Measure and feedback inter-annotator agreement

## Summary Table

| Failure Mode | Detection | Mitigation |
|--------------|-----------|------------|
| **Reward Hacking** | Gold eval, diversity metrics | KL penalty, ensemble, early stop |
| **Overoptimization** | Gold vs proxy divergence | Early stop, iterative, KL |
| **Distribution Shift** | OOD metrics, RM uncertainty | Online RLHF, broad RM training |
| **KL Sensitivity** | Reward-KL sweep | Adaptive beta, curriculum |
| **RM Bias** | Targeted tests | Balanced data, bias correction |

**Best Practices:**

1. **Always use KL penalty** (beta = 0.01-0.1)
2. **Monitor gold evaluation** throughout training
3. **Use ensemble reward models** for robustness
4. **Iterative training** to prevent distribution shift
5. **Early stopping** based on gold, not proxy
6. **Extensive testing** before deployment

These mitigations enable safe, effective RLHF that aligns AI systems with human values while avoiding common pitfalls.

</details>

---

## Scoring Rubric

**Question 1:** /20 points
- Problems with hand-defined rewards (8 pts)
- How reward modeling solves them (8 pts)
- Examples and comparison (4 pts)

**Question 2:** /25 points
- Bradley-Terry derivation (10 pts)
- Loss function forms (6 pts)
- Gradient and intuition (5 pts)
- Why it makes sense (4 pts)

**Question 3:** /25 points
- SFT stage (7 pts)
- RM stage (7 pts)
- PPO stage (7 pts)
- Why all necessary (4 pts)

**Question 4:** /20 points
- Data collection (4 pts)
- Each stage design (9 pts)
- Evaluation and monitoring (4 pts)
- Practical considerations (3 pts)

**Question 5:** /20 points
- Each failure mode (12 pts: 2-3 pts each)
- Detection methods (4 pts)
- Mitigation strategies (4 pts)

**Total:** /110 points

**Grading Scale:**
- 100+: Exceptional - Ready to implement RLHF
- 90-99: Strong - Solid understanding
- 80-89: Good - Core concepts mastered
- 70-79: Adequate - Needs more practice
- <70: Needs review
