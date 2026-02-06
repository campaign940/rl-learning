# Week 17: Reward Modeling and RLHF

## Overview

Reinforcement Learning from Human Feedback (RLHF) is the breakthrough technique that enables AI systems like ChatGPT, Claude, and GPT-4 to align with human values and preferences. Instead of hand-crafting reward functions, RLHF learns reward models from human preferences and uses RL to optimize policies. This week covers the complete RLHF pipeline from preference collection to policy optimization.

## Learning Objectives

- Understand why reward modeling is necessary for complex tasks
- Learn the Bradley-Terry preference model and reward model training
- Master the three-stage RLHF pipeline: SFT, reward modeling, and PPO
- Implement a complete RLHF training loop
- Analyze failure modes: reward hacking, distribution shift, and overoptimization

## Key Concepts

### 1. The Reward Specification Problem

**Why We Can't Just Write Reward Functions:**

For simple tasks, we can specify rewards:
```python
# CartPole: Stay upright
reward = 1.0 if angle < threshold else 0.0

# Robotics: Reach target
reward = -distance_to_target
```

For complex tasks (writing, conversation, creativity), this fails:

**Example: Text Summarization**
```python
# Attempt 1: Length-based
reward = -abs(len(summary) - target_length)
# Problem: Encourages gibberish of correct length

# Attempt 2: ROUGE score
reward = rouge_score(summary, reference)
# Problem: Can have high ROUGE but miss key points

# Attempt 3: Hand-crafted heuristics
reward = 0.1 * brevity + 0.3 * coverage + 0.2 * fluency + ...
# Problem: Weights are arbitrary, doesn't capture human judgment
```

**Fundamental Issue:**
- Human preferences are too complex to specify manually
- What we want: "helpful, harmless, and honest" responses
- These are high-level concepts, not easily formalized

**Solution: Learn from Preferences**
Instead of specifying reward, collect human comparisons:
```
Prompt: "Explain quantum computing"
Response A: [Technical jargon, hard to understand]
Response B: [Clear explanation with analogy]
Human: Prefers B

Learn reward model: r(B) > r(A)
```

### 2. The Bradley-Terry Model

**Preference Model:**

Given two options y_w (winner) and y_l (loser), the probability that humans prefer y_w is:

```
P(y_w > y_l) = sigma(r(y_w) - r(y_l))
             = exp(r(y_w)) / (exp(r(y_w)) + exp(r(y_l)))
```

where:
- r(y): Learned reward function (neural network)
- sigma: Logistic function
- Assumption: Human preferences follow this probabilistic model

**Intuition:**
- If r(y_w) >> r(y_l): P(y_w > y_l) ≈ 1 (almost certainly prefer winner)
- If r(y_w) ≈ r(y_l): P(y_w > y_l) ≈ 0.5 (no strong preference)
- If r(y_w) << r(y_l): P(y_w > y_l) ≈ 0 (model prediction is wrong)

**Why Bradley-Terry?**

1. **Probabilistic:** Captures that preferences are not deterministic
2. **Scale-invariant:** Only differences r(y_w) - r(y_l) matter
3. **Interpretable:** Difference in logits = log odds ratio
4. **Mathematically convenient:** Logistic function has nice properties

**Derivation from Utility Theory:**

Assume each response has latent utility: u(y) = r(y) + noise
If noise ~ Gumbel distribution:
```
P(y_w > y_l) = P(u(y_w) > u(y_l))
             = sigma(r(y_w) - r(y_l))
```

### 3. Reward Model Training

**Objective:**

Given dataset of preferences D = {(x, y_w, y_l)}, maximize likelihood:

```
L_RM = E[(x, y_w, y_l) ~ D][-log sigma(r(y_w) - r(y_l))]
```

**Equivalent to:**
```
L_RM = -E[log(exp(r(y_w)) / (exp(r(y_w)) + exp(r(y_l))))]
     = E[log(1 + exp(r(y_l) - r(y_w)))]  [log-sum-exp form]
```

**Gradient:**
```
∇_theta L_RM = E[(sigma(r(y_l) - r(y_w)) * (∇r(y_l) - ∇r(y_w))]
```

**Intuition:**
- If model predicts r(y_l) > r(y_w) (wrong): Large gradient to correct
- If model predicts r(y_w) >> r(y_l) (confident, correct): Small gradient
- Sigmoid weights the gradient by confidence

**Architecture:**

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model  # e.g., GPT-2
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from base model
        outputs = self.base(input_ids, attention_mask=attention_mask)

        # Use last token (or mean pooling)
        last_hidden = outputs.last_hidden_state[:, -1, :]

        # Scalar reward
        reward = self.value_head(last_hidden)
        return reward
```

**Training Loop:**

```python
for batch in preference_data:
    x, y_w, y_l = batch

    # Compute rewards
    r_w = reward_model(x, y_w)
    r_l = reward_model(x, y_l)

    # Bradley-Terry loss
    loss = -torch.log(torch.sigmoid(r_w - r_l))

    # Update
    loss.backward()
    optimizer.step()
```

**Key Considerations:**

1. **Data Quality:** Preferences must be reliable
2. **Model Capacity:** Reward model should capture nuanced preferences
3. **Regularization:** Prevent overfitting to preference data
4. **Normalization:** Reward scale affects RL training

### 4. The Three-Stage RLHF Pipeline

**Stage 1: Supervised Fine-Tuning (SFT)**

```
Goal: Teach model to generate reasonable responses
Data: High-quality demonstrations (prompt, response) pairs
Method: Standard supervised learning

Loss: L_SFT = -sum_t log p(y_t | y_{<t}, x)

Result: pi_SFT - policy that imitates demonstrations
```

**Why SFT First:**
- Bootstraps policy to reasonable behavior
- RL from random initialization would be too slow
- Provides good starting point for exploration

**Stage 2: Reward Model Training**

```
Goal: Learn human preferences
Data: Comparisons (x, y_w, y_l) from humans or AI
Method: Bradley-Terry model

Loss: L_RM = -log sigma(r(y_w) - r(y_l))

Result: r_theta - reward model predicting human preferences
```

**Stage 3: RL Fine-Tuning (PPO)**

```
Goal: Optimize policy to maximize learned rewards
Method: PPO with KL penalty from reference policy

Objective:
J(theta) = E[r_theta(x, y)] - beta * KL(pi_theta || pi_ref)

where:
- pi_ref: Reference policy (SFT model, fixed)
- beta: KL penalty coefficient
- r_theta: Learned reward model
```

**Complete Algorithm:**

```python
# Stage 1: SFT
for batch in demonstration_data:
    loss = -log_prob(response | prompt)
    sft_model.update(loss)

pi_ref = copy(sft_model)  # Save as reference

# Stage 2: Reward Model
for batch in preference_data:
    r_w, r_l = reward_model(prompt, response_w), reward_model(prompt, response_l)
    loss = -log(sigmoid(r_w - r_l))
    reward_model.update(loss)

# Stage 3: RL with PPO
for episode in range(num_episodes):
    prompt = sample_prompt()
    response = policy.generate(prompt)

    # Compute rewards
    reward = reward_model(prompt, response)
    kl_penalty = KL(policy.log_prob(response) - pi_ref.log_prob(response))

    total_reward = reward - beta * kl_penalty

    # PPO update
    policy.update_ppo(total_reward)
```

### 5. The KL Penalty

**Why KL Penalty is Critical:**

Without KL penalty:
```
Policy optimizes: max E[r(y)]
Result: Mode collapse, reward hacking, gibberish
```

With KL penalty:
```
Policy optimizes: max E[r(y) - beta * KL(pi || pi_ref)]
Result: High reward while staying close to reference policy
```

**KL Divergence:**

```
KL(pi || pi_ref) = E_pi[log pi(y|x) - log pi_ref(y|x)]
                 = E_pi[log(pi(y|x) / pi_ref(y|x))]
```

**Per-Token KL:**

For autoregressive policies:
```
KL(pi || pi_ref) = sum_t E[log pi(y_t | y_{<t}, x) - log pi_ref(y_t | y_{<t}, x)]
```

**Practical Implementation:**

```python
def compute_kl_penalty(log_probs_policy, log_probs_ref):
    """
    log_probs_policy: Log probs from current policy
    log_probs_ref: Log probs from reference policy (fixed)
    """
    kl = log_probs_policy - log_probs_ref
    return kl.mean()

# During RL training
with torch.no_grad():
    log_probs_ref = reference_model(prompt, response)

log_probs_policy = policy_model(prompt, response)
kl_penalty = compute_kl_penalty(log_probs_policy, log_probs_ref)

reward_total = reward_model(prompt, response) - beta * kl_penalty
```

**Choosing Beta:**

- **Too small (beta → 0):** Policy drifts far from reference
  - Risk: Mode collapse, reward hacking
- **Too large (beta → ∞):** Policy stays too close to reference
  - Risk: Limited improvement, ignores reward signal
- **Typical values:** beta = 0.01 to 0.1
- **Adaptive:** Start high, decrease over training

**KL vs Other Constraints:**

| Method | Formula | Properties |
|--------|---------|------------|
| **KL penalty** | E[r] - beta*KL | Soft constraint, smooth optimization |
| **KL constraint** | max E[r] s.t. KL < epsilon | Hard constraint, requires constrained optimization |
| **PPO clip** | Clip(ratio, 1-eps, 1+eps) | Prevents large updates, simpler than KL |

RLHF typically uses KL penalty for stability.

### 6. PPO for RLHF

**Standard PPO Objective:**

```
L_PPO = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]

where:
- ratio = pi_new(a|s) / pi_old(a|s)
- A = advantage estimate
- eps = clipping parameter (0.2)
```

**RLHF-PPO Objective:**

```
L_RLHF = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)]
         - beta * KL(pi || pi_ref)

where:
- Reward: r(x, y) from reward model
- KL penalty from reference policy (not old policy)
- Advantage: A = r(x,y) - V(x) - beta * KL(pi || pi_ref)
```

**Key Differences:**

1. **Reward Source:** Learned reward model, not environment
2. **KL Reference:** Fixed reference policy, not previous policy
3. **Episode Structure:** Generate full text, get single reward at end

**Implementation:**

```python
def ppo_rlhf_update(policy, value_net, reward_model, reference_policy,
                    prompts, responses, old_log_probs, beta=0.1):
    # Compute current log probs
    log_probs = policy.log_prob(prompts, responses)

    # Compute KL penalty
    with torch.no_grad():
        ref_log_probs = reference_policy.log_prob(prompts, responses)
    kl_penalty = log_probs - ref_log_probs

    # Compute rewards from reward model
    with torch.no_grad():
        rewards = reward_model(prompts, responses)

    # Total reward with KL penalty
    total_rewards = rewards - beta * kl_penalty

    # Compute advantages
    values = value_net(prompts)
    advantages = total_rewards - values

    # PPO ratio and clip
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages

    # PPO loss
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, total_rewards)

    # Total loss
    loss = policy_loss + 0.5 * value_loss

    return loss
```

### 7. Reward Hacking and Overoptimization

**Reward Hacking:**

Policy exploits reward model imperfections to get high predicted reward without being actually good.

**Example - Length Exploitation:**
```
Reward model: Prefers longer, detailed responses
Policy learns: Generate very long, repetitive text
Result: High r(y) but low actual quality
```

**Example - Keyword Stuffing:**
```
Reward model: Trained on professional responses with certain words
Policy learns: Stuff in keywords ("certainly", "absolutely", etc.)
Result: High reward but unnatural language
```

**Detection:**

```python
# Compare proxy reward vs gold reward
proxy_reward = reward_model(x, y)
gold_reward = human_evaluation(x, y)  # Expensive

if proxy_reward >> gold_reward:
    print("Reward hacking detected!")
```

**Overoptimization:**

As RL training progresses, proxy reward increases but true reward plateaus or decreases.

```
RL Steps: 0      1000    2000    3000    4000
Proxy:    0.5    0.7     0.8     0.9     0.95
Gold:     0.5    0.7     0.75    0.73    0.68 [decreases!]
```

**Goodhart's Law:**
"When a measure becomes a target, it ceases to be a good measure."

**Causes:**

1. **Reward Model Error:**
   - Model is imperfect approximation
   - Trained on finite data
   - Extrapolates poorly to optimized outputs

2. **Distribution Shift:**
   - RL policy generates different text than training data
   - Reward model unreliable out-of-distribution

3. **Mode Collapse:**
   - Policy finds single high-reward response
   - Generates same text for all prompts
   - Low diversity

**Solutions:**

1. **KL Penalty:**
   - Prevents policy from drifting too far
   - Most common solution

2. **Iterative Training:**
   - Collect new preferences on policy outputs
   - Retrain reward model
   - Prevents distribution shift

3. **Ensemble Reward Models:**
   - Train multiple reward models
   - Use minimum or disagreement-weighted reward
   - Penalize uncertainty

4. **Early Stopping:**
   - Monitor gold reward
   - Stop before overoptimization

5. **Reward Model Regularization:**
   - Encourage smooth, conservative rewards
   - Penalize high rewards on unusual inputs

## Implementation: Complete RLHF Pipeline

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import numpy as np

# ==================== Stage 1: SFT ====================

class SFTModel(nn.Module):
    """Supervised Fine-Tuning Model"""
    def __init__(self, model_name='gpt2'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

    def generate(self, prompt, max_length=50):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

def train_sft(model, train_data, num_epochs=3, lr=1e-5):
    """
    Stage 1: Supervised Fine-Tuning

    train_data: List of (prompt, response) pairs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for prompt, response in train_data:
            # Tokenize
            text = prompt + response
            tokens = model.tokenizer.encode(text, return_tensors='pt')

            # Forward pass
            loss, _ = model(input_ids=tokens, labels=tokens)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        print(f"SFT Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model

# ==================== Stage 2: Reward Model ====================

class RewardModel(nn.Module):
    """Reward Model for predicting human preferences"""
    def __init__(self, base_model_name='gpt2'):
        super().__init__()
        config = GPT2Config.from_pretrained(base_model_name)
        self.transformer = GPT2LMHeadModel.from_pretrained(base_model_name).transformer
        self.value_head = nn.Linear(config.n_embd, 1)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None):
        # Get transformer outputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # Use last non-padding token
        if attention_mask is not None:
            # Get last valid token for each sequence
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
        else:
            last_hidden = hidden_states[:, -1, :]

        # Compute scalar reward
        reward = self.value_head(last_hidden)
        return reward.squeeze(-1)

def train_reward_model(reward_model, preference_data, num_epochs=3, lr=1e-5):
    """
    Stage 2: Train reward model on preference comparisons

    preference_data: List of (prompt, response_winner, response_loser) tuples
    """
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for prompt, y_w, y_l in preference_data:
            # Tokenize both responses
            text_w = prompt + y_w
            text_l = prompt + y_l

            tokens_w = reward_model.tokenizer.encode(text_w, return_tensors='pt',
                                                      truncation=True, max_length=512)
            tokens_l = reward_model.tokenizer.encode(text_l, return_tensors='pt',
                                                      truncation=True, max_length=512)

            # Compute rewards
            r_w = reward_model(tokens_w)
            r_l = reward_model(tokens_l)

            # Bradley-Terry loss
            loss = -F.logsigmoid(r_w - r_l).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            correct += (r_w > r_l).sum().item()
            total += r_w.size(0)

        avg_loss = total_loss / len(preference_data)
        accuracy = correct / total
        print(f"RM Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.3f}")

    return reward_model

# ==================== Stage 3: PPO with RLHF ====================

class ValueNetwork(nn.Module):
    """Value network for PPO advantage estimation"""
    def __init__(self, base_model_name='gpt2'):
        super().__init__()
        config = GPT2Config.from_pretrained(base_model_name)
        self.transformer = GPT2LMHeadModel.from_pretrained(base_model_name).transformer
        self.value_head = nn.Linear(config.n_embd, 1)
        self.tokenizer = GPT2Tokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            last_token_indices = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
        else:
            last_hidden = hidden_states[:, -1, :]

        value = self.value_head(last_hidden)
        return value.squeeze(-1)

def compute_advantages(rewards, values, gamma=0.99, lam=0.95):
    """Compute GAE advantages"""
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    return torch.tensor(advantages)

def ppo_step(policy_model, value_model, reward_model, reference_model,
             prompts, responses, old_log_probs, beta=0.1, clip_eps=0.2):
    """
    Single PPO update step for RLHF
    """
    # Tokenize
    texts = [p + r for p, r in zip(prompts, responses)]
    tokens = policy_model.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    # Compute current log probs
    outputs = policy_model.model(**tokens)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Get log prob of actual tokens (simplified)
    current_log_probs = log_probs.mean(dim=(1, 2))

    # Compute KL penalty from reference
    with torch.no_grad():
        ref_outputs = reference_model.model(**tokens)
        ref_logits = ref_outputs.logits
        ref_log_probs = F.log_softmax(ref_logits, dim=-1).mean(dim=(1, 2))

    kl_penalty = (current_log_probs - ref_log_probs)

    # Compute rewards from reward model
    with torch.no_grad():
        rm_rewards = reward_model(tokens['input_ids'], tokens['attention_mask'])

    # Total reward
    total_rewards = rm_rewards - beta * kl_penalty

    # Compute values and advantages
    values = value_model(tokens['input_ids'], tokens['attention_mask'])
    advantages = total_rewards - values

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO ratio
    ratio = torch.exp(current_log_probs - old_log_probs)

    # Clipped surrogate objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss
    value_loss = F.mse_loss(values, total_rewards)

    # Total loss
    loss = policy_loss + 0.5 * value_loss

    return loss, {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'mean_reward': rm_rewards.mean().item(),
        'mean_kl': kl_penalty.mean().item()
    }

def train_rlhf(policy_model, value_model, reward_model, reference_model,
               prompts, num_episodes=100, beta=0.1):
    """
    Stage 3: RLHF training with PPO
    """
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-6)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=1e-5)

    for episode in range(num_episodes):
        # Generate responses for each prompt
        responses = []
        old_log_probs_list = []

        for prompt in prompts:
            # Generate response
            response = policy_model.generate(prompt)
            responses.append(response[len(prompt):])  # Remove prompt

            # Compute log prob (simplified)
            text = prompt + responses[-1]
            tokens = policy_model.tokenizer.encode(text, return_tensors='pt')
            with torch.no_grad():
                outputs = policy_model.model(tokens)
                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                old_log_probs_list.append(log_probs.mean())

        old_log_probs = torch.stack(old_log_probs_list)

        # PPO update
        loss, metrics = ppo_step(
            policy_model, value_model, reward_model, reference_model,
            prompts, responses, old_log_probs, beta
        )

        # Backward pass
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()
        loss.backward()
        policy_optimizer.step()
        value_optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
            print(f"  Mean KL: {metrics['mean_kl']:.4f}")
            print(f"  Sample: {prompts[0]} -> {responses[0][:50]}...")

    return policy_model

# ==================== Full Pipeline ====================

def full_rlhf_pipeline():
    """
    Complete RLHF pipeline demonstration
    """
    print("=" * 50)
    print("RLHF Pipeline Demo")
    print("=" * 50)

    # Synthetic data (in practice, use real datasets)
    sft_data = [
        ("What is AI? ", "AI is the simulation of human intelligence in machines."),
        ("Explain Python: ", "Python is a high-level programming language."),
        ("Tell me about ML: ", "Machine learning is a subset of AI."),
    ]

    preference_data = [
        ("What is AI? ",
         "AI is the simulation of human intelligence in machines.",
         "AI is stuff."),  # Winner vs loser
        ("Explain Python: ",
         "Python is a high-level programming language known for readability.",
         "Python is a language."),
    ]

    prompts = ["What is AI? ", "Explain Python: "]

    # Stage 1: SFT
    print("\n[Stage 1] Supervised Fine-Tuning...")
    sft_model = SFTModel('gpt2')
    sft_model = train_sft(sft_model, sft_data, num_epochs=2)

    # Save reference model
    reference_model = SFTModel('gpt2')
    reference_model.model.load_state_dict(sft_model.model.state_dict())
    reference_model.eval()

    # Stage 2: Reward Model
    print("\n[Stage 2] Training Reward Model...")
    reward_model = RewardModel('gpt2')
    reward_model = train_reward_model(reward_model, preference_data, num_epochs=2)

    # Stage 3: RLHF
    print("\n[Stage 3] RLHF with PPO...")
    value_model = ValueNetwork('gpt2')
    policy_model = train_rlhf(
        sft_model, value_model, reward_model, reference_model,
        prompts, num_episodes=20, beta=0.1
    )

    print("\n" + "=" * 50)
    print("RLHF Training Complete!")
    print("=" * 50)

    # Test final model
    test_prompt = "What is AI? "
    response = policy_model.generate(test_prompt)
    print(f"\nTest Prompt: {test_prompt}")
    print(f"Response: {response}")

    return policy_model

if __name__ == "__main__":
    # Run full pipeline
    model = full_rlhf_pipeline()
```

## Key Equations Summary

**Bradley-Terry Model:**
```
P(y_w > y_l) = exp(r(y_w)) / (exp(r(y_w)) + exp(r(y_l)))
             = sigma(r(y_w) - r(y_l))
```

**Reward Model Loss:**
```
L_RM = -E[(x, y_w, y_l)][ log sigma(r(y_w) - r(y_l)) ]
```

**RLHF Objective:**
```
max E_pi[r(x, y) - beta * KL(pi(y|x) || pi_ref(y|x))]
```

**KL Penalty:**
```
KL(pi || pi_ref) = E_pi[log pi(y|x) - log pi_ref(y|x)]
```

**PPO-RLHF Loss:**
```
L = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] - beta * KL
```

## Required Readings

1. **InstructGPT Paper (Ouyang et al., 2022):** "Training language models to follow instructions with human feedback"
2. **Deep RL from Human Preferences (Christiano et al., 2017)**
3. **Learning to Summarize (Stiennon et al., 2020)**
4. **CS234 RLHF Module**

## Exercises

1. Implement Bradley-Terry model and train on synthetic preferences
2. Build reward model from GPT-2 and train on comparison data
3. Add KL penalty to policy optimization and observe effects
4. Experiment with different beta values and plot reward vs KL
5. Implement reward model ensemble for uncertainty estimation

## Discussion Questions

1. Why can't we use hand-crafted reward functions for language models?
2. What are the failure modes of RLHF and how can we detect them?
3. How does the KL penalty prevent reward hacking?
4. Compare RLHF to imitation learning - when is each preferable?
5. What are the ethical considerations of learning from human feedback?

## Next Week Preview

Week 18 covers advanced alignment methods beyond RLHF: DPO (Direct Preference Optimization), Constitutional AI, KTO, and RLAIF. We'll see how to eliminate the reward model and improve upon RLHF's limitations.
