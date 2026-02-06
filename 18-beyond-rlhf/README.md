# Week 18: Beyond RLHF - Direct Preference Optimization and Modern Alignment

This is the culmination of the entire reinforcement learning course. We've journeyed from basic RL concepts through policy gradient methods, actor-critic algorithms, and RLHF. Now we explore the cutting edge of language model alignment: methods that improve upon or replace traditional RLHF.

## Learning Objectives

- [ ] Understand DPO derivation from RLHF objective
- [ ] Understand IPO, KTO, ORPO as DPO variants
- [ ] Understand Constitutional AI and RLAIF
- [ ] Compare online vs offline preference optimization
- [ ] Understand RLVR (RL with Verifiable Rewards)
- [ ] Implement DPO fine-tuning on a small model

## Prerequisites

Before starting this week, you should understand:
- RLHF pipeline (Week 17)
- PPO algorithm
- KL divergence and its role in constrained optimization
- Preference learning basics
- Language model fine-tuning

## Overview: The Evolution Beyond RLHF

Traditional RLHF (as practiced with PPO) has several challenges:
1. **Complexity**: Requires training a reward model, then a separate RL phase with actor-critic
2. **Instability**: PPO training can be unstable, hyperparameter-sensitive
3. **Computational Cost**: Multiple models, online sampling, advantage estimation
4. **Reward Hacking**: Models can exploit reward model weaknesses (Goodhart's Law)

The methods in this week address these challenges through various innovations:
- **DPO**: Eliminates reward model by optimizing preferences directly
- **Constitutional AI**: Replaces human feedback with AI feedback guided by principles
- **RLVR**: Uses verifiable rewards (e.g., math correctness) instead of learned rewards
- **Online methods**: Generate fresh data during training to avoid distribution shift

## Key Concepts

### 1. Direct Preference Optimization (DPO)

**The Core Insight**

DPO's breakthrough comes from analyzing the optimal policy under RLHF. Recall the RLHF objective:

```
max_π E_{x~D, y~π(·|x)} [r(x,y)] - β·KL(π || π_ref)
```

Where:
- `r(x,y)` is the reward model
- `β` is the KL penalty coefficient
- `π_ref` is the reference (SFT) model

**Mathematical Foundation**

The optimal policy π* for this constrained optimization has a closed form:

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)
```

Where Z(x) is a partition function (normalization constant).

Rearranging this equation to solve for the reward:

```
r(x,y) = β · log(π*(y|x)/π_ref(y|x)) + β·log Z(x)
```

**The DPO Transformation**

This means the reward model can be expressed in terms of the policy itself! We can substitute this back into the preference learning objective.

For preference data where y_w (chosen) is preferred over y_l (rejected):

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

Substituting our reward formula:

```
P(y_w ≻ y_l | x) = σ(β · log(π(y_w|x)/π_ref(y_w|x)) - β · log(π(y_l|x)/π_ref(y_l|x)))
```

The partition functions Z(x) cancel out!

**DPO Loss Function**

The final DPO loss directly optimizes this probability:

```python
L_DPO(π; π_ref) = -E_{(x,y_w,y_l)~D} [
    log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
]
```

**Key Properties**

1. **No reward model needed**: The policy itself implicitly represents rewards
2. **Stable training**: Simple classification-like loss, no actor-critic instability
3. **Same optimal solution**: DPO converges to the same solution as RLHF
4. **Offline algorithm**: Can train on fixed preference datasets
5. **Implicit reward**: Can extract rewards as `r(x,y) = β · log(π(y|x)/π_ref(y|x))`

**Practical Implementation**

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1):
    """
    Compute DPO loss.

    Args:
        policy_chosen_logps: log π(y_w|x) for chosen responses
        policy_rejected_logps: log π(y_l|x) for rejected responses
        reference_chosen_logps: log π_ref(y_w|x) for chosen responses
        reference_rejected_logps: log π_ref(y_l|x) for rejected responses
        beta: KL penalty coefficient

    Returns:
        DPO loss (scalar)
    """
    # Compute log ratios
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    reference_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO loss
    logits = beta * (policy_logratios - reference_logratios)
    loss = -F.logsigmoid(logits).mean()

    # Implicit reward for logging
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return loss, chosen_rewards, rejected_rewards
```

**Advantages over RLHF**

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Training phases | 3 (SFT, RM, RL) | 2 (SFT, DPO) |
| Models needed | Policy + Value + Reference | Policy + Reference |
| Stability | Can be unstable | Very stable |
| Complexity | High (PPO mechanics) | Low (classification-like) |
| Online sampling | Yes (required) | No (offline) |
| Memory usage | High (4 models in memory) | Lower (2 models) |
| Convergence | Same optimum | Same optimum |

**Limitations**

1. **Offline nature**: Fixed dataset can lead to distribution shift
2. **Reward extrapolation**: Can't adapt to new preference information
3. **Length bias**: May prefer longer responses (more tokens to increase probability ratio)
4. **Binary preferences**: Requires pairwise comparisons, not absolute ratings

### 2. DPO Variants and Improvements

#### IPO (Identity Preference Optimization)

**Motivation**: DPO can overfit to preference data, especially with limited datasets.

**Key Change**: Replace the logistic loss with a squared loss:

```python
L_IPO = E[(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)) - 1/2)²]
```

**Benefits**:
- Less prone to overfitting
- More robust to noise in preferences
- Better calibration of implicit rewards
- Smoother optimization landscape

**Trade-off**: May converge more slowly than DPO.

#### KTO (Kahneman-Tversky Optimization)

**Motivation**: Collecting pairwise preferences is expensive. Often we only have binary feedback: thumbs up/down, like/dislike.

**Key Insight**: Based on prospect theory from behavioral economics (Kahneman & Tversky). Humans evaluate gains and losses asymmetrically.

**Loss Function**:

For desirable examples (y ∈ D):
```
L_D = E[1 - σ(β · (log π(y|x)/π_ref(y|x) - z_ref))]
```

For undesirable examples (y ∈ U):
```
L_U = E[σ(β · (log π(y|x)/π_ref(y|x) - z_ref))]
```

Where `z_ref` is a running average of log-ratios (reference point).

**Benefits**:
- No pairwise comparisons needed
- Can use simple binary labels
- Better data efficiency
- Reflects human psychology (loss aversion)

**Use Case**: When you have user feedback (likes/dislikes) but not explicit comparisons.

#### ORPO (Odds Ratio Preference Optimization)

**Motivation**: Combine SFT and preference alignment into a single stage.

**Key Innovation**: Use odds ratio instead of probability ratio:

```python
L_ORPO = L_SFT + λ · L_OR

where L_OR = -E[log σ(log(odds_chosen(y_w|x) / odds_rejected(y_l|x)))]

odds(y|x) = P(y|x) / (1 - P(y|x))
```

**Benefits**:
- Single training stage (no separate SFT needed)
- Faster training pipeline
- Implicit penalty on rejected responses
- Good empirical performance

**Trade-off**: Less separation between instruction-following and preference alignment.

#### SimPO (Simple Preference Optimization)

**Motivation**: Remove the need for a reference model entirely.

**Key Change**: Use average log probability as implicit reward:

```python
r(x,y) = 1/|y| · Σ log π(y_i|x, y_{<i})

L_SimPO = -E[log σ((r(x,y_w) - r(x,y_l)) / β)]
```

**Benefits**:
- No reference model needed (saves memory)
- Naturally accounts for length
- Very simple implementation
- Competitive performance

**Limitation**: Can deviate from SFT initialization more freely.

### 3. Constitutional AI and RLAIF

**Constitutional AI** (Bai et al., 2022) represents a paradigm shift: replacing human feedback with AI feedback guided by principles.

#### The Constitutional AI Pipeline

**Phase 1: Supervised Learning with Critique and Revision**

1. Generate harmful/problematic responses via red teaming prompts
2. AI critiques its own response according to constitutional principles
3. AI revises the response based on the critique
4. Fine-tune on the revised responses

**Phase 2: RL from AI Feedback (RLAIF)**

1. Generate response pairs for prompts
2. AI evaluates which response better follows the constitution
3. Train reward model on AI preferences
4. Standard RLHF with this reward model

**The Constitution**

A set of principles guiding behavior. Example principles:

```
1. Choose the response that is most helpful, honest, and harmless.
2. Choose the response that is least likely to encourage illegal activity.
3. Choose the response that is most respectful and least discriminatory.
4. Choose the response that provides the most accurate information.
5. Choose the response that acknowledges uncertainty when appropriate.
...
16. Choose the response that promotes intellectual humility.
```

**Self-Improvement Loop**

```
Model → Red Teaming → Harmful Response → AI Critique → Revision → Training Data
  ↑                                                                      ↓
  └─────────────── Fine-tune on Revisions ──────────────────────────────┘
```

#### RLAIF (RL from AI Feedback)

**Core Idea**: Replace human annotators with AI evaluators.

**Process**:
1. Generate response pairs for prompts
2. Use a strong LLM (e.g., GPT-4) to judge which is better
3. Provide the constitution/rubric to the judge
4. Train reward model on AI preferences
5. Use RLHF or DPO as normal

**Advantages**:
- Scalable: No human annotation bottleneck
- Consistent: AI applies criteria uniformly
- Iterative: Can self-improve
- Principled: Explicit constitution guides behavior

**Challenges**:
- Quality depends on judge model capability
- May inherit judge model biases
- Risk of reward hacking (gaming the AI judge)
- Still requires human validation of constitution

**Empirical Results** (from Anthropic):
- RLAIF models comparable to RLHF in helpfulness
- Better at avoiding harmful content (due to explicit principles)
- More consistent in edge cases
- Scales to areas where human feedback is expensive

#### Constitutional AI vs Traditional RLHF

| Aspect | Traditional RLHF | Constitutional AI |
|--------|-----------------|-------------------|
| Feedback source | Human annotators | AI guided by principles |
| Scalability | Limited by humans | Nearly unlimited |
| Consistency | Varies by annotator | Very consistent |
| Transparency | Implicit in labels | Explicit constitution |
| Iteration speed | Slow (weeks) | Fast (hours/days) |
| Bias mitigation | Difficult to control | Can encode in principles |
| Human oversight | Continuous | One-time (constitution) |

### 4. Online vs Offline Preference Optimization

A critical distinction in modern alignment methods.

#### Offline Preference Optimization

**Definition**: Train on a fixed, pre-collected dataset of preferences.

**Examples**: Standard DPO, IPO, KTO on static datasets.

**Advantages**:
- Simple implementation
- Reproducible
- No expensive sampling during training
- Can use existing preference datasets

**Challenges**:
- **Distribution shift**: Model distribution diverges from data distribution
- **No exploration**: Can't discover new preferences
- **Stale data**: Preferences may not reflect current model capabilities
- **Reward hacking**: Can exploit weaknesses in fixed preference data

**When to use**:
- Limited compute
- High-quality existing preference data
- First iteration of alignment

#### Online Preference Optimization

**Definition**: Generate new responses during training and collect fresh preferences.

**Examples**: Online DPO, PPO-based RLHF, iterative DPO.

**Process**:
```python
for iteration in range(num_iterations):
    # 1. Generate new responses with current policy
    responses = policy.generate(prompts)

    # 2. Get preferences (human or AI)
    preferences = collect_preferences(prompts, responses)

    # 3. Update policy on fresh preferences
    policy.update(preferences)
```

**Advantages**:
- **No distribution shift**: Always train on current model distribution
- **Exploration**: Can discover new high-reward behaviors
- **Adaptation**: Responds to model's current weaknesses
- **Better final performance**: Empirically outperforms offline

**Challenges**:
- Expensive (continuous sampling)
- Requires preference annotation pipeline
- More complex implementation
- Longer training time

**Empirical Results** (Xiong et al., 2024):
- Online DPO outperforms offline DPO by 5-10% on benchmarks
- Gap increases with more iterations
- Benefits saturate after 3-5 iterations

#### Hybrid Approaches: Iterative DPO

**Compromise**: Periodically refresh the dataset, but not every step.

```python
dataset = initial_preference_data

for iteration in range(num_iterations):
    # Train on current dataset
    train_dpo(policy, dataset, num_steps=1000)

    # Generate new data with updated policy
    new_prompts = sample_prompts()
    new_responses = policy.generate(new_prompts)
    new_preferences = collect_preferences(new_prompts, new_responses)

    # Refresh dataset
    dataset = mix_datasets(dataset, new_preferences, ratio=0.5)
```

**Benefits**:
- Balance between cost and performance
- Mitigates distribution shift
- More practical than fully online
- Still benefits from exploration

**Best Practices**:
- 3-5 iterations usually sufficient
- Mix old and new data (e.g., 50/50)
- Monitor implicit reward drift
- Use AI feedback for scalability

### 5. RLVR (RL with Verifiable Rewards)

**Motivation**: For domains like math, coding, and logical reasoning, we can verify correctness automatically. We don't need human preferences or learned reward models.

**Key Insight**: When rewards are verifiable, RLHF's complexity is unnecessary.

#### Application Domains

**Mathematics**:
- Check if answer matches ground truth
- Verify symbolic solutions
- Test numerical accuracy

**Code Generation**:
- Run unit tests
- Check compilation
- Measure test coverage
- Verify correctness on test cases

**Logical Reasoning**:
- Verify proof validity
- Check constraint satisfaction
- Validate logical consistency

#### GRPO (Group Relative Policy Optimization)

Used in DeepSeek-R1, QwQ, and other reasoning models.

**Core Idea**: Generate multiple solutions, compare them, learn from relative quality.

**Algorithm**:

```python
for prompt in dataset:
    # 1. Generate multiple solutions
    solutions = [policy.generate(prompt) for _ in range(group_size)]

    # 2. Verify each solution
    rewards = [verify(prompt, solution) for solution in solutions]

    # 3. Compute advantages relative to group mean
    mean_reward = np.mean(rewards)
    advantages = [r - mean_reward for r in rewards]

    # 4. Update policy to increase probability of better solutions
    for solution, advantage in zip(solutions, advantages):
        if advantage > 0:
            policy.increase_probability(prompt, solution, weight=advantage)
        else:
            policy.decrease_probability(prompt, solution, weight=-advantage)
```

**Key Properties**:
- **No critic network**: Uses group statistics instead of learned baseline
- **Relative rewards**: Compares within group, not absolute
- **Simple implementation**: Just generate and compare
- **Effective**: Powers state-of-the-art reasoning models

**Advantages over RLHF**:
- No reward model training needed
- Perfect reward signal (verifiable)
- No reward hacking (can't game correctness)
- Cheaper (no human annotation)
- More reliable (objective verification)

#### DeepSeek-R1 Case Study

**Training Pipeline**:

1. **Cold Start**: Begin with supervised fine-tuning on reasoning traces
2. **RL Phase**: GRPO with verifiable rewards
   - Generate N solutions per problem
   - Verify correctness
   - Update policy to favor correct reasoning patterns
3. **Distillation**: Distill reasoning to smaller models

**Reward Function**:
```python
def verify_math_solution(problem, solution):
    # Extract final answer
    predicted_answer = extract_answer(solution)
    ground_truth = problem.answer

    # Binary reward
    if predicted_answer == ground_truth:
        return 1.0
    else:
        return 0.0
```

**Process Reward** (optional enhancement):
- Verify intermediate steps
- Reward correct reasoning, not just final answer
- Helps learn better reasoning patterns

**Results**:
- DeepSeek-R1 achieves GPT-4 level math performance
- Strong generalization to new problem types
- Emergent reasoning capabilities (chain-of-thought)
- Cost-effective training (no human feedback)

#### Limitations of RLVR

1. **Domain-specific**: Only works when verification is possible
2. **Binary rewards**: Often just correct/incorrect (sparse)
3. **Test set overfitting**: Can overfit to test cases if not careful
4. **Edge cases**: Some problems have multiple valid solutions
5. **Partial credit**: Hard to reward partially correct solutions

**Mitigation Strategies**:
- Use process rewards for intermediate steps
- Hold out test cases
- Verify reasoning process, not just answer
- Use multiple verification methods

### 6. Reward Model Overoptimization (Goodhart's Law)

**Goodhart's Law**: "When a measure becomes a target, it ceases to be a good measure."

**In RLHF Context**: The reward model is a proxy for human preferences. Optimizing it too aggressively degrades true quality.

#### The Overoptimization Phenomenon

**Observation** (Gao et al., 2023):

```
     True Quality
         ^
         |     ___
         |    /   \___
         |   /        \___
         |  /             \___
         | /                  \___
         |/________________________\___> RL Training Steps

         Reward Model Score
         ^
         |                    ___---
         |              ___---
         |        ___---
         |  ___---
         |--
         |_________________________> RL Training Steps
```

The reward model score keeps increasing, but true quality (measured by human eval) peaks then declines.

**Why Does This Happen?**

1. **Reward Model Errors**: The RM makes mistakes, especially out-of-distribution
2. **Exploitation**: The policy finds adversarial examples that fool the RM
3. **Mode Collapse**: The policy focuses on narrow high-reward behaviors
4. **Specification Gaming**: The policy satisfies the letter but not spirit of preferences

**Example**: A chatbot reward model might prefer longer, more verbose responses. Overoptimization leads to rambling, repetitive text that scores high but is actually worse.

#### Empirical Study Results

**Gao et al. (2023)** findings:

- Overoptimization begins after 10-50 KL divergence from reference model
- Effect is consistent across model sizes
- Stronger RMs delay but don't eliminate overoptimization
- Ensemble RMs reduce but don't solve the problem

**Gold Reward** (human eval) vs **Proxy Reward** (RM):

```python
# Typical training trajectory
KL = [0, 5, 10, 20, 50, 100]
proxy_reward = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]     # Monotonic increase
gold_reward  = [0.0, 0.2, 0.5, 0.6, 0.4, 0.1]     # Peaks then declines
```

#### Mitigation Strategies

**1. KL Penalty**

Constrain how far the policy can drift from the reference:

```python
objective = E[r(x,y)] - β·KL(π||π_ref)
```

Increase β to be more conservative. This is already standard in RLHF/DPO.

**2. Ensemble Reward Models**

Train multiple RMs, use average or conservative estimate:

```python
rewards = [rm1(x,y), rm2(x,y), rm3(x,y), rm4(x,y)]
conservative_reward = min(rewards)  # or percentile
```

**3. Online Training**

Collect fresh preferences on current model outputs:
- Catches RM exploitation early
- Grounds training in actual quality
- More expensive but more reliable

**4. Early Stopping**

Monitor gold reward (human eval) on held-out set:
- Stop when gold reward plateaus or declines
- Even if proxy reward is still improving
- Requires periodic human evaluation

**5. Uncertainty Quantification**

Use RM uncertainty to be conservative:

```python
reward_mean, reward_std = ensemble_rm(x, y)
conservative_reward = reward_mean - α * reward_std
```

**6. Regular Human Evaluation**

- Continuously collect human feedback on current outputs
- Detect overoptimization early
- Adjust training before quality degrades

#### Does DPO Solve Overoptimization?

**Partial answer**: DPO has no explicit reward model, so can't overoptimize one directly.

**But**: DPO has an *implicit* reward:

```python
r_implicit(x,y) = β · log(π(y|x) / π_ref(y|x))
```

**DPO can still overfit** to the preference data:
- Exploit noise in preference labels
- Overfit to specific examples
- Drift far from reference model

**Evidence**: IPO was developed specifically because DPO can overfit to noisy preferences.

**Mitigation**: Same strategies apply (KL constraints, online training, early stopping).

#### Does RLVR Solve Overoptimization?

**Much better**: When rewards are truly verifiable, there's no proxy to hack.

**Example**: A math solution is either correct or incorrect. No ambiguity.

**But**:
- Can still overfit to specific test cases
- May learn superficial patterns rather than reasoning
- Limited to domains with verifiable rewards

**Best practice**: Combine verifiable rewards (where available) with human/AI feedback (for style, safety, helpfulness).

## Implementation Guide: DPO Fine-Tuning

Let's implement a complete DPO training pipeline using the TRL (Transformer Reinforcement Learning) library.

### Setup

```bash
pip install transformers datasets trl peft accelerate bitsandbytes wandb
```

### Step 1: Prepare Preference Dataset

DPO requires triplets: (prompt, chosen_response, rejected_response).

```python
from datasets import load_dataset

# Load a preference dataset (e.g., Anthropic HH)
dataset = load_dataset("Anthropic/hh-rlhf")

# Format for DPO
def format_for_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

train_dataset = dataset["train"].map(format_for_dpo)
eval_dataset = dataset["test"].map(format_for_dpo)

print(f"Training examples: {len(train_dataset)}")
print(f"Eval examples: {len(eval_dataset)}")

# Example
print("\nExample preference pair:")
print(f"Prompt: {train_dataset[0]['prompt']}")
print(f"Chosen: {train_dataset[0]['chosen']}")
print(f"Rejected: {train_dataset[0]['rejected']}")
```

### Step 2: Load Model and Tokenizer

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "facebook/opt-350m"  # Small model for demonstration

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load reference model (frozen copy)
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
ref_model.eval()

print(f"Model loaded: {model_name}")
print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")
```

### Step 3: Configure DPO Trainer

```python
from trl import DPOTrainer, DPOConfig

# DPO hyperparameters
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    warmup_steps=100,

    # DPO specific
    beta=0.1,  # KL penalty coefficient
    loss_type="sigmoid",  # sigmoid (DPO) or hinge (IPO)

    # Logging
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    report_to="wandb",

    # Optimization
    bf16=True,
    gradient_checkpointing=True,
    max_length=512,
    max_prompt_length=256,
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

print("DPO Trainer initialized")
```

### Step 4: Train

```python
# Start training
trainer.train()

# Save final model
trainer.save_model("./dpo_final_model")
print("Training complete!")
```

### Step 5: Evaluation

```python
import numpy as np

# Generate with the trained model
def generate_response(prompt, model, tokenizer, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):]  # Remove prompt

# Compare base model vs DPO model
base_model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

test_prompts = [
    "How do I make a cake?",
    "What is the meaning of life?",
    "Explain quantum computing to a child.",
]

for prompt in test_prompts:
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*80}")

    base_response = generate_response(prompt, base_model, tokenizer)
    dpo_response = generate_response(prompt, model, tokenizer)

    print(f"\nBASE MODEL:\n{base_response}")
    print(f"\nDPO MODEL:\n{dpo_response}")
```

### Step 6: Extract Implicit Rewards

```python
def compute_implicit_reward(prompt, response, model, ref_model, tokenizer, beta=0.1):
    """Compute DPO's implicit reward."""
    text = prompt + response
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Get log probs from policy
        policy_outputs = model(**inputs)
        policy_logits = policy_outputs.logits

        # Get log probs from reference
        ref_outputs = ref_model(**inputs)
        ref_logits = ref_outputs.logits

    # Compute log probabilities
    policy_logprobs = torch.log_softmax(policy_logits, dim=-1)
    ref_logprobs = torch.log_softmax(ref_logits, dim=-1)

    # Get log prob of actual tokens
    labels = inputs["input_ids"][:, 1:]  # Shift for next token prediction
    policy_logprobs = policy_logprobs[:, :-1, :]
    ref_logprobs = ref_logprobs[:, :-1, :]

    # Gather log probs of actual tokens
    policy_lp = torch.gather(policy_logprobs, 2, labels.unsqueeze(-1)).squeeze(-1)
    ref_lp = torch.gather(ref_logprobs, 2, labels.unsqueeze(-1)).squeeze(-1)

    # Implicit reward
    reward = beta * (policy_lp.sum() - ref_lp.sum())

    return reward.item()

# Compute rewards for test responses
for prompt in test_prompts[:2]:
    dpo_response = generate_response(prompt, model, tokenizer)
    reward = compute_implicit_reward(prompt, dpo_response, model, ref_model, tokenizer)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {dpo_response[:100]}...")
    print(f"Implicit reward: {reward:.3f}")
```

### Advanced: Using LoRA for Efficiency

For larger models, use LoRA (Low-Rank Adaptation):

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Train as before - only LoRA parameters are updated
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # Full model as reference
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Full Pipeline Script

```python
#!/usr/bin/env python3
"""
Complete DPO training pipeline.
Usage: python train_dpo.py --model facebook/opt-350m --dataset Anthropic/hh-rlhf
"""

import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-350m")
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", default="./dpo_output")
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    train_data = dataset["train"]
    eval_data = dataset["test"]

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=500,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    print("Training complete!")

if __name__ == "__main__":
    main()
```

## Comparison Table: All Methods

| Method | Reward Model? | Pairwise Data? | Online/Offline | Phases | Complexity | Best For |
|--------|--------------|----------------|----------------|---------|-----------|----------|
| **RLHF (PPO)** | Yes | Yes | Online | 3 (SFT, RM, RL) | High | General alignment, when RM quality is high |
| **DPO** | No (implicit) | Yes | Offline | 2 (SFT, DPO) | Low | Fixed preference data, stable training |
| **IPO** | No | Yes | Offline | 2 | Low | Noisy preferences, small datasets |
| **KTO** | No | No | Offline | 2 | Low | Binary feedback only |
| **ORPO** | No | Yes | Offline | 1 (combined) | Low | Fast iteration, single-stage |
| **SimPO** | No | Yes | Offline | 2 | Very Low | Memory-constrained, no ref model |
| **Constitutional AI** | Yes (RLAIF) | Yes | Online | 4 | High | Principled alignment, scalable feedback |
| **RLVR/GRPO** | No (verifiable) | No | Online | 2 (SFT, RL) | Medium | Math, code, verifiable domains |
| **Online DPO** | No | Yes | Online | 2 (iterative) | Medium | Best performance, worth the cost |

## When to Use Each Method

### Use RLHF (PPO) when:
- You have high-quality reward model
- Need to adapt quickly to new rewards
- Have significant compute resources
- Online feedback collection is feasible

### Use DPO when:
- You have good preference dataset
- Want simple, stable training
- Limited compute budget
- Offline setting is acceptable

### Use IPO when:
- Preference data is noisy
- Dataset is small
- Robustness is critical

### Use KTO when:
- Only have binary feedback (thumbs up/down)
- Pairwise comparisons are expensive
- Have large user feedback dataset

### Use Constitutional AI when:
- Want principled, transparent alignment
- Need scalable feedback collection
- Can invest in designing constitution
- Have access to strong judge model

### Use RLVR/GRPO when:
- Working on math, code, or reasoning
- Rewards are verifiable
- Want objective training signal
- Need cost-effective scaling

### Use Online DPO when:
- Want best possible performance
- Can afford sampling and annotation
- Distribution shift is a concern
- Iterative improvement is valuable

## Practical Recommendations

### For Production Systems

1. **Start with DPO**: Simple, stable, effective baseline
2. **Iterate online**: Collect fresh data, retrain periodically
3. **Monitor gold metrics**: Track human eval, not just proxy rewards
4. **Use Constitutional AI**: Scale feedback collection
5. **Combine approaches**: RLVR for code/math, DPO for general chat

### For Research

1. **Understand DPO deeply**: It's the foundation of modern alignment
2. **Experiment with variants**: IPO, KTO, ORPO for different settings
3. **Study overoptimization**: Critical failure mode
4. **Explore online methods**: Frontier of performance
5. **Investigate RLVR**: Growing importance for reasoning

### Common Pitfalls

1. **Overoptimizing**: Stop early, monitor gold rewards
2. **Distribution shift**: Use online or iterative methods
3. **Length bias**: DPO can prefer longer responses, use length normalization
4. **Ignoring KL**: Always maintain reference constraint
5. **Poor SFT**: DPO assumes good SFT base, don't skip it

## The Future of Alignment

Emerging trends and open questions:

### Multimodal Alignment
- Extending DPO/RLHF to vision-language models
- Aligning text-to-image models
- Video generation alignment

### Personalized Alignment
- Learning user-specific preferences
- Federated preference learning
- Privacy-preserving alignment

### Continual Alignment
- Aligning models continuously during deployment
- Online learning from user interactions
- Detecting and correcting misalignment

### Scalable Oversight
- Aligning superhuman models
- Recursive reward modeling
- AI safety via debate

### Theoretical Understanding
- Convergence guarantees for DPO variants
- Sample complexity bounds
- Optimal KL penalties
- Addressing Goodhart's Law formally

## Summary

This week synthesizes the entire course. We've journeyed from basic RL (policy gradients, Q-learning) through advanced algorithms (PPO, SAC) to cutting-edge language model alignment.

**Key Takeaways**:

1. **DPO elegantly simplifies RLHF** by eliminating the reward model through mathematical insight
2. **Multiple variants** (IPO, KTO, ORPO) address specific limitations
3. **Constitutional AI** enables scalable, principled alignment
4. **Online methods** outperform offline but cost more
5. **RLVR** is ideal for verifiable domains (math, code)
6. **Goodhart's Law** is the central challenge: optimizing proxies degrades true objectives
7. **Practical alignment** combines multiple approaches strategically

**The Alignment Problem** remains open and critical. These methods are powerful tools, but:
- They don't solve deep alignment issues (value specification, robustness)
- Goodhart's Law is ever-present
- Scaling to superhuman capabilities is unsolved
- Safety guarantees remain elusive

Yet we've made tremendous progress. Modern language models are remarkably capable and increasingly helpful, harmless, and honest—thanks to these techniques.

Continue learning, stay curious, and contribute to making AI systems that benefit humanity.

## Further Exploration

- Implement Constitutional AI pipeline
- Experiment with different DPO variants
- Build RLVR system for coding tasks
- Study reward model overoptimization empirically
- Explore personalized preference learning
- Investigate multimodal alignment
- Read latest papers on scalable oversight

## Next Steps

This is the final week of structured content. From here:

1. **Build projects**: Apply what you've learned
2. **Read papers**: Stay current with arxiv.org
3. **Contribute**: Open source alignment tools
4. **Specialize**: Choose a subfield (safety, efficiency, theory)
5. **Teach**: Share your knowledge

Congratulations on completing this comprehensive RL course!
