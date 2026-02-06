# Week 18 Resources: Beyond RLHF

A comprehensive collection of papers, libraries, tutorials, and tools for modern AI alignment methods.

## Table of Contents

- [Foundational Papers](#foundational-papers)
- [DPO and Variants](#dpo-and-variants)
- [Constitutional AI and RLAIF](#constitutional-ai-and-rlaif)
- [RLVR and Verifiable Rewards](#rlvr-and-verifiable-rewards)
- [Reward Overoptimization](#reward-overoptimization)
- [Libraries and Tools](#libraries-and-tools)
- [Tutorials and Guides](#tutorials-and-guides)
- [Datasets](#datasets)
- [Blogs and Articles](#blogs-and-articles)
- [Courses and Lectures](#courses-and-lectures)
- [Research Groups](#research-groups)
- [Community](#community)

## Foundational Papers

### RLHF Background

**Training Language Models to Follow Instructions with Human Feedback (InstructGPT)**
- Authors: Ouyang et al. (OpenAI)
- Year: 2022
- Link: https://arxiv.org/abs/2203.02155
- Summary: The original InstructGPT paper that popularized RLHF. Describes the three-phase pipeline (SFT, RM, PPO) and empirical results showing alignment improvements.
- Key contributions: Established RLHF as standard practice, demonstrated effectiveness at scale

**Training a Helpful and Harmless Assistant with RLHF**
- Authors: Bai et al. (Anthropic)
- Year: 2022
- Link: https://arxiv.org/abs/2204.05862
- Summary: Anthropic's approach to RLHF, emphasizing both helpfulness and harmlessness. Discusses challenges and design choices.
- Key contributions: Multi-objective alignment, safety considerations

**Learning to Summarize from Human Feedback**
- Authors: Stiennon et al. (OpenAI)
- Year: 2020
- Link: https://arxiv.org/abs/2009.01325
- Summary: Early work on RLHF for summarization. Shows that RL from human feedback outperforms supervised learning on human evaluations.
- Key contributions: Demonstrated RLHF effectiveness on specific task

### Preference Learning Theory

**The Bradley-Terry Model**
- Original: Bradley & Terry (1952)
- Modern reference: "Modeling of pairwise comparisons"
- Summary: Statistical model for pairwise preferences used in RLHF and DPO
- Link: https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

**Preference Learning**
- Reference: Fürnkranz & Hüllermeier (2010)
- Book: "Preference Learning"
- Summary: Comprehensive treatment of learning from preferences
- Link: https://www.springer.com/gp/book/9783642141249

## DPO and Variants

### Direct Preference Optimization (DPO)

**Direct Preference Optimization: Your Language Model is Secretly a Reward Model**
- Authors: Rafailov et al. (Stanford)
- Year: 2023
- Link: https://arxiv.org/abs/2305.18290
- Code: https://github.com/eric-mitchell/direct-preference-optimization
- Summary: THE foundational paper for DPO. Derives the method from RLHF objective, shows equivalence to reward modeling, demonstrates effectiveness.
- Key contributions: Eliminates reward model phase, simplifies RLHF, same theoretical optimum
- Must read: Essential for understanding modern alignment

**DPO Blog Post (HuggingFace)**
- Link: https://huggingface.co/blog/dpo-trl
- Summary: Practical guide to using DPO with TRL library
- Includes: Code examples, hyperparameter tips, common pitfalls

### Identity Preference Optimization (IPO)

**A General Theoretical Paradigm to Understand Learning from Human Preferences**
- Authors: Azar et al. (Google DeepMind)
- Year: 2023
- Link: https://arxiv.org/abs/2310.12036
- Summary: Analyzes DPO theoretically, proposes IPO variant with squared loss for robustness to noise
- Key contributions: Theoretical framework, robustness improvements, overfitting mitigation

### Kahneman-Tversky Optimization (KTO)

**KTO: Model Alignment as Prospect Theoretic Optimization**
- Authors: Ethayarajh et al. (Stanford/Contextual AI)
- Year: 2024
- Link: https://arxiv.org/abs/2402.01306
- Code: https://github.com/ContextualAI/HALOs
- Summary: Uses prospect theory from behavioral economics to learn from binary feedback (not pairwise)
- Key contributions: Eliminates need for pairwise comparisons, based on psychological theory, effective with simple feedback
- Use case: When you only have thumbs up/down data

### Odds Ratio Preference Optimization (ORPO)

**ORPO: Monolithic Preference Optimization without Reference Model**
- Authors: Hong et al.
- Year: 2024
- Link: https://arxiv.org/abs/2403.07691
- Code: https://github.com/xfactlab/orpo
- Summary: Combines SFT and preference learning in single stage using odds ratio
- Key contributions: Single-phase training, no reference model needed, faster iteration

### Simple Preference Optimization (SimPO)

**SimPO: Simple Preference Optimization with a Reference-Free Reward**
- Authors: Meng et al.
- Year: 2024
- Link: https://arxiv.org/abs/2405.14734
- Summary: Removes reference model entirely, uses average log probability as implicit reward
- Key contributions: Simplest DPO variant, no reference model, length-normalized

### Comparative Studies

**Beyond DPO: Effective and Scalable LLM Alignment**
- Authors: Multiple (ongoing research)
- Year: 2024
- Summary: Compares DPO variants across benchmarks
- Key findings: Online methods best, IPO for noisy data, KTO for binary feedback

## Constitutional AI and RLAIF

### Constitutional AI

**Constitutional AI: Harmlessness from AI Feedback**
- Authors: Bai et al. (Anthropic)
- Year: 2022
- Link: https://arxiv.org/abs/2212.08073
- Summary: Replace human feedback with AI feedback guided by constitutional principles. Self-critique and revision pipeline.
- Key contributions: Scalable feedback collection, principled alignment, transparency
- Must read: Foundational for RLAIF approaches

**Anthropic's Constitution**
- Link: https://www.anthropic.com/index/claudes-constitution
- Summary: The actual principles used to train Claude
- Includes: Transparency about alignment process

### RLAIF (RL from AI Feedback)

**RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback**
- Authors: Lee et al. (Google)
- Year: 2023
- Link: https://arxiv.org/abs/2309.00267
- Summary: Systematic study of replacing human annotators with AI. Shows comparable performance.
- Key contributions: Scalability analysis, cost comparisons, quality validation

**RLHF vs RLAIF: An Empirical Study**
- Various authors
- Year: 2024
- Summary: Direct comparison of human vs AI feedback
- Key findings: AI feedback consistent, scalable, but needs strong judge model

### Self-Improvement

**Self-Rewarding Language Models**
- Authors: Yuan et al. (Meta)
- Year: 2024
- Link: https://arxiv.org/abs/2401.10020
- Summary: Models that generate their own training data and rewards
- Key contributions: Self-improvement loop, reduced human dependency

**Self-Taught Optimizer (STOP)**
- Authors: Zelikman et al. (Stanford)
- Year: 2024
- Link: https://arxiv.org/abs/2310.02304
- Summary: Models improve themselves through self-generated problems and solutions
- Key contributions: Bootstrapping capability, emergent improvement

## RLVR and Verifiable Rewards

### DeepSeek-R1

**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
- Authors: DeepSeek Team
- Year: 2025
- Link: https://arxiv.org/abs/2501.12948 (expected)
- Blog: https://api-docs.deepseek.com/news/news1226
- Summary: Uses GRPO (Group Relative Policy Optimization) with verifiable rewards for reasoning
- Key contributions: RL for reasoning, verifiable rewards, no reward model needed
- State-of-the-art: Achieves GPT-4 level performance on math/reasoning

### QwQ (Alibaba's Reasoning Model)

**QwQ: Exploring Reasoning in Large Language Models**
- Authors: Qwen Team (Alibaba)
- Year: 2024
- Link: https://qwenlm.github.io/blog/qwq-32b-preview/
- Summary: Similar approach to DeepSeek-R1, uses verifiable rewards for reasoning
- Key contributions: Open model, strong performance, transparent methodology

### GRPO (Group Relative Policy Optimization)

**Group Relative Policy Optimization**
- Context: Used in DeepSeek-R1
- Summary: Generate multiple solutions, compare within group, no critic network needed
- Key insight: Relative rewards within group, not absolute scores
- Advantages: Simple, effective, no reward model

### Mathematical Reasoning

**Let's Verify Step by Step**
- Authors: Lightman et al. (OpenAI)
- Year: 2023
- Link: https://arxiv.org/abs/2305.20050
- Summary: Process supervision (verify reasoning steps) outperforms outcome supervision
- Key contributions: Process rewards, intermediate verification, PRM800K dataset

**Solving Quantitative Reasoning Problems with Language Models**
- Authors: Lewkowycz et al. (Google)
- Year: 2022
- Link: https://arxiv.org/abs/2206.14858
- Summary: Minerva model for mathematical reasoning
- Key contributions: Verifiable rewards for math, dataset creation

### Code Generation

**AlphaCode: Competition-Level Code Generation**
- Authors: Li et al. (DeepMind)
- Year: 2022
- Link: https://arxiv.org/abs/2203.07814
- Summary: Generate code, verify with test cases
- Key contributions: Large-scale code generation, verification-based filtering

**CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning**
- Authors: Le et al.
- Year: 2022
- Link: https://arxiv.org/abs/2207.01780
- Summary: RL for code generation with unit test verification
- Key contributions: RL + verification, code-specific techniques

## Reward Overoptimization

### Empirical Studies

**Scaling Laws for Reward Model Overoptimization**
- Authors: Gao et al. (OpenAI)
- Year: 2023
- Link: https://arxiv.org/abs/2210.10760
- Summary: THE definitive study on reward overoptimization. Shows gold reward peaks then declines as proxy reward increases.
- Key contributions: Quantifies overoptimization, studies scaling, proposes mitigations
- Must read: Essential for understanding Goodhart's Law in RLHF

**The Alignment Problem from a Deep Learning Perspective**
- Authors: Ngo et al. (Google)
- Year: 2024
- Link: https://arxiv.org/abs/2209.00626
- Summary: Comprehensive analysis of alignment challenges including overoptimization
- Key contributions: Taxonomy of problems, theoretical analysis

### Goodhart's Law

**Goodhart's Law in Machine Learning**
- Reference: Manheim & Garrabrant (2018)
- Link: https://arxiv.org/abs/1803.04585
- Summary: Categorizes types of Goodhart's Law in ML context
- Four types: Regressional, extremal, causal, adversarial

**AI Alignment: A Comprehensive Survey**
- Authors: Ji et al.
- Year: 2024
- Link: https://arxiv.org/abs/2310.19852
- Summary: Comprehensive survey including overoptimization
- Covers: All alignment methods, failure modes, future directions

### Mitigation Strategies

**Ensemble Methods for Reward Modeling**
- Various papers
- Summary: Using multiple reward models reduces overoptimization
- Key insight: Harder to fool all models simultaneously

**Online RLHF**
- Authors: Various
- Summary: Collecting fresh data during training prevents distribution shift
- Key insight: Adaptation to current policy distribution

## Libraries and Tools

### TRL (Transformer Reinforcement Learning)

**HuggingFace TRL**
- GitHub: https://github.com/huggingface/trl
- Docs: https://huggingface.co/docs/trl
- Summary: THE standard library for RLHF, DPO, and variants
- Features:
  - PPO training
  - DPO training
  - IPO, KTO, ORPO support
  - Reward modeling
  - SFT utilities
  - Integration with HuggingFace ecosystem
- Examples: https://github.com/huggingface/trl/tree/main/examples

**Quick Start:**
```bash
pip install trl transformers datasets
```

```python
from trl import DPOTrainer, DPOConfig

# See README.md for full example
```

### Alignment Handbook

**HuggingFace Alignment Handbook**
- GitHub: https://github.com/huggingface/alignment-handbook
- Summary: Practical guide and recipes for aligning language models
- Includes:
  - Complete training recipes
  - Dataset preparation scripts
  - Evaluation pipelines
  - Best practices
- Scripts: Ready-to-use training scripts for DPO, RLHF, etc.

**Zephyr-7B Recipe:**
- One of the best-documented alignment pipelines
- Link: https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta

### OpenAI Alignment Tools

**OpenAI Evals**
- GitHub: https://github.com/openai/evals
- Summary: Framework for evaluating language model performance
- Useful for: Creating custom evaluations, benchmarking alignment

**OpenAI RLHF Code**
- Not publicly released in full
- Reference implementations in papers

### DeepSpeed

**DeepSpeed-Chat**
- GitHub: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat
- Summary: DeepSpeed's RLHF implementation with optimizations
- Features:
  - Efficient RLHF training
  - Hybrid Engine for inference optimization
  - Multi-GPU support

### RL Libraries

**Stable-Baselines3**
- GitHub: https://github.com/DLR-RM/stable-baselines3
- Docs: https://stable-baselines3.readthedocs.io/
- Summary: PPO and other RL algorithms (useful for understanding RLHF mechanics)

**Ray RLlib**
- GitHub: https://github.com/ray-project/ray
- Docs: https://docs.ray.io/en/latest/rllib/
- Summary: Scalable RL framework (for advanced use cases)

### Evaluation Tools

**HELM (Holistic Evaluation of Language Models)**
- GitHub: https://github.com/stanford-crfm/helm
- Link: https://crfm.stanford.edu/helm/
- Summary: Comprehensive benchmark suite for LLMs
- Includes: Safety, bias, toxicity, capabilities

**lm-evaluation-harness**
- GitHub: https://github.com/EleutherAI/lm-evaluation-harness
- Summary: Framework for evaluating language models across many tasks
- Widely used in research

**AlpacaEval**
- GitHub: https://github.com/tatsu-lab/alpaca_eval
- Link: https://tatsu-lab.github.io/alpaca_eval/
- Summary: Fast, cheap evaluation using LLM judges
- Useful for: Quick iteration, relative comparisons

### Annotation Tools

**Label Studio**
- Link: https://labelstud.io/
- GitHub: https://github.com/heartexlabs/label-studio
- Summary: Open-source data labeling platform
- Use case: Collecting preference data

**Argilla**
- Link: https://argilla.io/
- GitHub: https://github.com/argilla-io/argilla
- Summary: Open-source platform for data curation and labeling
- Features: LLM-specific workflows, preference annotation

### Infrastructure

**vLLM**
- GitHub: https://github.com/vllm-project/vllm
- Docs: https://vllm.readthedocs.io/
- Summary: Fast LLM inference engine
- Use case: Production deployment, generating responses for online training

**Text Generation Inference (TGI)**
- GitHub: https://github.com/huggingface/text-generation-inference
- Summary: HuggingFace's production-ready inference server
- Features: Fast, scalable, Docker-ready

## Datasets

### Preference Datasets

**Anthropic HH-RLHF**
- Link: https://huggingface.co/datasets/Anthropic/hh-rlhf
- Size: ~170K preference pairs
- Domain: Helpful and harmless assistant conversations
- Format: Prompt, chosen, rejected
- Most popular dataset for training DPO/RLHF

**OpenAssistant Conversations**
- Link: https://huggingface.co/datasets/OpenAssistant/oasst1
- Size: ~161K messages, human-annotated
- Domain: General assistant conversations
- Format: Conversation trees with quality ratings

**Stanford Human Preferences (SHP)**
- Link: https://huggingface.co/datasets/stanfordnlp/SHP
- Size: 385K preference pairs
- Domain: Reddit posts/comments
- Format: Question, human answer A, human answer B, preference

**HelpSteer**
- Link: https://huggingface.co/datasets/nvidia/HelpSteer
- Size: ~20K responses with detailed annotations
- Domain: Helpfulness ratings
- Format: Multi-dimensional ratings (helpfulness, correctness, etc.)

**UltraFeedback**
- Link: https://huggingface.co/datasets/openbmb/UltraFeedback
- Size: ~63K prompts, multiple responses
- Domain: General instructions
- Format: Ratings from multiple LLM judges

### Math and Reasoning

**PRM800K (Process Reward Model)**
- Link: https://github.com/openai/prm800k
- Size: 800K step-level labels
- Domain: Math problems
- Format: Problems with step-by-step solutions and correctness labels

**MATH Dataset**
- Link: https://github.com/hendrycks/math
- Size: 12.5K problems
- Domain: Competition mathematics
- Format: Problem, answer (verifiable)

**GSM8K**
- Link: https://github.com/openai/grade-school-math
- Size: 8.5K grade school math problems
- Domain: Elementary mathematics
- Format: Problem, solution, answer

### Code

**APPS**
- Link: https://github.com/hendrycks/apps
- Size: 10K problems
- Domain: Competitive programming
- Format: Problem, test cases, solutions (verifiable)

**CodeContests**
- Link: https://github.com/deepmind/code_contests
- Size: Programming competition problems
- Domain: Algorithmic problems
- Format: Problem description, test cases

**HumanEval**
- Link: https://github.com/openai/human-eval
- Size: 164 programming problems
- Domain: Python functions
- Format: Docstring, signature, test cases (verifiable)

### Safety and Toxicity

**RealToxicityPrompts**
- Link: https://allenai.org/data/real-toxicity-prompts
- Size: 100K prompts
- Domain: Potentially toxic completions
- Use case: Evaluating safety

**CivilComments**
- Link: https://huggingface.co/datasets/civil_comments
- Size: 2M comments with toxicity labels
- Domain: Online comments
- Use case: Training toxicity classifiers

## Tutorials and Guides

### Official Tutorials

**HuggingFace DPO Tutorial**
- Link: https://huggingface.co/blog/dpo-trl
- Summary: Step-by-step guide to training DPO models
- Includes: Code, explanations, best practices

**Alignment Handbook Recipes**
- Link: https://github.com/huggingface/alignment-handbook
- Summary: Complete training recipes for alignment
- Examples: Zephyr-7B, many others

**OpenAI Spinning Up in Deep RL**
- Link: https://spinningup.openai.com/
- Summary: Comprehensive RL course (background for RLHF)
- Includes: Theory, implementations, exercises

### Blog Posts

**Anthropic: Constitutional AI Blog**
- Link: https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback
- Summary: Explanation of Constitutional AI approach
- Insights: Design philosophy, results

**OpenAI: InstructGPT Blog**
- Link: https://openai.com/research/instruction-following
- Summary: How InstructGPT was trained
- Details: Three-phase process, results

**DeepMind: Sparrow**
- Link: https://www.deepmind.com/blog/building-safer-dialogue-agents
- Summary: DeepMind's approach to safe dialogue agents
- Methods: RLHF with external knowledge

**HuggingFace: LLM Training Guide**
- Link: https://huggingface.co/blog/rlhf
- Summary: Comprehensive guide to RLHF
- Covers: All phases, tools, tips

### Video Tutorials

**Andrej Karpathy: State of GPT**
- Link: https://www.youtube.com/watch?v=bZQun8Y4L2A
- Summary: Overview of GPT training including RLHF
- Length: ~1 hour
- Great: High-level understanding

**Stanford CS224N: RLHF Lecture**
- Link: https://web.stanford.edu/class/cs224n/
- Summary: Academic treatment of RLHF
- Details: Theory, math, applications

**DeepLearning.AI: RLHF Course**
- Link: https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/
- Summary: Short course on RLHF
- Features: Practical notebooks, exercises

## Blogs and Articles

### Research Blogs

**Anthropic Blog**
- Link: https://www.anthropic.com/research
- Focus: Safety, alignment, Constitutional AI
- Must follow: Cutting-edge safety research

**OpenAI Blog**
- Link: https://openai.com/research
- Focus: GPT models, RLHF, capabilities
- Must follow: State-of-the-art models

**DeepMind Blog**
- Link: https://www.deepmind.com/blog
- Focus: RL, game-playing, alignment
- Must follow: Strong theoretical work

**HuggingFace Blog**
- Link: https://huggingface.co/blog
- Focus: Practical ML, open source, tutorials
- Must follow: Best tutorials and tools

### Independent Blogs

**LessWrong: AI Alignment**
- Link: https://www.lesswrong.com/tag/ai-alignment
- Focus: AI safety, alignment theory
- Community: Active discussions, varied perspectives

**Alignment Forum**
- Link: https://www.alignmentforum.org/
- Focus: Technical AI alignment research
- Community: Researchers, serious discussions

**The Gradient**
- Link: https://thegradient.pub/
- Focus: ML research explained
- Quality: High-quality long-form articles

## Courses and Lectures

### Online Courses

**Stanford CS324: Large Language Models**
- Link: https://stanford-cs324.github.io/winter2022/
- Coverage: Comprehensive, includes alignment
- Materials: Lectures, readings, assignments

**Berkeley CS 285: Deep RL**
- Link: https://rail.eecs.berkeley.edu/deeprlcourse/
- Coverage: RL fundamentals (background for RLHF)
- Materials: Lectures, slides, homeworks

**DeepLearning.AI: Generative AI with LLMs**
- Link: https://www.deeplearning.ai/courses/generative-ai-with-llms/
- Coverage: End-to-end LLM training including RLHF
- Format: Video lectures, quizzes, exercises

### Academic Courses

**Stanford CS224N: NLP with Deep Learning**
- Link: https://web.stanford.edu/class/cs224n/
- Coverage: Modern NLP including LLMs
- Materials: Freely available

**MIT 6.S191: Deep Learning**
- Link: http://introtodeeplearning.com/
- Coverage: DL fundamentals, some RL
- Materials: Lectures, labs

## Research Groups

### Industry Labs

**Anthropic**
- Focus: AI safety, Constitutional AI, Claude
- Publications: High-quality safety research
- Link: https://www.anthropic.com/

**OpenAI**
- Focus: GPT models, RLHF, capabilities
- Publications: InstructGPT, GPT-4, etc.
- Link: https://openai.com/

**Google DeepMind**
- Focus: RL, AlphaGo, LLMs
- Publications: Strong theoretical work
- Link: https://www.deepmind.com/

**Meta AI (FAIR)**
- Focus: Open research, LLaMA models
- Publications: Open models and methods
- Link: https://ai.meta.com/

### Academic Groups

**Stanford NLP**
- Link: https://nlp.stanford.edu/
- Focus: NLP, LLMs, alignment
- Notable: Chris Manning, Percy Liang groups

**Berkeley RAIL**
- Link: https://rail.eecs.berkeley.edu/
- Focus: RL, alignment, robotics
- Notable: Sergey Levine group

**CMU Language Technologies Institute**
- Link: https://www.lti.cs.cmu.edu/
- Focus: NLP, dialog systems
- Notable: Graham Neubig group

**MIT CSAIL**
- Link: https://www.csail.mit.edu/
- Focus: AI broadly, some alignment work
- Notable: Various groups

## Community

### Forums and Discussion

**HuggingFace Discord**
- Link: https://huggingface.co/join/discord
- Activity: Very active, helpful community
- Topics: All things NLP/LLMs

**EleutherAI Discord**
- Link: https://www.eleuther.ai/get-involved
- Activity: Active open research community
- Topics: LLMs, open models, alignment

**r/MachineLearning**
- Link: https://www.reddit.com/r/MachineLearning/
- Activity: Active, research-focused
- Topics: ML papers, discussions

**r/LocalLLaMA**
- Link: https://www.reddit.com/r/LocalLLaMA/
- Activity: Very active
- Topics: Running LLMs locally, fine-tuning

### Twitter/X Accounts

**@AnthropicAI** - Anthropic updates
**@OpenAI** - OpenAI research
**@HuggingFace** - HF updates and tutorials
**@_philschmid** - Phil Schmid (HF, great tutorials)
**@abacaj** - Andrej Bauer (LLM insights)
**@karpathy** - Andrej Karpathy (when active)
**@srush_nlp** - Sasha Rush (NLP researcher)

### Conferences

**NeurIPS** - Neural Information Processing Systems
- Link: https://neurips.cc/
- Focus: Broad ML, including alignment
- Timing: December annually

**ICML** - International Conference on Machine Learning
- Link: https://icml.cc/
- Focus: ML theory and practice
- Timing: July annually

**ICLR** - International Conference on Learning Representations
- Link: https://iclr.cc/
- Focus: Deep learning, representation learning
- Timing: May annually

**ACL** - Association for Computational Linguistics
- Link: https://www.aclweb.org/
- Focus: NLP
- Timing: July/August annually

## Additional Resources

### Books

**Reinforcement Learning: An Introduction (Sutton & Barto)**
- Link: http://incompleteideas.net/book/the-book-2nd.html
- Summary: THE RL textbook (background for RLHF)
- Free online: Yes

**Deep Learning (Goodfellow, Bengio, Courville)**
- Link: https://www.deeplearningbook.org/
- Summary: Comprehensive DL textbook
- Free online: Yes

**Human-Compatible (Stuart Russell)**
- Link: https://www.penguinrandomhouse.com/books/566677/human-compatible-by-stuart-russell/
- Summary: AI alignment from philosophical perspective
- Popular: Accessible to general audience

### Newsletters

**Import AI (Jack Clark)**
- Link: https://jack-clark.net/
- Frequency: Weekly
- Content: AI research summaries, commentary

**The Batch (DeepLearning.AI)**
- Link: https://www.deeplearning.ai/the-batch/
- Frequency: Weekly
- Content: AI news, tutorials

**TLDR AI**
- Link: https://tldr.tech/ai
- Frequency: Daily
- Content: Quick AI news summaries

### Model Hubs

**HuggingFace Hub**
- Link: https://huggingface.co/models
- Content: Pre-trained models, datasets, spaces
- Search: Filter by task, license, size

**Model repositories:**
- Zephyr-7B-β (DPO-trained): https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
- Mistral-7B-Instruct: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- LLaMA-2-Chat (RLHF): https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

## How to Stay Current

The field moves fast. To stay up-to-date:

1. **Follow arxiv:**
   - https://arxiv.org/list/cs.CL/recent (NLP)
   - https://arxiv.org/list/cs.LG/recent (ML)
   - https://arxiv.org/list/cs.AI/recent (AI)
   - Set up alerts for keywords: RLHF, DPO, alignment

2. **Follow key researchers on Twitter/X:**
   - See list above

3. **Join Discord communities:**
   - HuggingFace, EleutherAI (most active)

4. **Read company blogs:**
   - Anthropic, OpenAI, DeepMind (cutting edge)

5. **Attend conferences (or watch recordings):**
   - NeurIPS, ICML, ICLR (recordings usually posted)

6. **Check GitHub trending:**
   - https://github.com/trending/python
   - Filter for ML repos

7. **Subscribe to newsletters:**
   - Import AI, The Batch (curated summaries)

## Recommended Learning Path

If you're new to this topic, follow this path:

### Week 1: Foundations
- Read InstructGPT paper
- Complete HuggingFace RLHF tutorial
- Understand Bradley-Terry model

### Week 2: DPO
- Read DPO paper (Rafailov et al.)
- Work through DPO tutorial
- Train a small DPO model

### Week 3: Variants
- Read IPO, KTO, ORPO papers
- Compare on a small dataset
- Understand tradeoffs

### Week 4: Constitutional AI
- Read Constitutional AI paper
- Understand RLAIF
- Explore self-improvement

### Week 5: RLVR
- Study DeepSeek-R1
- Understand verifiable rewards
- Try on math/code task

### Week 6: Advanced Topics
- Read reward overoptimization paper (Gao et al.)
- Study Goodhart's Law
- Explore mitigation strategies

### Week 7: Practice
- Build end-to-end alignment pipeline
- Deploy a model
- Set up monitoring

### Week 8: Research
- Read latest papers
- Join community discussions
- Start your own project

## Contributing

Found a useful resource not listed here? Open an issue or PR on the course repository!

## Final Notes

This resource list is extensive but not exhaustive. The field is rapidly evolving, with new papers, tools, and techniques emerging constantly.

Key takeaways:
- **TRL library** is the practical workhorse
- **DPO paper** is must-read for theory
- **Alignment Handbook** for best practices
- **Anthropic/OpenAI blogs** for cutting edge
- **HuggingFace community** for support

Good luck with your alignment research and implementations!

---

Last updated: 2025-02-05
Course: RL Learning (Week 18: Beyond RLHF)
