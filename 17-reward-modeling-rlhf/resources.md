# Week 17 Resources: Reward Modeling and RLHF

## Primary Readings

### Key Papers (Must Read)

1. **InstructGPT: Training language models to follow instructions with human feedback**
   - Authors: Ouyang et al. (OpenAI, 2022)
   - Paper: https://arxiv.org/abs/2203.02155
   - Key contribution: Complete RLHF pipeline for language models
   - Result: Foundation for ChatGPT
   - Impact: Defined modern approach to AI alignment

2. **Deep RL from Human Preferences**
   - Authors: Christiano et al. (OpenAI/DeepMind, 2017)
   - Paper: https://arxiv.org/abs/1706.03741
   - Key contribution: First successful RLHF for complex tasks
   - Result: Learned Atari and MuJoCo from preferences
   - Foundational paper for reward learning

3. **Learning to Summarize from Human Feedback**
   - Authors: Stiennon et al. (OpenAI, 2020)
   - Paper: https://arxiv.org/abs/2009.01325
   - Website: https://openai.com/research/learning-to-summarize-with-human-feedback
   - Key contribution: RLHF for summarization
   - Result: Better than supervised learning

4. **Fine-Tuning Language Models from Human Preferences**
   - Authors: Ziegler et al. (OpenAI, 2019)
   - Paper: https://arxiv.org/abs/1909.08593
   - Key contribution: Early LLM preference learning
   - Sentiment control and text continuation

## Supplementary Papers

5. **WebGPT: Browser-assisted question-answering with human feedback**
   - Paper: https://arxiv.org/abs/2112.09332
   - RLHF for web browsing and QA

6. **Anthropic's Constitutional AI paper**
   - Paper: https://arxiv.org/abs/2212.08073
   - AI-generated preferences (RLAIF)
   - Scalable oversight

7. **Scaling Laws for Reward Model Overoptimization**
   - Authors: Gao et al. (Anthropic, 2022)
   - Paper: https://arxiv.org/abs/2210.10760
   - Key insight: Goodhart's law quantified
   - When reward hacking becomes a problem

8. **Open Problems and Fundamental Limitations of RLHF**
   - Authors: Casper et al. (2023)
   - Paper: https://arxiv.org/abs/2307.15217
   - Critical analysis of RLHF
   - Limitations and future directions

## Blog Posts and Tutorials

1. **OpenAI: Illustrating Reinforcement Learning from Human Feedback (RLHF)**
   - URL: https://openai.com/research/instruction-following
   - Visual explanation of RLHF pipeline
   - InstructGPT details

2. **Hugging Face: Illustrating RLHF**
   - URL: https://huggingface.co/blog/rlhf
   - Comprehensive tutorial
   - Code examples

3. **Chip Huyen: RLHF - Reinforcement Learning from Human Feedback**
   - URL: https://huyenchip.com/2023/05/02/rlhf.html
   - Practical perspective
   - Productionization considerations

4. **Nathan Lambert: A Short Introduction to RLHF**
   - URL: https://www.interconnects.ai/p/rlhf-train-better-models
   - Clear conceptual overview
   - Recent developments

5. **Lil'Log: Learning from Human Preferences**
   - URL: https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/
   - Math-heavy explanation
   - Multiple techniques covered

## Video Lectures

1. **John Schulman (OpenAI): Reinforcement Learning from Human Feedback**
   - URL: https://www.youtube.com/watch?v=hhiLw5Q_UFg
   - Technical deep dive
   - InstructGPT insights from author

2. **CS25: Transformers United - RLHF (John Schulman)**
   - URL: https://www.youtube.com/watch?v=2MBJOuVq380
   - Stanford seminar
   - Comprehensive overview

3. **Stanford CS324: Understanding Language Models - RLHF**
   - Course materials on RLHF
   - Theory and practice

4. **DeepMind Podcast: The Alignment Problem**
   - URL: https://www.youtube.com/watch?v=0qfZcIdQfbE
   - High-level discussion
   - Motivation for RLHF

## Code and Libraries

1. **TRL: Transformer Reinforcement Learning**
   - URL: https://github.com/lvwerra/trl
   - Hugging Face's RLHF library
   - Easy-to-use tools for PPO fine-tuning
   - Examples: https://github.com/lvwerra/trl/tree/main/examples

2. **OpenAI's Learning to Summarize Code**
   - URL: https://github.com/openai/summarize-from-feedback
   - Reference implementation
   - Research-grade code

3. **CarperAI trlX**
   - URL: https://github.com/CarperAI/trlx
   - Distributed RLHF training
   - Production-scale

4. **DeepSpeed-Chat**
   - URL: https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat
   - Microsoft's RLHF pipeline
   - Efficient training

5. **RL4LMs**
   - URL: https://github.com/allenai/RL4LMs
   - AllenAI's RL for language models
   - Multiple algorithms

## Datasets

1. **Anthropic's HH-RLHF (Helpful & Harmless)**
   - URL: https://github.com/anthropics/hh-rlhf
   - 160K+ human preference labels
   - Helpfulness and harmlessness

2. **OpenAI's WebGPT Comparisons**
   - URL: https://huggingface.co/datasets/openai/webgpt_comparisons
   - 20K comparison pairs
   - Question answering task

3. **Stanford Human Preferences Dataset (SHP)**
   - URL: https://huggingface.co/datasets/stanfordnlp/SHP
   - 385K+ preferences from Reddit
   - Diverse topics

4. **OpenAssistant Conversations**
   - URL: https://huggingface.co/datasets/OpenAssistant/oasst1
   - Human-generated conversations
   - Multi-turn dialogues

5. **Anthropic's Helpful-Only and Harmless-Only**
   - Separate datasets for different objectives
   - Study trade-offs

## Tools and Frameworks

1. **Weights & Biases RLHF Dashboard**
   - Track RLHF metrics
   - Visualize training

2. **Cohere's For AI Toolkit**
   - Annotation interfaces
   - Preference collection tools

3. **Label Studio**
   - URL: https://labelstud.io/
   - Open-source data labeling
   - Support for pairwise comparisons

4. **Scale AI's RLHF Platform**
   - Commercial preference labeling
   - Quality control

## Advanced Topics

1. **Reward Model Ensembles**
   - "Deep RL at the Edge of the Statistical Precipice" (Agarwal et al.)
   - Paper: https://arxiv.org/abs/2108.13264
   - Uncertainty estimation

2. **Active Learning for RLHF**
   - "Active Learning for Deep RL" (Konyushkova et al.)
   - Paper: https://arxiv.org/abs/1909.12583
   - Sample-efficient preference collection

3. **Iterated RLHF**
   - "Constitutional AI" (Anthropic)
   - Continuous improvement loop
   - Online vs offline RLHF

4. **Multi-Objective RLHF**
   - Balancing helpfulness, harmlessness, honesty
   - Pareto optimal policies
   - Scalarization vs constrained optimization

5. **RLHF for Multi-Modal Models**
   - Vision-language models
   - Audio and speech
   - Robotics

## Theoretical Foundations

1. **Bradley-Terry Model**
   - Original paper (1952)
   - Paired comparison models
   - Statistical foundations

2. **Utility Theory and Preferences**
   - Von Neumann-Morgenstern utility theory
   - Connection to reward learning

3. **KL Divergence and Information Theory**
   - "Information Theory, Inference, and Learning Algorithms" (MacKay)
   - Free PDF: http://www.inference.org.uk/itprnn/book.pdf

4. **Policy Gradient Theorems**
   - "Policy Gradient Methods for Reinforcement Learning" (Sutton et al.)
   - Theoretical foundations of PPO

## Critiques and Limitations

1. **"Open Problems and Fundamental Limitations of RLHF"**
   - Paper: https://arxiv.org/abs/2307.15217
   - Comprehensive analysis
   - Future research directions

2. **Sycophancy in Language Models**
   - Paper: https://arxiv.org/abs/2310.13548
   - RLHF can make models agree with users incorrectly
   - Mitigation strategies

3. **Reward Hacking in Language Models**
   - Empirical studies
   - Detection and prevention

4. **Scalable Oversight Problem**
   - How to provide feedback on superhuman tasks?
   - Open research problem

## Practical Guides

1. **How to Train Your Own ChatGPT**
   - Step-by-step guide
   - Resource requirements
   - Common pitfalls

2. **RLHF at Scale: Engineering Challenges**
   - Distributed training
   - Data collection pipelines
   - Cost optimization

3. **Reward Model Debugging**
   - Testing for biases
   - Validation strategies
   - A/B testing

4. **Prompt Engineering for RLHF**
   - Instruction design
   - Evaluation prompts
   - Edge cases

## Community and Discussion

1. **EleutherAI Discord**
   - Active RLHF discussion
   - Open-source community

2. **Alignment Forum**
   - URL: https://www.alignmentforum.org/
   - Technical AI safety discussions
   - RLHF research

3. **Reddit r/MachineLearning**
   - RLHF paper discussions
   - Implementation questions

4. **Twitter/X #RLHF**
   - Latest research
   - Quick updates

## Benchmarks and Evaluations

1. **TruthfulQA**
   - URL: https://github.com/sylinrl/TruthfulQA
   - Test for truthfulness
   - Common RLHF benchmark

2. **HHH (Helpful, Honest, Harmless) Eval**
   - Anthropic's evaluation suite
   - Multi-objective assessment

3. **MT-Bench**
   - Multi-turn conversation evaluation
   - Human preference correlations

4. **AlpacaEval**
   - URL: https://github.com/tatsu-lab/alpaca_eval
   - Automated evaluation using LLM judges

## Related Courses

1. **CS224N: Natural Language Processing (Stanford)**
   - URL: https://web.stanford.edu/class/cs224n/
   - Guest lecture on RLHF

2. **CS234: Reinforcement Learning (Stanford)**
   - URL: https://web.stanford.edu/class/cs234/
   - RLHF module

3. **CS324: Large Language Models (Stanford)**
   - URL: https://stanford-cs324.github.io/winter2022/
   - Dedicated RLHF section

4. **Berkeley CS 285: Deep RL**
   - Lecture on learning from preferences

## Companies and Research Labs

1. **OpenAI**
   - Pioneers of RLHF for LLMs
   - InstructGPT, ChatGPT

2. **Anthropic**
   - Constitutional AI
   - Advanced RLHF research

3. **DeepMind**
   - Sparrow chatbot
   - Scalable alignment

4. **Cohere**
   - Commercial RLHF tools
   - For AI community

5. **CarperAI**
   - Open-source RLHF
   - trlX library

## Books and Comprehensive Resources

1. **"Reinforcement Learning: An Introduction" (Sutton & Barto)**
   - Chapter 13: Policy Gradient Methods
   - Foundational RL concepts

2. **"Deep Learning" (Goodfellow et al.)**
   - Neural network foundations
   - Free online: https://www.deeplearningbook.org/

3. **"The Alignment Problem" (Brian Christian)**
   - Popular science book
   - Motivation for RLHF
   - Historical context

## Practical Tips

### Data Collection

- Aim for 10K-100K demonstrations
- 10K-100K preference comparisons
- Use diverse labelers (avoid single annotator bias)
- Measure inter-annotator agreement (>70%)
- Include hard cases and edge cases

### Training

- Start with SFT (3-5 epochs, don't overfit)
- Reward model: 3-5 epochs, check validation accuracy
- PPO: Small learning rate (1e-6), beta=0.01-0.1
- Monitor KL divergence (target: 5-10)
- Regular human evaluation (every 1000 steps)

### Debugging

- If no improvement: Check reward model accuracy
- If mode collapse: Increase KL penalty (beta)
- If reward hacking: Use ensemble, early stopping
- If overoptimization: Stop training, use earlier checkpoint

### Evaluation

- Human evaluation is gold standard
- Use held-out test prompts
- Red team for safety
- Monitor win rate vs baselines
- Check for biases (length, verbosity, sycophancy)

## Future Directions

1. **Scalable Oversight**
   - How to evaluate superhuman performance?
   - Recursive reward modeling

2. **Debate and Amplification**
   - AI systems helping evaluate other AI
   - Iterated amplification

3. **Process vs Outcome Supervision**
   - Reward good reasoning, not just answers
   - Interpretability

4. **Multi-Agent RLHF**
   - Training multiple models together
   - Adversarial robustness

5. **Personalization**
   - Individual user preferences
   - Federated learning

## Next Week Preview

Week 18 covers advanced alignment methods beyond RLHF:
- **DPO (Direct Preference Optimization)**: Eliminates reward model
- **Constitutional AI**: AI-generated preferences (RLAIF)
- **KTO & ORPO**: Alternative objectives
- **Online RLHF**: Iterative improvement
- **Reward overoptimization**: Detection and mitigation

These methods address RLHF's limitations and push toward more robust alignment.
