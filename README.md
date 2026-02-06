# RL Learning: From Foundations to RLHF

A systematic reinforcement learning curriculum based on **Sutton & Barto** textbook, **Stanford CS234**, **UC Berkeley CS285**, **MIT 6.7950**, and **David Silver's UCL lectures**.

## Learning Loop

```
 1. Study        2. Implement      3. Quiz         4. Archive        5. Stay Updated
 ─────────       ──────────────    ─────────       ──────────        ───────────────
 Read weekly  →  Code the algo →  5 questions  →  Summary to     →  New papers auto-
 content         & run tests       per week        Google Drive      fetched & Slack
                                                   → NotebookLM      alert sent
```

## Curriculum (18 Weeks)

Based on **Sutton & Barto, Reinforcement Learning: An Introduction (2nd Ed.)** chapter progression, cross-referenced with top university courses.

| Phase | Weeks | Topic | Textbook | University Reference |
|-------|-------|-------|----------|---------------------|
| **Foundations** | 1-3 | RL Basics, MDP, DP | S&B Ch.1-4 | Silver L1-L3, CS234 W1-3 |
| **Tabular Methods** | 4-7 | MC, TD, n-step, Planning | S&B Ch.5-8 | Silver L4-L5,L8, CS234 W4-6 |
| **Function Approx** | 8-10 | FA, DQN, DQN Extensions | S&B Ch.9-11 | Silver L6, CS285 L7-8 |
| **Policy Optimization** | 11-14 | PG, Actor-Critic, PPO, SAC | S&B Ch.13 | Silver L7, CS285 L5,9 |
| **Advanced Topics** | 15-16 | Model-Based RL, Exploration | S&B Ch.8,14 | Silver L8-9, CS285 L11-12 |
| **RLHF & Alignment** | 17-18 | Reward Modeling, DPO | - | CS234 RLHF, InstructGPT |

## Progress

<!-- AUTO-GENERATED:START -->
![Progress](https://img.shields.io/badge/progress-0%25-red)

| Week | Topic | Status |
|------|-------|--------|
| 01 | Introduction to RL | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 02 | Markov Decision Processes | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 03 | Dynamic Programming | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 04 | Monte Carlo Methods | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 05 | Temporal-Difference Learning | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 06 | n-step Methods & Eligibility Traces | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 07 | Planning and Learning | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 08 | Value Function Approximation | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 09 | Deep Q-Networks | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 10 | DQN Extensions | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 11 | Policy Gradient Methods | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 12 | Actor-Critic Methods | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 13 | TRPO & PPO | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 14 | Continuous Control | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 15 | Model-Based RL | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 16 | Exploration & Exploitation | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 17 | Reward Modeling & RLHF | ![](https://img.shields.io/badge/-not_started-lightgrey) |
| 18 | Beyond RLHF | ![](https://img.shields.io/badge/-not_started-lightgrey) |
<!-- AUTO-GENERATED:END -->

## Primary References

| Resource | Author | Role |
|----------|--------|------|
| [Reinforcement Learning: An Introduction (2nd Ed.)](http://incompleteideas.net/book/the-book-2nd.html) | Sutton & Barto | Core textbook |
| [UCL RL Course](https://davidstarsilver.wordpress.com/teaching/) | David Silver | Lecture series |
| [Stanford CS234](https://web.stanford.edu/class/cs234/) | Emma Brunskill | University course |
| [UC Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) | Sergey Levine | Deep RL course |
| [MIT 6.7950](https://web.mit.edu/6.7950/www/) | MIT | Mathematical foundations |
| [OpenAI Spinning Up](https://spinningup.openai.com/) | OpenAI | Practical guide |

## Automation

- **Paper tracking**: Daily arxiv scan via GitHub Actions → `papers/auto-updates/`
- **Slack alerts**: New papers notified to `gdyr.ai #rl-learning-loop`
- **Progress tracking**: Auto-updated on push via GitHub Actions
- **NotebookLM sync**: Weekly summaries uploaded to Google Drive for NotebookLM sourcing

## Setup

```bash
# Clone
git clone https://github.com/<your-username>/rl-learning.git
cd rl-learning

# Install dependencies (for scripts)
pip install arxiv google-api-python-client google-auth-oauthlib

# Set secrets in GitHub repo settings:
# - SLACK_WEBHOOK_URL: Slack Incoming Webhook URL
# - GOOGLE_DRIVE_CREDENTIALS: Service account JSON for Drive upload
```

See [SETUP.md](SETUP.md) for detailed Slack and Google Drive configuration.
