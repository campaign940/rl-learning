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
![Overall Progress](https://img.shields.io/badge/Overall_Progress-0%25-lightgrey)
![Tasks Completed](https://img.shields.io/badge/Tasks_Completed-0%2F60-blue)

| Week | Topic | Progress | Status |
|------|-------|----------|--------|
| Week 01 | [Week 1: Introduction to Reinforcement Learning](01-introduction/) | 0/5 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 02 | [Week 2: Markov Decision Processes](02-mdp/) | 0/5 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 03 | [Week 3: Dynamic Programming](03-dynamic-programming/) | 0/5 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 04 | [Week 4: Monte Carlo Methods](04-monte-carlo/) | 0/0 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 05 | [Week 5: Temporal-Difference Learning](05-temporal-difference/) | 0/0 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 06 | [Week 6: n-step Methods & Eligibility Traces](06-nstep-eligibility/) | 0/0 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 07 | [Week 7: Planning and Learning](07-planning-and-learning/) | 0/0 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 08 | [Week 8: Value Function Approximation](08-value-function-approx/) | 0/6 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 09 | [Week 9: Deep Q-Networks (DQN)](09-dqn/) | 0/6 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 10 | [Week 10: DQN Extensions](10-dqn-extensions/) | 0/5 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 11 | [Week 11: Policy Gradient Methods](11-policy-gradient/) | 0/5 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 12 | [Week 12: Actor-Critic Methods](12-actor-critic/) | 0/5 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 13 | [Week 13: TRPO & PPO (Trust Region Policy Optimization & Proximal Policy Optimization)](13-trpo-ppo/) | 0/6 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 14 | [Week 14: Continuous Control (DDPG, TD3, SAC)](14-continuous-control/) | 0/6 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 15 | [Week 15: Model-Based Reinforcement Learning](15-model-based-rl/) | 0/0 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 16 | [Week 16: Exploration and Exploitation](16-exploration/) | 0/0 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 17 | [Week 17: Reward Modeling and RLHF](17-reward-modeling-rlhf/) | 0/0 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |
| Week 18 | [Week 18: Beyond RLHF - Direct Preference Optimization and Modern Alignment](18-beyond-rlhf/) | 0/6 (0%) | ![not_started](https://img.shields.io/badge/not started-lightgrey) |

### Summary

- **Completed Weeks:** 0/18
- **In Progress:** 0
- **Not Started:** 18
- **Total Tasks:** 0/60
- **Overall Completion:** 0.0%
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
