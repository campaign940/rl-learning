# Automation Setup Guide

This guide walks you through setting up all automation features for the RL Learning repository.

## Table of Contents

1. [Slack Integration](#slack-integration)
2. [Google Drive & NotebookLM Integration](#google-drive--notebooklm-integration)
3. [GitHub Actions](#github-actions)
4. [Local Development](#local-development)
5. [Troubleshooting](#troubleshooting)

---

## Slack Integration

Automated paper updates are sent to a Slack channel in the gdyr.ai workspace.

### Prerequisites
- Access to gdyr.ai Slack workspace
- Admin permissions to add apps

### Step-by-Step Setup

#### 1. Create Slack Channel
```
1. Open Slack (gdyr.ai workspace)
2. Click + next to "Channels"
3. Create channel: #rl-learning-loop
4. Set description: "Daily RL paper updates from arXiv"
```

#### 2. Create Slack App
```
1. Go to: https://api.slack.com/apps
2. Click "Create New App"
3. Select "From scratch"
4. App Name: "RL Learning Bot"
5. Workspace: "gdyr.ai"
6. Click "Create App"
```

#### 3. Enable Incoming Webhooks
```
1. In your app settings, go to "Incoming Webhooks"
2. Toggle "Activate Incoming Webhooks" to ON
3. Scroll down and click "Add New Webhook to Workspace"
4. Select channel: #rl-learning-loop
5. Click "Allow"
```

#### 4. Copy Webhook URL
```
1. You'll see the webhook URL (starts with https://hooks.slack.com/services/...)
2. Copy this URL - you'll need it for GitHub
3. IMPORTANT: Keep this URL secret!
```

#### 5. Add to GitHub Secrets
```
1. Go to your GitHub repository
2. Navigate to: Settings > Secrets and variables > Actions
3. Click "New repository secret"
4. Name: SLACK_WEBHOOK_URL
5. Value: (paste the webhook URL from step 4)
6. Click "Add secret"
```

#### 6. Test the Integration
```bash
# Test from command line
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🧪 Test message from RL Learning Bot"}' \
  YOUR_WEBHOOK_URL_HERE

# Expected result: Message appears in #rl-learning-loop
```

### Message Format

Daily updates will look like:

```
📚 5 new RL papers found today

• [Paper Title 1](https://arxiv.org/abs/2301.12345)
• [Paper Title 2](https://arxiv.org/abs/2301.12346)
• [Paper Title 3](https://arxiv.org/abs/2301.12347)
...and 2 more papers

View all papers: [link]
Auto-fetched from arXiv • 2024-01-15
```

---

## Google Drive & NotebookLM Integration

Upload weekly summaries to Google Drive for automatic syncing with NotebookLM.

### Prerequisites
- Google Account
- Access to Google Cloud Console
- Google Drive installed

### Step-by-Step Setup

#### 1. Create Google Cloud Project

```
1. Go to: https://console.cloud.google.com/
2. Click "Select a project" dropdown (top bar)
3. Click "New Project"
4. Project name: "rl-learning"
5. Organization: (leave default or select)
6. Click "Create"
7. Wait for project creation (check notifications)
```

#### 2. Enable Google Drive API

```
1. In Google Cloud Console, select your "rl-learning" project
2. Go to: APIs & Services > Library
3. Search: "Google Drive API"
4. Click "Google Drive API"
5. Click "Enable"
6. Wait for API to be enabled
```

#### 3. Create Service Account

```
1. Go to: APIs & Services > Credentials
2. Click "Create Credentials" > "Service Account"
3. Service account details:
   - Name: "rl-learning-automation"
   - ID: (auto-generated)
   - Description: "Automation for uploading RL learning summaries"
4. Click "Create and Continue"
5. Grant role: "Editor" (or "Drive File Editor" for minimal permissions)
6. Click "Continue"
7. Click "Done"
```

#### 4. Create Service Account Key

```
1. In Credentials page, find your service account
2. Click on the service account email
3. Go to "Keys" tab
4. Click "Add Key" > "Create new key"
5. Key type: JSON
6. Click "Create"
7. JSON file will download automatically
8. IMPORTANT: Store this file securely!
```

#### 5. Create Google Drive Folder

```
1. Open Google Drive: https://drive.google.com/
2. Click "+ New" > "Folder"
3. Name: "RL Learning Summaries"
4. Click "Create"
5. Right-click folder > "Share"
6. Add the service account email (from step 3)
   - Format: rl-learning-automation@rl-learning-XXXXX.iam.gserviceaccount.com
   - You can find this in the downloaded JSON file: "client_email"
7. Give "Editor" permissions
8. Uncheck "Notify people"
9. Click "Share"
```

#### 6. Get Folder ID

```
1. Open the "RL Learning Summaries" folder in Google Drive
2. Look at the URL in your browser
3. URL format: https://drive.google.com/drive/folders/FOLDER_ID_HERE
4. Copy the FOLDER_ID_HERE part
5. Example: 1aBcDeFgHiJkLmNoPqRsTuVwXyZ123456
```

#### 7. Add to GitHub Secrets

```
1. Go to: GitHub Repository > Settings > Secrets and variables > Actions
2. Add secret #1:
   - Name: GOOGLE_DRIVE_CREDENTIALS
   - Value: (paste ENTIRE contents of the JSON file from step 4)
   - Click "Add secret"

3. Add secret #2:
   - Name: GOOGLE_DRIVE_FOLDER_ID
   - Value: (paste folder ID from step 6)
   - Click "Add secret"
```

#### 8. Set Up NotebookLM

```
1. Go to: https://notebooklm.google.com/
2. Sign in with your Google Account
3. Click "New Notebook"
4. Name: "RL Learning"
5. Click "Create"
6. Click "Add Source"
7. Select "Google Drive"
8. Navigate to "RL Learning Summaries" folder
9. Click "Select"
10. NotebookLM will now automatically sync any files added to this folder
```

#### 9. Test the Integration

```bash
# Local test (requires setting environment variables first)
export GOOGLE_DRIVE_CREDENTIALS=/path/to/credentials.json
export GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here

# Run upload script
python scripts/upload-to-drive.py --week 01

# Expected result:
# - File appears in Google Drive folder
# - File automatically syncs to NotebookLM
```

### Verifying the Setup

After running the upload script:

1. **Check Google Drive:**
   - Go to "RL Learning Summaries" folder
   - You should see the uploaded markdown file

2. **Check NotebookLM:**
   - Open your "RL Learning" notebook
   - Sources should show the new file
   - Click on the source to verify content

### Folder Structure in Drive

```
RL Learning Summaries/
├── 01-introduction-README.md
├── 01-introduction-quiz.md
├── 02-mdp-README.md
├── 02-mdp-quiz.md
└── ...
```

---

## GitHub Actions

The repository includes two automated workflows:

### 1. Fetch Papers (`fetch-papers.yml`)

**Runs:** Daily at 09:00 UTC (6:00 PM KST)

**What it does:**
1. Fetches new RL papers from arXiv
2. Creates markdown entries in `papers/auto-updates/`
3. Commits changes to repository
4. Sends Slack notification with paper summaries

**Manual trigger:**
```
1. Go to: Repository > Actions
2. Select "Fetch RL Papers" workflow
3. Click "Run workflow"
4. Select branch: main
5. Click "Run workflow"
```

### 2. Update Progress (`update-progress.yml`)

**Runs:** On push to any `README.md` or `quiz.md` file

**What it does:**
1. Scans all week directories
2. Counts completed tasks (checked checkboxes)
3. Calculates progress percentages
4. Updates main `README.md` with progress badges
5. Commits changes back to repository

**Manual trigger:**
```
1. Go to: Repository > Actions
2. Select "Update Progress" workflow
3. Click "Run workflow"
4. Select branch: main
5. Click "Run workflow"
```

### Monitoring Workflows

**View workflow runs:**
```
1. Go to: Repository > Actions
2. See all workflow runs with status
3. Click on a run to see detailed logs
4. Red X = failed, Green check = succeeded
```

**Debugging failures:**
```
1. Click on failed workflow run
2. Click on the job name
3. Expand step that failed
4. Read error logs
5. Common issues:
   - Missing secrets
   - API rate limits
   - Network timeouts
   - Permission errors
```

---

## Local Development

### Installing Dependencies

#### Python Dependencies
```bash
# From repository root
pip install -r requirements.txt

# Or install individually
pip install arxiv
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

#### Make Scripts Executable
```bash
chmod +x scripts/new-week.sh
chmod +x scripts/new-experiment.sh
```

### Environment Variables

Create a `.env` file (don't commit this!):

```bash
# .env
GOOGLE_DRIVE_CREDENTIALS=/path/to/credentials.json
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here
```

Load in your shell:
```bash
# Bash/Zsh
source .env

# Or
export $(cat .env | xargs)
```

### Running Scripts Locally

#### Fetch Papers
```bash
# Fetch papers from last 24 hours
python scripts/fetch-papers.py --days 1

# Fetch papers from last week
python scripts/fetch-papers.py --days 7

# Custom output directory
python scripts/fetch-papers.py --days 1 --output papers/custom/
```

#### Update Progress
```bash
# Update progress for current repo
python scripts/update-progress.py

# Update progress for specific repo
python scripts/update-progress.py --repo /path/to/repo
```

#### Upload to Drive
```bash
# Upload week 01
python scripts/upload-to-drive.py --week 01

# Upload week 05
python scripts/upload-to-drive.py --week 05
```

#### Create New Week
```bash
# Create week directory
./scripts/new-week.sh 01-introduction 01 "Introduction to RL"

# Creates structure:
# 01-introduction/
#   ├── README.md
#   ├── quiz.md
#   ├── resources.md
#   ├── exercises/
#   ├── implementation/
#   └── notes/
```

#### Create New Experiment
```bash
# Create experiment
./scripts/new-experiment.sh "DQN Breakout"

# Creates:
# experiments/exp-001-dqn-breakout/
#   ├── config.yaml
#   ├── results.md
#   ├── src/
#   ├── figures/
#   └── logs/
```

---

## Troubleshooting

### Slack Integration Issues

**Problem: Webhook not working**
```
Solution:
1. Verify webhook URL is correct
2. Check if app is still installed in workspace
3. Verify channel still exists
4. Test with curl command
5. Check Slack app permissions
```

**Problem: Messages not appearing**
```
Solution:
1. Check #rl-learning-loop channel
2. Verify bot is in channel
3. Check workflow run logs for errors
4. Verify SLACK_WEBHOOK_URL secret is set correctly
```

### Google Drive Issues

**Problem: Authentication failed**
```
Solution:
1. Verify JSON credentials are valid
2. Check service account has correct permissions
3. Verify Drive API is enabled
4. Check credentials haven't expired
```

**Problem: Folder not found**
```
Solution:
1. Verify folder ID is correct
2. Check folder is shared with service account email
3. Verify service account has "Editor" permissions
4. Check folder wasn't deleted or moved
```

**Problem: Upload fails**
```
Solution:
1. Check internet connection
2. Verify Drive API quota not exceeded
3. Check file permissions
4. Verify folder exists and is accessible
```

### GitHub Actions Issues

**Problem: Workflow not running**
```
Solution:
1. Check Actions are enabled: Settings > Actions > General
2. Verify workflow file is in .github/workflows/
3. Check YAML syntax is valid
4. Verify trigger conditions are met
```

**Problem: Workflow failing**
```
Solution:
1. Check workflow run logs
2. Verify all secrets are set correctly
3. Check Python dependencies
4. Verify file paths are correct
5. Check for permission errors
```

**Problem: No papers found**
```
Solution:
This is normal! It means:
- No matching papers published in last 24 hours
- Or papers already captured in previous runs
- Not an error
```

### Script Issues

**Problem: Import errors**
```bash
# Solution: Install dependencies
pip install -r requirements.txt

# Or specific package
pip install arxiv
```

**Problem: Permission denied**
```bash
# Solution: Make script executable
chmod +x scripts/new-week.sh
chmod +x scripts/new-experiment.sh
```

**Problem: File not found**
```bash
# Solution: Run from repository root
cd /path/to/rl-learning
python scripts/fetch-papers.py
```

### Getting Help

If issues persist:

1. **Check logs:** Detailed error messages in GitHub Actions logs
2. **Verify setup:** Follow setup steps carefully
3. **Test locally:** Run scripts locally first
4. **Check documentation:** Google Cloud, Slack API docs
5. **Review secrets:** Ensure all secrets are set correctly

---

## Security Notes

### Secrets Management

**Never commit:**
- ❌ Webhook URLs
- ❌ API credentials
- ❌ Service account JSON files
- ❌ Access tokens
- ❌ Folder IDs (if private)

**Always use:**
- ✅ GitHub Secrets for CI/CD
- ✅ Environment variables for local development
- ✅ `.gitignore` to exclude sensitive files

### Best Practices

1. **Rotate credentials:** Periodically regenerate service account keys
2. **Minimal permissions:** Use least privilege principle
3. **Monitor access:** Check Google Cloud logs regularly
4. **Audit webhooks:** Review Slack app permissions
5. **Backup secrets:** Store securely in password manager

---

## Summary Checklist

### Slack Setup
- [ ] Created #rl-learning-loop channel
- [ ] Created Slack app "RL Learning Bot"
- [ ] Enabled incoming webhooks
- [ ] Added webhook URL to GitHub secrets
- [ ] Tested with curl command

### Google Drive Setup
- [ ] Created Google Cloud project
- [ ] Enabled Google Drive API
- [ ] Created service account
- [ ] Downloaded credentials JSON
- [ ] Created Google Drive folder
- [ ] Shared folder with service account
- [ ] Added secrets to GitHub
- [ ] Set up NotebookLM notebook
- [ ] Added Drive folder as source

### GitHub Actions
- [ ] Verified workflows are in `.github/workflows/`
- [ ] Confirmed secrets are set
- [ ] Tested workflows manually
- [ ] Verified Actions are enabled

### Local Development
- [ ] Installed Python dependencies
- [ ] Made scripts executable
- [ ] Set up environment variables
- [ ] Tested scripts locally

---

**Setup Complete! 🎉**

Your RL Learning repository is now fully automated. Papers will be fetched daily, progress will be tracked automatically, and weekly summaries can be uploaded to NotebookLM.
