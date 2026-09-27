# Team Setup Guide

Follow these steps **exactly** to get the project running on your machine.

## Step 1: Install Git

**Windows:**
Download from https://git-scm.com/download/win
During installation, select "Git from the command line and also from 3rd-party software"

**Mac:**
```bash
brew install git
```

**Linux:**
```bash
sudo apt update && sudo apt install git
```

After installing, configure your identity:
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@sliit.lk"
```

## Step 2: Clone the Repository

```bash
git clone https://github.com/<<your-org>>/proactive-anomaly-detection.git
cd proactive-anomaly-detection
```

## Step 3: Create Your Branch

**Member 1:**
```bash
git checkout -b member1/spatial
```

**Member 2:**
```bash
git checkout -b member2/loitering
```

**Member 3:**
```bash
git checkout -b member3/aggression
```

**Member 4:**
```bash
git checkout -b member4/critical-incidents
```

## Step 4: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate          # Mac / Linux
venv\Scripts\activate             # Windows

# Install dependencies
pip install -r requirements.txt
```

## Step 5: Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and set your local paths:
```
DATA_DIR=/home/yourname/datasets
CHECKPOINT_DIR=/home/yourname/checkpoints
LOG_DIR=/home/yourname/logs
```

## Step 6: Verify Installation

```bash
# Test Python imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"
python -c "import mediapipe; print('MediaPipe: OK')"

# Test shared modules
python shared/schemas/event_schema.py
python shared/utils/video_io.py sample_video.mp4
```

## Step 7: Create Dataset Directories

```bash
mkdir -p datasets/raw datasets/processed datasets/annotations
```

These directories are in `.gitignore` — your datasets stay local,
not uploaded to GitHub (they are too large).

## Daily Workflow

### Morning: Pull latest changes
```bash
git checkout develop
git pull origin develop
git checkout your-branch
git merge develop
```

### Working: Save your progress
```bash
git add .
git commit -m "[C4] Add fall baseline evaluation script"
git push origin your-branch
```

### When ready: Open a Pull Request
1. Go to GitHub
2. Click "Compare & pull request" for your branch
3. Set base branch to `develop` (NOT main)
4. Write a description of what you changed
5. Request review from at least 1 teammate
6. After approval, merge

### If you get a merge conflict
```bash
# Git will tell you which files conflict
# Open the file — look for these markers:
<<<<<<< HEAD
your version of the code
=======
their version of the code
>>>>>>> develop

# Keep the correct version, delete the markers, then:
git add .
git commit -m "[FIX] Resolve merge conflict in event_schema.py"
```

## Storing Large Files

Datasets and model checkpoints are **too large for GitHub**.
Store them on:
- Google Drive (shared folder with team)
- OneDrive (SLIIT account)
- University file server

Then set the path in your `.env` file.

## Getting Help

- **Git problems:** Ask in the team chat, or check https://ohshitgit.com
- **Python errors:** Share the full error message in team chat
- **Model issues:** Discuss in daily standup
