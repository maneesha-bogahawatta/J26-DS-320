# Git Cheat Sheet for Our Team

## The 5 Commands You'll Use Every Day

```bash
# 1. Check what you've changed
git status

# 2. Save your work
git add .
git commit -m "[C4] Your message here"

# 3. Upload to GitHub
git push origin your-branch-name

# 4. Download latest team changes
git pull origin develop

# 5. Switch between branches
git checkout member4/critical-incidents
```

## Our Branch Structure

```
main ──────────────────────── Protected (final tested code only)
  └── develop ─────────────── Integration branch (merge here first)
       ├── member1/spatial
       ├── member2/loitering
       ├── member3/aggression
       ├── member4/critical-incidents
       └── shared/perception
```

## Step-by-Step: Start of Day

```bash
# 1. Make sure you're on your branch
git checkout member4/critical-incidents

# 2. Get latest from develop
git pull origin develop

# 3. Merge develop into your branch (get teammates' changes)
git merge develop

# 4. Start working!
```

## Step-by-Step: End of Day

```bash
# 1. Check what you changed
git status

# 2. Stage everything
git add .

# 3. Commit with a clear message
git commit -m "[C4] Train fall CNN-LSTM model for 50 epochs"

# 4. Push to GitHub
git push origin member4/critical-incidents
```

## Step-by-Step: Share Your Work With Team

```bash
# 1. Push your branch
git push origin member4/critical-incidents

# 2. Go to GitHub.com
# 3. Click "Compare & pull request"
# 4. Set base: develop
# 5. Write what you did
# 6. Click "Create pull request"
# 7. Ask a teammate to review
# 8. After approval → Merge
```

## Common Problems & Fixes

### "I accidentally edited the wrong branch"
```bash
# Save your changes temporarily
git stash

# Switch to correct branch
git checkout member4/critical-incidents

# Apply your changes here
git stash pop
```

### "I want to undo my last commit"
```bash
# Undo commit but keep the files changed
git reset --soft HEAD~1
```

### "I have a merge conflict"
```bash
# Open the conflicting file
# Look for <<<<<<< and >>>>>>> markers
# Keep the correct code, delete the markers
# Then:
git add .
git commit -m "[FIX] Resolve conflict"
```

### "I want to see what changed"
```bash
# See changes not yet staged
git diff

# See commit history
git log --oneline -10
```

## Commit Message Examples

```
[C4] Add Le2i dataset download script
[C4] Implement bounding-box fall baseline
[C4] Train CNN-LSTM model — 87% F1 on validation
[C4] Add robustness test under low resolution
[C4] Fix false alarm on sitting-down action
[SHARED] Update event schema — add evidence_frame field
[DATA] Annotate 50 fall videos from URFD dataset
[DOCS] Update Component 4 README with evaluation results
[FIX] Fix ByteTrack crash on empty detection frame
[TEST] Add unit test for owner-object association
```

## Golden Rules

1. **Commit often** — small commits are easier to understand and fix
2. **Pull before push** — always get latest changes before uploading yours
3. **Never force push** — `git push --force` destroys teammates' work
4. **Write clear messages** — future you will thank present you
5. **Ask for help early** — a 5-minute question beats a 2-hour struggle
