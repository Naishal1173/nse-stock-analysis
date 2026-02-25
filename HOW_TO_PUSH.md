# How to Push to GitHub

## Simple One-Command Push

Just run this command to push all your changes to GitHub:

```bash
python git_push.py
```

That's it! The script will:
1. ✅ Check for changes
2. ✅ Show what will be committed
3. ✅ Add all changes
4. ✅ Commit with timestamp
5. ✅ Push to GitHub

## Example Output

```
==================================================
  AUTO GIT PUSH - NSE Stock Analysis
==================================================

📊 Checking for changes...
📝 Files to be committed:
 M app/api.py
 M static/script.js

✅ All changes staged
✅ Changes committed
🚀 Pushing to GitHub...

==================================================
  ✅ SUCCESS! All changes pushed to GitHub
==================================================
```

## Daily Workflow

```bash
# 1. Make your code changes
# 2. Run:
python git_push.py
```

## First Time Setup (if needed)

```bash
# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Troubleshooting

### If push fails:
```bash
# Pull latest changes first
git pull origin main

# Then push again
python git_push.py
```

### Check status manually:
```bash
git status
```

### View commit history:
```bash
git log --oneline -10
```

That's all you need to know! 🚀
