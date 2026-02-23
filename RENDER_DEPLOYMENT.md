# Render Deployment Guide

## 🚀 Deploy Your NSE Stock Analysis System on Render

This guide will help you deploy your complete system on Render with PostgreSQL database and automated daily updates.

## 📋 Prerequisites

1. GitHub account with your code pushed
2. Render account (sign up at https://render.com)
3. Your repository: `https://github.com/NaishaH173/nse-stock-analysis`

## 🎯 What Will Be Deployed

### 1. PostgreSQL Database
- Free tier PostgreSQL instance
- Stores all stock data and indicators
- Automatic backups

### 2. Web Service (FastAPI App)
- Your dashboard and API
- Accessible via public URL
- Auto-deploys on git push

### 3. Cron Job (Daily Automation)
- Runs `run_daily.py` every day at 6:30 PM IST
- Downloads NSE data automatically
- Updates indicators and signals

## 📝 Step-by-Step Deployment

### Step 1: Push to GitHub

Make sure your code is pushed to GitHub:

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### Step 2: Create Render Account

1. Go to https://render.com
2. Click "Get Started"
3. Sign up with GitHub
4. Authorize Render to access your repositories

### Step 3: Deploy Using Blueprint (Easiest Method)

1. Go to Render Dashboard
2. Click "New" → "Blueprint"
3. Connect your GitHub repository: `nse-stock-analysis`
4. Render will detect `render.yaml` automatically
5. Click "Apply"

Render will create:
- ✅ PostgreSQL database (`nse-postgres`)
- ✅ Web service (`nse-stock-analysis-web`)
- ✅ Cron job (`nse-daily-update`)

### Step 4: Wait for Deployment

- Database: ~2-3 minutes
- Web service: ~5-7 minutes (first time)
- Cron job: ~2-3 minutes

### Step 5: Setup Database Schema

After deployment, you need to create tables:

1. Go to your database in Render dashboard
2. Click "Connect" → Copy the External Database URL
3. Use a PostgreSQL client (pgAdmin, DBeaver, or psql) to connect
4. Run your database setup script to create tables

**OR** create a one-time job:

1. In Render dashboard, go to your web service
2. Click "Shell" tab
3. Run: `python database/setup_optimized_architecture_fast.py`

### Step 6: Verify Deployment

1. Click on your web service
2. Copy the URL (e.g., `https://nse-stock-analysis-web.onrender.com`)
3. Open in browser
4. You should see your dashboard!

## 🔧 Configuration

### Environment Variables (Auto-configured)

These are automatically set by Render from `render.yaml`:

- `DB_HOST` - Database host
- `DB_PORT` - Database port (5432)
- `DB_NAME` - Database name
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password
- `PYTHON_VERSION` - Python version (3.11.0)

### Cron Schedule

The daily job runs at **1:00 PM UTC** (6:30 PM IST):
- Schedule: `0 13 * * *`
- Runs: `python run_daily.py`
- Downloads NSE data and updates indicators

To change the schedule, edit `render.yaml`:
```yaml
schedule: "0 13 * * *"  # Change this line
```

## 📊 Render Free Tier Limits

### PostgreSQL Database
- Storage: 1 GB
- Connections: 97
- Backups: 90 days retention
- **Expires after 90 days** (need to upgrade or migrate)

### Web Service
- RAM: 512 MB
- CPU: Shared
- Bandwidth: 100 GB/month
- Auto-sleep after 15 min inactivity
- Cold start: ~30 seconds

### Cron Jobs
- Runs: Once per schedule
- Timeout: 15 minutes max
- RAM: 512 MB

## 🔄 Automatic Deployments

Every time you push to GitHub:
```bash
git add .
git commit -m "Your changes"
git push origin main
```

Render automatically:
1. Detects the push
2. Rebuilds your app
3. Deploys new version
4. Zero downtime deployment

## 🐛 Troubleshooting

### Build Failed

Check build logs in Render dashboard:
1. Go to your service
2. Click "Logs" tab
3. Look for error messages

Common issues:
- Missing dependencies in `requirements.txt`
- Python version mismatch
- Database connection errors

### Database Connection Failed

1. Check environment variables are set
2. Verify database is running
3. Check database URL format
4. Ensure `config.py` uses environment variables

### Cron Job Not Running

1. Check cron job logs in Render dashboard
2. Verify schedule format
3. Check if job timeout (15 min max)
4. Ensure database connection works

### App Sleeping (Cold Starts)

Free tier apps sleep after 15 minutes of inactivity:
- First request takes ~30 seconds (cold start)
- Subsequent requests are fast

Solutions:
- Upgrade to paid plan ($7/month for always-on)
- Use external uptime monitor (UptimeRobot) to ping every 14 minutes

## 🔐 Security

### Database Security
- Database is private by default
- Only accessible from Render services
- Use External URL for external access
- Credentials auto-generated and secure

### Environment Variables
- Never commit real credentials to git
- Use Render's environment variables
- Automatically injected at runtime

## 📈 Monitoring

### View Logs
1. Go to service in Render dashboard
2. Click "Logs" tab
3. Real-time log streaming

### Metrics
1. Go to service
2. Click "Metrics" tab
3. View CPU, memory, bandwidth usage

### Database Metrics
1. Go to database
2. View connections, storage, queries

## 💰 Cost Estimate

### Free Tier (Current Setup)
- Database: Free (90 days)
- Web Service: Free
- Cron Job: Free
- **Total: $0/month**

### After 90 Days (Database Expires)
- Database: $7/month (Starter plan)
- Web Service: Free or $7/month (for always-on)
- Cron Job: Free
- **Total: $7-14/month**

## 🚀 Going Live Checklist

- [ ] Code pushed to GitHub
- [ ] Render account created
- [ ] Blueprint deployed
- [ ] Database schema created
- [ ] Web service accessible
- [ ] Cron job scheduled
- [ ] Test dashboard functionality
- [ ] Test API endpoints
- [ ] Verify daily automation works
- [ ] Set up monitoring/alerts

## 🔗 Useful Links

- Render Dashboard: https://dashboard.render.com
- Render Docs: https://render.com/docs
- PostgreSQL Docs: https://render.com/docs/databases
- Cron Jobs Docs: https://render.com/docs/cronjobs

## 📞 Support

If you encounter issues:
1. Check Render documentation
2. Review service logs
3. Check Render community forum
4. Contact Render support (paid plans)

## 🎉 Success!

Once deployed, your system will be live at:
```
https://nse-stock-analysis-web.onrender.com
```

Share this URL with anyone to access your dashboard!

---

**Deployment Status**: Ready to deploy! 🚀

Follow the steps above to get your system live on Render.
