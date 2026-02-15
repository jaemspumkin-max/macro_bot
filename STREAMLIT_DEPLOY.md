# 🚀 Streamlit Deployment Guide

Complete guide to deploying your Macro Bias Engine with Streamlit Cloud.

---

## 📋 Prerequisites

1. ✅ GitHub repository with your code
2. ✅ Streamlit app file (`streamlit_app.py`)
3. ✅ `requirements.txt` with all dependencies
4. ✅ Free Streamlit Cloud account

---

## 🎯 Step-by-Step Deployment

### Step 1: Update Requirements.txt

Make sure your `requirements.txt` includes Streamlit:

```txt
# Core dependencies
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.28
pandas-datareader>=0.10.0

# Streamlit and visualization
streamlit>=1.28.0
plotly>=5.17.0

# Optional but recommended
requests>=2.31.0
```

### Step 2: Push to GitHub

```bash
# Add the new Streamlit app
git add streamlit_app.py .streamlit/ requirements.txt
git commit -m "Add Streamlit dashboard app"
git push origin main
```

### Step 3: Deploy to Streamlit Cloud (FREE!)

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Click **"Sign up"** (use your GitHub account)

2. **Create New App**
   - Click **"New app"** button
   - Choose your repository: `jaemspumkin-max/macro_bot`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click **"Deploy!"**

3. **Wait for Deployment** (2-3 minutes)
   - Streamlit will install dependencies
   - Build your app
   - Launch it live!

4. **Get Your URL**
   - You'll get a URL like: `https://macro-bias-engine.streamlit.app`
   - Share this URL with anyone!

---

## ✅ Quick Deploy Checklist

Before deploying, verify:

- [ ] `streamlit_app.py` is in repository root
- [ ] `macro_bias_engine.py` is in repository root
- [ ] `requirements.txt` includes `streamlit` and `plotly`
- [ ] Repository is public (or you have Streamlit Teams)
- [ ] No secrets/API keys in code

---

## 🧪 Test Locally First

Before deploying, test on your computer:

```bash
# Install Streamlit
pip install streamlit plotly

# Run the app
streamlit run streamlit_app.py
```

This opens your browser at `http://localhost:8501`

**Expected behavior:**
- Dashboard loads within 30-60 seconds (fetching data)
- All charts display correctly
- Metrics update properly
- No errors in terminal

---

## 🎨 Customize Your App

### Change App Name & URL

In Streamlit Cloud dashboard:
1. Click your app
2. Settings → General
3. Change **App URL** to: `macro-bias-engine` 
4. Result: `https://macro-bias-engine.streamlit.app`

### Add App Icon

Create `favicon.ico` in your repo:
```python
# In streamlit_app.py
st.set_page_config(
    page_title="Macro Bias Engine",
    page_icon="🌍",  # or path to favicon.ico
    layout="wide"
)
```

### Custom Domain (Optional)

In Streamlit Cloud:
1. Settings → Domains
2. Add custom domain (requires DNS setup)

---

## ⚡ Performance Tips

### 1. Enable Caching

Already implemented in `streamlit_app.py`:
```python
@st.cache_data(ttl=900)  # Cache for 15 minutes
def run_analysis():
    # Your analysis code
```

### 2. Optimize Data Fetching

Reduce data fetching frequency:
```python
# In macro_bias_engine.py MacroDataFetcher
def __init__(self, lookback_days=60):  # Reduced from 90
```

### 3. Add Loading Indicators

Already included:
```python
with st.spinner("🔄 Analyzing macro factors..."):
    results = run_analysis()
```

---

## 🔒 Security & Secrets

### If You Need API Keys (Optional)

Create `.streamlit/secrets.toml` locally:
```toml
# Don't commit this file!
[api_keys]
fred_api_key = "your-key-here"
alpha_vantage_key = "your-key-here"
```

Add to `.gitignore`:
```
.streamlit/secrets.toml
```

In Streamlit Cloud:
1. App Settings → Secrets
2. Paste your secrets in TOML format

Access in code:
```python
import streamlit as st
api_key = st.secrets["api_keys"]["fred_api_key"]
```

---

## 📊 Monitoring Your App

### View Logs

In Streamlit Cloud:
1. Click your app
2. Click **"Manage app"** → **"Logs"**
3. See real-time logs and errors

### Check Analytics

Streamlit provides:
- Number of viewers
- Usage patterns
- Error rates

Access: App → Analytics

### Set Up Alerts

Get notified when app goes down:
1. Settings → Notifications
2. Add your email
3. Get alerts for crashes

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"

**Problem:** Missing dependency

**Solution:** 
```bash
# Add to requirements.txt
missing-package>=version

# Then push to GitHub
git add requirements.txt
git commit -m "Add missing dependency"
git push
```

Streamlit auto-redeploys on push!

### "App Not Loading"

**Check:**
1. Logs for errors
2. `requirements.txt` has all packages
3. No syntax errors in `streamlit_app.py`

**Fix:**
```bash
# Test locally first
streamlit run streamlit_app.py

# If works locally, check Streamlit Cloud logs
```

### "Memory Limit Exceeded"

Free tier has 1GB RAM limit.

**Solutions:**
1. Reduce `lookback_days` in data fetcher
2. Use more aggressive caching
3. Upgrade to Streamlit Teams ($25/month)

### "Data Fetching Fails"

**Problem:** FRED or Yahoo Finance timeout

**Solution:** Add retry logic:
```python
import time

def fetch_with_retry(func, retries=3):
    for i in range(retries):
        try:
            return func()
        except:
            if i < retries - 1:
                time.sleep(2)
                continue
            raise
```

---

## 🔄 Updating Your App

After making changes:

```bash
# Make changes to streamlit_app.py
# Test locally
streamlit run streamlit_app.py

# Push to GitHub
git add .
git commit -m "Update: Improved dashboard layout"
git push

# Streamlit Cloud auto-redeploys (takes 1-2 minutes)
```

---

## 🌟 Make Your App Stand Out

### 1. Add README Badge

In your GitHub README.md:
```markdown
## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://macro-bias-engine.streamlit.app)

Try the live dashboard: **[macro-bias-engine.streamlit.app](https://macro-bias-engine.streamlit.app)**
```

### 2. Add Screenshots

Take screenshots and add to README:
```markdown
## 📸 Dashboard Preview

![Dashboard](screenshots/dashboard.png)
```

### 3. Create Demo Video

Record a 30-second demo:
1. Use Loom or OBS
2. Upload to YouTube
3. Embed in README

---

## 📱 Mobile Optimization

Streamlit apps work on mobile! But optimize:

```python
# Responsive layout
st.set_page_config(layout="wide")

# Mobile-friendly columns
cols = st.columns([1, 1] if st.session_state.get('mobile') else [2, 1])
```

---

## 💰 Cost Breakdown

**Streamlit Cloud (Free Tier):**
- ✅ Unlimited public apps
- ✅ 1GB RAM per app
- ✅ Auto-scaling
- ✅ SSL certificate
- ✅ Custom domains
- ❌ Private apps (Teams only)

**Streamlit Teams ($250/month):**
- Everything in Free
- ✅ Unlimited private apps
- ✅ 4GB RAM
- ✅ Priority support

**Recommendation:** Start with free tier!

---

## 📚 Additional Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Community Forum:** https://discuss.streamlit.io
- **Gallery:** https://streamlit.io/gallery
- **Blog:** https://blog.streamlit.io

---

## 🎯 Next Steps After Deployment

1. **Share Your App**
   - Post on Twitter/LinkedIn
   - Share on Reddit (r/Python, r/algotrading)
   - Submit to Streamlit Gallery

2. **Gather Feedback**
   - Add feedback form in sidebar
   - Monitor usage patterns
   - Iterate based on user needs

3. **Add Features**
   - Historical bias tracking
   - Email alerts
   - Export to PDF
   - Factor comparison over time

4. **Monetize (Optional)**
   - Premium features
   - API access
   - Consulting services

---

## 🎉 You're Ready!

Your deployment process:
1. Push `streamlit_app.py` to GitHub ✅
2. Go to share.streamlit.io ✅
3. Deploy in 3 clicks ✅
4. Get live URL ✅
5. Share with the world! 🚀

**Your app will be live at:**
`https://macro-bias-engine.streamlit.app`

---

## 📧 Need Help?

- Streamlit Community: https://discuss.streamlit.io
- GitHub Issues: Open an issue in your repo
- Twitter: Tweet @streamlit for support

Good luck! 🚀
