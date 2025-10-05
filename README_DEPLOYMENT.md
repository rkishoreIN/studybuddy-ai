# StudyBuddy AI - Web Deployment Guide

## 🚀 Deploy to Streamlit Cloud

This guide will help you deploy StudyBuddy AI to the web using GitHub and Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account** - [Sign up here](https://github.com)
2. **Streamlit Cloud Account** - [Sign up here](https://share.streamlit.io)
3. **OpenAI API Key** - [Get one here](https://platform.openai.com/api-keys)

## 🔧 Step 1: Create GitHub Repository

### 1.1 Create New Repository
1. Go to [GitHub](https://github.com) and click "New repository"
2. Repository name: `studybuddy-ai` (or your preferred name)
3. Description: "AI-powered study assistant with RAG and FAISS"
4. Make it **Public** (required for free Streamlit Cloud)
5. Check "Add a README file"
6. Click "Create repository"

### 1.2 Upload Your Code
1. **Option A: Upload via GitHub Web Interface**
   - Click "uploading an existing file"
   - Drag and drop these files:
     - `enhanced_studybuddy.py`
     - `streamlit_app.py`
     - `requirements.txt`
     - `.streamlit/config.toml`
     - `README_DEPLOYMENT.md`

2. **Option B: Use Git Commands**
   ```bash
   git clone https://github.com/YOUR_USERNAME/studybuddy-ai.git
   cd studybuddy-ai
   # Copy your files here
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

## 🌐 Step 2: Deploy to Streamlit Cloud

### 2.1 Connect to Streamlit Cloud
1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository: `YOUR_USERNAME/studybuddy-ai`

### 2.2 Configure Deployment
1. **Main file path**: `streamlit_app.py`
2. **App URL**: Choose your custom URL (e.g., `studybuddy-ai`)
3. **Python version**: 3.9 or higher
4. Click "Deploy!"

### 2.3 Set Environment Variables
1. In your Streamlit Cloud dashboard
2. Go to "Settings" → "Secrets"
3. Add these secrets:
   ```toml
   [secrets]
   OPENAI_API_KEY = "your_openai_api_key_here"
   STUDYBUDDY_DATA_DIR = "./studybuddy_data"
   ```

## 🔐 Step 3: Security Configuration

### 3.1 API Key Management
- **You**: Only you can upload study materials (they're stored in your GitHub repo)
- **Users**: Must provide their own OpenAI API key to use the app
- **No shared costs**: Each user pays for their own API usage

### 3.2 Study Materials Storage
- Study materials are stored in your GitHub repository
- Only you can add/remove materials by updating the repository
- Users cannot upload materials (security feature)

## 📁 Step 4: Add Study Materials

### 4.1 Upload Materials to GitHub
1. Go to your GitHub repository
2. Create a folder called `study_materials/`
3. Upload your PDF/text files there
4. The app will automatically detect and process them

### 4.2 Supported File Types
- PDF files (`.pdf`)
- Text files (`.txt`)
- Markdown files (`.md`)

## 🎯 Step 5: Customize Your App

### 5.1 Update App Information
Edit `enhanced_studybuddy.py`:
```python
# Update these lines
st.title(f"🎓 StudyBuddy AI {VERSION}")
st.caption(f"StudyBuddy AI {VERSION} - Built with RAG, FAISS, and Streamlit")
```

### 5.2 Add Your Study Materials
1. Create `study_materials/` folder in your repo
2. Add your PDF/text files
3. Commit and push to GitHub
4. The app will automatically process them

## 🔄 Step 6: Update and Maintain

### 6.1 Adding New Materials
1. Upload new files to `study_materials/` folder in GitHub
2. Push changes to GitHub
3. Streamlit Cloud will automatically redeploy

### 6.2 Updating Code
1. Make changes to your code
2. Commit and push to GitHub
3. Streamlit Cloud will automatically redeploy

## 🌟 Step 7: Share Your App

### 7.1 Get Your App URL
- Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`
- Share this URL with users

### 7.2 User Instructions
Tell your users:
1. Go to your app URL
2. Enter their OpenAI API key
3. Start asking questions!

## 🔧 Troubleshooting

### Common Issues:

1. **App won't start**
   - Check `requirements.txt` has all dependencies
   - Verify `streamlit_app.py` exists

2. **OpenAI API errors**
   - Users must provide their own API key
   - Check API key is valid and has credits

3. **Study materials not loading**
   - Ensure files are in `study_materials/` folder
   - Check file formats are supported

4. **Deployment fails**
   - Check GitHub repository is public
   - Verify all files are committed
   - Check Streamlit Cloud logs

## 📊 Features Available

✅ **RAG System**: Retrieval Augmented Generation with FAISS
✅ **Study Guide Generation**: AI-powered study guides
✅ **Q&A System**: Ask questions about your materials
✅ **Progress Tracking**: Monitor learning progress
✅ **Temperature Control**: Adjust AI creativity
✅ **Smart Fallback**: Uses general knowledge when materials insufficient
✅ **Multi-format Support**: PDF, TXT, MD files
✅ **Cost Tracking**: Monitor API usage and costs

## 🎉 You're Done!

Your StudyBuddy AI is now live on the web! Users can:
- Access your study materials
- Ask questions using their own API keys
- Generate study guides
- Track their learning progress

## 📞 Support

If you need help:
1. Check the troubleshooting section
2. Review Streamlit Cloud logs
3. Check GitHub repository for issues
4. Contact support if needed

---

**Happy Learning! 🎓**
