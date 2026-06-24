# 🎬 FunHub — AI Movie Recommendation Dashboard

> Dark cinematic AI-powered movie discovery platform built with Streamlit.

## ✨ Features
- 🤖 **ML Recommendations** — Collaborative + Content-Based hybrid engine
- 📊 **Visual Analytics** — Interactive Plotly charts, bento grid layout
- 🔍 **Smart Search** — Filter by genre, year, rating, mood
- 📚 **Personal Library** — Watchlist, favourites, reviews, ratings
- 🎨 **Dark Cinematic UI** — Glassmorphism + Spatial + Bento design

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/funhub.git
cd funhub
pip install -r requirements.txt
streamlit run app.py
```

🌐 **Demo Login**
- Username: `cinephile`
- Password: `funhub2024`

🎨 **Design System**
- Palette: Christmas Mulled Wine + Vanilla
- Typography: Cinzel Decorative · Playfair Display · Inter · JetBrains Mono
- Style: Glassmorphism · Spatial Design · Bento Grid

📁 **Project Structure**
```
funhub/
├── app.py              # Main Streamlit app
├── assets/style.css    # Global cinematic CSS
├── data/               # MovieLens dataset
├── models/             # ML recommendation engines
├── utils/              # Data loaders + analytics helpers
├── requirements.txt
└── README.md
```

🛠 **Tech Stack**
Python · Streamlit · Plotly · Pandas · NumPy · Scikit-Learn · Scikit-Surprise

🚢 **Streamlit Cloud Deployment**
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo → set main file to `app.py`
4. Deploy

⚠️ **Notes**
- Recommendation algorithms use the optimized KNN model to run instantly with low memory footprints.
