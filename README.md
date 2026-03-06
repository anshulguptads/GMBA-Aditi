# 🏦 Universal Bank — Customer Analytics Dashboard

End-to-end analytics dashboard for predicting personal loan acceptance and prescribing personalised offers.

## Features

### 📊 Descriptive Analytics
- Summary statistics (mean, median, std, min, max)
- Histograms with KDE for all numeric columns
- Pie/Bar charts for categorical & binary columns
- Product penetration rates
- Geographic distribution (ZIP code regions)
- Cross-tabulations (Personal Loan vs all features)
- Customer segmentation summaries

### 🔍 Diagnostic Analytics
- Correlation heatmap
- Acceptors vs Non-Acceptors comparison (box plots + mean table)
- Segment deep dive (interactive)
- Feature relationship explorer (scatter plots)
- Feature importance from trained models

### 🤖 Predictive Analytics
- **4 Classification Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Full Metrics Table**: Train & Test Accuracy, Precision, Recall, F1-Score, AUC
- **Combined ROC Curve**: All models plotted together
- **Confusion Matrices**: Visual for all 4 models
- **Overfit Check**: Train vs Test accuracy gap analysis
- **Interactive Prediction Calculator**: Enter any customer profile → get prediction + personalised offer

### 💡 Prescriptive Analytics
- Customer scoring & ranking with confidence tiers
- Personalised loan offers (Premium, Preferred, Family Flex, Digital Quick, Smart Start)
- Target segment recommendations with treemap
- Campaign strategy suggestions (5 strategies)
- What-If scenario analysis (threshold sensitivity)

### 📥 Data Export
- Download full predictions CSV
- Download campaign target list with offers
- Download model performance report

## Deployment on Streamlit Cloud

1. **Push to GitHub**: Upload these 3 files to a GitHub repo:
   - `app.py`
   - `requirements.txt`
   - `UniversalBank.csv`

2. **Go to** [share.streamlit.io](https://share.streamlit.io)

3. **Click** "New app" → Connect your GitHub repo

4. **Set**:
   - Repository: `your-username/your-repo-name`
   - Branch: `main`
   - Main file path: `app.py`

5. **Click Deploy** — your dashboard will be live in ~2 minutes!

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
- **Frontend**: Streamlit + Plotly
- **ML**: Scikit-learn (LogisticRegression, DecisionTree, RandomForest, GradientBoosting)
- **Data**: Pandas, NumPy, SciPy
