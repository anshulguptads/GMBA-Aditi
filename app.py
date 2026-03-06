import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from scipy.stats import gaussian_kde
import io, warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Universal Bank · Analytics Dashboard", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    .stApp { font-family: 'DM Sans', sans-serif; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label { font-weight: 500; color: #64748b !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 700; color: #0f172a !important; }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #0f172a;
        border-bottom: 3px solid #3b82f6; padding-bottom: 8px; margin-bottom: 4px; }
    .analytics-badge { display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 12px; }
    .badge-desc { background: #dbeafe; color: #1e40af; }
    .badge-diag { background: #fef3c7; color: #92400e; }
    .badge-pred { background: #dcfce7; color: #166534; }
    .badge-presc { background: #f3e8ff; color: #6b21a8; }
    .offer-card { background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #f0f9ff 100%);
        border: 1px solid #7dd3fc; border-radius: 14px; padding: 20px 24px;
        margin: 10px 0; box-shadow: 0 2px 8px rgba(56,189,248,0.1); }
    .offer-card h4 { margin: 0 0 6px 0; color: #0c4a6e; font-size: 1.05rem; }
    .offer-card p  { margin: 2px 0; color: #334155; font-size: 0.92rem; }
    .offer-tag { display: inline-block; background: #0ea5e9; color: white;
        padding: 2px 10px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
    .strategy-card { background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #c4b5fd; border-radius: 14px; padding: 18px 22px; margin: 8px 0; }
    .strategy-card h4 { margin: 0 0 4px 0; color: #5b21b6; }
    .strategy-card p  { margin: 2px 0; color: #334155; font-size: 0.9rem; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("UniversalBank.csv")
    df['Experience'] = df['Experience'].clip(lower=0)
    df['Education_Label'] = df['Education'].map({1: 'Undergrad', 2: 'Graduate', 3: 'Advanced/Professional'})
    df['Income_Bracket'] = pd.cut(df['Income'], bins=[0,50,100,150,250],
                                  labels=['Low (<50k)','Mid (50-100k)','High (100-150k)','Very High (>150k)'])
    df['Age_Group'] = pd.cut(df['Age'], bins=[20,30,40,50,60,70],
                             labels=['23-30','31-40','41-50','51-60','61-67'])
    df['CCAvg_Bracket'] = pd.cut(df['CCAvg'], bins=[-0.1,2,4,6,10],
                                 labels=['Low (<2k)','Mid (2-4k)','High (4-6k)','Very High (>6k)'])
    df['ZIP_Region'] = (df['ZIP Code'] // 100).astype(str)
    return df

df = load_data()
FEATURES = ['Age','Experience','Income','Family','CCAvg','Education','Mortgage',
            'Securities Account','CD Account','Online','CreditCard']
TARGET = 'Personal Loan'
NUMERIC = ['Age','Experience','Income','CCAvg','Mortgage']
BINARY  = ['Securities Account','CD Account','Online','CreditCard','Personal Loan']

# ─────────────────────────────────────────────────────────
# MODEL TRAINING — BOTH train & test metrics
# ─────────────────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    X = _df[FEATURES]; y = _df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    algos = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    }

    res = {}
    for name, clf in algos.items():
        scaled = name == 'Logistic Regression'
        Xtr = X_train_sc if scaled else X_train
        Xte = X_test_sc  if scaled else X_test
        clf.fit(Xtr, y_train)

        ytr_pred  = clf.predict(Xtr)
        ytr_proba = clf.predict_proba(Xtr)[:, 1]
        yte_pred  = clf.predict(Xte)
        yte_proba = clf.predict_proba(Xte)[:, 1]

        res[name] = {
            'model': clf,
            'train_accuracy':  accuracy_score(y_train, ytr_pred),
            'train_precision': precision_score(y_train, ytr_pred, zero_division=0),
            'train_recall':    recall_score(y_train, ytr_pred, zero_division=0),
            'train_f1':        f1_score(y_train, ytr_pred, zero_division=0),
            'train_auc':       roc_auc_score(y_train, ytr_proba),
            'test_accuracy':   accuracy_score(y_test, yte_pred),
            'test_precision':  precision_score(y_test, yte_pred, zero_division=0),
            'test_recall':     recall_score(y_test, yte_pred, zero_division=0),
            'test_f1':         f1_score(y_test, yte_pred, zero_division=0),
            'test_auc':        roc_auc_score(y_test, yte_proba),
            'y_test_pred': yte_pred, 'y_test_proba': yte_proba,
            'y_train_proba': ytr_proba,
            'cm': confusion_matrix(y_test, yte_pred),
        }
        if hasattr(clf, 'feature_importances_'):
            res[name]['importance'] = dict(zip(FEATURES, clf.feature_importances_))
        elif hasattr(clf, 'coef_'):
            res[name]['importance'] = dict(zip(FEATURES, np.abs(clf.coef_[0])))

    return res, scaler, X_train, X_test, y_train, y_test

results, scaler, X_train, X_test, y_train, y_test = train_models(df)
best_name  = max(results, key=lambda k: results[k]['test_auc'])
best_model = results[best_name]['model']

X_all = df[FEATURES]
X_sc  = scaler.transform(X_all) if best_name == 'Logistic Regression' else X_all
df['Loan_Probability']    = best_model.predict_proba(X_sc)[:, 1]
df['Predicted_Interested'] = (df['Loan_Probability'] >= 0.5).astype(int)

# ─────────────────────────────────────────────────────────
# OFFER ENGINE
# ─────────────────────────────────────────────────────────
def generate_offer(row, prob):
    inc, edu, fam = row['Income'], row['Education'], row['Family']
    ccavg, mort, cd, online = row['CCAvg'], row['Mortgage'], row['CD Account'], row['Online']
    conf = "High" if prob > 0.75 else "Medium" if prob > 0.5 else "Moderate"
    if inc >= 150:   base = 8.5
    elif inc >= 100: base = 9.5
    elif inc >= 50:  base = 10.5
    else:            base = 11.5
    if cd == 1:  base -= 0.5
    if edu == 3: base -= 0.25
    loan_amt = round(min(inc * 3.5, 500) * (0.6 if fam <= 2 else 0.8), 0)
    tenure  = "36-60 mo" if inc >= 100 else "24-48 mo" if inc >= 50 else "12-36 mo"
    channel = "Mobile App + Email" if online == 1 else "Branch + Direct Mail"
    if inc >= 120 and edu >= 2:
        otype, hook = "💎 Premium Personal Loan", "Pre-approved premium loan — priority processing, zero documentation."
    elif cd == 1 or (inc >= 80 and ccavg >= 3):
        otype, hook = "⭐ Preferred Customer Loan", "Special loyalty rate applied automatically for valued customers."
    elif fam >= 3 and mort > 0:
        otype, hook = "🏠 Family Flex Loan", "Flexible step-up EMI designed for families."
    elif ccavg >= 2 and online == 1:
        otype, hook = "📱 Digital Quick Loan", "Instant digital disbursement — apply in 2 min, funds in 4 hours."
    else:
        otype, hook = "✅ Smart Start Loan", "First personal loan with simple terms and a dedicated RM."
    return dict(offer_type=otype, hook=hook, rate=f"{base:.2f}%", max_amount=f"₹{loan_amt:.0f}k",
                tenure=tenure, channel=channel, confidence=conf, probability=prob)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Universal Bank")
    st.markdown("**Full Analytics Suite**")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠 Overview","📊 Descriptive Analytics","🔍 Diagnostic Analytics",
                                  "🤖 Predictive Analytics","💡 Prescriptive Analytics","📥 Data Export"])
    st.markdown("---")
    st.markdown(f"**Rows**: {len(df):,} · **Features**: {len(FEATURES)}")
    st.markdown(f"**Loan Rate**: {df[TARGET].mean()*100:.1f}%")
    st.markdown(f"**Best Model**: {best_name}")
    st.markdown(f"**Test AUC**: {results[best_name]['test_auc']:.4f}")

# ═══════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">Universal Bank · Customer Analytics Dashboard</p>', unsafe_allow_html=True)
    st.markdown("End-to-end analytics — from understanding customers to prescribing personalised loan offers.")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Loan Acceptors", f"{df[TARGET].sum():,}")
    c3.metric("Accept Rate", f"{df[TARGET].mean()*100:.1f}%")
    c4.metric("Avg Income", f"${df['Income'].mean():,.0f}k")
    c5.metric("Avg CC Spend", f"${df['CCAvg'].mean():,.1f}k/mo")
    c6.metric("Predicted Targets", f"{df['Predicted_Interested'].sum():,}")
    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<span class="analytics-badge badge-desc">DESCRIPTIVE</span>', unsafe_allow_html=True)
        st.markdown("**What happened?** — Summary stats, distributions with KDE, product penetration, geographic patterns, cross-tabulations, customer segmentation.")
        st.markdown("")
        st.markdown('<span class="analytics-badge badge-diag">DIAGNOSTIC</span>', unsafe_allow_html=True)
        st.markdown("**Why did it happen?** — Correlation heatmap, acceptor vs non-acceptor comparison, segment deep-dive, feature relationship explorer.")
    with col_b:
        st.markdown('<span class="analytics-badge badge-pred">PREDICTIVE</span>', unsafe_allow_html=True)
        st.markdown("**What will happen?** — 4 classifiers with full train/test metrics table, combined ROC curves, confusion matrices, interactive prediction calculator.")
        st.markdown("")
        st.markdown('<span class="analytics-badge badge-presc">PRESCRIPTIVE</span>', unsafe_allow_html=True)
        st.markdown("**What should we do?** — Customer scoring, personalised offers, target segments, campaign strategies, what-if scenario analysis.")
    st.markdown("---")

    fig = px.histogram(df, x='Income', color=df[TARGET].map({0:'Declined',1:'Accepted'}),
                       nbins=40, barmode='overlay', opacity=0.75,
                       color_discrete_map={'Declined':'#94a3b8','Accepted':'#3b82f6'},
                       labels={'color':'Personal Loan'}, title="Income Distribution by Loan Acceptance")
    fig.update_layout(template='plotly_white', font_family='DM Sans', height=380,
                      legend=dict(orientation='h', yanchor='top', y=1.12, x=0.5, xanchor='center'))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  DESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════
elif page == "📊 Descriptive Analytics":
    st.markdown('<p class="main-title">Descriptive Analytics</p>', unsafe_allow_html=True)
    st.markdown('<span class="analytics-badge badge-desc">WHAT HAPPENED?</span>', unsafe_allow_html=True)

    tabs = st.tabs(["Summary Statistics","Distributions (Histograms + KDE)",
                     "Categorical Breakdown","Product Penetration",
                     "Geographic Distribution","Cross-Tabulations","Customer Segmentation"])

    # ── 1. Summary Statistics ──
    with tabs[0]:
        st.markdown("#### Summary Statistics — All Numeric Columns")
        all_num = NUMERIC + ['Family','Education']
        summary = df[all_num].describe().T
        summary['median'] = df[all_num].median()
        summary = summary[['count','mean','median','std','min','25%','50%','75%','max']].round(2)
        st.dataframe(summary.style.format("{:.2f}").background_gradient(cmap='Blues', subset=['mean','std']),
                     use_container_width=True)
        st.markdown("#### Loan Acceptance Rate")
        la1, la2, la3 = st.columns(3)
        la1.metric("Overall Acceptance", f"{df[TARGET].mean()*100:.2f}%")
        la2.metric("Total Accepted", f"{df[TARGET].sum():,}")
        la3.metric("Total Declined", f"{(df[TARGET]==0).sum():,}")

    # ── 2. Distributions with KDE ──
    with tabs[1]:
        st.markdown("#### Histograms with KDE for Numeric Columns")
        for i in range(0, len(NUMERIC), 2):
            cols = st.columns(2)
            for j, col_name in enumerate(NUMERIC[i:i+2]):
                with cols[j]:
                    data = df[col_name].dropna()
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=data, nbinsx=40, name='Count',
                                               marker_color='#3b82f6', opacity=0.6,
                                               histnorm='probability density'))
                    kde_x = np.linspace(data.min(), data.max(), 300)
                    kde_y = gaussian_kde(data)(kde_x)
                    fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines',
                                             name='KDE', line=dict(color='#ef4444', width=2.5)))
                    fig.update_layout(template='plotly_white', font_family='DM Sans', height=340,
                                      title=f"{col_name} Distribution", showlegend=True,
                                      legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
                    st.plotly_chart(fig, use_container_width=True)

    # ── 3. Categorical Breakdown ──
    with tabs[2]:
        st.markdown("#### Pie / Bar Charts for Categorical & Binary Columns")
        c1, c2 = st.columns(2)
        with c1:
            edu_ct = df['Education_Label'].value_counts().reset_index()
            edu_ct.columns = ['Education','Count']
            fig = px.pie(edu_ct, names='Education', values='Count', hole=0.45,
                         title="Education Level", color_discrete_sequence=['#3b82f6','#f59e0b','#10b981'])
            fig.update_layout(template='plotly_white', font_family='DM Sans', height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fam_ct = df['Family'].value_counts().sort_index().reset_index()
            fam_ct.columns = ['Family Size','Count']
            fig = px.bar(fam_ct, x='Family Size', y='Count', text_auto=True,
                         title="Family Size", color_discrete_sequence=['#8b5cf6'])
            fig.update_layout(template='plotly_white', font_family='DM Sans', height=350)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Binary Columns")
        bin_data = []
        for b in BINARY:
            bin_data.append({'Column': b, 'Yes (1)': int(df[b].sum()),
                             'No (0)': int((df[b]==0).sum()), 'Yes %': f"{df[b].mean()*100:.1f}%"})
        st.dataframe(pd.DataFrame(bin_data), use_container_width=True, hide_index=True)

        melted_bin = pd.DataFrame({b: [df[b].sum(), (df[b]==0).sum()] for b in BINARY},
                                   index=['Yes','No']).T.reset_index()
        melted_bin.columns = ['Column','Yes','No']
        fig = px.bar(melted_bin, x='Column', y=['Yes','No'], barmode='group',
                     color_discrete_map={'Yes':'#3b82f6','No':'#cbd5e1'}, title="Binary Column Counts")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ── 4. Product Penetration ──
    with tabs[3]:
        products = {'Personal Loan': df['Personal Loan'].mean()*100,
                    'Securities Acct': df['Securities Account'].mean()*100,
                    'CD Account': df['CD Account'].mean()*100,
                    'Online Banking': df['Online'].mean()*100,
                    'Credit Card': df['CreditCard'].mean()*100}
        prod_df = pd.DataFrame(list(products.items()), columns=['Product','Adoption %']).sort_values('Adoption %')
        fig = px.bar(prod_df, y='Product', x='Adoption %', orientation='h', text_auto='.1f',
                     color='Adoption %', color_continuous_scale='blues', title="Product Adoption Rates")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Loan Acceptance vs Number of Products Held")
        df['Num_Products'] = df[['Securities Account','CD Account','Online','CreditCard']].sum(axis=1)
        pl = df.groupby('Num_Products')[TARGET].agg(['mean','count']).reset_index()
        pl.columns = ['Products Held','Accept Rate','Count']; pl['Accept Rate'] *= 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pl['Products Held'], y=pl['Count'], name='Customers', marker_color='#cbd5e1'), secondary_y=False)
        fig.add_trace(go.Scatter(x=pl['Products Held'], y=pl['Accept Rate'], name='Accept %',
                                 mode='lines+markers', marker=dict(size=10, color='#3b82f6'),
                                 line=dict(width=3, color='#3b82f6')), secondary_y=True)
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=380, title="Loan Acceptance by Products Held",
                          legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
        fig.update_yaxes(title_text="# Customers", secondary_y=False)
        fig.update_yaxes(title_text="Acceptance %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # ── 5. Geographic Distribution ──
    with tabs[4]:
        st.markdown("#### Loan Acceptance by ZIP Code Region (first 3 digits)")
        geo = df.groupby('ZIP_Region').agg(Count=('ID','count'), Accept_Rate=(TARGET,'mean')).reset_index()
        geo['Accept_Rate'] *= 100
        geo = geo[geo['Count'] >= 10].sort_values('Accept_Rate', ascending=False).head(30)
        fig = px.bar(geo, x='ZIP_Region', y='Accept_Rate', color='Count', color_continuous_scale='viridis',
                     text_auto='.1f', title="Top 30 ZIP Regions by Loan Acceptance Rate (min 10 customers)")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=420,
                          xaxis_title='ZIP Region', yaxis_title='Acceptance Rate %')
        st.plotly_chart(fig, use_container_width=True)

    # ── 6. Cross-Tabulations ──
    with tabs[5]:
        st.markdown("#### Cross-Tabulation: Personal Loan vs Other Features")
        cross_var = st.selectbox("Cross-tabulate Personal Loan with:",
                                 ['Education_Label','Family','Income_Bracket','Age_Group',
                                  'CCAvg_Bracket','Securities Account','CD Account','Online','CreditCard'])
        ct = pd.crosstab(df[cross_var], df[TARGET].map({0:'Declined',1:'Accepted'}), margins=True)
        ct_pct = pd.crosstab(df[cross_var], df[TARGET].map({0:'Declined',1:'Accepted'}), normalize='index').round(4) * 100
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Counts**")
            st.dataframe(ct, use_container_width=True)
        with c2:
            st.markdown("**Row Percentages (%)**")
            st.dataframe(ct_pct.style.format("{:.1f}%").background_gradient(cmap='Blues'), use_container_width=True)
        rate = df.groupby(cross_var, observed=True)[TARGET].mean().reset_index()
        rate.columns = [cross_var, 'Accept Rate']; rate['Accept Rate'] *= 100
        fig = px.bar(rate, x=cross_var, y='Accept Rate', text_auto='.1f',
                     color_discrete_sequence=['#3b82f6'], title=f"Loan Accept Rate by {cross_var}")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=360, yaxis_title='%')
        st.plotly_chart(fig, use_container_width=True)

    # ── 7. Customer Segmentation Summary ──
    with tabs[6]:
        st.markdown("#### Customer Segmentation Summary")
        for label, grp_col in [("By Education", 'Education_Label'), ("By Family Size", 'Family'), ("By Income Bracket", 'Income_Bracket')]:
            seg = df.groupby(grp_col, observed=True).agg(Count=('ID','count'), Avg_Income=('Income','mean'),
                        Avg_CCAvg=('CCAvg','mean'), Loan_Rate=(TARGET,'mean')).reset_index()
            seg['Loan_Rate'] = (seg['Loan_Rate']*100).round(1)
            seg['Avg_Income'] = seg['Avg_Income'].round(1)
            seg['Avg_CCAvg'] = seg['Avg_CCAvg'].round(2)
            st.markdown(f"**{label}**")
            st.dataframe(seg, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
#  DIAGNOSTIC ANALYTICS
# ═══════════════════════════════════════════════════════════
elif page == "🔍 Diagnostic Analytics":
    st.markdown('<p class="main-title">Diagnostic Analytics</p>', unsafe_allow_html=True)
    st.markdown('<span class="analytics-badge badge-diag">WHY DID IT HAPPEN?</span>', unsafe_allow_html=True)

    tabs = st.tabs(["Correlation Heatmap","Acceptors vs Non-Acceptors",
                     "Segment Deep Dive","Feature Relationship Explorer","Feature Importance"])

    with tabs[0]:
        corr = df[FEATURES + [TARGET]].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                        title="Feature Correlation Heatmap", aspect='auto', zmin=-1, zmax=1)
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=580)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("#### Loan Acceptors vs Non-Acceptors Comparison")
        compare_cols = ['Income','CCAvg','Mortgage','Age','Experience']
        for i in range(0, len(compare_cols), 2):
            cols = st.columns(2)
            for j, c in enumerate(compare_cols[i:i+2]):
                with cols[j]:
                    fig = px.box(df, x=df[TARGET].map({0:'Declined',1:'Accepted'}), y=c,
                                 color=df[TARGET].map({0:'Declined',1:'Accepted'}),
                                 color_discrete_map={'Declined':'#94a3b8','Accepted':'#3b82f6'},
                                 title=f"{c}: Acceptors vs Decliners")
                    fig.update_layout(template='plotly_white', font_family='DM Sans', height=360, showlegend=False, xaxis_title='')
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Mean Comparison Table")
        grp = df.groupby(TARGET)[FEATURES].mean().T.round(2)
        grp.columns = ['Declined (0)','Accepted (1)']
        grp['Difference'] = (grp['Accepted (1)'] - grp['Declined (0)']).round(2)
        grp['% Diff'] = ((grp['Difference'] / grp['Declined (0)'].replace(0, np.nan)) * 100).round(1)
        st.dataframe(grp.style.background_gradient(cmap='RdYlGn', subset=['% Diff']), use_container_width=True)

    with tabs[2]:
        seg_col = st.selectbox("Segment by:", ['Income_Bracket','Education_Label','Age_Group',
                                                'Family','CCAvg_Bracket','CD Account','Securities Account'])
        seg = df.groupby(seg_col, observed=True).agg(Count=('ID','count'), Accept_Rate=(TARGET,'mean'),
                    Avg_Income=('Income','mean'), Avg_CCAvg=('CCAvg','mean')).reset_index()
        seg['Accept_Rate'] *= 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=seg[seg_col].astype(str), y=seg['Count'], name='Customers',
                             marker_color='#cbd5e1', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=seg[seg_col].astype(str), y=seg['Accept_Rate'], name='Accept %',
                                 mode='lines+markers', marker=dict(size=10, color='#3b82f6'),
                                 line=dict(width=3, color='#3b82f6')), secondary_y=True)
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=420, title=f"Segment Analysis by {seg_col}",
                          legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
        fig.update_yaxes(title_text="# Customers", secondary_y=False)
        fig.update_yaxes(title_text="Accept Rate %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.markdown("#### Scatter Plot Explorer")
        c1, c2 = st.columns(2)
        with c1: x_feat = st.selectbox("X-axis:", FEATURES, index=FEATURES.index('Income'))
        with c2: y_feat = st.selectbox("Y-axis:", FEATURES, index=FEATURES.index('CCAvg'))
        fig = px.scatter(df, x=x_feat, y=y_feat,
                         color=df[TARGET].map({0:'Declined',1:'Accepted'}),
                         color_discrete_map={'Declined':'#94a3b8','Accepted':'#ef4444'},
                         opacity=0.5, title=f"{y_feat} vs {x_feat} (colored by Loan Acceptance)",
                         hover_data=['Age','Income','Education_Label'])
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=480,
                          legend=dict(orientation='h', y=1.08, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        model_sel = st.selectbox("Select model:", list(results.keys()))
        if 'importance' in results[model_sel]:
            imp_df = pd.DataFrame(list(results[model_sel]['importance'].items()),
                                  columns=['Feature','Importance']).sort_values('Importance')
            fig = px.bar(imp_df, y='Feature', x='Importance', orientation='h',
                         color='Importance', color_continuous_scale='viridis',
                         title=f"Feature Importance — {model_sel}")
            fig.update_layout(template='plotly_white', font_family='DM Sans', height=460, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PREDICTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════
elif page == "🤖 Predictive Analytics":
    st.markdown('<p class="main-title">Predictive Analytics</p>', unsafe_allow_html=True)
    st.markdown('<span class="analytics-badge badge-pred">WHAT WILL HAPPEN?</span>', unsafe_allow_html=True)

    tabs = st.tabs(["📋 Model Comparison Table","📈 ROC Curves (All Models)",
                     "🔲 Confusion Matrices","📊 Feature Importance","🎯 Loan Prediction Calculator"])

    # ────────────────────────────────────────────
    # TAB 1: COMPARISON TABLE (Train + Test)
    # ────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Classification Algorithms — Training & Testing Performance")
        st.markdown("*Algorithms in rows, metrics in columns — both train and test sets.*")

        rows = []
        for name, r in results.items():
            rows.append({
                'Algorithm': name,
                'Train Accuracy':  r['train_accuracy'],
                'Test Accuracy':   r['test_accuracy'],
                'Train Precision': r['train_precision'],
                'Test Precision':  r['test_precision'],
                'Train Recall':    r['train_recall'],
                'Test Recall':     r['test_recall'],
                'Train F1-Score':  r['train_f1'],
                'Test F1-Score':   r['test_f1'],
                'Train AUC':       r['train_auc'],
                'Test AUC':        r['test_auc'],
            })
        comp_df = pd.DataFrame(rows).set_index('Algorithm')

        def color_cells(val):
            if isinstance(val, (float, np.floating)):
                if val >= 0.95: return 'background-color: #bbf7d0; font-weight: 700;'
                elif val >= 0.90: return 'background-color: #d1fae5;'
                elif val >= 0.80: return 'background-color: #fef9c3;'
                elif val < 0.70: return 'background-color: #fecaca;'
            return ''

        styled = comp_df.style.format("{:.4f}").applymap(color_cells)
        st.dataframe(styled, use_container_width=True, height=230)

        st.success(f"🏆 **Best Model: {best_name}** — Test AUC = {results[best_name]['test_auc']:.4f}")

        # Visual comparison — test metrics bar chart
        st.markdown("#### Visual Comparison (Test Set Metrics)")
        test_only = comp_df[['Test Accuracy','Test Precision','Test Recall','Test F1-Score','Test AUC']]
        test_only.columns = ['Accuracy','Precision','Recall','F1-Score','AUC']
        melted = test_only.reset_index().melt(id_vars='Algorithm', var_name='Metric', value_name='Score')
        fig = px.bar(melted, x='Metric', y='Score', color='Algorithm', barmode='group',
                     color_discrete_sequence=['#3b82f6','#f59e0b','#10b981','#8b5cf6'],
                     title="Test Set — All Metrics Across All Algorithms")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=440, yaxis_range=[0,1.05],
                          legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

        # Overfit check
        st.markdown("#### Overfit Check — Train vs Test Accuracy Gap")
        gap_data = []
        for name, r in results.items():
            gap_data.append({'Algorithm': name, 'Train Accuracy': r['train_accuracy'],
                             'Test Accuracy': r['test_accuracy'],
                             'Gap': r['train_accuracy'] - r['test_accuracy']})
        gap_df = pd.DataFrame(gap_data)
        fig = px.bar(gap_df, x='Algorithm', y=['Train Accuracy','Test Accuracy'], barmode='group',
                     color_discrete_map={'Train Accuracy':'#93c5fd','Test Accuracy':'#3b82f6'},
                     title="Train vs Test Accuracy (smaller gap = less overfitting)")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=380, yaxis_range=[0.85,1.02],
                          legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
        for _, row in gap_df.iterrows():
            fig.add_annotation(x=row['Algorithm'], y=max(row['Train Accuracy'], row['Test Accuracy']) + 0.005,
                               text=f"Δ = {row['Gap']:.4f}", showarrow=False, font=dict(size=11, color='#ef4444'))
        st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # TAB 2: ROC CURVES (ALL TOGETHER)
    # ────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### ROC Curve — All Classification Algorithms Together")
        colors = {'Logistic Regression':'#3b82f6','Decision Tree':'#f59e0b',
                  'Random Forest':'#10b981','Gradient Boosting':'#8b5cf6'}
        fig = go.Figure()
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y_test, r['y_test_proba'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"{name} (AUC = {r['test_auc']:.4f})",
                                     line=dict(width=2.5, color=colors.get(name,'#64748b'))))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random Baseline',
                                 line=dict(dash='dash', color='#94a3b8', width=1.5)))
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=540,
                          title="Receiver Operating Characteristic (ROC) Curves — All Models",
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          legend=dict(orientation='h', yanchor='top', y=-0.12, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

        # AUC summary
        auc_df = pd.DataFrame([{'Model': n, 'Train AUC': r['train_auc'], 'Test AUC': r['test_auc']}
                               for n, r in results.items()])
        st.dataframe(auc_df.style.format({'Train AUC':'{:.4f}','Test AUC':'{:.4f}'}).highlight_max(
            subset=['Test AUC'], color='#bbf7d0'), use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────
    # TAB 3: CONFUSION MATRICES
    # ────────────────────────────────────────────
    with tabs[2]:
        cols = st.columns(2)
        for idx, (name, r) in enumerate(results.items()):
            with cols[idx % 2]:
                cm = r['cm']
                fig = px.imshow(cm, text_auto=True, color_continuous_scale='blues',
                                x=['Pred: No','Pred: Yes'], y=['Actual: No','Actual: Yes'],
                                title=f"{name}", aspect='auto')
                fig.update_layout(template='plotly_white', font_family='DM Sans', height=320)
                st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # TAB 4: FEATURE IMPORTANCE
    # ────────────────────────────────────────────
    with tabs[3]:
        model_sel2 = st.selectbox("Model:", list(results.keys()), key='fi_pred')
        if 'importance' in results[model_sel2]:
            imp_df = pd.DataFrame(list(results[model_sel2]['importance'].items()),
                                  columns=['Feature','Importance']).sort_values('Importance')
            fig = px.bar(imp_df, y='Feature', x='Importance', orientation='h',
                         color='Importance', color_continuous_scale='viridis',
                         title=f"Feature Importance — {model_sel2}")
            fig.update_layout(template='plotly_white', font_family='DM Sans', height=460, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ────────────────────────────────────────────
    # TAB 5: INTERACTIVE PREDICTION CALCULATOR
    # ────────────────────────────────────────────
    with tabs[4]:
        st.markdown("#### 🎯 Interactive Loan Prediction Calculator")
        st.markdown("Enter customer details below to predict their loan acceptance probability and get a personalised offer.")
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_age = st.slider("Age", 23, 67, 40)
            inp_exp = st.slider("Experience (yrs)", 0, 43, 15)
            inp_inc = st.slider("Income ($k)", 8, 224, 70)
            inp_fam = st.selectbox("Family Size", [1,2,3,4], index=1)
        with c2:
            inp_ccavg = st.slider("CC Avg Spend ($k/mo)", 0.0, 10.0, 2.0, 0.1)
            inp_edu = st.selectbox("Education", [1,2,3], format_func=lambda x: {1:'Undergrad',2:'Graduate',3:'Advanced'}[x])
            inp_mort = st.slider("Mortgage ($k)", 0, 635, 0)
        with c3:
            inp_sec = st.selectbox("Securities Account", [0,1], format_func=lambda x: 'Yes' if x else 'No')
            inp_cd  = st.selectbox("CD Account", [0,1], format_func=lambda x: 'Yes' if x else 'No')
            inp_onl = st.selectbox("Online Banking", [0,1], format_func=lambda x: 'Yes' if x else 'No', index=1)
            inp_cc  = st.selectbox("Credit Card", [0,1], format_func=lambda x: 'Yes' if x else 'No')

        input_row = pd.DataFrame([[inp_age, inp_exp, inp_inc, inp_fam, inp_ccavg,
                                   inp_edu, inp_mort, inp_sec, inp_cd, inp_onl, inp_cc]], columns=FEATURES)

        if st.button("🔮 Predict Loan Acceptance", type="primary"):
            inp_sc = scaler.transform(input_row) if best_name == 'Logistic Regression' else input_row
            prob = best_model.predict_proba(inp_sc)[0][1]
            pred = "✅ Likely to Accept" if prob >= 0.5 else "❌ Unlikely to Accept"

            pc1, pc2 = st.columns([1,2])
            with pc1:
                st.metric("Prediction", pred)
                st.metric("Probability", f"{prob:.1%}")
                st.metric("Model Used", best_name)
            with pc2:
                if prob >= 0.5:
                    offer_row = input_row.iloc[0].to_dict()
                    offer = generate_offer(pd.Series(offer_row), prob)
                    st.markdown(f"""
                    <div class="offer-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4>{offer['offer_type']}</h4>
                            <span class="offer-tag">{offer['confidence']} · {prob:.0%}</span>
                        </div>
                        <p>📋 <em>{offer['hook']}</em></p>
                        <p>💰 Rate: <strong>{offer['rate']}</strong> · Max: <strong>{offer['max_amount']}</strong> · Tenure: <strong>{offer['tenure']}</strong> · via <strong>{offer['channel']}</strong></p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.info("Customer not predicted as interested. Consider nurturing campaigns or revisiting after life events.")


# ═══════════════════════════════════════════════════════════
#  PRESCRIPTIVE ANALYTICS
# ═══════════════════════════════════════════════════════════
elif page == "💡 Prescriptive Analytics":
    st.markdown('<p class="main-title">Prescriptive Analytics</p>', unsafe_allow_html=True)
    st.markdown('<span class="analytics-badge badge-presc">WHAT SHOULD WE DO?</span>', unsafe_allow_html=True)

    pred_int = df[df['Predicted_Interested'] == 1].copy()

    tabs = st.tabs(["Customer Scoring & Ranking","Personalised Offers",
                     "Target Segment Recommendations","Campaign Strategy Suggestions",
                     "What-If Scenario Analysis"])

    # ── 1. Scoring ──
    with tabs[0]:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Predicted Interested", f"{len(pred_int):,}")
        c2.metric("Campaign Target %", f"{len(pred_int)/len(df)*100:.1f}%")
        c3.metric("Avg Probability", f"{pred_int['Loan_Probability'].mean():.1%}")
        c4.metric("High Confidence (>75%)", f"{(pred_int['Loan_Probability']>0.75).sum():,}")

        fig = px.histogram(df, x='Loan_Probability', nbins=50,
                           color=df['Predicted_Interested'].map({0:'Not Targeted',1:'Targeted'}),
                           color_discrete_map={'Not Targeted':'#94a3b8','Targeted':'#3b82f6'},
                           title="Loan Probability Distribution", barmode='overlay', opacity=0.75)
        fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold = 0.5")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=380,
                          legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Confidence Tier Breakdown")
        pred_int['Tier'] = pd.cut(pred_int['Loan_Probability'], bins=[0.5,0.65,0.80,1.01],
                                  labels=['Moderate (50-65%)','Medium (65-80%)','High (80-100%)'])
        tier = pred_int.groupby('Tier', observed=True).agg(Count=('ID','count'), Avg_Income=('Income','mean'),
                    Avg_CCAvg=('CCAvg','mean'), Avg_Prob=('Loan_Probability','mean')).reset_index()
        tier['Avg_Income'] = tier['Avg_Income'].round(1)
        tier['Avg_CCAvg']  = tier['Avg_CCAvg'].round(2)
        tier['Avg_Prob']   = (tier['Avg_Prob']*100).round(1)
        st.dataframe(tier, use_container_width=True, hide_index=True)

        st.markdown("#### Top 20 Customers by Score")
        top20 = pred_int.nlargest(20, 'Loan_Probability')[['ID','Age','Income','Education_Label',
                    'Family','CCAvg','CD Account','Online','Loan_Probability']].copy()
        top20['Loan_Probability'] = (top20['Loan_Probability']*100).round(1)
        top20.columns = ['ID','Age','Income ($k)','Education','Family','CC Avg','CD Acct','Online','Score (%)']
        st.dataframe(top20, use_container_width=True, hide_index=True)

    # ── 2. Personalised Offers ──
    with tabs[1]:
        st.markdown("#### Personalised Offers for Predicted-Interested Customers")
        n_show = st.slider("Number of customers:", 5, 50, 15)
        top_cust = pred_int.nlargest(n_show, 'Loan_Probability')
        for _, row in top_cust.iterrows():
            offer = generate_offer(row, row['Loan_Probability'])
            st.markdown(f"""
            <div class="offer-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4>{offer['offer_type']}</h4>
                    <span class="offer-tag">{offer['confidence']} Confidence · {offer['probability']:.0%}</span>
                </div>
                <p><strong>Customer #{int(row['ID'])}</strong> · Age {int(row['Age'])} · Income ${int(row['Income'])}k · {row['Education_Label']} · Family of {int(row['Family'])}</p>
                <p>📋 <em>{offer['hook']}</em></p>
                <p>💰 Rate: <strong>{offer['rate']}</strong> · Max: <strong>{offer['max_amount']}</strong> · Tenure: <strong>{offer['tenure']}</strong> · via <strong>{offer['channel']}</strong></p>
            </div>""", unsafe_allow_html=True)

    # ── 3. Target Segments ──
    with tabs[2]:
        st.markdown("#### Target Segment Recommendations")
        segs = []
        for edu in ['Undergrad','Graduate','Advanced/Professional']:
            for ib in ['Low (<50k)','Mid (50-100k)','High (100-150k)','Very High (>150k)']:
                subset = pred_int[(pred_int['Education_Label']==edu) & (pred_int['Income_Bracket']==ib)]
                if len(subset) >= 3:
                    segs.append({'Education': edu, 'Income Bracket': ib, 'Count': len(subset),
                                 'Avg Probability': round(subset['Loan_Probability'].mean()*100,1),
                                 'Avg CCAvg': round(subset['CCAvg'].mean(),2)})
        seg_df = pd.DataFrame(segs).sort_values('Avg Probability', ascending=False)
        st.dataframe(seg_df, use_container_width=True, hide_index=True)

        fig = px.treemap(seg_df, path=['Education','Income Bracket'], values='Count',
                         color='Avg Probability', color_continuous_scale='blues',
                         title="Target Segments — Size by Count, Color by Avg Probability")
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=480)
        st.plotly_chart(fig, use_container_width=True)

    # ── 4. Campaign Strategies ──
    with tabs[3]:
        st.markdown("#### Campaign Strategy Suggestions")
        strategies = [
            ("💎 Premium Channel Campaign", "High (80-100%)",
             "Target high-confidence customers (>80%) with dedicated RMs, pre-approved offers via personal calls and premium branch meetings.",
             f"{(pred_int['Loan_Probability']>0.80).sum()} customers",
             "Highest conversion — allocate top RMs and fast-track processing."),
            ("📱 Digital-First Campaign", "Medium (65-80%)",
             "Target medium-confidence online-active customers with in-app notifications, email sequences, and instant digital disbursement.",
             f"{((pred_int['Loan_Probability']>0.65) & (pred_int['Loan_Probability']<=0.80) & (pred_int['Online']==1)).sum()} customers",
             "Cost-effective — leverages digital channels with minimal branch overhead."),
            ("🏠 Family & Mortgage Cross-Sell", "Moderate (50-65%)",
             "Target families (3-4 members) with mortgages. Offer family-oriented packages with flexible EMIs.",
             f"{((pred_int['Loan_Probability']>0.50) & (pred_int['Loan_Probability']<=0.65) & (pred_int['Family']>=3)).sum()} customers",
             "Deepens existing relationship — builds on mortgage trust."),
            ("⭐ CD Holder Loyalty Campaign", "All tiers",
             "CD holders show dramatically higher conversion. Send exclusive loyalty offers with preferential rates and zero-doc processing.",
             f"{(pred_int['CD Account']==1).sum()} customers",
             "Very high ROI — these customers already trust the bank with significant deposits."),
            ("📧 Nurture Campaign (Near-Threshold)", "Below threshold (40-50%)",
             "Customers scoring 40-50% are near threshold. Run educational content campaigns followed by limited-time rate offers.",
             f"{((df['Loan_Probability']>=0.40) & (df['Loan_Probability']<0.50)).sum()} customers",
             "Long-term pipeline building — converts future prospects."),
        ]
        for title, tier, desc, size, note in strategies:
            st.markdown(f"""
            <div class="strategy-card">
                <h4>{title} <span style="font-size:0.8rem; color:#7c3aed;">({tier})</span></h4>
                <p>{desc}</p>
                <p>👥 <strong>Target Size:</strong> {size}</p>
                <p>💡 <strong>Note:</strong> {note}</p>
            </div>""", unsafe_allow_html=True)

    # ── 5. What-If Scenario ──
    with tabs[4]:
        st.markdown("#### What-If Scenario Analysis")
        st.markdown("##### 1. Threshold Sensitivity")
        threshold = st.slider("Prediction Threshold:", 0.1, 0.9, 0.5, 0.05)
        targeted = (df['Loan_Probability'] >= threshold).sum()
        actual_in = df[df['Loan_Probability'] >= threshold][TARGET].sum()
        prec_at = actual_in / targeted * 100 if targeted > 0 else 0
        rec_at  = actual_in / df[TARGET].sum() * 100

        wc1,wc2,wc3,wc4 = st.columns(4)
        wc1.metric("Targeted", f"{targeted:,}")
        wc2.metric("True Positives", f"{actual_in:,}")
        wc3.metric("Precision", f"{prec_at:.1f}%")
        wc4.metric("Recall", f"{rec_at:.1f}%")

        thresholds = np.arange(0.1, 0.95, 0.05)
        t_data = []
        for t in thresholds:
            targ = (df['Loan_Probability'] >= t).sum()
            act  = df[df['Loan_Probability'] >= t][TARGET].sum()
            prec = act/targ*100 if targ > 0 else 0
            rec  = act/df[TARGET].sum()*100
            t_data.append({'Threshold': round(t,2), 'Targeted': targ, 'Precision %': prec, 'Recall %': rec})
        t_df = pd.DataFrame(t_data)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=t_df['Threshold'], y=t_df['Precision %'], name='Precision',
                                 mode='lines+markers', line=dict(color='#3b82f6', width=2.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=t_df['Threshold'], y=t_df['Recall %'], name='Recall',
                                 mode='lines+markers', line=dict(color='#10b981', width=2.5)), secondary_y=False)
        fig.add_trace(go.Bar(x=t_df['Threshold'], y=t_df['Targeted'], name='# Targeted',
                             marker_color='#cbd5e1', opacity=0.4), secondary_y=True)
        fig.update_layout(template='plotly_white', font_family='DM Sans', height=400,
                          title="Precision / Recall / Volume vs Threshold",
                          legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'))
        fig.update_yaxes(title_text="% ", secondary_y=False)
        fig.update_yaxes(title_text="# Targeted", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 2. Individual What-If")
        st.markdown("Use the **Prediction Calculator** in the Predictive Analytics tab to test how changing individual customer attributes affects the prediction and offer.")


# ═══════════════════════════════════════════════════════════
#  DATA EXPORT
# ═══════════════════════════════════════════════════════════
elif page == "📥 Data Export":
    st.markdown('<p class="main-title">Data Export</p>', unsafe_allow_html=True)
    st.markdown("Download predictions, campaign lists, and model performance reports.")

    st.markdown("#### 1. Full Customer Predictions")
    export_df = df[['ID','Age','Income','Family','CCAvg','Education_Label','Mortgage',
                     'Securities Account','CD Account','Online','CreditCard',
                     TARGET,'Loan_Probability','Predicted_Interested']].copy()
    export_df['Loan_Probability'] = export_df['Loan_Probability'].round(4)
    st.dataframe(export_df.head(20), use_container_width=True, hide_index=True)
    st.download_button("📥 Download Full Predictions (CSV)", export_df.to_csv(index=False),
                       "universal_bank_predictions.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### 2. Campaign Target List with Personalised Offers")
    pred_int2 = df[df['Predicted_Interested'] == 1].copy()
    offer_rows = []
    for _, row in pred_int2.iterrows():
        o = generate_offer(row, row['Loan_Probability'])
        offer_rows.append({'Customer_ID': int(row['ID']), 'Age': int(row['Age']),
            'Income': int(row['Income']), 'Education': row['Education_Label'],
            'Family': int(row['Family']), 'Probability': round(row['Loan_Probability'],4),
            'Offer_Type': o['offer_type'], 'Rate': o['rate'], 'Max_Amount': o['max_amount'],
            'Tenure': o['tenure'], 'Channel': o['channel'], 'Confidence': o['confidence']})
    offer_df = pd.DataFrame(offer_rows).sort_values('Probability', ascending=False)
    st.dataframe(offer_df.head(20), use_container_width=True, hide_index=True)
    st.download_button("📥 Download Campaign List (CSV)", offer_df.to_csv(index=False),
                       "universal_bank_campaign_offers.csv", "text/csv")

    st.markdown("---")
    st.markdown("#### 3. Model Performance Report")
    perf_rows = []
    for name, r in results.items():
        perf_rows.append({'Algorithm': name,
            'Train_Accuracy': round(r['train_accuracy'],4), 'Test_Accuracy': round(r['test_accuracy'],4),
            'Train_Precision': round(r['train_precision'],4), 'Test_Precision': round(r['test_precision'],4),
            'Train_Recall': round(r['train_recall'],4), 'Test_Recall': round(r['test_recall'],4),
            'Train_F1': round(r['train_f1'],4), 'Test_F1': round(r['test_f1'],4),
            'Train_AUC': round(r['train_auc'],4), 'Test_AUC': round(r['test_auc'],4)})
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.download_button("📥 Download Model Report (CSV)", perf_df.to_csv(index=False),
                       "universal_bank_model_report.csv", "text/csv")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.8rem;'>Universal Bank Analytics Dashboard · Descriptive · Diagnostic · Predictive · Prescriptive · Built with Streamlit & Scikit-learn</p>", unsafe_allow_html=True)
