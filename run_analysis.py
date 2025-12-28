"""
Employee Sentiment Analysis - Standalone Script
Generates all visualizations and results
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print("="*70)
print("EMPLOYEE SENTIMENT ANALYSIS - STARTING")
print("="*70)

# Load data
df = pd.read_csv('data/test.csv')
print(f"\nLoaded dataset: {df.shape[0]} rows")

# Sentiment analysis
df_analysis = df.copy()
df_analysis['text'] = df_analysis['Subject'].fillna('') + ' ' + df_analysis['body'].fillna('')

def get_sentiment(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return 'Neutral'
        polarity = TextBlob(str(text)).sentiment.polarity
        return 'Positive' if polarity > 0.1 else 'Negative' if polarity < -0.1 else 'Neutral'
    except:
        return 'Neutral'

def get_polarity(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return 0.0
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0

print("🔄 Analyzing sentiment...")
df_analysis['Sentiment'] = df_analysis['text'].apply(get_sentiment)
df_analysis['Polarity'] = df_analysis['text'].apply(get_polarity)
print(f"✅ Sentiment labeling complete")

# Process dates
df_analysis['date'] = pd.to_datetime(df_analysis['date'], format='%m/%d/%Y', errors='coerce')
df_analysis['employee'] = df_analysis['from'].str.replace('@enron.com', '', regex=False)
df_analysis['year_month'] = df_analysis['date'].dt.to_period('M')

# Create visualizations
print("\n🎨 Generating visualizations...")

# 1. Sentiment distribution
sentiment_counts = df_analysis['Sentiment'].value_counts()
colors = {'Positive': '#2ecc71', 'Neutral': '#3498db', 'Negative': '#e74c3c'}
sentiment_order = ['Positive', 'Neutral', 'Negative']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bars = axes[0].bar(sentiment_order, [sentiment_counts.get(s, 0) for s in sentiment_order],
                   color=[colors[s] for s in sentiment_order], edgecolor='black')
axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
axes[1].pie([sentiment_counts.get(s, 0) for s in sentiment_order], labels=sentiment_order,
            colors=[colors[s] for s in sentiment_order], autopct='%1.1f%%')
plt.tight_layout()
plt.savefig('visualizations/sentiment_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ sentiment_distribution.png")

# 2. Monthly trends
monthly_sentiment = df_analysis.groupby(['year_month', 'Sentiment']).size().unstack(fill_value=0)
for col in sentiment_order:
    if col not in monthly_sentiment.columns:
        monthly_sentiment[col] = 0

fig, ax = plt.subplots(figsize=(14, 6))
x_labels = monthly_sentiment.index.astype(str)
ax.plot(x_labels, monthly_sentiment['Positive'], marker='o', label='Positive', color='#2ecc71')
ax.plot(x_labels, monthly_sentiment['Neutral'], marker='s', label='Neutral', color='#3498db')
ax.plot(x_labels, monthly_sentiment['Negative'], marker='^', label='Negative', color='#e74c3c')
ax.set_title('Monthly Sentiment Trends', fontsize=14, fontweight='bold')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/monthly_sentiment_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ monthly_sentiment_trends.png")

# 3. Top employees sentiment
top_employees = df_analysis['employee'].value_counts().head(10).index.tolist()
emp_sentiment = df_analysis[df_analysis['employee'].isin(top_employees)].groupby(
    ['employee', 'Sentiment']).size().unstack(fill_value=0)
for col in sentiment_order:
    if col not in emp_sentiment.columns:
        emp_sentiment[col] = 0

fig, ax = plt.subplots(figsize=(12, 6))
emp_sentiment[sentiment_order].loc[top_employees].plot(kind='barh', stacked=True, ax=ax,
                                                        color=[colors[s] for s in sentiment_order])
ax.set_title('Top 10 Employees - Sentiment Breakdown', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/top_employees_sentiment.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ top_employees_sentiment.png")

# 4. Polarity distribution
fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df_analysis['Polarity'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
ax.axvline(x=0.1, color='#2ecc71', linestyle='--', linewidth=2)
ax.axvline(x=-0.1, color='#e74c3c', linestyle='--', linewidth=2)
ax.set_title('Distribution of Polarity Scores', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/polarity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ polarity_distribution.png")

# Calculate scores
score_mapping = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
df_analysis['Score'] = df_analysis['Sentiment'].map(score_mapping)

overall_scores = df_analysis.groupby('employee').agg({
    'Score': 'sum', 'Sentiment': 'count'
}).rename(columns={'Sentiment': 'total_messages', 'Score': 'total_score'}).reset_index()

top_3_positive = overall_scores.sort_values(by=['total_score', 'employee'],
                                             ascending=[False, True]).head(3)
top_3_negative = overall_scores.sort_values(by=['total_score', 'employee'],
                                             ascending=[True, True]).head(3)

# 5. Employee rankings
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top_10_pos = overall_scores.sort_values('total_score', ascending=False).head(10)
axes[0].barh(top_10_pos['employee'], top_10_pos['total_score'], color='#2ecc71')
axes[0].set_title('Top 10 Positive Employees', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

top_10_neg = overall_scores.sort_values('total_score', ascending=True).head(10)
axes[1].barh(top_10_neg['employee'], top_10_neg['total_score'], color='#e74c3c')
axes[1].set_title('Top 10 Negative Employees', fontsize=12, fontweight='bold')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/employee_rankings.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ employee_rankings.png")

# Flight risk analysis
negative_msgs = df_analysis[df_analysis['Sentiment'] == 'Negative'].copy()
flight_risks = set()
for employee in negative_msgs['employee'].unique():
    emp_msgs = negative_msgs[negative_msgs['employee'] == employee].sort_values('date')
    if len(emp_msgs) < 4:
        continue
    dates = emp_msgs['date'].tolist()
    for start_date in dates:
        end_date = start_date + timedelta(days=30)
        if sum(1 for d in dates if start_date <= d <= end_date) >= 4:
            flight_risks.add(employee)
            break

flight_risk_list = sorted(list(flight_risks))
flight_risk_df = pd.DataFrame([{
    'Employee': emp,
    'Total Negative Messages': len(negative_msgs[negative_msgs['employee'] == emp])
} for emp in flight_risk_list]).sort_values('Total Negative Messages', ascending=False)

# 6. Flight risk visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
total_emp = df_analysis['employee'].nunique()
axes[0].pie([len(flight_risks), total_emp - len(flight_risks)],
            labels=['Flight Risk', 'Not at Risk'], colors=['#e74c3c', '#2ecc71'],
            autopct='%1.1f%%', startangle=90)
axes[0].set_title(f'Flight Risk Distribution\n({len(flight_risks)} of {total_emp} employees)')

if len(flight_risk_df) > 0:
    top_fr = flight_risk_df.head(10)
    axes[1].barh(top_fr['Employee'], top_fr['Total Negative Messages'], color='#e74c3c')
    axes[1].set_title('Top 10 Flight Risk Employees')
    axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/flight_risk_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ flight_risk_analysis.png")

# Predictive modeling
df_analysis['message_length'] = df_analysis['text'].str.len()
df_analysis['word_count'] = df_analysis['text'].str.split().str.len()

modeling_data = df_analysis.groupby(['employee', 'year_month']).agg({
    'Score': 'sum', 'text': 'count', 'message_length': 'mean', 'word_count': 'mean'
}).reset_index()
modeling_data.columns = ['employee', 'year_month', 'monthly_score', 'message_count',
                         'avg_message_length', 'avg_word_count']
modeling_data = modeling_data.dropna()

X = modeling_data[['message_count', 'avg_message_length', 'avg_word_count']]
y = modeling_data['monthly_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 7. Model performance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_pred, alpha=0.5, color='#3498db')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Actual vs Predicted (R² = {test_r2:.4f})')

axes[1].hist(y_test - y_pred, bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_title('Residuals Distribution')
plt.tight_layout()
plt.savefig('visualizations/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ model_performance.png")

# 8. Feature importance
fig, ax = plt.subplots(figsize=(10, 5))
features = ['message_count', 'avg_message_length', 'avg_word_count']
importance = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
colors_feat = ['#2ecc71' if c > 0 else '#e74c3c' for c in model.coef_]
ax.barh(features, model.coef_, color=colors_feat, edgecolor='black')
ax.axvline(x=0, color='gray')
ax.set_title('Feature Coefficients')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ feature_importance.png")

# Save data
df_analysis.to_csv('data/processed_sentiment_data.csv', index=False)
monthly_scores = df_analysis.groupby(['employee', 'year_month']).agg({
    'Score': 'sum'}).reset_index()
monthly_scores.to_csv('data/monthly_employee_scores.csv', index=False)
if len(flight_risk_df) > 0:
    flight_risk_df.to_csv('data/flight_risk_employees.csv', index=False)

import json
with open('data/summary_results.json', 'w') as f:
    json.dump({
        'top_3_positive': top_3_positive.to_dict('records'),
        'top_3_negative': top_3_negative.to_dict('records'),
        'flight_risk_count': len(flight_risks),
        'flight_risk_list': flight_risk_list,
        'model_r2': float(test_r2),
        'model_rmse': float(test_rmse)
    }, f, indent=2, default=str)

print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)
print(f"\n🏆 Top 3 Positive Employees:")
for i, row in top_3_positive.iterrows():
    print(f"   {i+1}. {row['employee']} (Score: {row['total_score']:+d})")
print(f"\n⚠️  Top 3 Negative Employees:")
for i, row in top_3_negative.iterrows():
    print(f"   {i+1}. {row['employee']} (Score: {row['total_score']:+d})")
print(f"\n🚨 Flight Risk Employees: {len(flight_risks)}")
print(f"\n📈 Model R² Score: {test_r2:.4f}")
print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
