1
# Employee Sentiment Analysis Project

## 📋 Overview
This project analyzes employee email messages from the dataset to assess sentiment and engagement levels. The analysis includes sentiment labeling, exploratory data analysis, employee scoring, ranking, flight risk identification, and predictive modeling.

## 🎯 Project Objectives
- **Sentiment Labeling**: Classify each message as Positive, Negative, or Neutral
- **Exploratory Data Analysis (EDA)**: Understand data structure, distributions, and trends
- **Employee Score Calculation**: Compute monthly sentiment scores per employee
- **Employee Ranking**: Identify top positive and negative employees
- **Flight Risk Identification**: Flag employees with 4+ negative messages in any 30-day period
- **Predictive Modeling**: Build linear regression model for sentiment trend analysis

## 📁 Project Structure
```
Final_Assessment/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data/
│   └── test.csv                        # Employee email dataset
├── notebooks/
│   └── sentiment_analysis.ipynb        # Main analysis notebook
├── visualizations/
│   ├── sentiment_distribution.png
│   ├── monthly_sentiment_trends.png
│   ├── employee_rankings.png
│   ├── flight_risk_analysis.png
│   └── model_performance.png
└── Final_Report.docx                   # Detailed project report
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd Final_Assessment
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download TextBlob corpora**
   ```bash
   python -m textblob.download_corpora
   ```

5. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook notebooks/sentiment_analysis.ipynb
   ```

## 📊 Methodology

### Sentiment Analysis Approach
- Used **TextBlob** library for Natural Language Processing
- Polarity scores range from -1 (negative) to +1 (positive)
- Classification thresholds:
  - **Positive**: polarity > 0.1
  - **Negative**: polarity < -0.1
  - **Neutral**: -0.1 ≤ polarity ≤ 0.1

### Employee Scoring System
- **Positive message**: +1 point
- **Negative message**: -1 point
- **Neutral message**: 0 points
- Scores aggregated monthly per employee

### Flight Risk Criteria
- Employee flagged if they send **4 or more negative messages** within any **rolling 30-day window**

## 📈 Key Findings

### Dataset Overview
- **Total Messages Analyzed**: 2,191
- **Unique Employees**: 11
- **Date Range**: March 2001 - November 2001

### Top 3 Positive Employees (Overall)
| Rank | Employee | Total Score | Messages |
|------|----------|-------------|----------|
| 1    | lydia.delgado | +101 | 284 |
| 2    | john.arnold | +97 | 256 |
| 3    | sally.beck | +89 | 227 |

### Top 3 Negative Employees (Lowest Positive Scores)
| Rank | Employee | Total Score | Messages |
|------|----------|-------------|----------|
| 1    | rhonda.denton | +50 | 172 |
| 2    | kayne.coulter | +61 | 174 |
| 3    | bobette.riner@ipgdirect.com | +72 | 217 |

*Note: All employees have positive overall scores. Those listed have the lowest positive scores.*

### Flight Risk Employees
**Total Flight Risk Employees**: 6

Employees flagged (4+ negative messages in any 30-day window):
1. lydia.delgado (28 negative messages)
2. patti.thompson (23 negative messages)
3. bobette.riner@ipgdirect.com (21 negative messages)
4. rhonda.denton (20 negative messages)
5. johnny.palmer (18 negative messages)
6. sally.beck (18 negative messages)

### Predictive Model Performance
- **R² Score**: 0.3615
- **RMSE**: 2.37
- **Model Type**: Linear Regression
- **Features**: Message count, average message length, average word count

## 🔧 Technologies Used
- **Python 3.8+**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **TextBlob**: Sentiment analysis
- **Scikit-learn**: Machine learning (Linear Regression)
- **WordCloud**: Text visualization

## 📝 Usage

### Running the Analysis
1. Open `notebooks/sentiment_analysis.ipynb` in Jupyter
2. Run all cells sequentially (Kernel → Restart & Run All)
3. Visualizations will be saved to the `visualizations/` folder
4. Results and insights are documented within the notebook

### Customization
- Adjust sentiment thresholds in the `get_sentiment()` function
- Modify flight risk threshold (default: 4 negative messages in 30 days)
- Add additional features for the regression model

