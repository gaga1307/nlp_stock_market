# Sentiment analysis and stock market prediction — Semiconductor sector

This repository contains the complete implementation of a thesis project analyzing the impact of public sentiment on stock market behavior for three major semiconductor companies: **NVIDIA (NVDA)**, **AMD**, and **Intel (INTC)**.

---

## Research Overview

The study investigates whether sentiment derived from Reddit discussions and financial news headlines improves the prediction of stock market indicators. Two prediction tasks are addressed:

1. **Stock price direction prediction** — binary classification of next-day price movement
2. **Realized volatility forecasting** — regression of 10-day annualized realized volatility

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_sentiment_analysis.ipynb` | Fine-tuning of BERT on Financial PhraseBank and Twitter Financial News datasets for three-class sentiment classification (positive / negative / neutral). Includes EDA, TF-IDF baseline with Logistic Regression, and BERT fine-tuning with weighted cross-entropy loss. |
| 02 | `02_reddit_headlines.ipynb` | Collection of Reddit posts related to NVDA, AMD, and INTC using the Arctic Shift API. Covers r/stocks, r/wallstreetbets, r/investing, and company-specific subreddits for the period 2016–2026. |
| 03 | `03_yahoo_finance_and_technical_analysis.ipynb` | Download of historical OHLCV data from Yahoo Finance. Computation of technical indicators (trend, momentum, volatility, volume, returns) and construction of prediction targets. |
| 04 | `04_macro_and_factor_features.ipynb` | Retrieval of macroeconomic features (FEDFUNDS, DGS10, DGS2, yield spread) from FRED API, market-wide features (VIX, Nasdaq, SOXX) from Yahoo Finance, and Fama-French 5 factors + Momentum from Kenneth French's Data Library. |
| 05 | `05_gdelt_news_headlines.ipynb` | Collection of financial news headlines from the GDELT Project via Google BigQuery. Covers 11 major financial news sources for the period 2016–2026. Includes headline cleaning, deduplication, and trading day alignment. |
| 06 | `06_final_dataset_construction.ipynb` | BERT inference on Reddit and news headlines to generate daily sentiment scores. Merging of all data sources (Yahoo Finance, macro features, Reddit sentiment, news sentiment) into a single company-level daily dataset. |
| 07 | `07_initial_feature_preprocessing.ipynb` | Cyclical temporal encoding of date variables and Spearman correlation analysis across all features. |
| 08 | `08_stock_price_direction_prediction.ipynb` | Binary classification of next-day stock price direction. Includes Mann-Whitney U feature relevance analysis, VIF-based multicollinearity reduction, Logistic Regression and XGBoost modeling, and ablation study across four sentiment configurations. |
| 09 | `09_target_volatility_10d.ipynb` | Regression of 10-day annualized realized volatility. Includes Spearman feature relevance analysis, Ridge Regression and XGBoost modeling, ablation study across four sentiment configurations. |
| 10 | `10_reverse_direction_analysis.ipynb` | Reverse-direction analysis examining whether market variables predict subsequent social media and news activity. XGBoost is trained to predict next-day values of Reddit post count, Reddit comment count, news count, Reddit sentiment, and news sentiment using market-based features only. |

---

## Data Sources

| Source | Data | Period |
|--------|------|--------|
| Yahoo Finance (`yfinance`) | OHLCV prices, technical indicators | 2016–2026 |
| FRED API | FEDFUNDS, DGS10, DGS2 | 2016–2026 |
| Kenneth French Data Library | Fama-French 5 factors + Momentum | 2016–2026 |
| Arctic Shift API | Reddit posts (NVDA, AMD, INTC) | 2016–2026 |
| GDELT Project (BigQuery) | Financial news headlines | 2016–2026 |

---

## Methodology

### Sentiment Analysis
A `bert-base-uncased` model was fine-tuned on a combined dataset of 
Financial PhraseBank and Twitter Financial News for three-class sentiment 
classification. The two datasets were selected to complement each other — 
Financial PhraseBank covers formal financial language suited for news 
headlines, while Twitter Financial News captures informal, colloquial 
expressions characteristic of social media platforms such as Reddit. 
Daily sentiment is obtained by averaging individual scores across all texts published on a given trading day for each company.

### Prediction Tasks

**Task 1 — Stock Price Direction (Classification)**
- Target: binary indicator of next-day price increase
- Models: Logistic Regression (baseline), XGBoost
- Evaluation: Accuracy, ROC-AUC, F1 macro

**Task 2 — Realized Volatility (Regression)**
- Target: annualized standard deviation of log returns over next 10 trading days
- Models: Ridge Regression (baseline), XGBoost
- Evaluation: RMSE, MAE, R²

### Ablation Study
Both tasks evaluate four feature configurations to isolate the contribution of sentiment:

| Configuration | Features |
|---|---|
| `no_sentiment` | Technical + macro features only |
| `reddit_only` | Adds Reddit sentiment, post count, comment count |
| `news_only` | Adds news sentiment and news count |
| `both` | Full feature set |

---

## Key Findings

- Neither Logistic Regression nor XGBoost achieved meaningful predictive 
  performance for next-day stock price direction (ROC-AUC ≈ 0.51–0.54), 
  consistent with the Efficient Market Hypothesis.

- Volatility forecasting also yielded near-zero R² values across all 
  configurations, partially attributable to a distribution shift between 
  the training and test periods driven by exceptional market conditions 
  in the semiconductor sector during 2024–2026.


- Activity-based features (`news_count`, `reddit_post_count`) showed 
  stronger associations with future volatility than sentiment scores, 
  indicating that the intensity of public attention may be more informative 
  than sentiment polarity for volatility forecasting.

- The reverse-direction analysis revealed an asymmetric causal structure: 
  market variables were substantially more predictive of subsequent media 
  activity (news count: R² = 0.367, Reddit post count: R² = 0.224) than 
  sentiment features were of future market outcomes. This suggests that 
  news and social media activity primarily reflect investor attention 
  generated by market developments rather than act as independent 
  predictors of future market behavior.

---
