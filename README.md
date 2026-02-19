# Predictive Marketing Analytics: Unlocking Growth in Tablespreads

## Executive Summary
This project delivers a data-driven strategy to optimize Conagra Brands' tablespread portfolio, focusing on refrigerated spreads. By analyzing IRI POS data across 2018 (pre-COVID), 2020 (crisis), and 2022 (recovery), we identified critical drivers of sales performance and regional disparities. Using advanced regression techniques, we quantified the impact of pricing, merchandising, and distribution on volume, providing actionable insights for portfolio management and growth acceleration.

## Business Problem
Conagra Brands sought to identify actionable insights to optimize its brand portfolio and accelerate growth within the tablespreads category. The core challenge was to understand the shifting dynamics of consumer behavior across different stages of the pandemic and to determine which factors—pricing, region, or brand strength—most significantly influence sales volume.

## Data Overview
The analysis utilizes **IRI POS Tablespreads Data** comprising approximately 270,000 observations across three years:
- **Timeframes:** 2018, 2020, and 2022.
- **Key Metrics:** Dollar Sales, Unit Sales, Volume Sales, ACV (All Commodity Volume) Weighted Distribution, and Price per Unit (Merched vs. No Merch).
- **Scope:** Regional US markets (excluding Total US to avoid aggregation bias).

## Methodology & Analysis

### 1. Data Preprocessing
- **Cleaning:** Rigorous handling of null values and duplicate removal.
- **Feature Engineering:** Extraction of brand identifiers from product descriptions and temporal features (Year/Week).
- **Categorization:** Segmentation of variables into qualitative (Geography, Time) and quantitative (Sales, ACV, Price) metrics.

### 2. Exploratory Data Analysis (EDA)
- **Trend Analysis:** Observed a steady increase in mean Dollar Sales and Price per Unit from 2018 to 2022, despite consistent brand count declines.
- **Correlation:** Strong positive correlation between Base Volume and Unit Sales; weak correlation between regionality and other variables, suggesting unique regional drivers.

### 3. Hypothesis Testing
Conducted T-tests and ANOVA to validate the significance of sales variations:
- **Result:** Confirmed statistically significant differences in sales across years and regions (p < 0.05), proving that the observed changes were not random.

### 4. Predictive Modeling
We developed three primary models to forecast **Unit Sales (No Merch)**:
- **Linear Regression:** Baseline model to understand direct coefficients of price and distribution.
- **Lasso Regression:** Used for feature selection, identifying **Imperial RFG**, **BlueBonnet**, and **Private Labels** as significant brand predictors.
- **Polynomial Regression:** Our final production-grade model, achieving **~81% accuracy** by capturing non-linear relationships and combinative effects.

## Key Insights & Business Impact
- **Regional Strategy:** The **Northeast** and **Southeast** emerged as the most positively impactful regions for sales. We recommend prioritizing resource allocation to these markets.
- **Brand Performance:** **BlueBonnet** demonstrated high significance and should be prioritized for investment over lower-performing brands like Smart Balance.
- **Pricing Elasticity:** Price per Unit (No Merch) showed a positive effect on sales during high-demand periods (COVID-19), suggesting a degree of brand loyalty that allows for price maintenance without volume loss.
- **Merchandising:** Non-merchandised distribution had a larger magnitude of impact than merchandised distribution in the Lasso model, indicating the strength of organic shelf presence.

## Recommendations
1. **Focus on High-Growth Regions:** Replicate successful Northeast/Southeast strategies in the Plains region.
2. **Portfolio Optimization:** Invest in high-significance brands (BlueBonnet, Imperial RFG) and consider phasing out brands with negative significance.
3. **Strategic Pricing:** Maintain consistent pricing during peak demand to leverage consumer loyalty and trust.

## Technologies Used
- **Languages:** Python (Pandas, NumPy, Scikit-learn, Statsmodels)
- **Visualization:** Matplotlib, Seaborn
- **Statistical Analysis:** ANOVA, T-Tests
- **Models:** Lasso, Linear & Polynomial Regression

---
*Developed by Group 16: Manoj Mareedu, Daniel Bond, Yashasvi Pamu, Nallam Paramkousam Nallam, Pooja Reddy Donda.*
