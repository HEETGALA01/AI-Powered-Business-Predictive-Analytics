# AI-Powered-Business-Predictive-Analytics
AI-powered Business Predictive Analytics leverages machine learning and large language models to analyze historical data, identify patterns, and forecast future business trends. It helps organizations make data-driven decisions by predicting outcomes like sales, customer behavior, and market shifts.
An end-to-end data science application that automates data cleaning, exploratory analysis, model selection, and provides AI-powered business insights through an interactive chat interface.

## Key Features

### 🧹 **Smart Data Cleaning**
- Hybrid cleaning with **AI recommendations** (Groq API) and **rule-based logic**
- Automatic handling of:
  - Missing values (imputation or removal)
  - Data type conversions
  - High-cardinality columns
  - Irrelevant features
- Detailed cleaning report with before/after comparison

### 📊 **Automated EDA**
- AI-generated statistical summaries
- Dynamic visualization of:
  - Temporal trends
  - Correlations
  - Distributions
  - Categorical breakdowns
- Custom query system for on-demand visualizations
- Interactive dashboard for all generated charts

### 🤖 **Intelligent Model Selection**
- Context-aware problem detection:
  - Time-series forecasting (LSTM)
  - Regression (XGBoost)
  - Classification (Random Forest)
- Automated feature engineering
- Model training and evaluation
- Business insights from predictions

### 💬 **AI Chat Assistant**
- Dataset-aware Q&A (LangChain agents)
- Technical and business question answering
- Conversation memory
- Clear chat functionality

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-business-analytics.git
   cd ai-business-analytics
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
## Setup API keys:
3. Create apikey.py with:
   ```bash
   apikey = "your_openai_api_key_here"
4. Run the application:
    ```bash
    streamlit run app.py


## Usage Guide
Upload your dataset (CSV format) via the sidebar

Navigate through tabs:

## Data Cleaning
Toggle between AI-powered and rules-based cleaning

Review cleaning recommendations

Download cleaned dataset

## EDA
View automated analysis

Explore generated visualizations

Create custom charts with natural language queries

## Model Selection
Select target variable

Let the app detect problem type

Train model and view predictions

Get business insights

## Chat Assistant
Ask questions about your data

Get technical explanations

Clear conversation history when needed

## Technical Architecture

graph TD

    A[User Uploads CSV] --> B[Data Cleaning]
    
    B --> C[Exploratory Analysis]
    
    C --> D[Model Selection]
    
    D --> E[Predictions]
    
    E --> F[Business Insights]
    
    G[Chat Assistant] <--> C
    
    G <--> D
## Dependencies
Package	--- Purpose

Streamlit ---	Web interface

Pandas/Numpy ---	Data manipulation

Scikit-learn ---	Machine learning

Plotly ---	Visualizations

XGBoost	--- Regression models

TensorFlow	--- LSTM networks

LangChain	--- AI chat agent

Groq	--- LLM API

## Configuration
Edit these key parameters:

apikey.py - OpenAI API key

Line 117 - Groq API key

Model parameters in Model Selection tab

## Troubleshooting
Missing values error: Use the cleaning module first

Visualization issues: Check for non-numeric columns

Model training failures: Ensure target variable is properly defined

API errors: Verify keys are valid and have sufficient quota

## Roadmap
Add more model options

Enable multi-file analysis

Implement user accounts

Add exportable reports

This README provides:
1. Clear feature overview
2. Setup instructions
3. Usage guide for each module
4. Technical architecture visualization
5. Dependency list
6. Configuration notes
7. Troubleshooting help
8. License information
9. Future roadmap

The file is structured to help both technical and non-technical users understand and use the application effectively.
