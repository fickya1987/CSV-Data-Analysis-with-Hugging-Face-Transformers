# This is task to analysis the sales data using LLM model from hugging face , 
# that take csv file and gives visual representation


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import numpy as np

# Initialize the Hugging Face model pipeline
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text, model):
    input_length = len(text.split())
    max_length = max(20, min(input_length, 200))
    min_length = max(5, min(input_length // 4, 50))
    return model(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

def preprocess_data(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def generate_automatic_visuals(df):
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not num_cols.empty:
        st.subheader("Basic Statistics")
        st.write(df[num_cols].describe())

        st.subheader("Correlation Heatmap")
        st.write("This heatmap shows the correlation between numerical features in the dataset. High correlation values (close to 1 or -1) indicate a strong relationship between features.")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt.gcf())

        st.subheader("Distribution of Numerical Features")
        for col in num_cols:
            st.write(f"**Distribution of {col}:**")
            st.write("This histogram displays the distribution of values for the selected numerical feature. The shape of the distribution can reveal insights into the underlying data, such as skewness or the presence of outliers.")
            plt.figure()
            sns.histplot(df[col], kde=True)
            st.pyplot(plt.gcf())
    else:
        st.warning("No numerical data found in the dataset for correlation and distribution plots.")

    cat_cols = df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        st.subheader("Count Plots for Categorical Features")
        for col in cat_cols:
            st.write(f"**Count Plot for {col}:**")
            st.write("This count plot displays the frequency of each category within the categorical feature. It helps in understanding the distribution and imbalance of categories.")
            plt.figure()
            sns.countplot(y=col, data=df, palette="viridis")
            st.pyplot(plt.gcf())
    else:
        st.warning("No categorical data found in the dataset for count plots.")

# Streamlit app
st.title('CSV Data Analysis with Hugging Face Transformers')

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)  

    df = preprocess_data(df)

    st.write("### Data Preview")
    st.write(df.head())

    model = load_model()

    text_columns = df.select_dtypes(include=['object']).columns
    if text_columns.empty:
        st.write("No text columns found in the data.")
    else:
        text_column = text_columns[0]
        st.write(f"### Summarizing Text from Column: {text_column}")

        summaries = df[text_column].dropna().apply(lambda x: generate_summary(x, model))
        
        st.write("### Summarized Texts")
        st.write(summaries.head())

        summary_lengths = summaries.apply(len)
        plt.figure(figsize=(10, 5))
        sns.histplot(summary_lengths, bins=30, kde=True)
        plt.title('Distribution of Summary Lengths')
        st.pyplot(plt)

        st.write(f"### Descriptive Statistics of '{text_column}'")
        st.write(df[text_column].describe())
        
        st.write("### Value Counts")
        for col in df.select_dtypes(include=['object']).columns:
            st.write(f"#### {col}")
            st.write(df[col].value_counts())

    with st.spinner("Generating automatic analysis and visualizations..."):
        try:
            generate_automatic_visuals(df)
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
