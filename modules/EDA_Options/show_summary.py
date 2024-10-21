# modules/EDA_Options/show_summary.py

import streamlit as st
import pandas as pd

def summarize_dataframe(df):
    """
    指定されたデータフレームのカラム名、データ型、NULLの数、0の数、ユニーク数を含む
    新しいデータフレームを返す関数。

    Parameters:
    df (pd.DataFrame): サマリーを作成するデータフレーム

    Returns:
    pd.DataFrame: サマリー情報を含むデータフレーム
    """
    # カラム名のリスト
    columns = df.columns

    # カラムのデータ型のリスト
    dtypes = df.dtypes

    # カラムごとのnullの数のリスト
    null_counts = df.isnull().sum()

    # カラムごとの0の数のリスト
    zero_counts = (df == 0).sum()

    # カラムごとのユニーク数のリスト
    unique_counts = df.nunique()

    # 新しいデータフレームの作成
    summary_df = pd.DataFrame({
        'Column Name': columns,
        'Data Type': dtypes.values,
        'Missing Values': null_counts.values,
        'Number Of "0" Values': zero_counts.values,
        'Unique Values': unique_counts.values
    })

    return summary_df

def show_summary(df):
    st.subheader("Show DataFrame")
    number_input = st.slider("Select number of rows to display:", min_value=1, max_value=len(df), value=10)
    st.dataframe(df.head(number_input))
    
    st.subheader("Missing Values etc...")
    st.dataframe(summarize_dataframe(df))
    
    st.subheader("Basic Statistics")
    st.write(df.describe().T)
