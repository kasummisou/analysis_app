# modules/Time_Series_Forecasting/initial_setting_quick_eda.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def initial_setting_quick_eda(df):
    st.subheader("Initial Setting & Quick EDA")
    
    # 日付カラムの選択
    date_column = st.selectbox("Select the Date column", df.columns)
    target_column = st.selectbox("Select the Target column", [col for col in df.columns if col != date_column])
    
    # 日付型への変換
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        st.success(f"Successfully converted `{date_column}` to datetime.")
    except Exception as e:
        st.error(f"Error converting `{date_column}` to datetime: {e}")
        st.stop()
    
    # 日付をインデックスに設定
    df.set_index(date_column, inplace=True)
    
    # データのソート
    df.sort_index(inplace=True)
    
    st.write("### Data after Initial Setting")
    st.dataframe(df.head())
    
    # 基本的な時系列プロット
    st.write("### Time Series Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[target_column], color='indianred')
    plt.title(f'Time Series of {target_column}')
    plt.xlabel('Date')
    plt.ylabel(target_column)
    plt.tight_layout()
    st.pyplot()
    
    # トレンドの可視化
    st.write("### Rolling Mean & Standard Deviation")
    rolling_window = st.slider("Select Rolling Window Size", min_value=1, max_value=365, value=30, step=1)
    
    rolling_mean = df[target_column].rolling(window=rolling_window).mean()
    rolling_std = df[target_column].rolling(window=rolling_window).std()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[target_column], label='Original', color='grey')
    plt.plot(rolling_mean, label='Rolling Mean', color='blue')
    plt.plot(rolling_std, label='Rolling Std', color='orange')
    plt.title(f'Rolling Mean & Std (Window = {rolling_window})')
    plt.xlabel('Date')
    plt.ylabel(target_column)
    plt.legend()
    plt.tight_layout()
    st.pyplot()
    
    # 季節性の可視化（オプション）
    if st.checkbox("Show Seasonal Decomposition"):
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(df[target_column], model='additive', period=rolling_window)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        decomposition.observed.plot(ax=axes[0], legend=False)
        axes[0].set_ylabel('Observed')
        decomposition.trend.plot(ax=axes[1], legend=False)
        axes[1].set_ylabel('Trend')
        decomposition.seasonal.plot(ax=axes[2], legend=False)
        axes[2].set_ylabel('Seasonal')
        decomposition.resid.plot(ax=axes[3], legend=False)
        axes[3].set_ylabel('Residual')
        plt.tight_layout()
        st.pyplot(fig)
