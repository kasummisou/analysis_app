import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import seaborn as sns

def run_stl_adf_autocorrelation(df):
    st.subheader("STL Decomposition, ADF Test, and Autocorrelation Analysis")
    
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
    
    # データの概要表示
    st.write("### Data after Initial Setting")
    st.dataframe(df.head())
    
    # STL分析の設定
    st.header("STL Decomposition")
    model_type = st.selectbox('Select STL Model', ['Additive', 'Multiplicative'])
    seasonal_period = st.number_input('Seasonal Period', min_value=1, value=13)
    
    if st.button('Run STL Decomposition'):
        with st.spinner("Performing STL Decomposition..."):
            try:
                stl = STL(df[target_column], period=seasonal_period, robust=True)
                result = stl.fit()
                
                # トレンド、季節性、残差のプロット
                fig, axs = plt.subplots(3, 1, figsize=(12, 9))
                axs[0].plot(result.trend, color='indianred')
                axs[0].set_title('Trend Component')
                axs[1].plot(result.seasonal, color='orangered')
                axs[1].set_title('Seasonal Component')
                axs[2].plot(result.resid, color='grey')
                axs[2].set_title('Residual Component')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during STL Decomposition: {e}")
    
    # ADF検定の設定
    st.header("ADF Test")
    difference_degree = st.number_input('Difference Degree', min_value=0, max_value=2, value=1)
    
    if st.button('Run ADF Test'):
        with st.spinner("Performing ADF Test..."):
            try:
                if difference_degree > 0:
                    differenced_series = df[target_column].diff(difference_degree).dropna()
                else:
                    differenced_series = df[target_column]
                adf_result = adfuller(differenced_series)
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"P-value: {adf_result[1]:.4f}")
                for key, value in adf_result[4].items():
                    st.write(f'Critical Value ({key}): {value:.4f}')
                
                # ADF検定結果のプロット
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(differenced_series, color='orangered')
                ax.set_title(f'Differenced Series (Degree={difference_degree})')
                ax.set_xlabel('Date')
                ax.set_ylabel(target_column)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during ADF Test: {e}")
    
    # 自己相関と偏自己相関の設定
    st.header("Autocorrelation Analysis")
    acf_lag = st.number_input('Select Lag', min_value=1, max_value=100, value=20)
    
    if st.button('Run Autocorrelation Analysis'):
        with st.spinner("Performing Autocorrelation Analysis..."):
            try:
                if difference_degree > 0:
                    differenced_series = df[target_column].diff(difference_degree).dropna()
                else:
                    differenced_series = df[target_column]
                
                fig, ax = plt.subplots(2, 1, figsize=(12, 8))
                plot_acf(differenced_series, lags=acf_lag, ax=ax[0], color='indianred')
                ax[0].set_title("Autocorrelation Function (ACF)")
                plot_pacf(differenced_series, lags=acf_lag, ax=ax[1], color='orangered')
                ax[1].set_title("Partial Autocorrelation Function (PACF)")
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during Autocorrelation Analysis: {e}")
