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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df[target_column], color='indianred')
    ax.set_title(f'Time Series of {target_column}')
    ax.set_xlabel('Date')
    ax.set_ylabel(target_column)
    plt.tight_layout()
    
    # fig 引数を st.pyplot に渡す
    st.pyplot(fig)
