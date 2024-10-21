# modules/Effectiveness_Verification/show_histogram_treatment.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_histogram_treatment(df):
    # カラムの選択
    target = st.selectbox('Select Target Column:', df.columns, key="ev_target_column")

    fig, ax = plt.subplots()
    df[target].astype('int').value_counts().sort_index().plot(kind='bar', ax=ax, color='orangered')
    ax.set_title('Target Column Bar Plot')
    ax.set_xlabel('Target')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    plt.close(fig)

    # 各カラムのプロット
    other_columns = [col for col in df.columns if col != target]
    for col in other_columns:
        st.subheader(f'{col} Column Plot')

        # カラムの値ごとのカウント
        value_counts = df[col].value_counts().sort_index()

        # カラムの値ごとのtarget=1の割合
        proportions = df[df[target] == 1][col].value_counts().sort_index() / value_counts

        fig, ax1 = plt.subplots()

        # バープロットの作成
        ax1.bar(value_counts.index, value_counts, color='lightgrey', label='Count')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='y')

        # Y軸の共有
        ax2 = ax1.twinx()
        ax2.plot(proportions.index, proportions, color='orangered', marker='o', label='Proportion of target=1')
        ax2.set_ylabel('Proportion of target=1', color='orangered')
        ax2.tick_params(axis='y')

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
