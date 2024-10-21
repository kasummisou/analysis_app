# modules/EDA_Options/show_crosstab_plot.py

import streamlit as st
import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

def show_crosstab_plot(df):
    # ユーザーに2つのカラムを選択させる
    columns = df.columns.tolist()
    col1 = st.selectbox("Please select the first column", columns, key="crosstab_col1")
    col2 = st.selectbox("Please select the second column", columns, key="crosstab_col2")

    if st.button("Generate Crosstab Plot and Test Independence"):
        if col1 and col2:
            # データフレームのコピーを作成して、元のデータを変更しないようにする
            df_copy = df.copy()

            # col1の処理
            if pd.api.types.is_numeric_dtype(df_copy[col1]):
                unique_vals_col1 = df_copy[col1].nunique(dropna=True)
                if unique_vals_col1 > 10:
                    # 10分割のビンに分割
                    df_copy[col1] = pd.qcut(df_copy[col1], q=10, duplicates='drop')
            else:
                unique_vals_col1 = df_copy[col1].nunique(dropna=True)

            # col2の処理
            if pd.api.types.is_numeric_dtype(df_copy[col2]):
                unique_vals_col2 = df_copy[col2].nunique(dropna=True)
                if unique_vals_col2 > 10:
                    # 10分割のビンに分割
                    df_copy[col2] = pd.qcut(df_copy[col2], q=10, duplicates='drop')
            else:
                unique_vals_col2 = df_copy[col2].nunique(dropna=True)

            # ビン分割後にNaNが発生する可能性があるため、欠損値を削除
            df_copy = df_copy.dropna(subset=[col1, col2])

            # クロス集計の計算
            crosstab = pd.crosstab(df_copy[col1], df_copy[col2])

            # カイ二乗検定の実施
            chi2, p, dof, expected = chi2_contingency(crosstab)

            # カイ二乗検定の結果を表で表示
            test_results = pd.DataFrame({
                'Statistic': ['Chi-squared value', 'Degrees of freedom', 'P-value'],
                'Value': [chi2, dof, p]
            })

            st.write("Chi-squared Test Result:")
            st.table(test_results)

            # クロス集計の表示（オプション）
            st.write("Crosstab table:")
            st.dataframe(crosstab)

            # プロットの作成
            fig, ax = plt.subplots(figsize=(8, 6))
            crosstab.plot(kind='barh', stacked=True, ax=ax, colormap='PuRd')
            ax.set_title(f'Crosstab of {col1} and {col2}')
            ax.set_xlabel('Frequency')
            ax.set_ylabel(col1)
            plt.tight_layout()

            # Streamlit上にプロットを表示
            st.pyplot(fig)
        else:
            st.warning("Please select two columns.")
