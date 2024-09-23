# modules/EDA_Options/show_histogram.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_distribution(df, feature, target, statistic):
    if target is not None:
        plot_feature_with_target(df, feature, target, statistic)
    else:
        plot_feature_without_target(df, feature)

def plot_feature_with_target(df, feature, target, statistic):
    # プロットの設定
    if pd.api.types.is_numeric_dtype(df[feature]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # 数値型特徴量のプロット
    if pd.api.types.is_numeric_dtype(df[feature]):
        unique_vals = df[feature].nunique()
        if unique_vals > 20:
            # 20分割のbinを作成
            bins = pd.qcut(df[feature], q=20, duplicates='drop')
            bin_centers = df.groupby(bins)[feature].mean()
            grouped = df.groupby(bins)[target]
        else:
            bin_centers = df[feature].unique()
            grouped = df.groupby(feature)[target]

        # 選択された統計量の計算
        if statistic == 'mean':
            target_stat = grouped.mean()
        elif statistic == 'max':
            target_stat = grouped.max()
        elif statistic == 'min':
            target_stat = grouped.min()
        elif statistic == 'mode':
            target_stat = grouped.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        else:
            target_stat = grouped.mean()  # デフォルトは平均

        # ヒストグラムのプロット
        sns.histplot(df[feature], bins=20 if unique_vals > 20 else unique_vals, kde=False, color="lightgrey",
                     label=feature, ax=ax1)
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel(feature)
        ax1.legend(loc='upper left')

        # ターゲット関連のプロット
        ax3 = ax1.twinx()
        if pd.api.types.is_numeric_dtype(df[feature]):
            # 数値型の場合は折れ線グラフ
            sns.lineplot(x=bin_centers, y=target_stat.values, color="orangered",
                         label=f'{target} {statistic}', ax=ax3, ci=None)
        else:
            # カテゴリ型の場合は点プロット
            sns.scatterplot(x=bin_centers, y=target_stat.values,
                            color="orangered", ax=ax3, label=f'{target} {statistic}')

        ax3.set_ylabel(f'{target} {statistic}')
        ax3.legend(loc='upper right')

        # 箱ひげ図の表示
        sns.boxplot(x=df[feature], ax=ax2, color="indianred")

    # カテゴリ型特徴量のプロット
    else:
        df[feature].value_counts().plot(kind='bar', color="grey",
                                        label=feature, ax=ax1)
        ax1.set_ylabel('Count')
        ax1.set_xlabel(feature)
        ax1.legend(loc='upper left')

        # ターゲット関連のプロット
        grouped = df.groupby(feature)[target]
        if statistic == 'mean':
            target_stat = grouped.mean()
        elif statistic == 'max':
            target_stat = grouped.max()
        elif statistic == 'min':
            target_stat = grouped.min()
        elif statistic == 'mode':
            target_stat = grouped.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
        else:
            target_stat = grouped.mean()

        ax3 = ax1.twinx()
        # カテゴリ型なので点プロット
        sns.scatterplot(x=target_stat.index, y=target_stat.values,
                        color="orangered", ax=ax3, label=f'{target} {statistic}')
        ax3.set_ylabel(f'{target} {statistic}')
        ax3.legend(loc='upper right')

    # タイトルの設定
    plt.suptitle(f'{feature} Distribution and {target} {statistic.capitalize()}', fontsize=19, color="grey")

    # レイアウトの調整と表示
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)
    plt.close(fig)

def plot_feature_without_target(df, feature):
    # プロットの設定
    if pd.api.types.is_numeric_dtype(df[feature]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})

        unique_vals = df[feature].nunique()
        sns.histplot(df[feature], bins=20 if unique_vals > 20 else unique_vals, kde=False, color="lightgrey",
                     label=feature, ax=ax1)

        ax1.set_ylabel('Frequency')
        ax1.set_xlabel(feature)
        ax1.legend(loc='upper left')

        # 箱ひげ図の表示
        sns.boxplot(x=df[feature], ax=ax2, color="indianred")

    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        df[feature].value_counts().plot(kind='bar', color="grey",
                                        label=feature, ax=ax1)
        ax1.set_ylabel('Count')
        ax1.set_xlabel(feature)
        ax1.legend(loc='upper left')

    # タイトルの設定
    plt.suptitle(f'{feature} Distribution', fontsize=19, color="grey")

    # レイアウトの調整と表示
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)
    plt.close(fig)

def show_histogram(df):
    st.subheader("Show Histogram")
    
    # ユーザーにプロットしたい変数とターゲット変数を選択させる
    x = st.selectbox("Select Column", df.columns.tolist())

    # 'None'を選択肢に追加
    columns_with_none = ['None'] + df.columns.tolist()
    target = st.selectbox("Select Target Column", columns_with_none)

    if target == 'None':
        target = None

    # 統計量の選択
    statistic_options = ['mean', 'max', 'min', 'mode']
    selected_statistic = st.selectbox("Select Statistic", statistic_options)

    if st.button("Generate Plot"):
        if x:
            plot_feature_distribution(df, x, target, selected_statistic)
