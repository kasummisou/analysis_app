# modules/Effectiveness_Verification/run_t_statistic_ab_test.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_t_statistic_ab_test(df):
    st.subheader("T-Statistic A/B Test")
    
    # テストタイプの選択
    test_type = st.selectbox("Select Test Type", ["A/B Test", "A/A Test"])
    
    if test_type == "A/B Test":
        st.header("A/B Test")
        
        # グループとターゲットの列を選択
        group_column = st.selectbox("Select Group Column", df.columns)
        target_column = st.selectbox("Select Target Column", df.columns)
        
        # 有意水準の選択
        alpha = st.slider("Select Significance Level (alpha)", 0.0, 0.25, 0.05)
        
        # テスト実行ボタン
        if st.button("Run T-Statistic A/B Test"):
            # 進行状況バーとステータステキストの初期化
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing...")
            
            # グループごとにデータを分割
            status_text.text("Splitting data into groups...")
            unique_groups = df[group_column].unique()
            if len(unique_groups) != 2:
                st.error(f"The selected group column must contain exactly 2 groups, but found {len(unique_groups)} groups.")
                progress_bar.empty()
                return
            else:
                group_A_label, group_B_label = unique_groups
                group_A_data = df[df[group_column] == group_A_label][target_column]
                group_B_data = df[df[group_column] == group_B_label][target_column]
                progress_bar.progress(10)
            
            # グループの統計量の計算
            status_text.text("Calculating group statistics...")
            mean_A = group_A_data.mean()
            mean_B = group_B_data.mean()
            std_A = group_A_data.std()
            std_B = group_B_data.std()
            
            # 統計量の表示
            data = {
                "Group": ["Control (A)", "Treatment (B)"],
                "Mean": [mean_A, mean_B],
                "Standard Deviation": [std_A, std_B]
            }
            df_stats = pd.DataFrame(data)
            st.subheader("Group Statistics")
            st.table(df_stats)
            progress_bar.progress(20)
            
            # t検定の実行
            status_text.text("Performing T-Test...")
            t_stat, p_value = stats.ttest_ind(group_A_data, group_B_data, equal_var=False)
            df_t = len(group_A_data) + len(group_B_data) - 2  # 自由度
            
            # t分布の生成
            x = np.linspace(-4, 4, 1000)
            t_dist = stats.t.pdf(x, df_t)
            
            # プロットの作成
            status_text.text("Plotting results...")
            st.write("Standardized Distribution of Mean Difference Between Groups")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # t分布のプロット
            ax.plot(x, t_dist, color='orangered', label='T-distribution')
            
            # 臨界領域の塗りつぶし
            critical_value = stats.t.ppf(1 - alpha/2, df_t)
            ax.fill_between(x, t_dist, where=(x >= critical_value) | (x <= -critical_value), color='indianred', alpha=0.6)
            
            # t統計量のプロット
            ax.axvline(x=t_stat, color='grey', linestyle='-')
            ax.text(t_stat, max(t_dist), f'T-Statistic: {t_stat:.2f}', 
                    verticalalignment='bottom', horizontalalignment='right', color='black')
            
            # 凡例の追加
            ax.legend()
            
            # プロットの表示
            st.pyplot(fig)
            plt.close(fig)
            progress_bar.progress(40)
            
            # t検定の結果の表示
            status_text.text("Displaying T-Test results...")
            Ttest_results = {
                "T-Statistic": [f"{t_stat:.4f}"],
                "P-Value": [f"{p_value:.4f}"],
            }
            df_ttest = pd.DataFrame(Ttest_results)
            st.write("T-Test Results")
            st.dataframe(df_ttest)
            progress_bar.progress(50)
            
    elif test_type == "A/A Test":
        st.header("A/A Test")
        
        # グループとターゲットの列を選択
        group_column = st.selectbox("Select Group Column", df.columns)
        target_column = st.selectbox("Select Target Column", df.columns)
        
        # 有意水準の選択
        alpha = st.slider("Select Significance Level (alpha)", 0.0, 0.25, 0.05)
        
        # テスト実行ボタン
        if st.button("Run T-Statistic A/A Test"):
            # 進行状況バーとステータステキストの初期化
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Initializing...")
            
            # グループごとにデータを分割し、サイズを均等化
            status_text.text("Splitting and equalizing data...")
            unique_groups = df[group_column].unique()
            if len(unique_groups) != 2:
                st.error(f"The selected group column must contain exactly 2 groups, but found {len(unique_groups)} groups.")
                progress_bar.empty()
                return
            else:
                group_0_label, group_1_label = unique_groups
                group_0_data = df[df[group_column] == group_0_label][target_column]
                group_1_data = df[df[group_column] == group_1_label][target_column]
                min_size = min(len(group_0_data), len(group_1_data))
                
                # ランダムサンプリング
                group_A_sample = group_0_data.sample(n=min_size, random_state=1)
                group_B_sample = group_1_data.sample(n=min_size, random_state=1)
                progress_bar.progress(10)
            
            # グループの統計量の計算
            status_text.text("Calculating group statistics...")
            mean_A = group_A_sample.mean()
            mean_B = group_B_sample.mean()
            std_A = group_A_sample.std()
            std_B = group_B_sample.std()
            
            # 統計量の表示
            group_stats = {
                "Group": ["Control (A)", "Treatment (B)"],
                "Mean": [mean_A, mean_B],
                "Standard Deviation": [std_A, std_B]
            }
            df_group_stats = pd.DataFrame(group_stats)
            st.subheader("Group Statistics")
            st.table(df_group_stats)
            progress_bar.progress(20)
            
            # t検定の実行
            status_text.text("Performing T-Test...")
            t_stat, p_value = stats.ttest_ind(group_A_sample, group_B_sample, equal_var=False)
            df_t = len(group_A_sample) + len(group_B_sample) - 2  # 自由度
            
            # t分布の生成
            x = np.linspace(-4, 4, 1000)
            t_dist = stats.t.pdf(x, df_t)
            
            # プロットの作成
            status_text.text("Plotting results...")
            st.write("Standardized Distribution of Mean Difference Between Groups")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # t分布のプロット
            ax.plot(x, t_dist, color='orangered', label='T-distribution')
            
            # 臨界領域の塗りつぶし
            critical_value = stats.t.ppf(1 - alpha/2, df_t)
            ax.fill_between(x, t_dist, where=(x >= critical_value) | (x <= -critical_value), color='indianred', alpha=0.6)
            
            # t統計量のプロット
            ax.axvline(x=t_stat, color='grey', linestyle='-')
            ax.text(t_stat, max(t_dist), f'T-Statistic: {t_stat:.2f}', 
                    verticalalignment='bottom', horizontalalignment='right', color='black')
            
            # 凡例の追加
            ax.legend()
            
            # プロットの表示
            st.pyplot(fig)
            plt.close(fig)
            progress_bar.progress(40)
            
            # t検定の結果の表示
            status_text.text("Displaying T-Test results...")
            Ttest_results = {
                "T-Statistic": [f"{t_stat:.4f}"],
                "P-Value": [f"{p_value:.4f}"],
            }
            df_ttest = pd.DataFrame(Ttest_results)
            st.write("T-Test Results")
            st.dataframe(df_ttest)
            progress_bar.progress(50)
            
            # 複数のA/Aテスト結果の表示
            st.subheader('Multiple A/A Tests to Check the Distribution of T-Statistics')
            # サンプル数の指定
            num_samples = st.number_input("Number of A/A Tests Conducted", min_value=1, value=100, step=1)
            
            # 複数A/Aテスト実行ボタン
            if st.button("Run Multiple A/A Tests"):
                status_text = st.empty()
                progress_bar = st.progress(0)
                status_text.text("Running multiple A/A tests...")
                t_statistics = []
                
                for i in range(num_samples):
                    # グループごとにデータを分割し、サイズを均等化
                    group_0_sample = group_0_data.sample(n=min_size, random_state=None)
                    group_1_sample = group_1_data.sample(n=min_size, random_state=None)
                    
                    # t検定の実行
                    if group_0_sample.std() != 0 and group_1_sample.std() != 0:
                        t_stat_i, _ = stats.ttest_ind(group_0_sample, group_1_sample, equal_var=False)
                        t_statistics.append(t_stat_i)
                    else:
                        # 標準偏差がゼロの場合はNaNを追加
                        t_statistics.append(np.nan)
                    
                    # 進行状況バーの更新
                    if (i + 1) % max(1, num_samples // 10) == 0:
                        progress = int((i + 1) / num_samples * 90)
                        progress_bar.progress(progress)
                
                # NaNを除外
                t_statistics = [t for t in t_statistics if not np.isnan(t)]
                
                # サマリー統計量の計算
                if t_statistics:
                    t_stat_mean = np.mean(t_statistics)
                    t_stat_std = np.std(t_statistics)
                else:
                    t_stat_mean = np.nan
                    t_stat_std = np.nan
                
                # サマリーの表示
                ttest_summary = {
                    "Target Column": [target_column],
                    "Mean T-Statistic": [f"{t_stat_mean:.4f}" if not np.isnan(t_stat_mean) else 'NaN'],
                    "Standard Deviation of T-Statistic": [f"{t_stat_std:.4f}" if not np.isnan(t_stat_std) else 'NaN'],
                }
                df_ttest_summary = pd.DataFrame(ttest_summary)
                
                status_text.text("Displaying summary of multiple A/A test results...")
                st.write("Summary of T-Test Results")
                st.dataframe(df_ttest_summary)
                progress_bar.progress(85)
                
                # T統計量のヒストグラムの作成
                if t_statistics:
                    st.write(f"Histogram of T-Statistics from {num_samples} A/A Tests")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(t_statistics, kde=True, color='orangered', ax=ax)
                    ax.set_xlabel("T-Statistic")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("No valid T-Statistics were calculated.")
                
                progress_bar.progress(100)
                status_text.text("Multiple A/A Tests completed successfully!")
