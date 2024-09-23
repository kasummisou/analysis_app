# modules/Effectiveness_Verification/run_bayesian_ab_test.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import seaborn as sns

def run_bayesian_ab_test(df):
    st.subheader("Bayesian A/B Test")
    
    # グループとターゲットの列を選択
    group_column = st.selectbox("Group Column", df.columns)
    target_column = st.selectbox("Target Column", df.columns)
    
    # アウトカムがバイナリか連続かを選択
    outcome_type = st.selectbox("Outcome Type", ["Binary (0/1)", "Continuous (numeric)"])
    
    # サンプリング回数のスライダー設定
    st.subheader("Select the Number of MCMC Samples")
    num_samples = st.slider("Number of samples", min_value=1000, max_value=10000, value=2000, step=100)
    
    # バーンイン期間の設定
    burn_in = st.slider("Burn-in Period (Number of samples to discard)", min_value=0, max_value=num_samples//2, value=num_samples//10)
    
    if st.button("Run Bayesian AB Test"):
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
            group_A_data = df[df[group_column] == group_A_label][target_column].values
            group_B_data = df[df[group_column] == group_B_label][target_column].values
            progress_bar.progress(10)
        
        # ベイジアンモデルの構築とサンプリング
        try:
            status_text.text("Building the Bayesian model...")
            with pm.Model() as model:
                if outcome_type == "Binary (0/1)":
                    prior_A = pm.Beta('prior_A', alpha=1, beta=1)
                    prior_B = pm.Beta('prior_B', alpha=1, beta=1)
                    likelihood_A = pm.Bernoulli('likelihood_A', p=prior_A, observed=group_A_data)
                    likelihood_B = pm.Bernoulli('likelihood_B', p=prior_B, observed=group_B_data)
                else:
                    prior_mean_A = pm.Normal('prior_mean_A', mu=0, sigma=20)
                    prior_mean_B = pm.Normal('prior_mean_B', mu=0, sigma=20)
                    prior_std = pm.HalfNormal('prior_std', sigma=10)
                    likelihood_A = pm.Normal('likelihood_A', mu=prior_mean_A, sigma=prior_std, observed=group_A_data)
                    likelihood_B = pm.Normal('likelihood_B', mu=prior_mean_B, sigma=prior_std, observed=group_B_data)
                progress_bar.progress(30)
    
                # サンプリング
                status_text.text("Running MCMC sampling...")
                trace = pm.sample(draws=num_samples, tune=burn_in, progressbar=False, return_inferencedata=True)
                progress_bar.progress(60)
        except Exception as e:
            st.error(f"An error occurred during model sampling: {e}")
            progress_bar.empty()
            return
    
        # プロットスタイルの設定
        status_text.text("Setting plot styles...")
        sns.set_theme(style='darkgrid')
        plt.rcParams.update({
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "figure.facecolor": "white",
            "figure.edgecolor": "black",
            "lines.color": "darkgray",
            "patch.edgecolor": "black",
            "patch.force_edgecolor": True,
            "grid.color": "gray"
        })
        progress_bar.progress(70)
    
        # MCMCトレースと事後分布の表示
        status_text.text("Plotting MCMC trace and posterior distributions...")
        st.subheader("MCMC Trace and Posterior Distributions")
        
        # トレース内の変数名を表示
        st.write("Trace Variables:", list(trace.posterior.data_vars.keys()))
    
        az.plot_trace(trace, combined=True, 
                     trace_kwargs={"color": "indianred"}, 
                     hist_kwargs={"color": "indianred"})
        st.pyplot(plt.gcf())
        plt.close()
        progress_bar.progress(80)
    
        # 結果のサマリー表示
        status_text.text("Displaying summary of results...")
        st.subheader("Results")
        st.write(az.summary(trace, round_to=4))
        progress_bar.progress(90)
    
        # 処置群と対照群の事後分布のヒストグラムと箱ひげ図の作成
        status_text.text("Creating histograms and boxplots...")
        st.subheader("Posterior Distributions for Control and Treatment Groups")
        if outcome_type == "Binary (0/1)":
            data = {
                "Control (A)": trace.posterior.prior_A.values.flatten(),
                "Treatment (B)": trace.posterior.prior_B.values.flatten()
            }
        else:
            data = {
                "Control (A)": trace.posterior.prior_mean_A.values.flatten(),
                "Treatment (B)": trace.posterior.prior_mean_B.values.flatten()
            }
        
        df_hist = pd.DataFrame(data)
        fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 8))
    
        # ヒストグラムの描画
        axs[0].hist(df_hist['Control (A)'], bins=30, color='grey', alpha=0.7, label='Control')
        axs[0].hist(df_hist['Treatment (B)'], bins=30, color='OrangeRed', alpha=0.7, label='Treatment')
        axs[0].set_title("Histogram of Posterior Distributions")
        axs[0].legend()
    
        # 箱ひげ図の描画（横向き）
        sns.boxplot(data=df_hist, orient='h', ax=axs[1], palette=["grey", "OrangeRed"])
        axs[1].set_title("Boxplot of Posterior Distributions")
        axs[1].set_yticklabels(['Control (A)', 'Treatment (B)'])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        progress_bar.progress(95)
    
        # 処置群と対照群の差分計算
        status_text.text("Calculating differences between groups...")
        if outcome_type == "Binary (0/1)":
            delta = trace.posterior.prior_B.values.flatten() - trace.posterior.prior_A.values.flatten()
        else:
            delta = trace.posterior.prior_mean_B.values.flatten() - trace.posterior.prior_mean_A.values.flatten()
    
        # 差分のプロット
        st.subheader("Difference between Treatment and Control")
        fig, ax = plt.subplots()
    
        # 差分を正と負に分ける
        delta_pos = delta[delta >= 0]
        delta_neg = delta[delta < 0]
    
        # 正の差分と負の差分のヒストグラムを描画
        ax.hist(delta_neg, bins=50, density=True, alpha=0.6, color='grey', label='Delta <= 0')
        ax.hist(delta_pos, bins=50, density=True, alpha=0.6, color='OrangeRed', label='Delta > 0')
    
        # ゼロの位置に縦線を引く
        ax.axvline(x=0, color='indianred', linestyle='--')
        ax.set_xlabel("Difference in Means (B - A)" if outcome_type == "Continuous (numeric)" else "Difference in Proportions (B - A)")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        progress_bar.progress(100)
    
        # 処置群が対照群より優れている確率を計算
        prob = (delta > 0).mean()
        st.write(f"Probability that Treatment is better than Control: {prob:.2%}")
    
        # 最終メッセージ
        status_text.text("Bayesian AB Test completed successfully!")
