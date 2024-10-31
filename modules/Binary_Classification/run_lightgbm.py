# run_lightgbm.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def run_lightgbm(df):
    st.subheader("Run LightGBM for Binary Classification")

    # ターゲット列、ID列、不均衡データの処理方法を選択
    id_column = st.multiselect(
        "Select ID column (optional)",  
        df.columns.tolist(), 
        key="lgbm_id_column"
    )
    target_column = st.selectbox(
        "Select the target column", 
        [col for col in df.columns if df[col].nunique() == 2], 
        key="lgbm_target_column"
    )
    imbalance_method = st.multiselect(
        "Select imbalance handling method", 
        ["Undersampling"],  # SMOTEを削除
        key="lgbm_imbalance_method"
    )
    validation_method = st.selectbox(
        "Select validation method", 
        ["KFold"],  # Stratified KFoldを削除
        key="lgbm_validation_method"
    )

    # KFoldの分割数を設定
    if validation_method == "KFold":
        n_splits = st.number_input(
            "Number of splits (K)", 
            min_value=2, 
            max_value=20, 
            value=5, 
            key="lgbm_n_splits"
        )

    # スコアリングメトリックを選択
    scoring_metric = st.selectbox(
        "Select scoring metric", 
        ["LogLoss", "ROC_AUC", "PR_AUC"], 
        key="lgbm_scoring_metric"
    )

    # モデル実行ボタン
    if st.button("Run LightGBM", key="lgbm_run_button"):
        progress = st.progress(0)
        status_text = st.empty()

        try:
            # データをJSON形式に変換
            data_json = df.to_dict(orient='records')

            # FastAPIのエンドポイントにリクエストを送信
            response = requests.post("http://localhost:8080/train", json={
                "data": data_json,
                "target_column": target_column,
                "id_column": id_column,
                "imbalance_method": imbalance_method,
                "n_splits": n_splits,
                "scoring_metric": scoring_metric
            })

            if response.status_code == 200:
                result = response.json()
                st.session_state.model_trained = True
                st.session_state.best_params = result["best_params"]
                st.session_state.y_test = result["y_test"]
                st.session_state.y_pred_proba = result["y_pred_proba"]
                st.session_state.roc_auc = result["roc_auc"]

                progress.progress(100)
                status_text.text("Modeling process completed!")
            else:
                st.error(f"An error occurred: {response.text}")
                progress.empty()
                status_text.empty()
                return

        except Exception as e:
            st.error(f"An error occurred: {e}")
            progress.empty()
            status_text.empty()
            return

    # モデルがトレーニングされた場合の表示
    if st.session_state.get('model_trained', False):
        y_pred_proba = np.array(st.session_state.y_pred_proba)
        y_test = np.array(st.session_state.y_test)
        best_params = st.session_state.best_params
        roc_auc = st.session_state.roc_auc

        # 以下、予測結果の可視化や評価指標の計算を行う
        # （元のコードから必要な部分を移植）

        # スライダーを定義して閾値を取得
        threshold = st.slider(
            'Select Threshold for Classification',
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )

        # 現在の閾値での予測を再計算
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)

        # 混同行列と評価メトリックの計算
        from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

        conf_matrix_threshold = confusion_matrix(y_test, y_pred_threshold)
        accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
        precision_threshold = precision_score(y_test, y_pred_threshold)
        recall_threshold = recall_score(y_test, y_pred_threshold)
        f1_threshold = f1_score(y_test, y_pred_threshold)

        # ROC曲線の計算
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # ROC曲線のプロット
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.2f})',
            line=dict(color='indianred')
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='gray', dash='dash')
        ))
        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc)

        # 混同行列の表示
        conf_matrix_df = pd.DataFrame(
            conf_matrix_threshold,
            index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )
        st.write("### Confusion Matrix")
        st.table(conf_matrix_df)

        # 評価メトリックの表示
        evaluation_metrics_threshold = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [
                accuracy_threshold,
                precision_threshold,
                recall_threshold,
                f1_threshold
            ]
        }
        eval_df_threshold = pd.DataFrame(evaluation_metrics_threshold)
        st.write(f"### Evaluation Metrics (Threshold = {threshold:.2f})")
        st.table(eval_df_threshold)

        # ハイパーパラメータ表示ボタン
        if st.button("Show Best Hyperparameters"):
            st.write("### Best Parameters:")
            st.write(best_params)

        # 特徴量重要度の表示
        if st.button("Show Feature Importance"):
            # モデルをロード
            import joblib
            model = joblib.load("lightgbm_model.pkl")

            importances = model.feature_importances_
            feature_names = model.feature_name_
            indices = np.argsort(importances)[::-1]
            sorted_features = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=sorted_importances, y=sorted_features, color='indianred', ax=ax3)
            ax3.set_title("Feature Importance")
            ax3.set_xlabel("Importance")
            ax3.set_ylabel("Features")

            st.pyplot(fig3)
            plt.close(fig3)
