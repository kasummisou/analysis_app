import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve,
    roc_auc_score, log_loss, precision_recall_curve, auc, confusion_matrix
)
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

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
    tuning_method = st.selectbox(
        "Select hyperparameter tuning method", 
        ["Random Search"],  # Optunaを削除
        key="lgbm_tuning_method"
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

    # セッションステートの初期化
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'y_pred_proba' not in st.session_state:
        st.session_state.y_pred_proba = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'pr_thresholds' not in st.session_state:
        st.session_state.pr_thresholds = None
    if 'f1_scores' not in st.session_state:
        st.session_state.f1_scores = None
    if 'best_threshold' not in st.session_state:
        st.session_state.best_threshold = None
    if 'best_f1' not in st.session_state:
        st.session_state.best_f1 = None
    if 'fpr' not in st.session_state:
        st.session_state.fpr = None
    if 'tpr' not in st.session_state:
        st.session_state.tpr = None
    if 'roc_thresholds' not in st.session_state:
        st.session_state.roc_thresholds = None
    if 'youden_index' not in st.session_state:
        st.session_state.youden_index = None
    if 'youden_threshold' not in st.session_state:
        st.session_state.youden_threshold = None
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.5  # 初期値を0.5に設定

    # メトリックのマッピング
    metric_mapping = {
        "LogLoss": "neg_log_loss",
        "ROC_AUC": "roc_auc",
        "PR_AUC": "average_precision"
    }

    # モデル実行ボタン
    if st.button("Run LightGBM", key="lgbm_run_button"):
        progress = st.progress(0)
        status_text = st.empty()

        try:
            # ID列の処理
            id_column_value = id_column[0] if len(id_column) > 0 else None

            # データの準備
            X = df.drop(columns=[target_column] + ([id_column_value] if id_column_value else []))
            category_columns = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            X[category_columns] = X[category_columns].astype('category')
            y = df[target_column]

            # データの分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            progress.progress(10)

            # 不均衡データの処理
            if "Undersampling" in imbalance_method:
                rus = RandomUnderSampler(random_state=42)
                X_train, y_train = rus.fit_resample(X_train, y_train)
                progress.progress(20)

            # バリデーション方法の設定
            if validation_method == "KFold":
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            progress.progress(40)

            # ハイパーパラメータチューニング（Random Searchのみ）
            if tuning_method == "Random Search":
                param_grid = {
                    'n_estimators': list(range(50, 1001, 50)),
                    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
                    'num_leaves': list(range(31, 151, 10)),
                    'boosting_type': ['gbdt', 'dart', 'goss'],
                    'max_depth': list(range(-1, 51, 5)),
                    'min_child_samples': list(range(10, 101, 10)),
                    'reg_alpha': [round(x, 2) for x in np.linspace(0.0, 2.0, 21)],
                    'reg_lambda': [round(x, 2) for x in np.linspace(0.0, 2.0, 21)]
                }
                search = RandomizedSearchCV(
                    LGBMClassifier(random_state=42),
                    param_distributions=param_grid,
                    n_iter=50,
                    scoring=metric_mapping.get(scoring_metric, 'roc_auc'),
                    cv=cv,
                    verbose=0,
                    random_state=42,
                    n_jobs=-1
                )
                search.fit(X_train, y_train, categorical_feature=category_columns)
                best_params = search.best_params_
                model = search.best_estimator_
                progress.progress(60)

                # ハイパーパラメータをセッションステートに保存
                st.session_state.best_params = best_params

            # モデル予測
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            progress.progress(70)

            # デフォルトの閾値（0.5）でのメトリック計算
            y_pred = (y_pred_proba >= 0.5).astype(int)
            precision_default = precision_score(y_test, y_pred)
            recall_default = recall_score(y_test, y_pred)
            f1_default = f1_score(y_test, y_pred)
            accuracy_default = accuracy_score(y_test, y_pred)

            # Precision-Recall曲線の計算とベストなF1スコアの閾値取得
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (precision_curve[:-1] + recall_curve[:-1])
            if len(pr_thresholds) > 0:
                best_index = np.argmax(f1_scores)
                best_threshold = pr_thresholds[best_index]
                best_f1 = f1_scores[best_index]
            else:
                best_threshold = 0.5
                best_f1 = f1_default

            # ROC曲線とYouden's J統計量の計算
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
            youden_index = tpr - fpr
            youden_best_index = np.argmax(youden_index)
            youden_best_threshold = roc_thresholds[youden_best_index]

            # セッションステートへの保存
            st.session_state.model_trained = True
            st.session_state.y_pred_proba = y_pred_proba
            st.session_state.y_test = y_test
            st.session_state.model = model
            st.session_state.pr_thresholds = pr_thresholds
            st.session_state.f1_scores = f1_scores
            st.session_state.best_threshold = best_threshold
            st.session_state.best_f1 = best_f1
            st.session_state.fpr = fpr
            st.session_state.tpr = tpr
            st.session_state.roc_thresholds = roc_thresholds
            st.session_state.youden_index = youden_index
            st.session_state.youden_threshold = youden_best_threshold

            progress.progress(90)
            status_text.text("Modeling process completed!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            progress.empty()
            status_text.empty()

    # モデルがトレーニングされた場合の表示
    if st.session_state.get('model_trained', False):
        y_pred_proba = st.session_state.y_pred_proba
        y_test = st.session_state.y_test
        model = st.session_state.model
        pr_thresholds = st.session_state.pr_thresholds
        f1_scores = st.session_state.f1_scores
        best_threshold = st.session_state.best_threshold
        best_f1 = st.session_state.best_f1
        fpr = st.session_state.fpr
        tpr = st.session_state.tpr
        roc_thresholds = st.session_state.roc_thresholds
        youden_index = st.session_state.youden_index
        youden_best_threshold = st.session_state.youden_threshold

        # スライダーを最初に定義して値を取得
        threshold = st.slider(
            'Select Threshold for Classification',
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.01
        )
        st.session_state.threshold = threshold  # スライダーの値をセッションステートに保存

        # 現在の閾値での予測を再計算
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)

        # 現在の閾値でのFPRとTPRを計算
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        current_fpr = fp / (fp + tn)
        current_tpr = tp / (tp + fn)

        # ROC曲線のプロット
        fig_roc = go.Figure()

        # ROC曲線の追加
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})',
            line=dict(color='indianred')
        ))

        # ランダム予測のライン
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guess',
            line=dict(color='gray', dash='dash')
        ))

        # Youden's J統計量最大のポイントを追加（青色の点）
        tn_youden, fp_youden, fn_youden, tp_youden = confusion_matrix(y_test, y_pred_proba >= youden_best_threshold).ravel()
        youden_fpr = fp_youden / (fp_youden + tn_youden)
        youden_tpr = tp_youden / (tp_youden + fn_youden)

        fig_roc.add_trace(go.Scatter(
            x=[youden_fpr], y=[youden_tpr],
            mode='markers',
            name=f'Optimal Threshold (Youden\'s J = {youden_best_threshold:.2f})',
            marker=dict(color='blue', size=10)
        ))

        # 現在の閾値のポイントを追加（緑色の点）
        fig_roc.add_trace(go.Scatter(
            x=[current_fpr], y=[current_tpr],
            mode='markers',
            name=f'Current Threshold ({threshold:.2f})',
            marker=dict(color='green', size=10)
        ))

        fig_roc.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            legend=dict(x=0.6, y=0.1)
        )

        # ROC曲線の表示
        st.plotly_chart(fig_roc)

        # 混同行列
        conf_matrix_threshold = confusion_matrix(y_test, y_pred_threshold)
        conf_matrix_df = pd.DataFrame(
            conf_matrix_threshold,
            index=['Actual Negative', 'Actual Positive'],
            columns=['Predicted Negative', 'Predicted Positive']
        )

        # 評価メトリック
        accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
        precision_threshold = precision_score(y_test, y_pred_threshold)
        recall_threshold = recall_score(y_test, y_pred_threshold)
        f1_threshold = f1_score(y_test, y_pred_threshold)

        # 評価メトリックのデータフレーム
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

        # 予測確率の分布
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.hist(y_pred_proba[y_test == 1], bins=20, color='indianred', alpha=0.6, label='Positive')
        ax1.hist(y_pred_proba[y_test == 0], bins=20, color='grey', alpha=0.6, label='Negative')
        ax1.set_title('Prediction Probability Distribution')
        ax1.legend()

        sns.boxplot(
            y=y_test.astype(str),          # y_testを文字列に変換
            x=y_pred_proba,                # 予測確率
            ax=ax2,                        # 2つ目のサブプロット
            palette={'0': 'grey', '1': 'indianred'}  # カラーマッピング
        )
        ax2.set_title('Boxplot of Predictions')

        st.pyplot(fig)
        plt.close(fig)

        # 混同行列の表示
        st.write("### Confusion Matrix")
        st.table(conf_matrix_df)

        # 評価メトリックの表示
        st.write(f"### Evaluation Metrics (Threshold = {threshold:.2f})")
        st.table(eval_df_threshold)

        # ハイパーパラメータ表示ボタン
        if st.button("Show Best Hyperparameters"):
            st.write("### Best Parameters (Random Search):")
            st.write(st.session_state.best_params)

        # 特徴量重要度のプロット
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

        st.progress(100)

        # モデリングプロセス完了
        st.success("Modeling process completed!")
