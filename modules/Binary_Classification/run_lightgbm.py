# modules/Binary_Classification/run_lightgbm.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve,
    roc_auc_score, log_loss, precision_recall_curve, auc, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.integration import LightGBMPruningCallback

def run_lightgbm(df):
    st.subheader("Run LightGBM for Binary Classification")
    
    # Select target column, ID column, and imbalance handling method
    id_column = st.multiselect("Select ID column (optional)",  df.columns.tolist(), key="lgbm_id_column")
    target_column = st.selectbox("Select the target column", [col for col in df.columns if df[col].nunique() == 2], key="lgbm_target_column")
    imbalance_method = st.multiselect("Select imbalance handling method", ["Undersampling", "SMOTE"], key="lgbm_imbalance_method")
    tuning_method = st.selectbox("Select hyperparameter tuning method", ["Random Search", "Bayesian Optimization (Optuna)"], key="lgbm_tuning_method")
    validation_method = st.selectbox("Select validation method", ["KFold", "Stratified KFold"], key="lgbm_validation_method")

    # Set number of splits for KFold or Stratified KFold
    if validation_method in ["KFold", "Stratified KFold"]:
        n_splits = st.number_input("Number of splits (K)", min_value=2, max_value=20, value=5, key="lgbm_n_splits")

    # Select scoring metric
    scoring_metric = st.selectbox("Select scoring metric", ["LogLoss", "ROC_AUC", "PR_AUC"], key="lgbm_scoring_metric")

    # Model run button
    if st.button("Run LightGBM", key="lgbm_run_button"):
        progress = st.progress(0)
        status_text = st.empty()

        # ID column handling
        id_column_value = id_column[0] if len(id_column) > 0 else None

        # Prepare data
        X = df.drop(columns=[target_column] + ([id_column_value] if id_column_value else []))
        category_columns = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        X[category_columns] = X[category_columns].astype('category')
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        progress.progress(10)

        # Handle imbalanced data
        if "Undersampling" in imbalance_method:
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            progress.progress(20)
        if "SMOTE" in imbalance_method:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            progress.progress(30)

        # Setup validation method
        if validation_method == "KFold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif validation_method == "Stratified KFold":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        progress.progress(40)

        # Hyperparameter tuning with Optuna or Random Search
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 31, 150),
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'max_depth': trial.suggest_int('max_depth', -1, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0)
            }
            model = LGBMClassifier(**params, random_state=42)
            model.fit(X_train, y_train, categorical_feature=category_columns, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            if scoring_metric == "LogLoss":
                return log_loss(y_test, y_pred_proba)
            elif scoring_metric == "ROC_AUC":
                return roc_auc_score(y_test, y_pred_proba)
            elif scoring_metric == "PR_AUC":
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                return auc(recall, precision)

        if tuning_method == "Bayesian Optimization (Optuna)":
            study = optuna.create_study(direction='minimize' if scoring_metric == "LogLoss" else 'maximize')
            study.optimize(objective, n_trials=100, callbacks=[LightGBMPruningCallback(study, "auc")])
            best_params = study.best_params
            st.write("Best Parameters: ", best_params)
            model = LGBMClassifier(**best_params, random_state=42)
            model.fit(X_train, y_train, categorical_feature=category_columns)
            progress.progress(60)

        elif tuning_method == "Random Search":
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
                scoring='neg_log_loss' if scoring_metric == "LogLoss" else 'roc_auc' if scoring_metric == "ROC_AUC" else 'average_precision',
                cv=cv,
                verbose=0,
                random_state=42,
                n_jobs=-1
            )
            search.fit(X_train, y_train, categorical_feature=category_columns)
            best_params = search.best_params_
            st.write("Best Parameters: ", best_params)
            model = search.best_estimator_
            progress.progress(60)

        # モデルの予測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        progress.progress(70)

        # 各種メトリクスを計算
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)

        # Prediction probability distribution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.hist(y_pred_proba[y_test == 1], bins=20, color='indianred', alpha=0.6, label='Positive')
        ax1.hist(y_pred_proba[y_test == 0], bins=20, color='grey', alpha=0.6, label='Negative')
        ax1.set_title('Prediction Probability Distribution')
        ax1.legend()

        sns.boxplot(
            y=y_test.astype(str),          # y_testを0と1の文字列に変換
            x=y_pred_proba,                # 予測確率
            ax=ax2,                        # 2番目のプロット
            palette={'0': 'grey', '1': 'indianred'}  # 0はgrey、1はindianred
        )
        ax2.set_title('Boxplot of Predictions')
        st.pyplot(fig)
        plt.close(fig)
        progress.progress(80)

        # ROC curve and F1 Score calculation
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

        # F1 scores
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve)

        # 最大のF1スコアを持つ閾値を見つける
        best_threshold = pr_thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)

        # F1スコアが最大となる閾値に最も近いROC曲線の閾値を見つける
        closest_threshold_index = np.argmin(np.abs(roc_thresholds - best_threshold))
        best_fpr = fpr[closest_threshold_index]
        best_tpr = tpr[closest_threshold_index]

        # ROC curve with F1 Score marker
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}', color='indianred')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

        # F1 scoreが最大の閾値の点をプロット
        plt.scatter(best_fpr, best_tpr, color='indianred', label=f'Best F1 = {best_f1:.2f}', zorder=5)
        plt.title('ROC Curve with Best F1 Score')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()
        plt.close()
        progress.progress(90)

        # Confusion Matrix as heatmap
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', cbar=False)
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot()
        plt.close()
        progress.progress(95)

        # 表の作成 (Precision, Recall, F1 Score, Accuracy)
        evaluation_metrics = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred),
                f1_score(y_test, y_pred)
            ]
        }
        eval_df = pd.DataFrame(evaluation_metrics)
        st.table(eval_df)

        # Feature importance plot
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=X.columns[indices], color='indianred')
        plt.title("Feature Importance")
        st.pyplot()
        plt.close()
        progress.progress(100)

        # モデルプロセス完了
        status_text.text("Modeling process completed!")
