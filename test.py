import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, f1_score, precision_recall_curve, auc, classification_report, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

# データセットの生成
np.random.seed(42)
n_samples = 1000
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(1, 2, n_samples)
X3 = np.random.normal(-1, 1, n_samples)
X4 = np.random.normal(2, 3, n_samples)
X5 = np.random.choice(['A', 'B', 'C'], n_samples)
X6 = np.random.randint(18, 70, n_samples)
y = (X1 + X2 - X3 + (X6 / 10) + np.random.normal(0, 0.5, n_samples)) > 3
y = y.astype(int)
X5_dummies = pd.get_dummies(X5, prefix='Category')

df = pd.DataFrame({
    'Feature_1': X1,
    'Feature_2': X2,
    'Feature_3': X3,
    'Feature_4': X4,
    'Age': X6,
    'Target': y
})
df = pd.concat([df, X5_dummies], axis=1)

# データセットを表示
st.write("Dataset:")
st.dataframe(df.head())

# LightGBMのパラメータ設定
if st.sidebar.checkbox("Run LightGBM"):
    # カテゴリカルカラムの自動検出
    category_columns = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # ターゲットが2クラスのものだけを選択肢として表示
    target_column = st.selectbox("Select the target column", [col for col in df.columns if df[col].nunique() == 2])

    # チューニング手法の選択
    tuning_method = st.selectbox("Select hyperparameter tuning method", ["Random Search", "Bayesian Optimization (Optuna)"])

    # 検証方法を選択
    validation_method = st.selectbox("Select validation method", ["KFold", "Stratified KFold"])

    if validation_method in ["KFold", "Stratified KFold"]:
        n_splits = st.number_input("Number of splits (K)", min_value=2, max_value=20, value=5)

    # 評価指標を選択
    scoring_metric = st.selectbox("Select scoring metric", ["LogLoss", "ROC_AUC", "PR_AUC"])

    # モデル実行ボタン
    if st.button("Run LightGBM"):
        progress = st.progress(0)
        status_text = st.empty()

        # データを分割
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # カテゴリカル特徴量の指定
        categorical_features = [X.columns.get_loc(col) for col in category_columns]

        # 不均衡データを処理
        status_text.text("Handling imbalanced data...")
        imbalance_method = st.sidebar.selectbox("Select imbalance handling method", ["None", "Undersampling", "SMOTE"])
        if imbalance_method == "Undersampling":
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        elif imbalance_method == "SMOTE":
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        progress.progress(25)

        # 検証方法を設定
        status_text.text("Setting up validation method...")
        if validation_method == "KFold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif validation_method == "Stratified KFold":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        progress.progress(50)

        # ハイパーパラメータチューニング
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 31, 150),
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                'max_depth': trial.suggest_int('max_depth', -1, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                'cat_smooth': trial.suggest_float('cat_smooth', 0.01, 1.0),
                'cat_l2': trial.suggest_float('cat_l2', 1.0, 100.0)
            }
            model = LGBMClassifier(**params, random_state=42)
            model.fit(X_train, y_train, categorical_feature=categorical_features)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            if scoring_metric == "LogLoss":
                return log_loss(y_test, y_pred_proba)
            elif scoring_metric == "ROC_AUC":
                return roc_auc_score(y_test, y_pred_proba)
            elif scoring_metric == "PR_AUC":
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                return auc(recall, precision)

        # Optunaを使用したベイズ最適化
        if tuning_method == "Bayesian Optimization (Optuna)":
            study = optuna.create_study(direction='minimize' if scoring_metric == "LogLoss" else 'maximize')
            study.optimize(objective, n_trials=100)
            best_params = study.best_params
            st.write("Best Parameters: ", best_params)

            # 最適なパラメータでモデルを再実行
            model = LGBMClassifier(**best_params, random_state=42)
            model.fit(X_train, y_train, categorical_feature=categorical_features)

        elif tuning_method == "Random Search":
            # RandomizedSearchCVによるランダムサーチ
            param_grid = {
                'n_estimators': np.arange(50, 1000, 50),
                'learning_rate': np.linspace(0.001, 0.3, 10),
                'num_leaves': np.arange(31, 151, 10),
                'boosting_type': ['gbdt', 'dart', 'goss'],
                'max_depth': np.arange(-1, 51, 5),
                'min_child_samples': np.arange(10, 101, 10),
                'reg_alpha': np.linspace(0.0, 2.0, 10),
                'reg_lambda': np.linspace(0.0, 2.0, 10),
                'cat_smooth': np.linspace(0.01, 1.0, 10),
                'cat_l2': np.linspace(1.0, 100.0, 10)
            }
            search = RandomizedSearchCV(LGBMClassifier(random_state=42), param_distributions=param_grid, n_iter=50, scoring=scoring_metric.lower(), cv=cv)
            search.fit(X_train, y_train, categorical_feature=categorical_features)
            best_params = search.best_params_
            st.write("Best Parameters: ", best_params)

            # 最適なパラメータでモデルを再実行
            model = search.best_estimator_

        # 予測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # ROC曲線の表示
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}', color='indianred')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()

        # 混合行列の表示
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot()

        # 予測値のヒストグラムとボックスプロット
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
        ax1.hist(y_pred_proba[y_test == 1], bins=20, color='indianred', alpha=0.6, label='Positive')
        ax1.hist(y_pred_proba[y_test == 0], bins=20, color='grey', alpha=0.6, label='Negative')
        ax1.set_title('Prediction Probability Distribution')
        ax1.legend()

        sns.boxplot(x=y_test, y=y_pred_proba, ax=ax2, palette={0: 'grey', 1: 'indianred'})
        ax2.set_title('Boxplot of Predictions')
        st.pyplot(fig)

        # 特徴量重要度の表示
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=X.columns[indices], palette="OrRd")
        plt.title("Feature Importance")
        st.pyplot()

        # モデル完了
        progress.progress(100)
        status_text.text("Modeling process completed!")

# CatBoostのパラメータ設定
if st.sidebar.checkbox("Run CatBoost"):
    # ラベル列を選択（任意）
    label_columns = st.multiselect("Select columns to id (optional)", df.columns)

    # ターゲットが2クラスのものだけを選択肢として表示
    target_column = st.selectbox("Select the target column", [col for col in df.columns if df[col].nunique() == 2])

    # 自動的に数値型の特徴量を選択
    features = df.drop(columns=[target_column])
    feature_columns = features.select_dtypes(include=['number']).columns.tolist()

    # 必要であればターゲットをエンコード
    le = LabelEncoder()
    df[target_column] = le.fit_transform(df[target_column])

    # 不均衡データの処理方法を選択(任意)
    imbalance_method = st.multiselect("Select imbalance handling method（optional）", ["Undersampling", "SMOTE"])

    # 検証方法を選択
    validation_method = st.selectbox("Select validation method", ["KFold", "Stratified KFold"])

    if validation_method in ["KFold", "Stratified KFold"]:
        n_splits = st.number_input("Number of splits (K)", min_value=2, max_value=20, value=5)

    # 評価指標を選択
    scoring_metric = st.selectbox("Select scoring metric of tuning method", ["ROC_AUC", "PR_AUC", "Logloss", "F1 Score"])
    param_grid_catboost = {
        'iterations': st.sidebar.slider("iterations", 100, 1000, step=100),
        'learning_rate': st.sidebar.select_slider("learning_rate", options=[0.001, 0.01, 0.05, 0.1, 0.2, 0.3]),
        'depth': st.sidebar.slider("depth", 4, 10, step=1),
        'l2_leaf_reg': st.sidebar.slider("l2_leaf_reg", 1, 9, step=1),
        'bagging_temperature': st.sidebar.slider("bagging_temperature", 0.0, 2.0, step=0.5),
        'random_strength': st.sidebar.slider("random_strength", 1, 10, step=1)
    }

    # モデル実行ボタン
    if st.button("Run CatBoost"):
        progress = st.progress(0)
        status_text = st.empty()

        # データを分割
        X = df.drop(columns=[target_column] + label_columns)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 不均衡データを処理
        status_text.text("Handling imbalanced data...")
        if imbalance_method == "Undersampling":
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        elif imbalance_method == "SMOTE":
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        progress.progress(25)

        # 検証方法を設定
        status_text.text("Setting up validation method...")
        if validation_method == "KFold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif validation_method == "Stratified KFold":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        progress.progress(50)

        # CatBoostモデルのインスタンス作成と実行
        model = CatBoostClassifier(random_state=42, **param_grid_catboost, silent=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # モデルの評価
        status_text.text("Evaluating CatBoost model...")
        if scoring_metric == "ROC_AUC":
            score = roc_auc_score(y_test, y_pred_proba)
        elif scoring_metric == "PR_AUC":
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            score = auc(recall, precision)
        elif scoring_metric == "Logloss":
            score = log_loss(y_test, y_pred_proba)
        elif scoring_metric == "F1 Score":
            score = f1_score(y_test, y_pred)

        st.subheader('CatBoost Model Performance')
        st.write(f"{scoring_metric}: {score:.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # ROC曲線の表示
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'CatBoost (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})', color='orangered')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot()

        progress.progress(100)
        status_text.text("CatBoost modeling completed!")
