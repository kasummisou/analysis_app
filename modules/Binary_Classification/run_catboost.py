# modules/Binary_Classification/run_catboost.py
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    log_loss,
    f1_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def run_catboost(df):
    st.header("Run CatBoost for Binary Classification")
    
    # ラベル列の選択（任意）
    label_columns = st.multiselect("Select columns to exclude from training (optional)", df.columns)
    
    # ターゲットが2クラスのものだけを選択肢として表示
    target_column = st.selectbox("Select the target column", [col for col in df.columns if df[col].nunique() == 2])
    
    # 自動的に数値型の特徴量を選択
    features = df.drop(columns=[target_column] + label_columns)
    feature_columns = features.select_dtypes(include=['number']).columns.tolist()
    
    # ターゲットのエンコード
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
    
    # モデル実行ボタン
    if st.button("Run CatBoost"):
        progress = st.progress(0)
        status_text = st.empty()
        status_text.text("Initializing...")
        
        # データを分割
        X = df.drop(columns=[target_column] + label_columns)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 不均衡データを処理
        status_text.text("Handling imbalanced data...")
        if "Undersampling" in imbalance_method:
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        if "SMOTE" in imbalance_method:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        progress.progress(10)
        
        # 検証方法を設定
        status_text.text("Setting up validation method...")
        if validation_method == "KFold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif validation_method == "Stratified KFold":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        progress.progress(20)
        
        # CatBoostモデルのインスタンス作成と実行
        status_text.text("Fitting CatBoost model...")
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            eval_metric='AUC',
            verbose=0
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)
        progress.progress(40)
        
        # モデルの予測
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        progress.progress(50)
        
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
        progress.progress(60)
        
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
        progress.progress(70)
        
        # Feature Importanceの表示
        st.subheader("Feature Importance")
        importances = model.get_feature_importance()
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='orangered')
        plt.title("Feature Importance")
        st.pyplot()
        progress.progress(80)
        
        # モデルプロセス完了
        progress.progress(100)
        status_text.text("CatBoost modeling completed successfully!")
