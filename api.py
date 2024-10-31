# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve,
    roc_auc_score, log_loss, precision_recall_curve, auc, confusion_matrix
)
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
import joblib

app = FastAPI()

# リクエスト用のデータモデル
class TrainRequest(BaseModel):
    data: list
    target_column: str
    id_column: list = None
    imbalance_method: list = []
    tuning_method: str = "Random Search"
    validation_method: str = "KFold"
    n_splits: int = 5
    scoring_metric: str = "ROC_AUC"

class PredictRequest(BaseModel):
    data: list

# モデルと関連データを保存するためのグローバル変数
model = None
category_columns = []
best_params = None

@app.post("/train")
def train_model(request: TrainRequest):
    global model, category_columns, best_params
    try:
        df = pd.DataFrame(request.data)
        target_column = request.target_column
        id_column = request.id_column
        imbalance_method = request.imbalance_method
        tuning_method = request.tuning_method
        validation_method = request.validation_method
        n_splits = request.n_splits
        scoring_metric = request.scoring_metric

        # メトリックのマッピング
        metric_mapping = {
            "LogLoss": "neg_log_loss",
            "ROC_AUC": "roc_auc",
            "PR_AUC": "average_precision"
        }

        # ID列の処理
        id_column_value = id_column[0] if id_column else None

        # データの準備
        X = df.drop(columns=[target_column] + ([id_column_value] if id_column_value else []))
        category_columns = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        X[category_columns] = X[category_columns].astype('category')
        y = df[target_column]

        # データの分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 不均衡データの処理
        if "Undersampling" in imbalance_method:
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)

        # バリデーション方法の設定
        if validation_method == "KFold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # ハイパーパラメータチューニング
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

            # モデルを保存
            joblib.dump(model, 'lightgbm_model.pkl')

        return {"message": "Model trained successfully", "best_params": best_params}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_model(request: PredictRequest):
    global model, category_columns
    try:
        if model is None:
            # モデルをロード
            model = joblib.load('lightgbm_model.pkl')
        X_new = pd.DataFrame(request.data)
        X_new[category_columns] = X_new[category_columns].astype('category')
        y_pred_proba = model.predict_proba(X_new)[:, 1]
        return {"predictions": y_pred_proba.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
