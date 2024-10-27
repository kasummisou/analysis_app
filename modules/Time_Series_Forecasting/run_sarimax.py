import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go  # Plotlyのインポート
import plotly.express as px  # Plotly Expressのインポート

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mase(y_true, y_pred, naive_forecast=None):
    """Mean Absolute Scaled Error"""
    n = len(y_true)
    if naive_forecast is None:
        d = np.abs(np.diff(y_true)).sum() / (n - 1)
    else:
        d = np.mean(np.abs(y_true - naive_forecast))
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def run_sarimax(df):
    st.subheader("SARIMAX Model")
    
    # 初期設定とクイックEDA
    st.header("Initial Setting & Quick EDA for SARIMAX")
    
    # 日付列の選択
    date_column = st.selectbox("Please select the Date column", df.columns)
    # ターゲット列の選択
    target_column = st.selectbox("Please select the Target column", [col for col in df.columns if col != date_column])
    
    # datetime に変換
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        st.success(f"Successfully converted `{date_column}` to datetime.")
    except Exception as e:
        st.error(f"Error converting `{date_column}` to datetime: {e}")
        st.stop()
    
    # 日付をインデックスとして設定
    df.set_index(date_column, inplace=True)
    
    # データの概要表示
    st.subheader("Data Overview")
    st.write(df.head())
    st.write(f"Data points: {len(df)}")
    
    # 欠損値のチェック
    st.subheader("Missing Values Check")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    
    if missing_values.sum() > 0:
        st.warning("There are missing values in the dataset. Please handle them before modeling.")
        # 欠損値を処理するオプションを提供
        handle_missing = st.selectbox("How would you like to handle missing values?", ["Drop", "Interpolate", "Fill with Mean"])
        if handle_missing == "Drop":
            df = df.dropna()
            st.success("Dropped rows with missing values.")
        elif handle_missing == "Interpolate":
            df = df.interpolate()
            st.success("Interpolated missing values.")
        elif handle_missing == "Fill with Mean":
            df = df.fillna(df.mean())
            st.success("Filled missing values with column means.")
    
    # データ型の表示
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    # 数値列と非数値列を区別
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    
    if non_numeric_cols:
        st.warning("Non-numeric columns detected. They will be excluded from modeling.")
    
    # ID列の選択（オプション）
    id_column = st.multiselect("Please select ID columns (optional)", df.columns.tolist())
    
    # 外生変数の選択（オプション）
    exogenous_columns = st.multiselect(
        "Please select Exogenous variables (optional)",
        [col for col in numeric_cols if col not in id_column + [target_column]]
    )
    
    # 予測期間の設定
    forecast_periods = st.number_input("Please enter the number of forecast periods (days)", min_value=1, max_value=365, value=30)
    
    # パラメータ設定方法の選択
    param_method = st.radio("Please select the parameter setting method", ["AutoARIMA", "Manual"])
    
    # 季節性パラメータの設定
    seasonal_period = st.number_input("Seasonal Period (m)", min_value=1, max_value=24, value=12)
    
    # 評価指標の選択
    st.subheader("Select Evaluation Metric for Hyperparameter Tuning")
    eval_metric = st.selectbox("Evaluation Metric", ["MSE", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "AIC", "BIC"])
    
    # 評価指標の関数を定義
    def evaluate_metric(y_true, y_pred, metric, result=None):
        if metric == "MSE":
            return mean_squared_error(y_true, y_pred)
        elif metric == "MAE":
            return mean_absolute_error(y_true, y_pred)
        elif metric == "RMSE":
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == "MAPE":
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif metric == "sMAPE":
            return smape(y_true, y_pred)
        elif metric == "MASE":
            return mase(y_true, y_pred)
        elif metric == "AIC" and result is not None:
            return result.aic
        elif metric == "BIC" and result is not None:
            return result.bic
        else:
            return None
    
    if param_method == "Manual":
        st.subheader("Set SARIMAX Parameters Manually")
        p = st.number_input("Order of AR (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Order of Integration (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("Order of MA (q)", min_value=0, max_value=5, value=1)
        seasonal_p = st.number_input("Seasonal AR (P)", min_value=0, max_value=2, value=1)
        seasonal_d = st.number_input("Seasonal Integration (D)", min_value=0, max_value=1, value=0)
        seasonal_q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=2, value=1)
    else:
        st.subheader("Hyperparameter Tuning Range")
        p_range = st.slider("AR order (p) range", 0, 5, (0, 5))
        d_range = st.slider("Differencing order (d) range", 0, 2, (0, 2))
        q_range = st.slider("MA order (q) range", 0, 5, (0, 5))
        seasonal_p_range = st.slider("Seasonal AR (P) range", 0, 2, (0, 2))
        seasonal_d_range = st.slider("Seasonal Differencing (D) range", 0, 1, (0, 1))
        seasonal_q_range = st.slider("Seasonal MA (Q) range", 0, 2, (0, 2))
    
    # モデルの実行ボタン
    if st.button("Run SARIMAX"):
        with st.spinner("Fitting SARIMAX model..."):
            try:
                # データの準備
                selected_columns = [target_column] + exogenous_columns
                df_filtered = df[selected_columns].copy()
                
                # 外生変数の処理
                if exogenous_columns:
                    exog = df_filtered[exogenous_columns]
                else:
                    exog = None
                
                # ターゲット変数が定数でないことを確認
                if df_filtered[target_column].nunique() == 1:
                    st.error("The target variable is constant. ARIMA models require variability in the target.")
                    st.stop()
                
                # 定常性の確認（ADFテスト）
                st.subheader("Stationarity Check (ADF Test)")
                adf_result = adfuller(df_filtered[target_column].dropna())
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"p-value: {adf_result[1]:.4f}")
                if adf_result[1] < 0.05:
                    st.write("The series is stationary.")
                else:
                    st.write("The series is non-stationary. Differencing will be applied if needed.")
                
                # データの分割
                st.subheader("Train-Test Split")
                train_size = len(df_filtered) - forecast_periods
                if train_size <= 0:
                    st.error("Not enough data points for the specified forecast period. Please reduce the forecast period.")
                    st.stop()
                train = df_filtered.iloc[:train_size]
                test = df_filtered.iloc[train_size:]
                st.write(f"Training data points: {len(train)}")
                st.write(f"Testing data points: {len(test)}")
                
                # 外生変数の分割
                if exogenous_columns:
                    exog_train = train[exogenous_columns]
                    exog_test = test[exogenous_columns]
                else:
                    exog_train = None
                    exog_test = None
                
                # ハイパーパラメータのチューニング
                if param_method == "AutoARIMA":
                    st.write("Performing hyperparameter tuning using AIC/BIC...")
                    # パラメータ範囲の定義
                    p_values = range(p_range[0], p_range[1]+1)
                    d_values = range(d_range[0], d_range[1]+1)
                    q_values = range(q_range[0], q_range[1]+1)
                    P_values = range(seasonal_p_range[0], seasonal_p_range[1]+1)
                    D_values = range(seasonal_d_range[0], seasonal_d_range[1]+1)
                    Q_values = range(seasonal_q_range[0], seasonal_q_range[1]+1)
                    
                    best_score = float('inf') if eval_metric not in ["AIC", "BIC"] else None
                    best_order = None
                    best_seasonal_order = None
                    for p in p_values:
                        for d in d_values:
                            for q in q_values:
                                for P in P_values:
                                    for D in D_values:
                                        for Q in Q_values:
                                            order = (p, d, q)
                                            seasonal_order = (P, D, Q, seasonal_period)
                                            try:
                                                model = SARIMAX(
                                                    train[target_column],
                                                    order=order,
                                                    seasonal_order=seasonal_order,
                                                    exog=exog_train,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False
                                                )
                                                result = model.fit(disp=False)
                                                # 予測
                                                pred = result.get_forecast(steps=len(test), exog=exog_test)
                                                y_pred = pred.predicted_mean
                                                y_true = test[target_column]
                                                
                                                # 評価指標を計算
                                                if eval_metric in ["AIC", "BIC"]:
                                                    metric_value = evaluate_metric(y_true, y_pred, eval_metric, result=result)
                                                else:
                                                    metric_value = evaluate_metric(y_true, y_pred, eval_metric)
                                                
                                                # ベストスコアの更新
                                                if eval_metric in ["AIC", "BIC"]:
                                                    if best_score is None or metric_value < best_score:
                                                        best_score = metric_value
                                                        best_order = order
                                                        best_seasonal_order = seasonal_order
                                                else:
                                                    if metric_value < best_score:
                                                        best_score = metric_value
                                                        best_order = order
                                                        best_seasonal_order = seasonal_order
                                            except Exception as e:
                                                continue
                    if best_order is not None and best_seasonal_order is not None:
                        st.write(f"Best parameters found: order={best_order}, seasonal_order={best_seasonal_order}")
                        if eval_metric in ["AIC", "BIC"]:
                            st.write(f"Best {eval_metric}: {best_score:.4f}")
                        else:
                            st.write(f"Best {eval_metric}: {best_score:.4f}")
                        # 最良モデルでフィッティング
                        model = SARIMAX(
                            train[target_column],
                            order=best_order,
                            seasonal_order=best_seasonal_order,
                            exog=exog_train,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        result = model.fit(disp=False)
                    else:
                        st.error("No suitable model found. Please adjust the hyperparameter ranges or data.")
                        st.stop()
                else:
                    # 手動パラメータ
                    model = SARIMAX(
                        train[target_column],
                        order=(p, d, q),
                        seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
                        exog=exog_train,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    st.write(f"Using manual parameters: order=({p},{d},{q}), seasonal_order=({seasonal_p},{seasonal_d},{seasonal_q},{seasonal_period})")
                    result = model.fit(disp=False)
                
                # 予測の実行
                pred = result.get_forecast(steps=forecast_periods, exog=exog_test)
                pred_ci = pred.conf_int()
                pred_mean = pred.predicted_mean
                
                # 予測結果のプロット
                fig_forecast = go.Figure()
                # トレーニングデータ
                fig_forecast.add_trace(go.Scatter(
                    x=train.index,
                    y=train[target_column],
                    mode='lines',
                    name='Training Data',
                    line=dict(color='grey')
                ))
                # テストデータ
                fig_forecast.add_trace(go.Scatter(
                    x=test.index,
                    y=test[target_column],
                    mode='lines',
                    name='Actual Test Data',
                    line=dict(color='blue')
                ))
                # 予測データ
                fig_forecast.add_trace(go.Scatter(
                    x=pred_mean.index,
                    y=pred_mean,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='orangered')
                ))
                # 信頼区間の追加
                fig_forecast.add_trace(go.Scatter(
                    x=pred_ci.index,
                    y=pred_ci.iloc[:, 1],
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Upper Confidence Interval',
                    showlegend=False
                ))
                fig_forecast.add_trace(go.Scatter(
                    x=pred_ci.index,
                    y=pred_ci.iloc[:, 0],
                    fill='tonexty',
                    fillcolor='rgba(255, 69, 0, 0.2)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Lower Confidence Interval',
                    showlegend=False
                ))
                # レイアウトの更新
                fig_forecast.update_layout(
                    title="Forecast Results vs Actual",
                    xaxis_title="Date",
                    yaxis_title=target_column,
                    hovermode="x unified"
                )
                # 範囲スライダーとボタンの追加
                fig_forecast.update_xaxes(
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=3, label="3m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
                # プロットの表示
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # 予測精度を評価
                st.subheader("Forecast Accuracy Metrics")
                if len(test) > 0:
                    y_true = test[target_column]
                    y_pred = pred_mean[:len(test)]
                    mse = mean_squared_error(y_true, y_pred)
                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    smape_value = smape(y_true, y_pred)
                    mase_value = mase(y_true, y_pred)
                    aic = result.aic
                    bic = result.bic
                    
                    # メトリクスの DataFrame を作成します
                    metrics = {
                        "Metric": ["MSE", "MAE", "RMSE", "MAPE", "sMAPE", "MASE", "AIC", "BIC"],
                        "Value": [mse, mae, rmse, mape, smape_value, mase_value, aic, bic]
                    }
                    df_metrics = pd.DataFrame(metrics)
                    
                    # メトリクスをテーブルとして表示します
                    st.table(df_metrics)
                else:
                    st.warning("Insufficient data for evaluating forecast accuracy. Please reduce the forecast period.")
                
                # モデルの概要を表示
                with st.expander("Model Summary"):
                    st.text(result.summary())
                
                # 診断プロットを表示
                with st.expander("Model Diagnostics"):
                    fig_diag = result.plot_diagnostics(figsize=(15, 12))
                    
                    # 各サブプロットの色をindianredに変更
                    for ax in fig_diag.axes:
                        for line in ax.get_lines():
                            line.set_color('indianred')
                        for patch in ax.patches:
                            patch.set_color('indianred')
                        for collection in ax.collections:
                            if hasattr(collection, 'set_color'):
                                collection.set_color('indianred')
                    
                    st.pyplot(fig_diag)
            
            except Exception as e:
                st.error(f"Error during SARIMAX modeling: {e}")
                st.write("Please refer to the troubleshooting guide: [pmdarima Troubleshooting](http://alkaline-ml.com/pmdarima/no-successful-model.html)")

