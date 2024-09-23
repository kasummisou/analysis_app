# modules/Time_Series_Forecasting/run_sarimax.py

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def run_sarimax(df):
    st.subheader("SARIMAX Model")
    
    # 必要なセッションステートからの情報を取得
    if 'target_column' not in st.session_state or 'date_column' not in st.session_state:
        st.error("Please perform initial setting and Quick EDA first.")
        st.stop()
    
    target_column = st.session_state['target_column']
    date_column = st.session_state['date_column']
    
    # SARIMAXモデルの設定
    st.header("SARIMAX Model Configuration")
    
    # ID列の選択（オプション）
    id_column = st.multiselect("Select ID column (optional)", df.columns.tolist())
    
    # エクソジェネス変数の選択（オプション）
    exogenous_columns = st.multiselect(
        "Select Exogenous variables (optional)",
        [col for col in df.columns if col not in id_column + [date_column, target_column]]
    )
    
    # フォーキャスト期間の設定
    forecast_periods = st.number_input("Number of periods to forecast", min_value=1, max_value=365, value=30)
    
    # パラメータ設定方法の選択
    param_method = st.radio("Choose parameter setting method", ["Manual", "AutoARIMA"])
    
    if param_method == "Manual":
        st.write("Set SARIMAX parameters:")
        p = st.number_input("Order of AR (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("Order of Integration (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("Order of MA (q)", min_value=0, max_value=5, value=1)
        seasonal_p = st.number_input("Seasonal AR (P)", min_value=0, max_value=5, value=1)
        seasonal_d = st.number_input("Seasonal Integration (D)", min_value=0, max_value=2, value=1)
        seasonal_q = st.number_input("Seasonal MA (Q)", min_value=0, max_value=5, value=1)
        seasonal_period = st.number_input("Seasonal Period", min_value=1, max_value=24, value=12)
    else:
        st.write("Using AutoARIMA to automatically determine SARIMAX parameters.")
    
    # モデル実行ボタン
    if st.button("Run SARIMAX"):
        st.spinner("Fitting SARIMAX model...")
        try:
            # データの準備
            selected_columns = [date_column, target_column] + exogenous_columns
            df_filtered = df[selected_columns].copy()
            
            # 外因変数がある場合の処理
            if exogenous_columns:
                exog = df_filtered[exogenous_columns]
            else:
                exog = None
            
            # モデルのフィッティング
            if param_method == "Manual":
                model = SARIMAX(
                    df_filtered[target_column],
                    order=(p, d, q),
                    seasonal_order=(seasonal_p, seasonal_d, seasonal_q, seasonal_period),
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                # AutoARIMAで最適なSARIMAXパラメータを探索
                auto_model = auto_arima(
                    df_filtered[target_column],
                    exogenous=exog,
                    seasonal=True,
                    m=12,
                    stepwise=True,
                    suppress_warnings=True,
                    trace=False
                )
                model = SARIMAX(
                    df_filtered[target_column],
                    order=auto_model.order,
                    seasonal_order=auto_model.seasonal_order,
                    exog=exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                st.write(f"Best parameters found by AutoARIMA: order={auto_model.order}, seasonal_order={auto_model.seasonal_order}")
            
            result = model.fit(disp=False)
            
            # 予測の実行
            pred = result.get_forecast(steps=forecast_periods, exog=exog[-forecast_periods:] if exog is not None else None)
            pred_ci = pred.conf_int()
            
            # 予測結果のプロット
            ax = df_filtered[target_column].plot(label='Observed', figsize=(14, 7))
            pred.predicted_mean.plot(ax=ax, label='Forecast', color='indianred')
            ax.fill_between(pred_ci.index,
                            pred_ci.iloc[:, 0],
                            pred_ci.iloc[:, 1], color='indianred', alpha=0.2)
            ax.set_xlabel('Date')
            ax.set_ylabel(target_column)
            plt.legend()
            st.pyplot()
            
            # 予測精度の評価（トレーニングデータの最後の部分をテストデータとして使用）
            st.subheader("Forecast Accuracy Metrics")
            y_true = df_filtered[target_column].iloc[-forecast_periods:]
            y_pred = pred.predicted_mean
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            st.write(f"MSE: {mse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAPE: {mape:.2f}%")
        
        except Exception as e:
            st.error(f"Error during SARIMAX modeling: {e}")
