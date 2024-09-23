# modules/Time_Series_Forecasting/run_prophet.py

import streamlit as st
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def run_prophet(df):
    st.subheader("Prophet Model")
    
    # 必要なセッションステートからの情報を取得
    if 'target_column' not in st.session_state or 'date_column' not in st.session_state:
        st.error("Please perform initial setting and Quick EDA first.")
        st.stop()
    
    target_column = st.session_state['target_column']
    date_column = st.session_state['date_column']
    
    # Prophetモデルの設定
    st.header("Prophet Model Configuration")
    
    # エクソジェネス変数の選択（オプション）
    exogenous_columns = st.multiselect(
        "Select Exogenous variables (optional)",
        [col for col in df.columns if col not in [date_column, target_column]]
    )
    
    # フォーキャスト期間の設定
    forecast_periods = st.number_input("Number of periods to forecast", min_value=1, max_value=365, value=30)
    
    # パラメータ設定方法の選択
    param_method = st.radio("Choose parameter setting method", ["Default", "Manual"])
    
    if param_method == "Manual":
        st.write("Set Prophet parameters:")
        changepoint_prior_scale = st.number_input(
            "Changepoint Prior Scale",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.01
        )
        seasonality_prior_scale = st.number_input(
            "Seasonality Prior Scale",
            min_value=0.1,
            max_value=10.0,
            value=10.0,
            step=0.1
        )
    else:
        changepoint_prior_scale = 0.05
        seasonality_prior_scale = 10.0
    
    # モデル実行ボタン
    if st.button("Run Prophet"):
        st.spinner("Fitting Prophet model...")
        try:
            # データの準備
            df_prophet = df[[date_column, target_column] + exogenous_columns].copy()
            df_prophet.rename(columns={date_column: 'ds', target_column: 'y'}, inplace=True)
            
            # エクソジェネス変数がある場合
            if exogenous_columns:
                exog = df_prophet[exogenous_columns]
            else:
                exog = None
            
            # Prophetモデルのインスタンス作成
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale
            )
            
            # エクソジェネス変数がある場合のモデルフィッティング
            if exogenous_columns:
                model.add_regressor(exogenous_columns)
                model.fit(df_prophet, exog=exog)
            else:
                model.fit(df_prophet)
            
            # 予測の作成
            future = model.make_future_dataframe(periods=forecast_periods)
            if exogenous_columns:
                future_exog = df_prophet[exogenous_columns].iloc[-forecast_periods:].reset_index(drop=True)
                future = pd.concat([future, future_exog], axis=1)
                forecast = model.predict(future, exog=future_exog)
            else:
                forecast = model.predict(future)
            
            # 予測結果のプロット
            st.subheader("Prophet Forecast")
            fig1 = model.plot(forecast)
            plt.title("Prophet Forecast")
            st.pyplot(fig1)
            
            # コンポーネントのプロット
            st.subheader("Prophet Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
            
            # 予測精度の評価（トレーニングデータの最後の部分をテストデータとして使用）
            st.subheader("Forecast Accuracy Metrics")
            y_true = df_prophet['y'].iloc[-forecast_periods:]
            y_pred = forecast['yhat'].iloc[-forecast_periods:]
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            st.write(f"MSE: {mse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAPE: {mape:.2f}%")
        
        except Exception as e:
            st.error(f"Error during Prophet modeling: {e}")
